"""rosbag 索引封装。

两项能力：

1. 不读消息内容，只靠 bag 的索引记录拿到每个话题所有消息的时间戳；
2. 通过 ``(chunk_pos, offset)`` 直接读取指定索引的一条消息，不用从头扫描。

这样即使 bag 有 8 GB，取一帧也只需要读 1~2 个 chunk。

``rosbag`` 在反序列化时会先用 ``genpy.dynamic`` 把 bag 内嵌的消息定义
（connection 记录里的 message_definition 字段）动态生成 Python 类，
所以本机没装 ``livox_ros_driver`` 也能正常读出 ``livox_ros_driver/CustomMsg``。
"""

from __future__ import annotations

import bisect
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rosbag


class BagIndexError(RuntimeError):
    """bag 打不开 / 没有索引 / 话题不存在等错误。"""


class BagIndex:
    """只读的 bag 索引访问器，可用作上下文管理器。"""

    def __init__(self, bag_path: str) -> None:
        self.bag_path = str(bag_path)
        try:
            self.bag = rosbag.Bag(self.bag_path, "r")
        except Exception as exc:  # noqa: BLE001 - 统一转成自己的异常
            raise BagIndexError("无法打开 bag: %s (%s)" % (self.bag_path, exc)) from exc

        self._entries: Dict[str, List[Any]] = {}
        self._times: Dict[str, np.ndarray] = {}
        self._indexed = False
        self._read_index()

    # ------------------------------------------------------------------ 索引
    def _read_index(self) -> None:
        """把 bag 末尾的索引记录读进内存（3529 个 chunk 只需几十毫秒）。"""
        if self._indexed:
            return
        if getattr(self.bag, "_connection_indexes_read", False):
            self._indexed = True
            return

        # rosbag 的公开 API 没有「只读索引」的入口，这里走内部方法；
        # 用 getattr 取，既能兼容不同版本的实现，也避免把私有成员写进类型检查。
        reader = getattr(self.bag, "_reader", None)
        read_index = getattr(reader, "_read_connection_index_records", None)
        if not callable(read_index):
            raise BagIndexError(
                "bag 没有索引（unindexed），请先执行 `rosbag reindex %s`" % self.bag_path
            )
        try:
            read_index()
        except Exception as exc:  # noqa: BLE001
            raise BagIndexError("读取 bag 索引失败: %s" % exc) from exc

        if not getattr(self.bag, "_connection_indexes", None):
            raise BagIndexError("bag 索引为空: %s" % self.bag_path)
        self._indexed = True

    def close(self) -> None:
        self.bag.close()

    def __enter__(self) -> "BagIndex":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -------------------------------------------------------------- 基础信息
    @property
    def start_time(self) -> float:
        """bag 记录时间的最小值（本 bag 因前几帧时间戳为 0 而显示为 0）。"""
        return float(self.bag.get_start_time())

    @property
    def end_time(self) -> float:
        return float(self.bag.get_end_time())

    def topic_info(self) -> Dict[str, Tuple[str, int, float]]:
        """{topic: (消息类型, 消息数, 频率 Hz)}"""
        info = self.bag.get_type_and_topic_info()
        return {
            topic: (value.msg_type, value.message_count, value.frequency)
            for topic, value in info.topics.items()
        }

    def connection(self, topic: str) -> Any:
        """该话题的连接信息（rosbag ConnectionInfo）。"""
        for conn in self.bag._connections.values():  # noqa: SLF001
            if conn.topic == topic:
                return conn
        known = ", ".join(sorted(c.topic for c in self.bag._connections.values()))  # noqa: SLF001
        raise BagIndexError("bag 中不存在话题 %r；现有话题: %s" % (topic, known))

    # ------------------------------------------------------------ 时间戳/读取
    def entries(self, topic: str) -> List[Any]:
        """该话题按时间排序的索引条目列表（rosbag IndexEntry）。"""
        if topic not in self._entries:
            conn = self.connection(topic)
            entries: List[Any] = list(self.bag._connection_indexes.get(conn.id, []))  # noqa: SLF001
            entries.sort(key=lambda entry: entry.time.to_nsec())
            self._entries[topic] = entries
            self._times[topic] = np.asarray(
                [entry.time.to_sec() for entry in entries], dtype=np.float64
            )
        return self._entries[topic]

    def times(self, topic: str) -> np.ndarray:
        """该话题所有消息的 bag 记录时间（秒，float64，升序）。"""
        if topic not in self._times:
            self.entries(topic)
        return self._times[topic]

    def count(self, topic: str) -> int:
        return len(self.entries(topic))

    def read(self, topic: str, index: int) -> Tuple[Any, float]:
        """按话题内的索引读取一条消息。

        返回 ``(msg, bag 记录时间)``；``msg`` 已由 rosbag 反序列化完成。
        """
        entries = self.entries(topic)
        if not -len(entries) <= index < len(entries):
            raise IndexError(
                "话题 %s 只有 %d 条消息，索引 %d 越界" % (topic, len(entries), index)
            )
        entry = entries[index]
        bag_msg = self.bag._read_message((entry.chunk_pos, entry.offset))  # noqa: SLF001
        return bag_msg.message, float(entry.time.to_sec())

    def nearest(self, topic: str, stamp: float) -> int:
        """返回 bag 记录时间最接近 ``stamp`` 的消息索引。"""
        times = self.times(topic)
        if times.shape[0] == 0:
            raise BagIndexError("话题 %s 没有消息" % topic)
        return nearest_index(times, stamp)


def nearest_index(times: np.ndarray, stamp: float) -> int:
    """在升序时间数组里找最接近 ``stamp`` 的位置。"""
    position = int(bisect.bisect_left(times, stamp))
    if position <= 0:
        return 0
    if position >= len(times):
        return len(times) - 1
    before, after = times[position - 1], times[position]
    return position - 1 if abs(before - stamp) <= abs(after - stamp) else position


def header_stamp(msg: Any) -> Optional[float]:
    """取消息头时间戳（秒）；没有 header 或时间戳非法时返回 None。"""
    header = getattr(msg, "header", None)
    if header is None:
        return None
    try:
        stamp = float(header.stamp.to_sec())
    except Exception:  # noqa: BLE001
        return None
    return stamp if stamp > 0.0 else None
