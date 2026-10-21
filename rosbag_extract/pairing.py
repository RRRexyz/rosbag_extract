"""图像 / 点云的时间戳配对。

策略分两步，兼顾「准」和「快」：

1. 用 bag 索引里的记录时间做初选：``nearest_index`` 找到记录时间最接近点云
   时间戳的图像，只读这一条消息（1 次 chunk 读取）。
2. 用消息头 ``header.stamp`` 复核：若与点云时间戳的差值已经小于半个图像
   周期，认为已是最近帧，结束；否则在 ±``radius`` 范围内继续读候选帧，
   取 ``|image_stamp - lidar_stamp|`` 最小者。

由于记录时间与本机传感器时间戳通常只差零点几毫秒，绝大多数帧第 1 步就命中，
只有时间戳异常（例如录制时钟跳变）时才会触发第 2 步的多帧比较。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Tuple

import numpy as np

from .bag_index import BagIndexError, header_stamp, nearest_index


class ImageSource(Protocol):
    """配对需要的两个能力：按话题取时间戳、按索引读消息。

    :class:`rosbag_extract.bag_index.BagIndex` 天然满足；测试里也用它做替身。
    """

    def times(self, topic: str) -> np.ndarray:
        """该话题所有消息的记录时间（秒，升序）。"""
        ...

    def read(self, topic: str, index: int) -> Tuple[Any, float]:
        """按索引读取消息，返回 (msg, 记录时间)。"""
        ...


@dataclass
class ImageMatch:
    """配对结果。"""

    index: int                     # 图像话题内的索引
    stamp: float                   # 图像 header.stamp (s)
    record_time: float             # 图像 bag 记录时间 (s)
    delta_s: float                 # image_stamp - lidar_stamp
    msg: Any = field(repr=False, default=None)
    candidates_checked: int = 1
    stamp_is_fallback: bool = False

    @property
    def delta_ms(self) -> float:
        return self.delta_s * 1000.0


def camera_period(times: np.ndarray) -> float:
    """图像话题的中位周期（秒）；不足两帧时返回 0。"""
    if int(times.shape[0]) < 2:
        return 0.0
    diffs: np.ndarray = np.diff(times)
    diffs = diffs[diffs > 0]
    if int(diffs.shape[0]) == 0:
        return 0.0
    return float(np.median(diffs))


def match_image(
    index: ImageSource,
    image_topic: str,
    lidar_stamp: float,
    radius: int = 2,
    tol_ratio: float = 0.5,
    period: Optional[float] = None,
) -> ImageMatch:
    """为给定点云时间戳找时间上最接近的图像。

    :param index: 提供 ``times`` / ``read`` 的消息源（通常是 ``BagIndex``）
    :param lidar_stamp: 点云 ``header.stamp``（秒）
    :param radius: 复核时向两侧扩展的候选数
    :param tol_ratio: 判定「已足够近」的阈值 = ``tol_ratio * period``
    """
    times = index.times(image_topic)
    count = int(times.shape[0])
    if count == 0:
        raise BagIndexError("图像话题 %s 没有消息" % image_topic)
    if period is None:
        period = camera_period(times)
    tolerance = tol_ratio * period if period > 0 else 0.0

    first = nearest_index(times, lidar_stamp)
    order = [first]
    for offset in range(1, max(1, int(radius)) + 1):
        if first - offset >= 0:
            order.append(first - offset)
        if first + offset < count:
            order.append(first + offset)

    best: Optional[ImageMatch] = None
    checked = 0
    for position in order:
        msg, record_time = index.read(image_topic, position)
        checked += 1
        stamp = header_stamp(msg)
        fallback = stamp is None
        if stamp is None:
            stamp = record_time
        candidate = ImageMatch(
            index=position,
            stamp=stamp,
            record_time=record_time,
            delta_s=stamp - lidar_stamp,
            msg=msg,
            candidates_checked=checked,
            stamp_is_fallback=fallback,
        )
        if best is None or abs(candidate.delta_s) < abs(best.delta_s):
            best = candidate
        if tolerance > 0 and abs(candidate.delta_s) <= tolerance:
            break

    if best is None:  # 理论上 order 非空，这里只是让类型检查器放心
        raise BagIndexError("图像话题 %s 没有可用候选帧" % image_topic)
    best.candidates_checked = checked
    return best
