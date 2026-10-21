"""``livox_ros_driver/CustomMsg`` -> numpy 点云。

CustomMsg 结构（Livox ROS Driver 1.x）::

    Header header
    uint64  timebase        # 本帧第一个点的时间（ns）
    uint32  point_num
    uint8   lidar_id
    uint8[3] rsvd
    CustomPoint[] points    # 每个点: offset_time, x, y, z, reflectivity, tag, line

其中 ``offset_time`` 是相对 ``timebase`` 的纳秒偏移，所以某点的绝对时间
= ``timebase + offset_time`` (ns)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from .bag_index import header_stamp

#: 写 PCD 用的点云 dtype（x/y/z 与反射强度，均为 float32）
CLOUD_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")]
)


@dataclass
class LidarFrame:
    """一帧点云及其元信息（便于写日志 / json 侧车文件）。"""

    cloud: np.ndarray                      # (N,) structured array, 见 CLOUD_DTYPE
    stamp: float                           # header.stamp (s)
    record_time: float                     # bag 记录时间 (s)
    timebase_ns: int = 0
    lidar_id: int = 0
    frame_id: str = ""
    num_points: int = 0
    dropped_points: int = 0
    offset_time_range_ms: Tuple[float, float] = (0.0, 0.0)
    stamp_is_fallback: bool = False        # header.stamp 非法、退化为记录时间
    arrays: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)


def extract_arrays(msg: Any, drop_invalid: bool = True) -> Dict[str, np.ndarray]:
    """把 CustomMsg 的 ``points`` 拆成 numpy 数组（不复制 msg 本身）。"""
    points = msg.points
    num = len(points)
    x = np.fromiter((p.x for p in points), dtype=np.float32, count=num)
    y = np.fromiter((p.y for p in points), dtype=np.float32, count=num)
    z = np.fromiter((p.z for p in points), dtype=np.float32, count=num)
    intensity = np.fromiter(
        (p.reflectivity for p in points), dtype=np.float32, count=num
    )
    offset = np.fromiter((p.offset_time for p in points), dtype=np.float64, count=num)
    line = np.fromiter((p.line for p in points), dtype=np.int16, count=num)
    tag = np.fromiter((p.tag for p in points), dtype=np.int16, count=num)

    if drop_invalid and num:
        # Livox 对「无回波」点输出 (0, 0, 0)，这类点对建图/投影没有意义，默认剔除。
        valid = ~((x == 0.0) & (y == 0.0) & (z == 0.0))
        valid &= np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        keep = np.flatnonzero(valid)
        x, y, z = x[keep], y[keep], z[keep]
        intensity, offset = intensity[keep], offset[keep]
        line, tag = line[keep], tag[keep]

    return {
        "x": x,
        "y": y,
        "z": z,
        "intensity": intensity,
        "offset_time_ns": offset,
        "line": line,
        "tag": tag,
    }


def build_cloud(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    """按 :data:`CLOUD_DTYPE` 组装结构化点云数组。"""
    num = int(arrays["x"].shape[0])
    cloud = np.empty(num, dtype=CLOUD_DTYPE)
    cloud["x"] = arrays["x"]
    cloud["y"] = arrays["y"]
    cloud["z"] = arrays["z"]
    cloud["intensity"] = arrays["intensity"]
    return cloud


def frame_from_msg(msg: Any, record_time: float, drop_invalid: bool = True) -> LidarFrame:
    """CustomMsg -> :class:`LidarFrame`。"""
    stamp = header_stamp(msg)
    fallback = stamp is None
    if stamp is None:
        stamp = float(record_time)

    arrays = extract_arrays(msg, drop_invalid=drop_invalid)
    cloud = build_cloud(arrays)

    offsets_ms = arrays["offset_time_ns"] / 1e6
    if int(offsets_ms.shape[0]):
        offset_range: Tuple[float, float] = (
            float(offsets_ms.min()),
            float(offsets_ms.max()),
        )
    else:
        offset_range = (0.0, 0.0)

    point_num = int(getattr(msg, "point_num", len(msg.points)))
    return LidarFrame(
        cloud=cloud,
        stamp=stamp,
        record_time=float(record_time),
        timebase_ns=int(getattr(msg, "timebase", 0)),
        lidar_id=int(getattr(msg, "lidar_id", 0)),
        frame_id=str(getattr(getattr(msg, "header", None), "frame_id", "")),
        num_points=int(cloud.shape[0]),
        dropped_points=point_num - int(cloud.shape[0]),
        offset_time_range_ms=offset_range,
        stamp_is_fallback=fallback,
        arrays=arrays,
    )
