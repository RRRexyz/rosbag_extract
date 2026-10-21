"""PCD 文件写出（PCD v0.7，支持 binary / ascii）。

只写「点云」部分：FIELDS 由传入的结构化数组 dtype 决定，
例如 lidar.CLOUD_DTYPE -> ``x y z intensity``。
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

# numpy 类型 -> (SIZE, TYPE) 中的 TYPE
_TYPE_CODE = {
    ("f", 4): "F",
    ("f", 8): "F",
    ("i", 1): "I",
    ("i", 2): "I",
    ("i", 4): "I",
    ("i", 8): "I",
    ("u", 1): "U",
    ("u", 2): "U",
    ("u", 4): "U",
    ("u", 8): "U",
}

DEFAULT_VIEWPOINT = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def _field_names(dtype: np.dtype) -> Tuple[str, ...]:
    names = dtype.names
    if names is None:
        raise ValueError("PCD 需要结构化数组（带字段名）的 dtype")
    return names


def _field_spec(dtype: np.dtype) -> List[Tuple[str, int, str]]:
    """[(field, size, type_code), ...]"""
    spec: List[Tuple[str, int, str]] = []
    for name in _field_names(dtype):
        sub = dtype[name]
        if sub.subdtype is not None:
            raise ValueError("暂不支持子数组字段: %s" % name)
        key = (sub.kind, sub.itemsize)
        if key not in _TYPE_CODE:
            raise ValueError("PCD 不支持的数据类型: %s (%s)" % (name, sub))
        spec.append((name, sub.itemsize, _TYPE_CODE[key]))
    return spec


def header_text(
    dtype: np.dtype,
    num_points: int,
    binary: bool = True,
    viewpoint: Optional[Iterable[float]] = None,
) -> str:
    """生成 PCD 文本头。"""
    spec = _field_spec(dtype)
    fields = " ".join(name for name, _, _ in spec)
    sizes = " ".join(str(size) for _, size, _ in spec)
    types = " ".join(code for _, _, code in spec)
    counts = " ".join("1" for _ in spec)
    vp = DEFAULT_VIEWPOINT if viewpoint is None else tuple(viewpoint)

    lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS %s" % fields,
        "SIZE %s" % sizes,
        "TYPE %s" % types,
        "COUNT %s" % counts,
        "WIDTH %d" % num_points,
        "HEIGHT 1",
        "VIEWPOINT " + " ".join("%g" % value for value in vp),
        "POINTS %d" % num_points,
        "DATA %s" % ("binary" if binary else "ascii"),
    ]
    return "\n".join(lines) + "\n"


def _ascii_format(dtype: np.dtype) -> str:
    formats = []
    for name in _field_names(dtype):
        kind, itemsize = dtype[name].kind, dtype[name].itemsize
        if kind == "f" and itemsize == 4:
            formats.append("%.6f")
        elif kind == "f":
            formats.append("%.12g")
        else:
            formats.append("%d")
    return " ".join(formats)


def write_pcd(
    path: str,
    data: np.ndarray,
    binary: bool = True,
    viewpoint: Optional[Iterable[float]] = None,
) -> int:
    """写 PCD 文件，返回写入的点数。

    ``data`` 必须是 1 维结构化数组，字段名即 PCD 的 FIELDS。
    """
    if data.ndim != 1:
        raise ValueError("write_pcd 需要 1 维结构化数组，收到 shape=%s" % (data.shape,))

    names = _field_names(data.dtype)
    expected = sum(data.dtype[name].itemsize for name in names)
    if data.dtype.itemsize != expected:
        raise ValueError(
            "结构化数组不是紧凑布局（itemsize=%d, 字段和=%d）"
            % (data.dtype.itemsize, expected)
        )

    num_points = int(data.shape[0])
    header = header_text(data.dtype, num_points, binary=binary, viewpoint=viewpoint)

    with open(path, "wb") as file:
        file.write(header.encode("ascii"))
        if num_points == 0:
            return 0
        if binary:
            file.write(np.ascontiguousarray(data).tobytes())
        else:
            columns = np.column_stack([data[name] for name in names])
            np.savetxt(file, columns, fmt=_ascii_format(data.dtype))
    return num_points
