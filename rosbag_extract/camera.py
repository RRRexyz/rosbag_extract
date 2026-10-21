"""``sensor_msgs/CompressedImage`` 的解码与保存。

保存原则：源数据本来就是 JPEG 时，直接把原始压缩字节写成 .jpg，
避免解压再压缩造成二次损失；只有显式指定 ``quality`` 或源格式不是 JPEG
（例如 png）时才重新编码。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np


def format_string(msg: Any) -> str:
    """``CompressedImage.format``，例如 ``"jpeg"`` 或 ``"bgr8; jpeg compressed bgr8"``。"""
    return str(getattr(msg, "format", "") or "")


def is_jpeg(fmt: str) -> bool:
    low = fmt.lower()
    return "jpeg" in low or "jpg" in low


def to_uint8_buffer(msg: Any) -> np.ndarray:
    """``msg.data``（bytes / array('B') / list）-> uint8 视图。"""
    data = msg.data
    try:
        return np.frombuffer(data, dtype=np.uint8)
    except TypeError:  # 少数实现给出 list
        return np.asarray(data, dtype=np.uint8)


def decode(msg: Any) -> np.ndarray:
    """解码成 numpy 图像（BGR / 灰度 / 原通道数）。"""
    buffer = to_uint8_buffer(msg)
    if buffer.shape[0] == 0:
        raise ValueError("CompressedImage.data 为空")
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(
            "图像解码失败 (format=%r, %d bytes)"
            % (format_string(msg), int(buffer.shape[0]))
        )
    return image


def save_jpeg(
    msg: Any, image: np.ndarray, path: str, quality: Optional[int] = None
) -> str:
    """把图像写成 .jpg，返回写入方式 ``"copy"``（原始字节）或 ``"encode"``。"""
    fmt = format_string(msg)
    if quality is None and is_jpeg(fmt):
        with open(path, "wb") as file:
            file.write(to_uint8_buffer(msg).tobytes())
        return "copy"

    params = []
    if quality is not None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))]
    ok, encoded = cv2.imencode(".jpg", image, params)
    if not ok:
        raise RuntimeError("JPEG 编码失败: %s" % path)
    with open(path, "wb") as file:
        file.write(encoded.tobytes())
    return "encode"


def describe(image: np.ndarray) -> Tuple[int, int, int]:
    """(width, height, channels)"""
    height, width = int(image.shape[0]), int(image.shape[1])
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    return width, height, channels
