"""四元数、齐次变换与 btc_desc ``pose.txt`` 行格式。

约定
----
- 四元数一律 ``(x, y, z, w)``，与 ROS ``geometry_msgs/Quaternion`` 一致；
- 位姿 ``T_parent_child`` 表示把点从 child 系变到 parent 系：
  ``p_parent = R * p_child + t``；
- btc_desc 的 ``pose.txt`` 存 ``T_world_lidar``，列序为
  ``id x y z qx qy qz qw``（w 在最后）。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

_EPS = 1e-12


def normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    """归一化四元数；零范数/NaN 直接报错，避免静默写出缩放矩阵。"""
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape != (4,):
        raise ValueError("四元数必须是 4 个数 (x, y, z, w)，收到 %r" % (q,))
    if not np.all(np.isfinite(q)):
        raise ValueError("四元数含 NaN/Inf: %r" % (q,))
    norm = float(np.linalg.norm(q))
    if norm < _EPS:
        raise ValueError("四元数范数为 0: %r" % (q,))
    return q / norm


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """``(x, y, z, w)`` -> 3x3 旋转矩阵。"""
    x, y, z, w = normalize_quaternion(quaternion)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> ``(x, y, z, w)``（归一化，不保证 w 的符号）。"""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("旋转矩阵必须是 3x3，收到 shape=%s" % (rotation.shape,))
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    return normalize_quaternion([x, y, z, w])


def quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    """Hamilton 积 ``q_left * q_right``（与旋转矩阵复合顺序一致）。"""
    x1, y1, z1, w1 = normalize_quaternion(left)
    x2, y2, z2, w2 = normalize_quaternion(right)
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def compose_transforms(
    t_parent_child: Sequence[float],
    q_parent_child: Sequence[float],
    t_child_grandchild: Sequence[float],
    q_child_grandchild: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """``T_parent_grandchild = T_parent_child @ T_child_grandchild``。"""
    first_t = np.asarray(t_parent_child, dtype=np.float64).reshape(3)
    second_t = np.asarray(t_child_grandchild, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(first_t)) or not np.all(np.isfinite(second_t)):
        raise ValueError("平移量含 NaN/Inf")
    first_rotation = quaternion_to_matrix(q_parent_child)
    translation = first_rotation.dot(second_t) + first_t
    quaternion = normalize_quaternion(
        quaternion_multiply(q_parent_child, q_child_grandchild)
    )
    return translation, quaternion


def pose_to_matrix(translation: Sequence[float], quaternion: Sequence[float]) -> np.ndarray:
    """``(t, q)`` -> 4x4 齐次矩阵。"""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_matrix(quaternion)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """4x4 齐次矩阵 -> ``(t, q)``。"""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("齐次矩阵必须是 4x4，收到 shape=%s" % (transform.shape,))
    return transform[:3, 3].copy(), matrix_to_quaternion(transform[:3, :3])


def format_pose_line(index: int, translation: Sequence[float], quaternion: Sequence[float]) -> str:
    """按 btc_desc 要求写成一行 8 个 token（单空格分隔，w 在最后）。"""
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    q = normalize_quaternion(quaternion)
    if not np.all(np.isfinite(t)):
        raise ValueError("平移量含 NaN/Inf: %r" % (t,))
    return "%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f" % (
        int(index),
        float(t[0]),
        float(t[1]),
        float(t[2]),
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
    )


def parse_pose_line(line: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """解析 ``format_pose_line`` 写出的行；用于自检/单测。"""
    tokens = line.split(" ")
    if len(tokens) != 8 or "" in tokens:
        raise ValueError("pose 行必须恰好 8 个 token 且单空格分隔: %r" % (line,))
    values = [float(token) for token in tokens]
    return int(values[0]), np.asarray(values[1:4]), normalize_quaternion(values[4:8])


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _numbers(text: str) -> List[float]:
    cleaned = text.replace("[", " ").replace("]", " ").replace(",", " ")
    # YAML 列表行的 "-" 是标记，不是数字；负数的 "-1.0" 是一个 token，不受影响。
    return [float(part) for part in cleaned.split() if part != "-"]


def read_lidar_extrinsic(
    path: str,
    section: str = "li_extrinsic",
    rotation_key: str = "R_i_l",
    translation_key: str = "t_i_l",
) -> Tuple[np.ndarray, np.ndarray]:
    """从标定 YAML 读 ``T_i_l``（lidar -> imu/body）。

    只解析 :file:`building.yaml` 这一种结构，不依赖 PyYAML：

    .. code-block:: yaml

        li_extrinsic:
          R_i_l:
            - [1.0, 0.0, 0.0]
            ...
          t_i_l: [0.04165, 0.02326, -0.0284]

    返回 ``(R, t)``，满足 ``p_i = R * p_l + t``。
    """
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    section_index = None
    for index, line in enumerate(lines):
        if _strip_comment(line) == section + ":":
            section_index = index
            break
    if section_index is None:
        raise ValueError("%s 里找不到 %s: 段" % (path, section))

    block_end = len(lines)
    for index in range(section_index + 1, len(lines)):
        stripped = _strip_comment(lines[index])
        if not stripped:
            continue
        if not lines[index][:1].isspace() and stripped.endswith(":"):
            block_end = index
            break
    block = lines[section_index + 1 : block_end]

    rotation = None
    for index, line in enumerate(block):
        stripped = _strip_comment(line)
        if not stripped.startswith(rotation_key + ":"):
            continue
        rest = stripped[len(rotation_key) + 1 :].strip()
        rows: List[List[float]] = []
        if rest:
            values = _numbers(rest)
            if len(values) == 9:
                rotation = np.asarray(values, dtype=np.float64).reshape(3, 3)
            elif len(values) == 3:
                rows.append(values)
            else:
                raise ValueError("%s 的 %s 不是 3x3 矩阵" % (path, rotation_key))
        if rotation is None:
            for follow in block[index + 1 :]:
                follow_text = _strip_comment(follow)
                if not follow_text:
                    continue
                if follow_text.startswith("-"):
                    rows.append(_numbers(follow_text))
                    if len(rows) == 3:
                        break
                else:
                    break
            if len(rows) != 3 or any(len(row) != 3 for row in rows):
                raise ValueError("%s 的 %s 不是 3x3 矩阵" % (path, rotation_key))
            rotation = np.asarray(rows, dtype=np.float64)
        break
    if rotation is None:
        raise ValueError("%s 里找不到 %s" % (path, rotation_key))

    translation = None
    for index, line in enumerate(block):
        stripped = _strip_comment(line)
        if not stripped.startswith(translation_key + ":"):
            continue
        rest = stripped[len(translation_key) + 1 :].strip()
        if not rest:
            for follow in block[index + 1 :]:
                follow_text = _strip_comment(follow)
                if follow_text:
                    rest = follow_text
                    break
        values = _numbers(rest)
        if len(values) != 3:
            raise ValueError("%s 的 %s 必须是 3 个数" % (path, translation_key))
        translation = np.asarray(values, dtype=np.float64)
        break
    if translation is None:
        raise ValueError("%s 里找不到 %s" % (path, translation_key))

    orthonormal_error = float(np.abs(rotation.dot(rotation.T) - np.eye(3)).max())
    determinant = float(np.linalg.det(rotation))
    if orthonormal_error > 1e-3 or determinant <= 0.0:
        raise ValueError(
            "%s 的 %s 不是合法旋转矩阵（|R R^T - I|max=%.3g, det=%.6f）"
            % (path, rotation_key, orthonormal_error, determinant)
        )
    return rotation, translation
