#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 btc_desc 的 PCD 输入数据集是否满足格式要求。

检查项：

- ``velodyne/%06d.pcd`` 是否从 000000 连续、帧数是否与 ``pose.txt`` 行数一致；
- PCD 头（FIELDS/SIZE/TYPE/COUNT/WIDTH/HEIGHT/POINTS/DATA）、binary/ascii 长度、
  x/y/z 是否有限、点数与半径是否合理；
- ``pose.txt`` 是否每行 8 个单空格分隔的 token、四元数是否单位长度、轨迹是否连续；
- ``config.yaml`` 首行是否 ``%YAML:1.0``、btc 关键键是否存在（含拼写坑
  ``max_constrait_dis``）；
- ``run_params.yaml`` 的 ``read_bin`` 是否为 false、路径是否存在；
- ``meta.csv`` 行数是否与帧数一致（可选）。

退出码：0 = 没有 FAIL；1 = 至少一个 FAIL。WARN 不影响退出码。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REQUIRED_CONFIG_KEYS = (
    "useful_corner_num",
    "plane_detection_thre",
    "plane_merge_normal_thre",
    "plane_merge_dis_thre",
    "voxel_size",
    "voxel_init_num",
    "proj_plane_num",
    "proj_image_resolution",
    "proj_image_high_inc",
    "proj_dis_min",
    "proj_dis_max",
    "summary_min_thre",
    "line_filter_enable",
    "descriptor_near_num",
    "descriptor_min_len",
    "descriptor_max_len",
    "max_constrait_dis",
    "triangle_resolution",
    "skip_near_num",
    "candidate_num",
    "similarity_threshold",
    "rough_dis_threshold",
    "normal_threshold",
    "dis_threshold",
    "icp_threshold",
)


class Report:
    def __init__(self) -> None:
        self.errors = 0
        self.warnings = 0

    def section(self, title: str) -> None:
        print("")
        print("[%s]" % title)

    def ok(self, message: str) -> None:
        print("  [ OK ] %s" % message)

    def warn(self, message: str) -> None:
        self.warnings += 1
        print("  [WARN] %s" % message)

    def fail(self, message: str) -> None:
        self.errors += 1
        print("  [FAIL] %s" % message)


def _numpy_dtype(type_name: str, size: int) -> Optional[str]:
    if type_name == "F":
        return {4: "<f4", 8: "<f8"}.get(size)
    if type_name == "I":
        return {1: "i1", 2: "<i2", 4: "<i4", 8: "<i8"}.get(size)
    if type_name == "U":
        return {1: "u1", 2: "<u2", 4: "<u4", 8: "<u8"}.get(size)
    return None


def read_pcd(path: str) -> Tuple[Dict[str, Any], List[str], bytes]:
    """读取 PCD 头 + body，返回 (info, header_lines, body)。"""
    info: Dict[str, Any] = {}
    header_lines: List[str] = []
    with open(path, "rb") as file:
        while True:
            raw_line = file.readline()
            if not raw_line:
                raise ValueError("%s 的 PCD 头不完整（没有 DATA 行）" % path)
            text = raw_line.decode("ascii", "replace").strip()
            header_lines.append(text)
            if not text:
                continue
            parts = text.split()
            key = parts[0].upper()
            if key in ("FIELDS", "SIZE", "TYPE", "COUNT"):
                info[key] = parts[1:]
            elif key in ("VERSION", "WIDTH", "HEIGHT", "POINTS", "VIEWPOINT"):
                info[key] = parts[1:]
            elif key == "DATA":
                info["DATA"] = parts[1].lower() if len(parts) > 1 else ""
                break
        body = file.read()
    return info, header_lines, body


def check_pcd(path: str, report: Report) -> Optional[Dict[str, Any]]:
    name = os.path.basename(path)
    try:
        info, header_lines, body = read_pcd(path)
    except (OSError, ValueError) as exc:
        report.fail("%s: %s" % (name, exc))
        return None

    for key in ("FIELDS", "SIZE", "TYPE", "COUNT", "WIDTH", "HEIGHT", "POINTS", "DATA"):
        if key not in info:
            report.fail("%s: PCD 头缺少 %s" % (name, key))
            return None

    fields = [str(item) for item in info["FIELDS"]]
    try:
        sizes = [int(item) for item in info["SIZE"]]
        counts = [int(item) for item in info["COUNT"]]
    except ValueError:
        report.fail("%s: SIZE/COUNT 不是整数" % name)
        return None
    types = [str(item).upper() for item in info["TYPE"]]
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        report.fail(
            "%s: FIELDS/SIZE/TYPE/COUNT 列数不一致 (%d/%d/%d/%d)"
            % (name, len(fields), len(sizes), len(types), len(counts))
        )
        return None

    lower_fields = [item.lower() for item in fields]
    if not all(key in lower_fields for key in ("x", "y", "z")):
        report.fail("%s: FIELDS 必须包含 x y z，实际 %s" % (name, fields))
        return None
    data_type = str(info.get("DATA", "")).lower()
    if data_type not in ("ascii", "binary"):
        report.fail("%s: DATA 只能是 ascii/binary，实际 %r" % (name, data_type))
        return None

    try:
        width = int(info["WIDTH"][0])
        height = int(info["HEIGHT"][0])
        num_points = int(info["POINTS"][0])
    except (ValueError, IndexError):
        report.fail("%s: WIDTH/HEIGHT/POINTS 格式错误" % name)
        return None
    if width * height != num_points:
        report.fail(
            "%s: WIDTH*HEIGHT(%d*%d) != POINTS(%d)" % (name, width, height, num_points)
        )
    if num_points <= 0:
        report.fail("%s: POINTS=0" % name)
        return None

    if header_lines and not header_lines[0].startswith("# .PCD"):
        report.warn("%s: 第一行不是 '# .PCD ...'" % name)
    version = " ".join(str(item) for item in info.get("VERSION", []))
    if "0.7" not in version:
        report.warn("%s: VERSION 不是 0.7（实际 %r）" % (name, version))

    point_step = sum(size * count for size, count in zip(sizes, counts))
    offsets: List[int] = []
    offset = 0
    for size, count in zip(sizes, counts):
        offsets.append(offset)
        offset += size * count

    try:
        if data_type == "binary":
            need = num_points * point_step
            if len(body) < need:
                report.fail(
                    "%s: binary body 只有 %d 字节，POINTS*point_step=%d"
                    % (name, len(body), need)
                )
                return None
            if len(body) > need:
                report.warn("%s: binary body 比预期多 %d 字节" % (name, len(body) - need))
            formats: List[Any] = []
            for type_name, size, count in zip(types, sizes, counts):
                dtype = _numpy_dtype(type_name, size)
                if dtype is None:
                    report.fail("%s: 不支持的 TYPE/SIZE %s/%d" % (name, type_name, size))
                    return None
                formats.append((dtype, (count,)) if count > 1 else dtype)
            dtype = np.dtype(
                {
                    "names": fields,
                    "formats": formats,
                    "offsets": offsets,
                    "itemsize": point_step,
                }
            )
            points = np.frombuffer(body[:need], dtype=dtype, count=num_points)
            scalars = {
                field.lower(): points[field]
                for field in fields
            }
        else:
            text = body.decode("ascii", "replace").replace("\r", " ").replace("\n", " ")
            values = np.fromstring(text, sep=" ")
            expected = num_points * sum(counts)
            if values.size < expected:
                report.fail(
                    "%s: ascii 只有 %d 个数，需要 %d" % (name, values.size, expected)
                )
                return None
            if values.size > expected:
                report.warn("%s: ascii 比预期多 %d 个数" % (name, values.size - expected))
            table = values[:expected].reshape(num_points, sum(counts))
            scalar_offsets: List[int] = []
            scalar_offset = 0
            for count in counts:
                scalar_offsets.append(scalar_offset)
                scalar_offset += count
            scalars = {
                field.lower(): table[:, scalar_offsets[index]]
                for index, field in enumerate(fields)
            }
    except (ValueError, TypeError) as exc:
        report.fail("%s: 解析 body 失败: %s" % (name, exc))
        return None

    for axis in ("x", "y", "z"):
        index = lower_fields.index(axis)
        value = scalars[axis]
        if value.ndim != 1:
            report.fail("%s: 字段 %s 的 COUNT>1，btc 只支持标量" % (name, axis))
            return None
        if not np.all(np.isfinite(value.astype(np.float64))):
            report.fail("%s: %s 含 NaN/Inf" % (name, axis))
            return None
        if counts[index] != 1:
            report.fail("%s: %s 的 COUNT=%d" % (name, axis, counts[index]))
            return None

    x = scalars["x"].astype(np.float64)
    y = scalars["y"].astype(np.float64)
    z = scalars["z"].astype(np.float64)
    radius = np.sqrt(x * x + y * y + z * z)
    max_radius = float(radius.max())
    if max_radius < 1.0:
        report.warn("%s: 最远点只有 %.3f m，可能点云不在米制/传感器系" % (name, max_radius))
    if num_points < 1000:
        report.warn("%s: 只有 %d 点，btc 建议 1e4~1e5" % (name, num_points))

    return {
        "name": name,
        "points": num_points,
        "fields": tuple(lower_fields),
        "point_step": point_step,
        "data_type": data_type,
        "max_radius": max_radius,
        "bbox": (
            (float(x.min()), float(x.max())),
            (float(y.min()), float(y.max())),
            (float(z.min()), float(z.max())),
        ),
    }


def read_pose(path: str, report: Report) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    poses: List[Tuple[int, np.ndarray, np.ndarray]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, 1):
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line.strip():
                report.warn("%s: 第 %d 行是空行（btc 会忽略）" % (os.path.basename(path), line_number))
                continue
            tokens = line.split(" ")
            if len(tokens) != 8 or any(token == "" for token in tokens):
                report.fail(
                    "%s: 第 %d 行不是 8 个单空格分隔的 token（实际 %d 个）"
                    % (os.path.basename(path), line_number, len(tokens))
                )
                continue
            try:
                values = [float(token) for token in tokens]
            except ValueError:
                report.fail("%s: 第 %d 行含非数字 token" % (os.path.basename(path), line_number))
                continue
            quaternion = np.asarray(values[4:8], dtype=np.float64)
            norm = float(np.linalg.norm(quaternion))
            if not np.isfinite(norm) or abs(norm - 1.0) > 1e-3:
                report.fail(
                    "%s: 第 %d 行四元数范数 %.6f，不是单位四元数"
                    % (os.path.basename(path), line_number, norm)
                )
                continue
            index = int(values[0])
            if index != len(poses):
                report.warn(
                    "%s: 第 %d 行首列是 %d，期望 %d（btc 不使用该列，仅提示）"
                    % (os.path.basename(path), line_number, index, len(poses))
                )
            poses.append(
                (
                    index,
                    np.asarray(values[1:4], dtype=np.float64),
                    quaternion / norm,
                )
            )
    return poses


def list_pcds(pcd_dir: str, report: Report) -> List[str]:
    if not os.path.isdir(pcd_dir):
        report.fail("PCD 目录不存在: %s" % pcd_dir)
        return []
    names = sorted(name for name in os.listdir(pcd_dir) if name.endswith(".pcd"))
    if not names:
        report.fail("PCD 目录里没有 .pcd: %s" % pcd_dir)
        return []
    indices: List[int] = []
    for name in names:
        stem = name[:-4]
        if len(stem) != 6 or not stem.isdigit():
            report.fail("PCD 文件名不是 6 位数字: %s" % name)
            continue
        indices.append(int(stem))
    if not indices:
        return names
    if min(indices) != 0:
        report.fail("PCD 编号不是从 000000 开始（最小 %06d）" % min(indices))
    if len(set(indices)) != len(indices):
        report.fail("PCD 编号有重复")
    missing = sorted(set(range(max(indices) + 1)) - set(indices))
    if missing:
        report.fail(
            "PCD 编号不连续，缺 %d 个（前几个: %s）"
            % (len(missing), ", ".join("%06d" % item for item in missing[:10]))
        )
    return names


def check_config(path: str, frame_count: int, report: Report) -> None:
    if not os.path.isfile(path):
        report.fail("config.yaml 不存在: %s" % path)
        return
    with open(path, "r", encoding="utf-8") as file:
        lines = file.read().splitlines()
    if not lines or lines[0].strip() != "%YAML:1.0":
        report.fail("config.yaml 首行必须是 %%YAML:1.0（实际 %r）"
                    % (lines[0] if lines else ""))
        return
    values: Dict[str, float] = {}
    for line in lines[1:]:
        text = line.split("#", 1)[0].strip()
        if not text or ":" not in text:
            continue
        key, _, raw_value = text.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        try:
            values[key] = float(raw_value)
        except ValueError:
            report.fail("config.yaml 的 %s 不是数值: %r" % (key, raw_value))
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in values]
    if missing:
        report.fail("config.yaml 缺少关键键: %s" % ", ".join(missing))
    if "max_constraint_dis" in values:
        report.warn(
            "config.yaml 出现 max_constraint_dis；btc 源码用的是拼写错误的 "
            "max_constrait_dis，这个键不会生效"
        )
    for key in ("voxel_size", "summary_min_thre", "descriptor_near_num",
                "candidate_num", "proj_image_resolution"):
        if key in values and values[key] <= 0.0:
            report.fail("config.yaml 的 %s=%g，必须 > 0" % (key, values[key]))
    if frame_count > 0 and "skip_near_num" in values:
        skip = int(values["skip_near_num"])
        if 0 < skip >= frame_count:
            report.warn(
                "skip_near_num=%d >= 帧数 %d，可能选不到回环候选" % (skip, frame_count)
            )
    report.ok(
        "config.yaml: %%YAML:1.0，%d 个键（key 拼写 max_constrait_dis 已检查）"
        % len(values)
    )


def read_key_values(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file.read().splitlines():
            text = line.split("#", 1)[0].strip()
            if not text or ":" not in text:
                continue
            key, _, value = text.partition(":")
            values[key.strip()] = value.strip()
    return values


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            return str(json.loads(value))
        except ValueError:
            return value[1:-1]
    return value


def check_run_params(path: str, data_dir: Optional[str], report: Report) -> None:
    if not os.path.isfile(path):
        report.warn("run_params.yaml 不存在（btc 本身不需要它，只是少了一键加载）: %s" % path)
        return
    values = read_key_values(path)
    pcds_dir = _unquote(values.get("pcds_dir", ""))
    pose_file = _unquote(values.get("pose_file", ""))
    setting_path = _unquote(values.get("setting_path", ""))
    read_bin = values.get("read_bin", "").strip().lower()
    overlap = values.get("cloud_overlap_thr", "")
    if not pcds_dir or not os.path.isdir(pcds_dir):
        report.fail("run_params.yaml 的 pcds_dir 不存在: %r" % pcds_dir)
    if not pose_file or not os.path.isfile(pose_file):
        report.fail("run_params.yaml 的 pose_file 不存在: %r" % pose_file)
    if not setting_path or not os.path.isfile(setting_path):
        report.fail("run_params.yaml 的 setting_path 不存在: %r" % setting_path)
    if read_bin not in ("false", "0", "no"):
        report.fail("run_params.yaml 的 read_bin=%r，PCD 模式必须是 false" % read_bin)
    try:
        overlap_value = float(overlap)
        if not 0.0 < overlap_value <= 1.0:
            report.warn("cloud_overlap_thr=%g 不在 (0,1] 内" % overlap_value)
    except ValueError:
        report.fail("run_params.yaml 的 cloud_overlap_thr 不是数值: %r" % overlap)
    if data_dir:
        expected = {
            "pcds_dir": os.path.join(data_dir, "velodyne"),
            "pose_file": os.path.join(data_dir, "pose.txt"),
            "setting_path": os.path.join(data_dir, "config.yaml"),
        }
        actual = {
            "pcds_dir": pcds_dir,
            "pose_file": pose_file,
            "setting_path": setting_path,
        }
        for key, expected_path in expected.items():
            if os.path.realpath(actual[key]) != os.path.realpath(expected_path):
                report.warn(
                    "run_params.yaml 的 %s 指向 %s，不是数据集里的 %s"
                    % (key, actual[key], expected_path)
                )
    report.ok("run_params.yaml: read_bin=false，路径检查完成")


def check_meta(path: str, frame_count: int, report: Report) -> None:
    if not os.path.isfile(path):
        report.warn("meta.csv 不存在（可选）: %s" % path)
        return
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if len(rows) != frame_count:
        report.fail("meta.csv 有 %d 行，pose/PCD 有 %d 帧" % (len(rows), frame_count))
        return
    if not rows:
        return
    required = {"frame_index", "lidar_index", "lidar_stamp", "odom_index", "pcd_file"}
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        report.fail("meta.csv 缺少列: %s" % ", ".join(missing))
        return
    for expected, row in enumerate(rows):
        try:
            actual = int(row["frame_index"])
        except (TypeError, ValueError):
            report.fail("meta.csv 第 %d 行 frame_index 不是整数" % (expected + 2))
            break
        if actual != expected:
            report.fail(
                "meta.csv 第 %d 行 frame_index=%d，期望 %d"
                % (expected + 2, actual, expected)
            )
            break
    report.ok("meta.csv: %d 行，frame_index 连续" % len(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_btc_input.py",
        description="检查 btc_desc 的 PCD 输入格式（帧连续性/PCD 头/pose/config）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例::

    python3 check_btc_input.py --data btc_input/building03
    python3 check_btc_input.py --data btc_input/building03 --sample 50
    python3 check_btc_input.py --pcds-dir .../velodyne --pose-file .../pose.txt
""",
    )
    parser.add_argument("--data", help="数据集根目录（含 velodyne/、pose.txt、config.yaml）")
    parser.add_argument("--pcds-dir", help="PCD 目录（默认 <data>/velodyne）")
    parser.add_argument("--pose-file", help="pose.txt（默认 <data>/pose.txt）")
    parser.add_argument("--config", help="config.yaml（默认 <data>/config.yaml）")
    parser.add_argument("--run-params", help="run_params.yaml（默认 <data>/run_params.yaml）")
    parser.add_argument("--meta", help="meta.csv（默认 <data>/meta.csv）")
    parser.add_argument(
        "--sample", type=int, default=20, help="抽查多少个 PCD（默认 20，0=全部）"
    )
    parser.add_argument(
        "--max-jump", type=float, default=5.0, help="相邻位姿最大位移，米（默认 5.0）"
    )
    parser.add_argument("--quiet", action="store_true", help="只输出最终结果")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = Report()
    data_dir = os.path.abspath(args.data) if args.data else None
    pcds_dir = os.path.abspath(args.pcds_dir) if args.pcds_dir else (
        os.path.join(data_dir, "velodyne") if data_dir else None
    )
    pose_file = os.path.abspath(args.pose_file) if args.pose_file else (
        os.path.join(data_dir, "pose.txt") if data_dir else None
    )
    config_path = os.path.abspath(args.config) if args.config else (
        os.path.join(data_dir, "config.yaml") if data_dir else None
    )
    run_params_path = os.path.abspath(args.run_params) if args.run_params else (
        os.path.join(data_dir, "run_params.yaml") if data_dir else None
    )
    meta_path = os.path.abspath(args.meta) if args.meta else (
        os.path.join(data_dir, "meta.csv") if data_dir else None
    )
    if not pcds_dir or not pose_file:
        print("错误: 至少给 --data，或同时给 --pcds-dir 与 --pose-file", file=sys.stderr)
        return 2
    if args.sample < 0:
        print("错误: --sample 必须 >= 0", file=sys.stderr)
        return 2

    print("检查 btc_desc PCD 输入")
    print("  PCD 目录 : %s" % pcds_dir)
    print("  pose     : %s" % pose_file)
    print("  config   : %s" % (config_path or "(未指定)"))
    print("  抽检帧数 : %s" % ("全部" if args.sample == 0 else args.sample))

    report.section("PCD 文件")
    names = list_pcds(pcds_dir, report)
    if names:
        report.ok("PCD 文件: %d 个，编号连续" % len(names))

    report.section("pose.txt")
    if not os.path.isfile(pose_file):
        report.fail("pose.txt 不存在: %s" % pose_file)
        poses: List[Tuple[int, np.ndarray, np.ndarray]] = []
    else:
        poses = read_pose(pose_file, report)
        if not poses:
            report.fail("pose.txt 没有有效位姿")
        else:
            quaternions = np.asarray([item[2] for item in poses])
            positions = np.asarray([item[1] for item in poses])
            norm_error = float(np.abs(np.linalg.norm(quaternions, axis=1) - 1.0).max())
            if norm_error > 1e-3:
                report.fail("pose.txt 四元数最大范数误差 %.3g > 1e-3" % norm_error)
            if len(poses) >= 2:
                steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                total_length = float(steps.sum())
                max_step = float(steps.max())
                if total_length < 1e-6:
                    report.fail("pose.txt 所有位姿相同，轨迹长度为 0")
                elif max_step > args.max_jump:
                    report.warn(
                        "pose.txt 相邻位姿最大位移 %.3f m > %.1f m，确认没有跳变"
                        % (max_step, args.max_jump)
                    )
                z_values = positions[:, 2]
                if np.allclose(z_values, z_values[0], atol=1e-9) and abs(float(z_values[0])) > 5.0:
                    report.warn(
                        "pose.txt 的 z 恒定且 |z|=%.3f > 5，确认单位是米而不是毫米"
                        % abs(float(z_values[0]))
                    )
                report.ok(
                    "pose.txt: %d 行，四元数最大误差 %.2g，轨迹 %.2f m，最大步长 %.3f m"
                    % (len(poses), norm_error, total_length, max_step)
                )
            else:
                report.ok("pose.txt: 1 行")

    report.section("帧数一致性")
    if names and poses:
        if len(names) != len(poses):
            report.fail(
                "PCD 帧数 %d != pose 行数 %d；btc 会按较小值处理"
                % (len(names), len(poses))
            )
        else:
            report.ok("PCD 帧数 == pose 行数 == %d" % len(names))

    report.section("PCD 内容抽检")
    if names:
        if args.sample == 0 or args.sample >= len(names):
            chosen = names
        else:
            step = max(1, len(names) // args.sample)
            chosen = names[::step][: args.sample]
            first_last = {names[0], names[-1]}
            chosen = sorted(set(chosen) | first_last)
        stats: List[Dict[str, Any]] = []
        for name in chosen:
            result = check_pcd(os.path.join(pcds_dir, name), report)
            if result is not None:
                stats.append(result)
        if stats:
            point_counts = [item["points"] for item in stats]
            layouts = {item["fields"] for item in stats}
            if len(layouts) > 1:
                report.warn("抽检到的 PCD 字段布局不一致: %s" % sorted(layouts))
            if min(point_counts) < 1000:
                report.warn(
                    "抽检最少点数 %d，btc 建议 1e4~1e5" % min(point_counts)
                )
            if min(item["max_radius"] for item in stats) < 1.0:
                report.warn("抽检到最远点 < 1 m 的帧，确认点云是传感器系/米制")
            report.ok(
                "PCD 抽检: %d 帧，点数 %d ~ %d（平均 %.0f），字段 %s"
                % (
                    len(stats),
                    min(point_counts),
                    max(point_counts),
                    float(np.mean(point_counts)),
                    ", ".join(layouts.pop()) if len(layouts) == 1 else "(多种)",
                )
            )

    report.section("config.yaml")
    if config_path:
        check_config(config_path, len(poses), report)
    else:
        report.warn("未指定 config.yaml，跳过检查")

    report.section("run_params.yaml")
    if run_params_path:
        check_run_params(run_params_path, data_dir, report)
    else:
        report.warn("未指定 run_params.yaml，跳过检查")

    report.section("meta.csv")
    if meta_path:
        check_meta(meta_path, len(poses), report)
    else:
        report.warn("未指定 meta.csv，跳过检查")

    print("")
    if report.errors:
        print("结果: %d 个 FAIL，%d 个 WARN" % (report.errors, report.warnings))
        return 1
    print("结果: 全部通过（%d 个 WARN）" % report.warnings)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        sys.exit(130)
