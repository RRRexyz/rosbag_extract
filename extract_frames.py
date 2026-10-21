#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 rosbag 中提取时间戳最接近的「图像 + 点云」对，分别保存为 jpg 与 pcd。

用法示例::

    python3 extract_frames.py                          # 默认 bag，取第 0 帧
    python3 extract_frames.py --frame-index 12         # 指定帧
    python3 extract_frames.py --num-frames 5           # 全程均匀取 5 帧
    python3 extract_frames.py --timestamp 1760225200.5 # 取离该时刻最近的一帧
    python3 extract_frames.py --all                    # 全部帧（3496 对，较慢）
    python3 extract_frames.py --list-topics            # 只看 bag 里有什么
    python3 extract_frames.py --dry-run --num-frames 5 # 只算时间差，不写文件

输出目录（默认 ``<本项目>/output``，可用 ``--output-dir`` 修改）::

    output/
    ├── images/frame_000000.jpg    # 图像
    ├── pcd/frame_000000.pcd       # 点云（x y z intensity, binary）
    ├── meta/frame_000000.json     # 该帧的完整时间戳信息
    └── pairs.csv                  # 所有帧的配对表
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from rosbag_extract import camera, lidar, pairing, pcd
from rosbag_extract.bag_index import BagIndex, BagIndexError, nearest_index

DEFAULT_BAG = "/home/rex/data/shitong/campus/campus1.bag"
DEFAULT_LIDAR_TOPIC = "/livox/lidar"
DEFAULT_IMAGE_TOPIC = "/camera/image/compressed"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

EPILOG = """用法示例::

    python3 extract_frames.py                          # 默认 bag，取第 0 帧
    python3 extract_frames.py --frame-index 12         # 指定帧
    python3 extract_frames.py --num-frames 5           # 全程均匀取 5 帧
    python3 extract_frames.py --timestamp 1760225200.5 # 取离该时刻最近的一帧
    python3 extract_frames.py --all                    # 全部帧（3496 对，较慢）
    python3 extract_frames.py --list-topics            # 只看 bag 里有什么
    python3 extract_frames.py --dry-run --num-frames 5 # 只算时间差，不写文件

输出目录（默认 <本项目>/output，可用 --output-dir 修改）::

    output/
    ├── images/frame_000000.jpg    # 图像
    ├── pcd/frame_000000.pcd       # 点云（x y z intensity, binary）
    ├── meta/frame_000000.json     # 该帧的完整时间戳信息
    └── pairs.csv                  # 所有帧的配对表
"""

CSV_FIELDS = [
    "frame_index",
    "lidar_record_time",
    "lidar_stamp",
    "image_record_time",
    "image_stamp",
    "delta_ms",
    "num_points",
    "image_file",
    "pcd_file",
]


# --------------------------------------------------------------------- 参数
def _int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_frames.py",
        description="从 rosbag 中提取时间戳最接近的图像(jpg)与点云(pcd)对",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument(
        "--bag", default=DEFAULT_BAG, help="bag 路径（默认: %(default)s）"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_DIR, "output"),
        help="输出目录（默认: <项目>/output）",
    )
    parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC, help="点云话题")
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC, help="图像话题")

    selection = parser.add_argument_group("帧选择")
    selection.add_argument("--frame-index", help="点云帧序号，可逗号分隔，如 0,120,240")
    selection.add_argument("--timestamp", help="绝对时间戳(秒)，取最近帧；可逗号分隔")
    selection.add_argument(
        "--num-frames", type=int, default=1, help="在全程均匀取多少帧（默认 1）"
    )
    selection.add_argument("--all", action="store_true", help="提取全部有效帧")

    output = parser.add_argument_group("输出")
    output.add_argument(
        "--pcd-format", choices=("binary", "ascii"), default="binary", help="PCD 编码"
    )
    output.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="重编码 JPEG 的质量 1-100；不给则直接拷贝原始 JPEG 字节（无二次损失）",
    )
    output.add_argument(
        "--keep-zero-points", action="store_true", help="保留 Livox 的 (0,0,0) 无效点"
    )
    output.add_argument(
        "--skip-existing", action="store_true", help="jpg 与 pcd 都已存在时跳过该帧"
    )

    misc = parser.add_argument_group("其他")
    misc.add_argument(
        "--search-radius",
        type=int,
        default=2,
        help="时间戳复核时向两侧扩展的候选图像数（默认 2）",
    )
    misc.add_argument("--dry-run", action="store_true", help="只计算配对，不写文件")
    misc.add_argument("--list-topics", action="store_true", help="列出 bag 话题后退出")
    misc.add_argument("--quiet", action="store_true", help="只输出最终汇总")
    return parser


# ----------------------------------------------------------------- 帧选择
def select_positions(times: np.ndarray, args: argparse.Namespace) -> List[int]:
    """返回所选点云帧在「有效帧列表」中的位置，按时间升序。"""
    count = int(times.shape[0])
    if count == 0:
        raise BagIndexError("点云话题没有有效帧")

    picks: List[int] = []
    if args.all:
        picks = list(range(count))
    elif args.frame_index:
        for index in _int_list(args.frame_index):
            if not 0 <= index < count:
                raise SystemExit(
                    "帧序号 %d 超出范围：有效点云帧共 %d 帧（0 ~ %d）"
                    % (index, count, count - 1)
                )
            picks.append(index)
    elif args.timestamp:
        for stamp in _float_list(args.timestamp):
            picks.append(nearest_index(times, stamp))
    else:
        wanted = max(1, int(args.num_frames))
        if wanted >= count:
            picks = list(range(count))
        elif wanted == 1:
            picks = [0]
        else:
            picks = sorted(
                {int(round(v)) for v in np.linspace(0.0, count - 1.0, wanted)}
            )

    seen = set()
    ordered: List[int] = []
    for pick in picks:
        if pick not in seen:
            seen.add(pick)
            ordered.append(pick)
    return ordered


# ------------------------------------------------------------------ 工具
def _fmt_stamp(value: float) -> str:
    return "%.6f" % value


def _print_topic_table(index: BagIndex) -> None:
    info = index.topic_info()
    print(
        "%-30s %-34s %8s %8s %18s %18s"
        % ("topic", "type", "count", "freq", "first", "last")
    )
    print("-" * 122)
    for topic in sorted(info):
        msg_type, count, frequency = info[topic]
        times = index.times(topic)
        first = float(times[0]) if times.shape[0] else float("nan")
        last = float(times[-1]) if times.shape[0] else float("nan")
        print(
            "%-30s %-34s %8d %8.2f %18.6f %18.6f"
            % (topic, msg_type, count, frequency, first, last)
        )
    print("-" * 122)
    print("提示: bag 记录时间（first/last）取自索引；前几帧若为 0 表示录制时钟异常。")


def _dump_csv(output_dir: str) -> int:
    """扫描 meta/*.json，重写 pairs.csv（支持多次运行 / 断点续提）。"""
    meta_dir = os.path.join(output_dir, "meta")
    rows: List[Dict[str, Any]] = []
    if os.path.isdir(meta_dir):
        for name in sorted(os.listdir(meta_dir)):
            if not (name.startswith("frame_") and name.endswith(".json")):
                continue
            with open(os.path.join(meta_dir, name), "r", encoding="utf-8") as file:
                meta = json.load(file)
            rows.append(
                {
                    "frame_index": meta["frame_index"],
                    "lidar_record_time": "%.6f" % meta["lidar"]["record_time"],
                    "lidar_stamp": "%.9f" % meta["lidar"]["header_stamp"],
                    "image_record_time": "%.6f" % meta["image"]["record_time"],
                    "image_stamp": "%.9f" % meta["image"]["header_stamp"],
                    "delta_ms": "%+.3f" % meta["delta_ms"],
                    "num_points": meta["lidar"]["num_points"],
                    "image_file": meta["files"]["image"],
                    "pcd_file": meta["files"]["pcd"],
                }
            )
    rows.sort(key=lambda row: int(row["frame_index"]))
    path = os.path.join(output_dir, "pairs.csv")
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


# -------------------------------------------------------------------- 主流程
def run(args: argparse.Namespace) -> int:
    with BagIndex(args.bag) as index:
        if args.list_topics:
            _print_topic_table(index)
            return 0

        all_lidar_times = index.times(args.lidar_topic)
        image_times = index.times(args.image_topic)

        # 剔除记录时间为 0 的无效帧（本 bag 开头有 4 帧空点云）
        valid_mask = all_lidar_times > 0.0
        entry_positions = np.flatnonzero(valid_mask)
        times = all_lidar_times[valid_mask]
        if times.shape[0] == 0:
            raise BagIndexError("点云话题 %s 没有有效帧" % args.lidar_topic)

        positions = select_positions(times, args)
        period = pairing.camera_period(image_times)
        warn_threshold = 0.5 * period if period > 0 else 0.0

        images_dir = os.path.join(args.output_dir, "images")
        pcd_dir = os.path.join(args.output_dir, "pcd")
        meta_dir = os.path.join(args.output_dir, "meta")
        if not args.dry_run:
            for directory in (images_dir, pcd_dir, meta_dir):
                os.makedirs(directory, exist_ok=True)

        print("bag        : %s" % args.bag)
        print(
            "点云/图像  : %s (%d 帧)  <->  %s (%d 帧)"
            % (
                args.lidar_topic,
                int(times.shape[0]),
                args.image_topic,
                int(image_times.shape[0]),
            )
        )
        print(
            "有效点云帧 : %d（已跳过记录时间无效帧 %d 帧）"
            % (int(times.shape[0]), int(all_lidar_times.shape[0] - times.shape[0]))
        )
        print(
            "图像周期   : %.4f s   本次提取 %d 帧%s"
            % (period, len(positions), "（dry-run）" if args.dry_run else "")
        )
        print("")

        verbose = (not args.quiet) and len(positions) <= 25
        if verbose:
            header = "%-7s %-19s %-19s %10s %8s  %s" % (
                "frame",
                "lidar_stamp",
                "image_stamp",
                "dt[ms]",
                "points",
                "files",
            )
            print(header)
            print("-" * len(header))
        step = max(1, len(positions) // 20) if positions else 1

        rows: List[Dict[str, Any]] = []
        skipped = 0
        deltas: List[float] = []

        for order, position in enumerate(positions):
            frame_index = int(position)
            stem = "frame_%06d" % frame_index
            image_path = os.path.join(images_dir, stem + ".jpg")
            pcd_path = os.path.join(pcd_dir, stem + ".pcd")
            if (
                args.skip_existing
                and not args.dry_run
                and os.path.exists(image_path)
                and os.path.exists(pcd_path)
            ):
                skipped += 1
                if not args.quiet:
                    print("%-7s 已存在，跳过" % stem)
                continue

            lidar_msg, lidar_record = index.read(
                args.lidar_topic, int(entry_positions[position])
            )
            frame = lidar.frame_from_msg(
                lidar_msg, lidar_record, drop_invalid=not args.keep_zero_points
            )
            if frame.num_points == 0:
                skipped += 1
                print("%-7s 空点云（point_num=0），跳过" % stem, file=sys.stderr)
                continue

            match = pairing.match_image(
                index,
                args.image_topic,
                frame.stamp,
                radius=args.search_radius,
                period=period,
            )
            image = camera.decode(match.msg)

            if not args.dry_run:
                camera.save_jpeg(
                    match.msg, image, image_path, quality=args.jpeg_quality
                )
                pcd.write_pcd(
                    pcd_path, frame.cloud, binary=(args.pcd_format == "binary")
                )
                width, height, channels = camera.describe(image)
                _write_json(
                    os.path.join(meta_dir, stem + ".json"),
                    {
                        "frame_index": frame_index,
                        "bag": os.path.abspath(args.bag),
                        "delta_ms": round(match.delta_ms, 3),
                        "lidar": {
                            "topic": args.lidar_topic,
                            "record_time": round(frame.record_time, 6),
                            "header_stamp": round(frame.stamp, 9),
                            "timebase_ns": frame.timebase_ns,
                            "frame_id": frame.frame_id,
                            "lidar_id": frame.lidar_id,
                            "num_points": frame.num_points,
                            "dropped_invalid_points": frame.dropped_points,
                            "offset_time_range_ms": list(frame.offset_time_range_ms),
                        },
                        "image": {
                            "topic": args.image_topic,
                            "record_time": round(match.record_time, 6),
                            "header_stamp": round(match.stamp, 9),
                            "format": camera.format_string(match.msg),
                            "width": width,
                            "height": height,
                            "channels": channels,
                        },
                        "files": {
                            "image": "images/" + stem + ".jpg",
                            "pcd": "pcd/" + stem + ".pcd",
                        },
                    },
                )
                del image, lidar_msg

            rows.append(
                {
                    "frame_index": frame_index,
                    "lidar_stamp": frame.stamp,
                    "image_stamp": match.stamp,
                    "delta_ms": match.delta_ms,
                    "num_points": frame.num_points,
                }
            )
            deltas.append(abs(match.delta_ms))

            if verbose:
                print(
                    "%-7s %-19s %-19s %+10.3f %8d  %s"
                    % (
                        stem,
                        _fmt_stamp(frame.stamp),
                        _fmt_stamp(match.stamp),
                        match.delta_ms,
                        frame.num_points,
                        (stem + ".jpg / " + stem + ".pcd") if not args.dry_run else "",
                    )
                )
            elif not args.quiet and ((order + 1) % step == 0 or order == 0):
                print(
                    "[%d/%d] %s  dt=%+.2f ms  %d 点"
                    % (order + 1, len(positions), stem, match.delta_ms, frame.num_points)
                )
            if warn_threshold > 0 and abs(match.delta_ms) > warn_threshold * 1000.0:
                print(
                    "        警告: 最近图像与本帧相差 %.1f ms，超过半个图像周期"
                    % match.delta_ms,
                    file=sys.stderr,
                )

        if rows and not verbose and not args.quiet:
            print("")
            print(
                "%-7s %-19s %-19s %10s %8s"
                % ("frame", "lidar_stamp", "image_stamp", "dt[ms]", "points")
            )
            print("-" * 66)
            shown: List[Optional[Dict[str, Any]]] = list(rows)
            if len(rows) > 13:
                shown = list(rows[:10]) + [None] + list(rows[-3:])
            for row in shown:
                if row is None:
                    print("   ...")
                    continue
                print(
                    "%-7s %-19s %-19s %+10.3f %8d"
                    % (
                        "frame_%06d" % row["frame_index"],
                        _fmt_stamp(row["lidar_stamp"]),
                        _fmt_stamp(row["image_stamp"]),
                        row["delta_ms"],
                        row["num_points"],
                    )
                )

    print("")
    if deltas:
        print(
            "完成: %d 帧, 平均 |dt| = %.2f ms, 最大 |dt| = %.2f ms%s"
            % (
                len(deltas),
                statistics.fmean(deltas),
                max(deltas),
                "，跳过 %d 帧" % skipped if skipped else "",
            )
        )
    else:
        print("没有需要提取的帧（全部跳过）")

    if not args.dry_run:
        total = _dump_csv(args.output_dir)
        print("输出目录: %s" % os.path.abspath(args.output_dir))
        print("            images/  (%d 个 jpg)" % total)
        print("            pcd/     (%d 个 pcd)" % total)
        print("            meta/    (%d 个 json)" % total)
        print("            pairs.csv (%d 行)" % total)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except BagIndexError as exc:
        print("错误: %s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
