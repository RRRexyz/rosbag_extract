#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 raw Livox bag + LIO 里程计转成 btc_desc 的输入格式（PCD）。

默认数据源是石通 building03：

- 点云：``/home/rex/data/shitong/building/building03.bag`` 的 ``/livox/lidar``（24000 点/帧原始 Livox CustomMsg）；
- 位姿：``/home/rex/data/shitong/building/lio_building03.bag`` 的 ``/Odometry``（FAST-LIO 结果，header.stamp 与 raw lidar 对齐）；
- 外参：``/home/rex/data/shitong/building/building.yaml`` 的 ``li_extrinsic``（lidar → imu/body）。

输出目录（默认 ``<本项目>/btc_input/building03``）::

    btc_input/building03/
    ├── velodyne/000000.pcd ...    # 传感器系点云，x y z intensity, binary
    ├── pose.txt                   # id x y z qx qy qz qw（T_world_lidar，w 最后）
    ├── config_indoor.yaml         # btc 室内参数
    ├── config_outdoor.yaml        # btc 室外参数
    ├── config.yaml                # 按 --scene 选中的一份
    ├── meta.csv                   # 输出帧 ↔ 原始 lidar/odom 帧的映射
    ├── run_params.yaml            # 可直接 rosparam load 的全局参数
    └── dataset.json               # 本次生成的元数据/统计
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from rosbag_extract import btc_input
from rosbag_extract.bag_index import BagIndex, BagIndexError

DEFAULT_LIDAR_BAG = "/home/rex/data/shitong/campus/campus1.bag"
DEFAULT_ODOM_BAG = "/home/rex/data/shitong/campus/lio_campus1.bag"
DEFAULT_CALIBRATION = "/home/rex/data/shitong/campus/campus.yaml"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "btc_input", "campus1")

EPILOG = """用法示例::

    python3 extract_btc_input.py --dry-run              # 只统计能写多少帧，不落盘
    python3 extract_btc_input.py --limit 20 --clean     # 先生成 20 帧小样本
    python3 extract_btc_input.py --clean                # 全量（building03 约 1707 帧）
    python3 extract_btc_input.py --scene outdoor --start 0 --end 500
    python3 check_btc_input.py --data btc_input/building03

说明::

    - btc_desc 要求帧名从 000000 连续；脚本按输出顺序重新编号，原始帧号写进 meta.csv。
    - raw lidar 前 5 帧没有 20 ms 内的里程计（FAST-LIO 预热），默认会被跳过；
      1708 条 odom 中最后 1 条比 raw 最后一帧晚约 95 ms，也不会被用到。
    - pose.txt 语义为 T_world_lidar：p_world = R * p_lidar + t，点云保持传感器系。
    - 运行 btc_desc 前必须先启动 roscore，然后 rosparam load run_params.yaml，
      再 rosrun btc_desc btc_place_recognition（read_bin=false 读 PCD）。
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_btc_input.py",
        description="把 raw Livox bag + LIO 里程计转成 btc_desc 的 PCD/pose/config 输入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument(
        "--lidar-bag", default=DEFAULT_LIDAR_BAG, help="原始点云 bag（默认: %(default)s）"
    )
    parser.add_argument(
        "--odom-bag", default=DEFAULT_ODOM_BAG, help="LIO 里程计 bag（默认: %(default)s）"
    )
    parser.add_argument(
        "--calibration",
        default=DEFAULT_CALIBRATION,
        help="含 li_extrinsic 的标定 YAML（默认: %(default)s）",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录（默认: <项目>/btc_input/building03）",
    )
    parser.add_argument("--lidar-topic", default=btc_input.DEFAULT_LIDAR_TOPIC, help="点云话题")
    parser.add_argument("--odom-topic", default=btc_input.DEFAULT_ODOM_TOPIC, help="里程计话题")

    selection = parser.add_argument_group("帧选择")
    selection.add_argument("--start", type=int, default=0, help="raw 起始帧下标（默认 0）")
    selection.add_argument("--end", type=int, default=None, help="raw 结束帧下标（含）")
    selection.add_argument("--step", type=int, default=1, help="raw 抽帧步长（默认 1）")
    selection.add_argument(
        "--limit", type=int, default=0, help="最多写出多少帧（0=不限制）"
    )
    selection.add_argument(
        "--max-time-diff",
        type=float,
        default=btc_input.DEFAULT_MAX_TIME_DIFF,
        help="raw/odom header.stamp 配对容差，秒（默认 0.02）",
    )

    output = parser.add_argument_group("输出")
    output.add_argument(
        "--pcd-format", choices=("binary", "ascii"), default=btc_input.DEFAULT_PCD_FORMAT,
        help="PCD 编码（默认 binary）",
    )
    output.add_argument(
        "--scene", choices=("indoor", "outdoor"), default=btc_input.DEFAULT_SCENE,
        help="config.yaml 用室内还是室外模板（默认 indoor）",
    )
    output.add_argument(
        "--keep-zero-points", action="store_true", help="保留 Livox 的 (0,0,0) 无效点"
    )
    output.add_argument(
        "--clean", action="store_true", help="先删除输出目录里旧的 PCD 与元数据"
    )

    misc = parser.add_argument_group("其他")
    misc.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    misc.add_argument("--list-topics", action="store_true", help="列出两个 bag 的话题后退出")
    misc.add_argument("--quiet", action="store_true", help="只输出最终汇总")
    return parser


def _print_topics(label: str, bag_path: str, topic_info: dict) -> None:
    print("%s: %s" % (label, os.path.abspath(bag_path)))
    print("%-42s %-44s %8s %9s" % ("topic", "type", "count", "freq[Hz]"))
    print("-" * 106)
    for topic in sorted(topic_info):
        message_type, count, frequency = topic_info[topic]
        print("%-42s %-44s %8d %9.1f" % (topic, message_type, count, frequency))
    print("")


def run(args: argparse.Namespace) -> int:
    if args.list_topics:
        with BagIndex(args.lidar_bag) as lidar_index, BagIndex(args.odom_bag) as odom_index:
            _print_topics("lidar bag", args.lidar_bag, lidar_index.topic_info())
            _print_topics("odom bag", args.odom_bag, odom_index.topic_info())
        return 0

    return btc_input.run_build(
        lidar_bag=args.lidar_bag,
        odom_bag=args.odom_bag,
        calibration=args.calibration,
        output_dir=args.output_dir,
        lidar_topic=args.lidar_topic,
        odom_topic=args.odom_topic,
        start=args.start,
        end=args.end,
        step=args.step,
        limit=args.limit,
        max_time_diff=args.max_time_diff,
        pcd_format=args.pcd_format,
        scene=args.scene,
        keep_zero_points=args.keep_zero_points,
        clean=args.clean,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except BagIndexError as exc:
        print("错误: %s" % exc, file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print("错误: %s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
