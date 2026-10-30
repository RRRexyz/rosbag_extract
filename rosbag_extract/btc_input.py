"""把 raw Livox bag + LIO 里程计转成 btc_desc 的输入格式。

btc_desc 需要三样磁盘输入：

- ``velodyne/%06d.pcd``：从 000000 连续编号的传感器系点云；
- ``pose.txt``：每行 ``id x y z qx qy qz qw``，语义 ``T_world_lidar``；
- ``config.yaml``：OpenCV FileStorage，首行 ``%YAML:1.0``。

本模块只读写磁盘，不依赖 ROS 运行时（只需要 ``rosbag`` 能反序列化消息）。
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from . import odometry, pcd
from .bag_index import BagIndex, header_stamp, nearest_index
from .lidar import LidarFrame, frame_from_msg

DEFAULT_LIDAR_TOPIC = "/livox/lidar"
DEFAULT_ODOM_TOPIC = "/Odometry"
DEFAULT_MAX_TIME_DIFF = 0.02
DEFAULT_CLOUD_OVERLAP_THR = 0.5
DEFAULT_PCD_FORMAT = "binary"
DEFAULT_SCENE = "indoor"

PCD_SUBDIR = "velodyne"
POSE_FILENAME = "pose.txt"
META_FILENAME = "meta.csv"
RUN_PARAMS_FILENAME = "run_params.yaml"
DATASET_FILENAME = "dataset.json"
CONFIG_FILENAME = "config.yaml"
CONFIG_INDOOR_FILENAME = "config_indoor.yaml"
CONFIG_OUTDOOR_FILENAME = "config_outdoor.yaml"

META_FIELDS = [
    "frame_index",
    "lidar_index",
    "lidar_stamp",
    "lidar_record_time",
    "odom_index",
    "odom_stamp",
    "dt_ms",
    "num_points",
    "dropped_points",
    "pcd_file",
]

# 与 /home/rex/project/btc_descriptor/src/config/config_indoor.yaml 保持一致。
# 注意键名 max_constrait_dis 是 btc_desc 源码里的拼写，不能改成 constraint。
INDOOR_CONFIG = """%YAML:1.0

# binary descriptor
useful_corner_num: 500  # Maximum number of triangle vertices
plane_detection_thre: 0.01  # Maximum allowed value for the minimum eigenvalue of the covariance matrix to consider as a plane
plane_merge_normal_thre: 0.1 # Normal threshold for plane merging
plane_merge_dis_thre: 0.3   # Distance threshold for plane merging
voxel_size: 0.5      # Side length of each voxel during plane extraction
voxel_init_num: 10   # Minimum number of points in the voxel for plane extraction
proj_plane_num: 2    # Number of projection planes
proj_image_resolution: 0.2  # In meters
proj_image_high_inc: 0.1    # In meters
proj_dis_min: -1.0   # Minimum distance to the projection plane
proj_dis_max: 4.0  # Maximum distance to the projection plane
summary_min_thre: 6  # Minimum number of points in one image grid
line_filter_enable: 0  # Enable line point filtering; turn off in environments lacking features (indoor and outdoor), otherwise enable to reduce computation

# std descriptor
descriptor_near_num: 15   # Number of neighboring points to save for each point when generating triangles
descriptor_min_len: 1     # Minimum edge length
descriptor_max_len: 30    # Maximum edge length
max_constrait_dis: 1      # Neighborhood range for non-maximum suppression
triangle_resolution: 0.2  # Edge length amplification coefficient

# candidate search
skip_near_num: 100   # Number of frames to skip; set to a positive value for intra-machine loops, and -1000000 for inter-machine loops
candidate_num: 50   # Maximum number of reference frames
similarity_threshold: 0.7   # BTC similarity coefficient
rough_dis_threshold: 0.01   # Threshold coefficient for BTC edge length distance
normal_threshold: 0.2  # Threshold for plane ICP and geometric validation: point-to-plane distance
dis_threshold: 0.5  # Distance threshold for point-to-plane validation
icp_threshold: 0.2  # Threshold for plane-to-plane ICP to triggle loop.
"""

# 与 /home/rex/project/btc_descriptor/src/config/config_outdoor.yaml 保持一致。
OUTDOOR_CONFIG = """%YAML:1.0

# binary descriptor
useful_corner_num: 500  # Maximum number of useful corners
plane_detection_thre: 0.01  # Threshold for plane detection
plane_merge_normal_thre: 0.1 # Normal threshold for plane merging
plane_merge_dis_thre: 0.3   # Distance threshold for plane merging
voxel_size: 2  # Size of each voxel
voxel_init_num: 10  # Minimum number of points in a voxel for extraction
proj_plane_num: 1  # Number of projection planes
proj_image_resolution: 0.5  # Resolution of the projected image in meters
proj_image_high_inc: 0.1    # Height increment for the projected image in meters
proj_dis_min: -1.0  # Minimum distance to the projection plane
proj_dis_max: 4.0  # Maximum distance to the projection plane
summary_min_thre: 10  # Minimum number of points in one image grid
line_filter_enable: 1  # Enable line point filtering (1: enabled, 0: disabled)

# std descriptor
descriptor_near_num: 10  # Number of neighboring points to save for each point when generating triangles
descriptor_min_len: 2    # Minimum edge length
descriptor_max_len: 50   # Maximum edge length
max_constrait_dis: 2     # Neighborhood range for non-maximum suppression
triangle_resolution: 0.2  # Edge length amplification coefficient

# candidate search
skip_near_num: 100  # Number of frames to skip
candidate_num: 50   # Maximum number of reference frames
similarity_threshold: 0.7  # BTC similarity coefficient
rough_dis_threshold: 0.01   # Threshold coefficient for BTC edge length distance
normal_threshold: 0.2  # Threshold for plane ICP and geometric validation: point-to-plane distance
dis_threshold: 0.5  # Distance threshold for plane-to-plane validation.
icp_threshold: 0.2  # Threshold for plane-to-plane ICP to triggle loop.
"""


class BagReader(Protocol):
    """``load_odometry`` 需要的最小 bag 接口（``BagIndex`` 结构上满足）。"""

    def count(self, topic: str) -> int:
        ...

    def read(self, topic: str, index: int) -> Tuple[Any, float]:
        ...


@dataclass
class OdometrySeries:
    """按 header.stamp 排好序的里程计序列。"""

    stamps: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    record_times: np.ndarray
    bag_indices: np.ndarray
    max_quaternion_error: float = 0.0
    skipped_invalid: int = 0

    def __len__(self) -> int:
        return int(self.stamps.shape[0])


def load_odometry(bag: BagReader, topic: str = DEFAULT_ODOM_TOPIC) -> OdometrySeries:
    """读取 ``nav_msgs/Odometry``，统一归一化四元数并按 header.stamp 排序。"""
    count = int(bag.count(topic))
    stamps: List[float] = []
    positions: List[List[float]] = []
    quaternions: List[np.ndarray] = []
    record_times: List[float] = []
    bag_indices: List[int] = []
    max_quaternion_error = 0.0
    skipped_invalid = 0

    for index in range(count):
        message, record_time = bag.read(topic, index)
        stamp = header_stamp(message)
        if stamp is None:
            skipped_invalid += 1
            continue
        pose = message.pose.pose
        position = pose.position
        orientation = pose.orientation
        quaternion = np.asarray(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm < 1e-12:
            skipped_invalid += 1
            continue
        max_quaternion_error = max(max_quaternion_error, abs(norm - 1.0))
        stamps.append(float(stamp))
        positions.append([float(position.x), float(position.y), float(position.z)])
        quaternions.append(quaternion / norm)
        record_times.append(float(record_time))
        bag_indices.append(index)

    if not stamps:
        raise ValueError("里程计话题 %s 没有可用位姿" % topic)

    stamp_array = np.asarray(stamps, dtype=np.float64)
    position_array = np.asarray(positions, dtype=np.float64)
    quaternion_array = np.asarray(quaternions, dtype=np.float64)
    record_array = np.asarray(record_times, dtype=np.float64)
    index_array = np.asarray(bag_indices, dtype=np.int64)

    order = np.argsort(stamp_array, kind="mergesort")
    return OdometrySeries(
        stamps=stamp_array[order],
        positions=position_array[order],
        quaternions=quaternion_array[order],
        record_times=record_array[order],
        bag_indices=index_array[order],
        max_quaternion_error=max_quaternion_error,
        skipped_invalid=skipped_invalid,
    )


def match_odometry(
    stamps: np.ndarray, stamp: float, max_time_diff: float = DEFAULT_MAX_TIME_DIFF
) -> Optional[int]:
    """返回 header.stamp 最近的里程计下标；超过容差返回 None。"""
    if stamps.size == 0 or not np.isfinite(stamp):
        return None
    index = int(nearest_index(stamps, float(stamp)))
    if abs(float(stamps[index]) - float(stamp)) > float(max_time_diff):
        return None
    return index


def select_frame_positions(
    count: int, start: int = 0, end: Optional[int] = None, step: int = 1
) -> List[int]:
    """按 raw 帧下标选帧（闭区间）；越界自动截断，step<1 抛 ValueError。"""
    if count <= 0:
        return []
    if step < 1:
        raise ValueError("step 必须 >= 1")
    first = max(0, int(start))
    if first >= count:
        return []
    last = count - 1 if end is None else min(count - 1, int(end))
    if last < first:
        return []
    return list(range(first, last + 1, int(step)))


def write_configs(output_dir: str, scene: str = DEFAULT_SCENE) -> str:
    """写出室内/室外两份模板，并把选中的一份复制为 ``config.yaml``。"""
    if scene not in ("indoor", "outdoor"):
        raise ValueError("scene 只能是 indoor 或 outdoor，收到 %r" % (scene,))
    indoor_path = os.path.join(output_dir, CONFIG_INDOOR_FILENAME)
    outdoor_path = os.path.join(output_dir, CONFIG_OUTDOOR_FILENAME)
    selected_path = os.path.join(output_dir, CONFIG_FILENAME)
    _write_text(indoor_path, INDOOR_CONFIG)
    _write_text(outdoor_path, OUTDOOR_CONFIG)
    _write_text(selected_path, INDOOR_CONFIG if scene == "indoor" else OUTDOOR_CONFIG)
    return selected_path


def run_params_text(
    pcds_dir: str,
    pose_file: str,
    setting_path: str,
    cloud_overlap_thr: float = DEFAULT_CLOUD_OVERLAP_THR,
) -> str:
    """生成可直接 ``rosparam load`` 的全局参数（btc 用 ros::NodeHandle nh）。"""
    lines = [
        "# btc_desc 全局参数（源码用 ros::NodeHandle nh，不是 ~）",
        "pcds_dir: %s" % json.dumps(os.path.abspath(pcds_dir)),
        "pose_file: %s" % json.dumps(os.path.abspath(pose_file)),
        "setting_path: %s" % json.dumps(os.path.abspath(setting_path)),
        "read_bin: false",
        "cloud_overlap_thr: %g" % float(cloud_overlap_thr),
    ]
    return "\n".join(lines) + "\n"


def write_meta_csv(rows: Iterable[Dict[str, Any]], path: str) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=META_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    _write_text(path, buffer.getvalue())


def write_dataset_json(path: str, payload: Dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _write_pcd_atomic(path: str, cloud: np.ndarray, binary: bool) -> int:
    temporary = path + ".tmp"
    try:
        size = pcd.write_pcd(temporary, cloud, binary=binary)
        os.replace(temporary, path)
        return size
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _clean_output(output_dir: str) -> None:
    pcd_dir = os.path.join(output_dir, PCD_SUBDIR)
    if os.path.isdir(pcd_dir):
        for name in os.listdir(pcd_dir):
            if name.endswith(".pcd") or name.endswith(".pcd.tmp"):
                os.remove(os.path.join(pcd_dir, name))
    for name in (
        POSE_FILENAME,
        META_FILENAME,
        RUN_PARAMS_FILENAME,
        DATASET_FILENAME,
        CONFIG_FILENAME,
        CONFIG_INDOOR_FILENAME,
        CONFIG_OUTDOOR_FILENAME,
    ):
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            os.remove(path)


def _iter_frames(
    lidar_index: BagIndex,
    lidar_topic: str,
    positions: Sequence[int],
    odometry_series: OdometrySeries,
    max_time_diff: float,
    extrinsic_translation: np.ndarray,
    extrinsic_quaternion: np.ndarray,
    keep_zero_points: bool,
) -> Iterable[Tuple[int, LidarFrame, Optional[int], Optional[np.ndarray], Optional[np.ndarray]]]:
    """逐帧读取 raw lidar，返回 (raw 下标, frame, odom 下标, t, q)。"""
    for raw_index in positions:
        message, record_time = lidar_index.read(lidar_topic, raw_index)
        frame = frame_from_msg(message, record_time, drop_invalid=not keep_zero_points)
        if frame.num_points == 0:
            yield raw_index, frame, None, None, None
            continue
        odom_index = match_odometry(odometry_series.stamps, frame.stamp, max_time_diff)
        if odom_index is None:
            yield raw_index, frame, None, None, None
            continue
        translation, quaternion = odometry.compose_transforms(
            odometry_series.positions[odom_index],
            odometry_series.quaternions[odom_index],
            extrinsic_translation,
            extrinsic_quaternion,
        )
        yield raw_index, frame, odom_index, translation, quaternion


def run_build(
    lidar_bag: str,
    odom_bag: str,
    calibration: str,
    output_dir: str,
    lidar_topic: str = DEFAULT_LIDAR_TOPIC,
    odom_topic: str = DEFAULT_ODOM_TOPIC,
    start: int = 0,
    end: Optional[int] = None,
    step: int = 1,
    limit: int = 0,
    max_time_diff: float = DEFAULT_MAX_TIME_DIFF,
    pcd_format: str = DEFAULT_PCD_FORMAT,
    scene: str = DEFAULT_SCENE,
    keep_zero_points: bool = False,
    clean: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
) -> int:
    """执行转换；返回 0 成功、1 没有写出任何帧。异常由 CLI 统一捕获。"""
    if pcd_format not in ("binary", "ascii"):
        raise ValueError("pcd_format 只能是 binary 或 ascii，收到 %r" % (pcd_format,))
    if scene not in ("indoor", "outdoor"):
        raise ValueError("scene 只能是 indoor 或 outdoor，收到 %r" % (scene,))
    if max_time_diff <= 0.0:
        raise ValueError("max_time_diff 必须 > 0")
    if step < 1:
        raise ValueError("step 必须 >= 1")
    if limit < 0:
        raise ValueError("limit 必须 >= 0")
    if start < 0:
        raise ValueError("start 必须 >= 0")
    if end is not None and end < 0:
        raise ValueError("end 必须 >= 0")

    output_dir = os.path.abspath(output_dir)
    pcd_dir = os.path.join(output_dir, PCD_SUBDIR)
    pose_path = os.path.join(output_dir, POSE_FILENAME)
    meta_path = os.path.join(output_dir, META_FILENAME)
    run_params_path = os.path.join(output_dir, RUN_PARAMS_FILENAME)
    dataset_path = os.path.join(output_dir, DATASET_FILENAME)

    def log(message: str) -> None:
        if not quiet:
            print(message)

    log("btc_desc 输入生成")
    log("  lidar bag : %s" % os.path.abspath(lidar_bag))
    log("  odom bag  : %s" % os.path.abspath(odom_bag))
    log("  calib     : %s" % os.path.abspath(calibration))
    log("  output    : %s" % output_dir)
    log("  scene     : %s  (config.yaml)" % scene)
    log("  PCD       : %s, x y z intensity, 剔除无效点=%s"
        % (pcd_format, "否" if keep_zero_points else "是"))

    rotation, translation = odometry.read_lidar_extrinsic(calibration)
    extrinsic_quaternion = odometry.matrix_to_quaternion(rotation)
    log("  T_i_l     : t=[%.6f, %.6f, %.6f]  R=I? %s"
        % (translation[0], translation[1], translation[2],
           "是" if np.allclose(rotation, np.eye(3), atol=1e-9) else "否"))

    with BagIndex(lidar_bag) as lidar_index, BagIndex(odom_bag) as odom_index:
        lidar_count = int(lidar_index.count(lidar_topic))
        odom_count = int(odom_index.count(odom_topic))
        odometry_series = load_odometry(odom_index, odom_topic)
        positions = select_frame_positions(lidar_count, start, end, step)
        if not positions:
            raise ValueError("没有选中任何 raw 帧（count=%d, start=%d, end=%s）"
                             % (lidar_count, start, end))
        log("  lidar     : %s  %d 条" % (lidar_topic, lidar_count))
        log("  odom      : %s  %d 条（可用 %d 条）"
            % (odom_topic, odom_count, len(odometry_series)))
        log("  raw 选帧  : %d 个（%d..%d，step=%d%s）"
            % (len(positions), positions[0], positions[-1], step,
               ", limit=%d" % limit if limit else ""))

        if dry_run:
            matched = 0
            skipped_no_pose = 0
            skipped_empty = 0
            point_counts: List[int] = []
            max_abs_dt_ms = 0.0
            for raw_index, frame, odom_index, _translation, _quaternion in _iter_frames(
                lidar_index,
                lidar_topic,
                positions,
                odometry_series,
                max_time_diff,
                translation,
                extrinsic_quaternion,
                keep_zero_points,
            ):
                if frame.num_points == 0:
                    skipped_empty += 1
                    continue
                if odom_index is None:
                    skipped_no_pose += 1
                    continue
                matched += 1
                point_counts.append(int(frame.num_points))
                max_abs_dt_ms = max(
                    max_abs_dt_ms,
                    abs(float(frame.stamp) - float(odometry_series.stamps[odom_index])) * 1000.0,
                )
                if limit and matched >= limit:
                    break
                if not quiet and matched % 100 == 0:
                    log("  ... 已扫描 %d 帧" % matched)

            log("")
            log("dry-run: 可写出 %d 帧" % matched)
            log("  跳过: 无里程计 %d 帧，空点云 %d 帧" % (skipped_no_pose, skipped_empty))
            if point_counts:
                log("  点数: %d ~ %d，平均 %.0f"
                    % (min(point_counts), max(point_counts), float(np.mean(point_counts))))
                log("  最大时间差: %.3f ms" % max_abs_dt_ms)
            log("  未写入任何文件")
            return 0

        os.makedirs(pcd_dir, exist_ok=True)
        if clean:
            _clean_output(output_dir)
            log("  已清理旧输出（--clean）")
        else:
            existing = [
                name for name in os.listdir(pcd_dir)
                if name.endswith(".pcd") or name.endswith(".pcd.tmp")
            ]
            if existing:
                raise ValueError(
                    "%s 已有 %d 个 PCD；加 --clean 覆盖，或换一个 --output-dir"
                    % (pcd_dir, len(existing))
                )

        pose_lines: List[str] = []
        meta_rows: List[Dict[str, Any]] = []
        point_counts = []
        skipped_no_pose = 0
        skipped_empty = 0
        max_abs_dt_ms = 0.0
        stamp_fallback_count = 0

        for raw_index, frame, odom_index, frame_translation, frame_quaternion in _iter_frames(
            lidar_index,
            lidar_topic,
            positions,
            odometry_series,
            max_time_diff,
            translation,
            extrinsic_quaternion,
            keep_zero_points,
        ):
            if frame.num_points == 0:
                skipped_empty += 1
                continue
            if (
                odom_index is None
                or frame_translation is None
                or frame_quaternion is None
            ):
                skipped_no_pose += 1
                continue

            out_index = len(pose_lines)
            pcd_name = "%06d.pcd" % out_index
            _write_pcd_atomic(
                os.path.join(pcd_dir, pcd_name),
                frame.cloud,
                binary=(pcd_format == "binary"),
            )
            pose_lines.append(
                odometry.format_pose_line(out_index, frame_translation, frame_quaternion)
            )
            odom_stamp = float(odometry_series.stamps[odom_index])
            dt_ms = (float(frame.stamp) - odom_stamp) * 1000.0
            max_abs_dt_ms = max(max_abs_dt_ms, abs(dt_ms))
            meta_rows.append(
                {
                    "frame_index": out_index,
                    "lidar_index": raw_index,
                    "lidar_stamp": "%.9f" % float(frame.stamp),
                    "lidar_record_time": "%.6f" % float(frame.record_time),
                    "odom_index": int(odometry_series.bag_indices[odom_index]),
                    "odom_stamp": "%.9f" % odom_stamp,
                    "dt_ms": "%+.3f" % dt_ms,
                    "num_points": int(frame.num_points),
                    "dropped_points": int(frame.dropped_points),
                    "pcd_file": pcd_name,
                }
            )
            point_counts.append(int(frame.num_points))
            if frame.stamp_is_fallback:
                stamp_fallback_count += 1

            if not quiet and (len(pose_lines) <= 3 or len(pose_lines) % 100 == 0):
                log("  [%5d] raw %04d -> %s  %5d 点  dt=%+.2f ms"
                    % (len(pose_lines), raw_index, pcd_name, frame.num_points, dt_ms))
            if limit and len(pose_lines) >= limit:
                break

        if not pose_lines:
            log("错误: 没有写出任何帧（无匹配里程计或全部为空点云）")
            return 1

        _write_text(pose_path, "\n".join(pose_lines) + "\n")
        write_meta_csv(meta_rows, meta_path)
        config_path = write_configs(output_dir, scene)
        _write_text(
            run_params_path,
            run_params_text(pcd_dir, pose_path, config_path, DEFAULT_CLOUD_OVERLAP_THR),
        )

        warnings: List[str] = []
        if skipped_no_pose:
            warnings.append(
                "%d 个 raw 帧在 %.1f ms 内没有里程计，已跳过"
                % (skipped_no_pose, max_time_diff * 1000.0)
            )
        if skipped_empty:
            warnings.append("%d 个 raw 帧点云为空，已跳过" % skipped_empty)
        if stamp_fallback_count:
            warnings.append(
                "%d 帧的 header.stamp 无效，退回使用 bag 记录时间"
                % stamp_fallback_count
            )
        if odometry_series.max_quaternion_error > 1e-6:
            warnings.append(
                "里程计四元数最大范数误差 %.3g，已逐条归一化"
                % odometry_series.max_quaternion_error
            )
        if odometry_series.skipped_invalid:
            warnings.append(
                "里程计有 %d 条无效位姿被跳过" % odometry_series.skipped_invalid
            )
        if limit and len(pose_lines) >= limit:
            warnings.append("只生成了前 %d 帧（--limit）" % limit)

        dataset_payload: Dict[str, Any] = {
            "generator": "extract_btc_input.py",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "frame_count": len(pose_lines),
            "source": {
                "lidar_bag": os.path.abspath(lidar_bag),
                "lidar_topic": lidar_topic,
                "odom_bag": os.path.abspath(odom_bag),
                "odom_topic": odom_topic,
                "calibration": os.path.abspath(calibration),
                "extrinsic": {
                    "R_i_l": rotation.tolist(),
                    "t_i_l": translation.tolist(),
                    "meaning": "T_i_l: p_imu = R_i_l * p_lidar + t_i_l",
                },
                "quaternion_order": "x y z w",
            },
            "output": {
                "pcds_dir": pcd_dir,
                "pose_file": pose_path,
                "setting_path": config_path,
                "read_bin": False,
                "cloud_overlap_thr": float(DEFAULT_CLOUD_OVERLAP_THR),
                "pcd_format": pcd_format,
                "scene": scene,
            },
            "stats": {
                "raw_frames": lidar_count,
                "selected_raw_frames": len(positions),
                "written_frames": len(pose_lines),
                "skipped_no_odometry": skipped_no_pose,
                "skipped_empty": skipped_empty,
                "points_min": int(min(point_counts)),
                "points_mean": round(float(np.mean(point_counts)), 1),
                "points_max": int(max(point_counts)),
                "max_abs_time_diff_ms": round(max_abs_dt_ms, 3),
            },
            "warnings": warnings,
        }
        write_dataset_json(dataset_path, dataset_payload)

    log("")
    log("完成: 写出 %d 帧到 %s" % (len(pose_lines), output_dir))
    log("  点数: %d ~ %d，平均 %.0f"
        % (min(point_counts), max(point_counts), float(np.mean(point_counts))))
    log("  跳过: 无里程计 %d 帧，空点云 %d 帧" % (skipped_no_pose, skipped_empty))
    log("  文件: %s/  %s  %s  %s  %s  %s"
        % (PCD_SUBDIR, POSE_FILENAME, CONFIG_FILENAME,
           META_FILENAME, RUN_PARAMS_FILENAME, DATASET_FILENAME))
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log("  自检: python3 %s --data %s"
        % (os.path.join(project_dir, "check_btc_input.py"), output_dir))
    log("  运行: source /home/rex/project/btc_descriptor/devel/setup.bash")
    log("        rosparam load %s" % run_params_path)
    log("        rosrun btc_desc btc_place_recognition   # 需要先启动 roscore")
    return 0
