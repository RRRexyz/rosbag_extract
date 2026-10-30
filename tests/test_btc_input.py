# -*- coding: utf-8 -*-
"""btc_desc 输入生成/检查的离线单测：不依赖 bag 文件。

运行: python3 -m unittest discover -s tests -v
"""

import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rosbag_extract import btc_input, odometry  # noqa: E402


class _Stamp:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def to_sec(self) -> float:
        return self.seconds


class _Header:
    def __init__(self, seconds: float) -> None:
        self.stamp = _Stamp(seconds)


class _Vector:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _Quaternion:
    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class _Pose:
    def __init__(self, position: _Vector, orientation: _Quaternion) -> None:
        self.position = position
        self.orientation = orientation


class _PoseWithCovariance:
    def __init__(self, pose: _Pose) -> None:
        self.pose = pose


class _Odometry:
    def __init__(
        self,
        seconds: float,
        position: Tuple[float, float, float],
        quaternion: Tuple[float, float, float, float],
    ) -> None:
        self.header = _Header(seconds)
        self.pose = _PoseWithCovariance(
            _Pose(_Vector(*position), _Quaternion(*quaternion))
        )


class _FakeOdometryBag:
    def __init__(self, messages: List[Any], record_times: List[float]) -> None:
        self.messages = messages
        self.record_times = record_times

    def count(self, topic: str) -> int:
        return len(self.messages)

    def read(self, topic: str, index: int) -> Tuple[Any, float]:
        return self.messages[index], self.record_times[index]


class TestOdometryMath(unittest.TestCase):
    def test_normalize(self):
        np.testing.assert_allclose(
            odometry.normalize_quaternion([0.0, 0.0, 0.0, 2.0]),
            [0.0, 0.0, 0.0, 1.0],
        )
        with self.assertRaises(ValueError):
            odometry.normalize_quaternion([0.0, 0.0, 0.0, 0.0])

    def test_yaw_90(self):
        half = np.pi / 4.0
        q = [0.0, 0.0, np.sin(half), np.cos(half)]
        rotation = odometry.quaternion_to_matrix(q)
        np.testing.assert_allclose(rotation.dot([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)

    def test_matrix_roundtrip(self):
        half = np.pi / 4.0
        q = odometry.normalize_quaternion([0.0, np.sin(half), 0.0, np.cos(half)])
        back = odometry.matrix_to_quaternion(odometry.quaternion_to_matrix(q))
        self.assertAlmostEqual(abs(float(np.dot(q, back))), 1.0, places=12)

    def test_compose(self):
        half = np.pi / 4.0
        q_yaw = [0.0, 0.0, np.sin(half), np.cos(half)]
        translation, quaternion = odometry.compose_transforms(
            [1.0, 2.0, 3.0], q_yaw, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        )
        np.testing.assert_allclose(translation, [1.0, 3.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(quaternion, q_yaw, atol=1e-12)

    def test_pose_matrix_roundtrip(self):
        q = odometry.normalize_quaternion([0.1, -0.2, 0.3, 0.9])
        matrix = odometry.pose_to_matrix([1.0, 2.0, 3.0], q)
        translation, quaternion = odometry.matrix_to_pose(matrix)
        np.testing.assert_allclose(translation, [1.0, 2.0, 3.0], atol=1e-12)
        self.assertAlmostEqual(abs(float(np.dot(q, quaternion))), 1.0, places=12)


class TestPoseLine(unittest.TestCase):
    def test_format(self):
        line = odometry.format_pose_line(7, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(
            line,
            "7 1.000000000 2.000000000 3.000000000 "
            "0.000000000 0.000000000 0.000000000 1.000000000",
        )
        self.assertEqual(len(line.split(" ")), 8)

    def test_parse_roundtrip(self):
        line = odometry.format_pose_line(3, [1.5, -2.5, 0.25], [0.0, 0.0, 0.0, 1.0])
        index, translation, quaternion = odometry.parse_pose_line(line)
        self.assertEqual(index, 3)
        np.testing.assert_allclose(translation, [1.5, -2.5, 0.25], atol=1e-9)
        np.testing.assert_allclose(quaternion, [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_reject_bad_spacing(self):
        with self.assertRaises(ValueError):
            odometry.parse_pose_line(
                "0  0.000000000 0.000000000 0.000000000 "
                "0.000000000 0.000000000 0.000000000 1.000000000"
            )
        with self.assertRaises(ValueError):
            odometry.parse_pose_line("0 0 0 0 0 0 1")


class TestExtrinsic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="btc_calib_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, text: str) -> str:
        path = os.path.join(self.tmp, "calib.yaml")
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
        return path

    def test_read_identity(self):
        path = self._write(
            "li_extrinsic:\n"
            "  R_i_l:\n"
            "    - [1.0, 0.0, 0.0]\n"
            "    - [0.0, 1.0, 0.0]\n"
            "    - [0.0, 0.0, 1.0]\n"
            "  t_i_l: [0.04165, 0.02326, -0.0284]\n"
        )
        rotation, translation = odometry.read_lidar_extrinsic(path)
        np.testing.assert_allclose(rotation, np.eye(3))
        np.testing.assert_allclose(translation, [0.04165, 0.02326, -0.0284])

    def test_reject_non_rotation(self):
        path = self._write(
            "li_extrinsic:\n"
            "  R_i_l:\n"
            "    - [2.0, 0.0, 0.0]\n"
            "    - [0.0, 1.0, 0.0]\n"
            "    - [0.0, 0.0, 1.0]\n"
            "  t_i_l: [0.0, 0.0, 0.0]\n"
        )
        with self.assertRaises(ValueError):
            odometry.read_lidar_extrinsic(path)

    def test_missing_section(self):
        path = self._write("camera_intrinsic:\n  width: 1280\n")
        with self.assertRaises(ValueError):
            odometry.read_lidar_extrinsic(path)


class TestOdometrySeries(unittest.TestCase):
    def test_load_and_normalize(self):
        bag = _FakeOdometryBag(
            [
                _Odometry(100.0, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)),
                _Odometry(100.1, (2.0, 3.0, 4.0), (0.0, 0.0, 0.0, 2.0)),
            ],
            [200.0, 200.1],
        )
        series = btc_input.load_odometry(bag, "/Odometry")
        self.assertEqual(len(series), 2)
        np.testing.assert_allclose(series.stamps, [100.0, 100.1])
        np.testing.assert_allclose(series.positions, [[1, 2, 3], [2, 3, 4]])
        np.testing.assert_allclose(series.quaternions, [[0, 0, 0, 1], [0, 0, 0, 1]])
        np.testing.assert_array_equal(series.bag_indices, [0, 1])
        self.assertAlmostEqual(series.max_quaternion_error, 1.0)


class TestSelectionAndMatching(unittest.TestCase):
    def test_select(self):
        self.assertEqual(btc_input.select_frame_positions(10, 2, 8, 2), [2, 4, 6, 8])
        self.assertEqual(btc_input.select_frame_positions(5, 0, None, 1), [0, 1, 2, 3, 4])
        self.assertEqual(btc_input.select_frame_positions(5, 10, None, 1), [])
        with self.assertRaises(ValueError):
            btc_input.select_frame_positions(5, 0, None, 0)

    def test_match(self):
        stamps = np.asarray([100.0, 100.1, 100.2])
        self.assertEqual(btc_input.match_odometry(stamps, 100.09, 0.02), 1)
        self.assertIsNone(btc_input.match_odometry(stamps, 100.15, 0.02))
        self.assertEqual(btc_input.match_odometry(stamps, 100.0, 0.02), 0)
        self.assertIsNone(btc_input.match_odometry(np.asarray([]), 100.0, 0.02))


class TestOutputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="btc_output_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_configs(self):
        selected = btc_input.write_configs(self.tmp, "indoor")
        self.assertEqual(selected, os.path.join(self.tmp, "config.yaml"))
        for name in ("config.yaml", "config_indoor.yaml", "config_outdoor.yaml"):
            path = os.path.join(self.tmp, name)
            self.assertTrue(os.path.isfile(path))
            with open(path, "r", encoding="utf-8") as file:
                self.assertEqual(file.readline().strip(), "%YAML:1.0")
        with open(os.path.join(self.tmp, "config.yaml"), "r", encoding="utf-8") as file:
            indoor_text = file.read()
        with open(os.path.join(self.tmp, "config_indoor.yaml"), "r", encoding="utf-8") as file:
            self.assertEqual(indoor_text, file.read())
        self.assertIn("max_constrait_dis", indoor_text)
        self.assertNotIn("max_constraint_dis", indoor_text)

    def test_run_params(self):
        text = btc_input.run_params_text(
            os.path.join(self.tmp, "velodyne"),
            os.path.join(self.tmp, "pose.txt"),
            os.path.join(self.tmp, "config.yaml"),
        )
        self.assertIn("read_bin: false", text)
        self.assertIn("cloud_overlap_thr: 0.5", text)
        self.assertIn("pcds_dir:", text)
        self.assertIn("setting_path:", text)

    def test_meta_csv(self):
        rows = []
        for index in range(2):
            rows.append(
                {
                    "frame_index": index,
                    "lidar_index": index + 5,
                    "lidar_stamp": "%.9f" % (100.0 + index),
                    "lidar_record_time": "%.6f" % (200.0 + index),
                    "odom_index": index,
                    "odom_stamp": "%.9f" % (100.0 + index),
                    "dt_ms": "+0.000",
                    "num_points": 100 + index,
                    "dropped_points": 10,
                    "pcd_file": "%06d.pcd" % index,
                }
            )
        path = os.path.join(self.tmp, "meta.csv")
        btc_input.write_meta_csv(rows, path)
        with open(path, "r", encoding="utf-8", newline="") as file:
            lines = file.read().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("frame_index,lidar_index,"))
        self.assertIn("000001.pcd", lines[2])


if __name__ == "__main__":
    unittest.main()
