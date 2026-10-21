# -*- coding: utf-8 -*-
"""PCD 写出/时间戳配对的离线单测：不依赖 bag 文件。

运行: python3 -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest
from typing import Any, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPcd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pcd_test_")
        self.data = np.zeros(3, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
        self.data["x"] = [1.0, 2.0, 3.0]
        self.data["y"] = [0.5, 0.25, 0.125]
        self.data["z"] = [-1.0, -2.0, -3.0]
        self.data["intensity"] = [10, 20, 255]

    def tearDown(self):
        for name in os.listdir(self.tmp):
            os.remove(os.path.join(self.tmp, name))
        os.rmdir(self.tmp)

    def _header(self, path):
        with open(path, "rb") as fh:
            head = []
            for _ in range(11):
                head.append(fh.readline().decode("ascii").strip())
            offset = fh.tell()
        return head, offset

    def test_binary_roundtrip(self):
        from rosbag_extract import pcd

        path = os.path.join(self.tmp, "cloud.pcd")
        n = pcd.write_pcd(path, self.data, binary=True)
        self.assertEqual(n, 3)

        head, offset = self._header(path)
        self.assertIn("FIELDS x y z intensity", head)
        self.assertIn("SIZE 4 4 4 4", head)
        self.assertIn("TYPE F F F F", head)
        self.assertIn("COUNT 1 1 1 1", head)
        self.assertIn("WIDTH 3", head)
        self.assertIn("HEIGHT 1", head)
        self.assertIn("POINTS 3", head)
        self.assertIn("DATA binary", head)

        back = np.fromfile(path, dtype=self.data.dtype, offset=offset)
        np.testing.assert_array_equal(back, self.data)

    def test_ascii_roundtrip(self):
        from rosbag_extract import pcd

        path = os.path.join(self.tmp, "cloud_ascii.pcd")
        n = pcd.write_pcd(path, self.data, binary=False)
        self.assertEqual(n, 3)

        head, offset = self._header(path)
        self.assertIn("DATA ascii", head)
        with open(path, "r", encoding="ascii") as fh:
            fh.seek(offset)
            lines = [ln.split() for ln in fh.read().strip().splitlines()]
        self.assertEqual(len(lines), 3)
        self.assertAlmostEqual(float(lines[0][0]), 1.0, places=5)
        self.assertAlmostEqual(float(lines[2][3]), 255.0, places=5)

    def test_empty_cloud(self):
        from rosbag_extract import pcd

        path = os.path.join(self.tmp, "empty.pcd")
        empty = np.zeros(0, dtype=self.data.dtype)
        self.assertEqual(pcd.write_pcd(path, empty, binary=True), 0)
        head, offset = self._header(path)
        self.assertIn("POINTS 0", head)
        self.assertEqual(os.path.getsize(path), offset)


class _Stamp:
    def __init__(self, sec: float) -> None:
        self._sec = float(sec)

    def to_sec(self) -> float:
        return self._sec


class _Header:
    def __init__(self, sec: float) -> None:
        self.stamp = _Stamp(sec)


class _ImgMsg:
    def __init__(self, sec: float) -> None:
        self.header = _Header(sec)
        self.data = b"jpeg-bytes"


class _FakeIndex:
    """最小化的 BagIndex 替身：记录时间与 header.stamp 可以故意错开。"""

    def __init__(self, record_times: List[float], stamps: List[float]) -> None:
        self._times = np.asarray(record_times, dtype=np.float64)
        self._stamps = list(stamps)
        self.reads = 0

    def times(self, topic: str) -> np.ndarray:
        return self._times

    def read(self, topic: str, index: int) -> Tuple[_ImgMsg, float]:
        self.reads += 1
        return _ImgMsg(self._stamps[index]), float(self._times[index])


class TestPairing(unittest.TestCase):
    def test_camera_period(self):
        from rosbag_extract import pairing

        times = np.array([0.0, 0.1, 0.2, 0.3])
        self.assertAlmostEqual(pairing.camera_period(times), 0.1, places=9)
        self.assertEqual(pairing.camera_period(np.array([1.0])), 0.0)

    def test_exact_nearest_single_read(self):
        from rosbag_extract import pairing

        times = [100.0, 100.1, 100.2, 100.3]
        index = _FakeIndex(times, times)
        match = pairing.match_image(index, "/cam", 100.18, radius=2)
        self.assertEqual(match.index, 2)
        self.assertAlmostEqual(match.delta_ms, 20.0, places=6)
        self.assertEqual(index.reads, 1, "最近帧命中后不应继续读候选")

    def test_refine_when_record_time_is_off(self):
        from rosbag_extract import pairing

        # 记录时间把 lidar 引向 1 号帧，但真正的最近帧是 2 号帧
        record = [99.0, 100.05, 101.5]
        stamps = [99.0, 100.30, 100.02]
        index = _FakeIndex(record, stamps)
        match = pairing.match_image(index, "/cam", 100.0, radius=2, tol_ratio=0.0)
        self.assertEqual(match.index, 2)
        self.assertAlmostEqual(match.delta_ms, 20.0, places=6)
        self.assertEqual(index.reads, 3)

    def test_clamps_at_edges(self):
        from rosbag_extract import pairing

        times = [100.0, 100.1]
        index = _FakeIndex(times, times)
        self.assertEqual(pairing.match_image(index, "/cam", 99.0, radius=3).index, 0)
        self.assertEqual(pairing.match_image(index, "/cam", 200.0, radius=3).index, 1)


class TestLidar(unittest.TestCase):
    def _fake_msg(
        self,
        points: Sequence[Tuple[int, float, float, float, int, int]],
        timebase: int = 1760225178832200926,
    ) -> Any:
        from types import SimpleNamespace

        header = SimpleNamespace(stamp=_Stamp(1760225178.832201), frame_id="livox_frame")
        return SimpleNamespace(
            header=header,
            timebase=timebase,
            point_num=len(points),
            lidar_id=0,
            rsvd=[0, 0, 0],
            points=[
                SimpleNamespace(
                    offset_time=ot,
                    x=x,
                    y=y,
                    z=z,
                    reflectivity=r,
                    tag=0,
                    line=line,
                )
                for (ot, x, y, z, r, line) in points
            ],
        )

    def test_extract_and_drop_invalid(self):
        from rosbag_extract import lidar

        msg = self._fake_msg(
            [
                (0, 1.0, 2.0, 3.0, 10, 1),
                (1000, 0.0, 0.0, 0.0, 0, 2),   # 无回波点，默认剔除
                (2000, -1.0, 0.0, 0.5, 200, 3),
            ]
        )
        frame = lidar.frame_from_msg(msg, record_time=1760225178.832433)
        self.assertEqual(frame.num_points, 2)
        self.assertEqual(frame.dropped_points, 1)
        np.testing.assert_allclose(frame.cloud["x"], [1.0, -1.0])
        np.testing.assert_allclose(frame.cloud["intensity"], [10.0, 200.0])
        self.assertEqual(frame.timebase_ns, msg.timebase)
        self.assertAlmostEqual(frame.offset_time_range_ms[1], 0.002, places=9)
        self.assertFalse(frame.stamp_is_fallback)

    def test_keep_zero_points(self):
        from rosbag_extract import lidar

        msg = self._fake_msg([(0, 0.0, 0.0, 0.0, 0, 0), (1, 1.0, 1.0, 1.0, 5, 1)])
        frame = lidar.frame_from_msg(msg, record_time=0.0, drop_invalid=False)
        self.assertEqual(frame.num_points, 2)
        self.assertEqual(frame.dropped_points, 0)

    def test_stamp_fallback(self):
        from rosbag_extract import lidar

        msg = self._fake_msg([(0, 1.0, 1.0, 1.0, 5, 1)])
        msg.header = _Header(0.0)  # header.stamp 非法
        frame = lidar.frame_from_msg(msg, record_time=123.5)
        self.assertAlmostEqual(frame.stamp, 123.5)
        self.assertTrue(frame.stamp_is_fallback)


if __name__ == "__main__":
    unittest.main(verbosity=2)
