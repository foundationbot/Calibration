"""calib_coverage's stream checks: the head IMU line goes red when the IMU is silent (2026-10-20:
two camera-IMU takes recorded 0 IMU samples because EtherCAT was down and nothing said so)."""
from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_coverage as cc  # noqa: E402


class ImuStatus(unittest.TestCase):
    def test_rate_zero_is_missing(self):
        ok, text = cc.imu_status(0.0, 0.0)
        self.assertFalse(ok)
        self.assertIn("IMU MISSING", text)
        self.assertIn("EtherCAT", text)

    def test_stale_sample_is_missing(self):
        self.assertFalse(cc.imu_status(500.0, 1.5)[0])

    def test_streaming_is_ok(self):
        ok, text = cc.imu_status(500.0, 0.01)
        self.assertTrue(ok)
        self.assertIn("500 Hz", text)


class RingRateFromHead(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp()
        os.write(fd, bytes(128))
        os.close(fd)

    def tearDown(self):
        os.unlink(self.path)

    def _head(self, n):
        with open(self.path, "r+b") as f:
            f.seek(64)
            f.write(struct.pack("<Q", n))

    def test_advancing_head_gives_rate_and_fresh_age(self):
        r = cc.RingRate(self.path, 0.0)
        self._head(100)
        r.poll(0.0)
        self._head(600)
        rate, age = r.poll(1.0)
        self.assertAlmostEqual(rate, 500.0)
        self.assertEqual(age, 0.0)
        self.assertTrue(cc.imu_status(rate, age)[0])

    def test_frozen_head_goes_missing(self):
        """A ring left behind by a dead producer: the head never moves."""
        self._head(30330)
        r = cc.RingRate(self.path, 0.0)
        r.poll(0.0)
        rate, age = r.poll(2.0)
        self.assertEqual(rate, 0.0)
        self.assertFalse(cc.imu_status(rate, age)[0])

    def test_absent_ring_is_missing(self):
        r = cc.RingRate(self.path + ".absent", 0.0)
        self.assertFalse(cc.imu_status(*r.poll(2.0))[0])


if __name__ == "__main__":
    unittest.main()
