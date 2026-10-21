"""Step 10's camera clock handling.

CI's python-tests job installs pytest only, so every dependency of the package is
skipped-if-missing here rather than imported at module level.
"""
import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
build_trees = importlib.import_module("10_build_trees")

HOST_EPOCH_NS = 1_792_020_994_777_996_032  # a real phantom12 capture stamp, 2026-10-14


def _rows(capture0, delays_ms, period_ns=66_666_667):
    return [(capture0 + i * period_ns, capture0 + i * period_ns + int(d * 1e6))
            for i, d in enumerate(delays_ms)]


def test_host_clock_stamps_are_not_rebased():
    # DMA.video's producer already stamps capture time on the host clock; publish
    # follows 29-30 ms later (measured on phantom12). Adding min(publish - capture)
    # would shift every frame ~29 ms late: Kalibr then reports it as timeshift and
    # step 70 bakes it into image_delay.
    full = {"cam1": _rows(HOST_EPOCH_NS, [29.4, 30.1, 29.2, 31.0]),
            "cam2": _rows(HOST_EPOCH_NS, [29.6, 29.9, 29.3, 30.4])}
    assert build_trees.clock_offset_ns(full) == 0


def test_device_clock_stamps_are_rebased_by_the_common_minimum_delay():
    # A recording whose capture stamps really are a device clock (seconds since
    # the camera booted) still needs the old estimator: the fastest publish is the
    # best estimate of the clock difference, applied identically to both cameras.
    device0 = 5_000_000_000
    host0 = HOST_EPOCH_NS
    cam1 = [(device0 + i * 66_666_667, host0 + i * 66_666_667 + int(d * 1e6))
            for i, d in enumerate([30.0, 28.5, 29.0])]
    cam2 = [(device0 + i * 66_666_667, host0 + i * 66_666_667 + int(d * 1e6))
            for i, d in enumerate([29.5, 28.9, 31.0])]
    expected = min(min(p - c for c, p in cam1), min(p - c for c, p in cam2))
    assert build_trees.clock_offset_ns({"cam1": cam1, "cam2": cam2}) == expected
