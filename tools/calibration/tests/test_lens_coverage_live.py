"""Step 06's counting logic, without a camera: holds, cells, distance, the text map and the PNG."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("yaml")
pytest.importorskip("scipy")               # lens_synth imports stereofit, which needs scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = importlib.import_module("lens_synth")
cov = importlib.import_module("06_coverage_live")
PP = (639.5, 399.5)


def test_a_still_stretch_counts_once_when_it_has_lasted():
    tr = cov.StillTracker(still_px=2.0, still_s=1.0)
    seq = [(0.0, (500, 300)), (0.5, (501, 300)), (1.0, (500.5, 301)), (1.5, (500, 300)),     # counts at t 1.0
           (2.0, (530, 300)),                                                               # moved: a new stretch
           (2.5, (530, 300)), (2.9, None), (3.0, (530, 300)), (3.5, (530, 300)),            # board lost: restarts
           (4.0, (531, 300.5))]                                                             # 1.0 s after 3.0
    got = [tr.update(t, c) for t, c in seq]
    assert got == [False, False, True, False, False, False, False, False, False, True]


def test_holds_land_in_cells_and_distance_bins():
    counts = np.zeros((9, 3), int)
    assert cov.add_hold(counts, (100, 80), 0.5, PP) == ("top-left", 0)
    assert cov.add_hold(counts, (640, 400), 1.0, PP) == ("centre", 2)
    assert cov.add_hold(counts, (640, 400), 1.5, PP) is None                     # beyond 1.3 m: no bin
    assert counts.sum() == 2


def test_board_depth_from_its_known_size_within_one_percent():
    """a board rendered by phantom12's true lens, read with the nominal one (f 568, centre of the image)"""
    d = S.views([S.I0, S.I1], S.C_TRUE, 3, seed=31, noise=0.1, targets=[(643, 385)], z_range=(0.7, 0.7), jitter=(1, 1))
    for img in range(3):
        m = (d["det_img"] == img) & (d["det_cam"] == 0)
        det = {(int(t), int(k)): (u, v) for t, k, u, v in zip(d["det_tag"][m], d["det_k"][m], d["det_u"][m], d["det_v"][m])}
        centre, z = cov.observe(det, S.BOARD, 568.0, PP)
        assert z == pytest.approx(0.7, rel=0.015)                     # the whole board is seen: its centre at 0.7 m
        assert np.hypot(*(centre - [643, 385])) < 40


def test_the_text_map_says_what_is_missing_and_the_png_is_written(tmp_path):
    counts = np.zeros((9, 3), int)
    counts[4] = [0, 9, 3]                                                         # centre done
    counts[0] = [5, 0, 0]
    txt = cov.text_map(counts, 8)
    assert "0/9/3" in txt and "need 3" in txt and "still short: top-left" in txt and "centre" not in txt.split("short:")[1]
    cov.heat_png(counts, 8, PP, (1280, 800), tmp_path / "c.png")
    assert cv2.imread(str(tmp_path / "c.png")).shape == (200, 320, 3)
