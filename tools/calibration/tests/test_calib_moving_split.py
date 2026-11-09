"""Step 15's moving-board selection (steady_pairs, spaced) and step 16's distance split (rough_z, split)."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
d15 = importlib.import_module("15_dump_corners")
d16 = importlib.import_module("16_split_distance")
rigcfg = importlib.import_module("rigcfg")
RIG = rigcfg.load(SCRIPTS.parent / "config" / "rig-phantom12.yaml")


def _det(shift_px, n=12):
    """a fake detection: n corners on a grid, all moved by shift_px in u"""
    return {(t, k): (100.0 + 20 * t + shift_px, 200.0 + 30 * k) for t in range(n // 4) for k in range(4)}


def test_steady_pairs_keeps_slow_boards_and_drops_fast_ones():
    t = np.arange(6, dtype=np.int64) * 40_000_000          # 40 ms frames
    # speeds px/ms: frames 0-3 move 2 px per frame (0.05 px/ms: blur 0.1 px at 2 ms), then 40 px per frame
    pos = [0, 2, 4, 6, 46, 86]
    dets = [_det(p) for p in pos]
    kept = d15.steady_pairs(t, dets, max_blur_px=0.5, exposure_ms=2.0)
    assert kept == [1, 2]                                   # 3 has a fast neighbour (4), ends never have two


def test_steady_pairs_needs_both_neighbours_and_known_gaps():
    t = np.array([0, 40, 80, 400, 440], np.int64) * 1_000_000   # a 320 ms gap: dropped frames
    dets = [_det(0), None, _det(0), _det(0), _det(0)]
    assert d15.steady_pairs(t, dets, 0.5, 2.0) == []        # 1 has no board; 2 and 3 straddle the gap


def test_steady_pairs_blur_scales_with_exposure():
    t = np.arange(3, dtype=np.int64) * 40_000_000
    dets = [_det(0), _det(4), _det(8)]                      # 0.1 px/ms
    assert d15.steady_pairs(t, dets, 0.5, 2.0) == [1]       # 0.2 px blur
    assert d15.steady_pairs(t, dets, 0.5, 8.0) == []        # 0.8 px blur


def test_spaced_keeps_pairs_at_least_gap_apart():
    t = np.array([0, 100, 200, 300, 600, 650], np.int64) * 1_000_000
    assert d15.spaced([0, 1, 2, 3, 4, 5], t, 0.25) == [0, 3, 4]   # 300 -> 600 ms is >= 250 ms; 650 is not


def _corner_file(path, depths, f, res):
    """a step-15 file whose pairs see the board fronto-parallel at the given depths through a plain equidistant lens
    (r = f theta, the rig's pinhole-equi), camera 1 a 44 mm baseline to the right"""
    b = RIG.board
    rows = []
    for j, z in enumerate(depths):
        for tag in range(b.n_tags):
            for cam, bx in ((0, 0.0), (1, 0.044)):
                P = b.obj_points(tag) - [0.42 + bx, 0.30, 0] + [0, 0, z]
                rho = np.hypot(P[:, 0], P[:, 1])
                r = f * np.arctan2(rho, P[:, 2])
                uv = np.array([res[0] / 2, res[1] / 2]) + (r / rho)[:, None] * P[:, :2]
                for k in range(4):
                    rows.append((j, cam, tag, k, uv[k, 0], uv[k, 1]))
    r = np.array(rows)
    n = len(depths)
    np.savez(path, img_hold=np.arange(n), img_t0=np.arange(n) * 10, img_t1=np.arange(n) * 10 + 1,
             det_img=r[:, 0].astype(int), det_cam=r[:, 1].astype(int), det_tag=r[:, 2].astype(int),
             det_k=r[:, 3].astype(int), det_u=r[:, 4], det_v=r[:, 5])


def test_rough_z_recovers_the_board_depth(tmp_path):
    f, res = RIG.nominal_focal_px, RIG.resolution
    _corner_file(tmp_path / "a.npz", [0.6, 1.0, 1.8], f, res)
    z = d16.rough_z(dict(np.load(tmp_path / "a.npz")), RIG.board, f, res)
    assert np.allclose(z, [0.6, 1.0, 1.8], rtol=1e-3)


def test_split_puts_near_pairs_in_fit_far_in_test_and_keeps_corners_with_their_pair(tmp_path):
    f, res = RIG.nominal_focal_px, RIG.resolution
    _corner_file(tmp_path / "a.npz", [0.6, 1.5, 1.0], f, res)
    _corner_file(tmp_path / "b.npz", [2.0, 0.8], f, res)
    fit, test = d16.split([tmp_path / "a.npz", tmp_path / "b.npz"], RIG.board, f, res, 1.3)
    assert np.allclose(sorted(fit["img_z"]), [0.6, 0.8, 1.0], rtol=1e-3)
    assert np.allclose(sorted(test["img_z"]), [1.5, 2.0], rtol=1e-3)
    for d in (fit, test):
        n = len(d["img_t0"])
        assert set(d["det_img"].tolist()) == set(range(n))  # every pair keeps its corners, renumbered from 0
        assert np.allclose(d16.rough_z(d, RIG.board, f, res), d["img_z"], rtol=1e-3)  # and they are its own
    # each pair (one hold per pair here; b's holds offset past a's) lands in exactly one set
    assert sorted(fit["img_hold"].tolist() + test["img_hold"].tolist()) == [0, 1, 2, 3, 4]


def test_near_holdout_takes_every_nth_near_hold_in_time_order_and_no_far_one():
    holds = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    t0 = np.array([0, 1, 50, 20, 30, 40, 60, 70, 80, 90])          # hold 1 comes after hold 4 in time
    near = np.array([True] * 8 + [False, False])                    # holds 7, 8 are far
    held = d16.near_holdout(holds, t0, near, 2)
    # near holds in time order: 0, 2, 3, 4, 1, 5, 6 -> every 2nd: 2, 4, 5 (all pairs of each)
    assert sorted(set(holds[held].tolist())) == [2, 4, 5]
    assert not held[~near].any()
    assert not d16.near_holdout(holds, t0, near, 0).any()


def test_split_holds_out_near_boards_too(tmp_path):
    f, res = RIG.nominal_focal_px, RIG.resolution
    _corner_file(tmp_path / "a.npz", [0.6, 0.7, 0.8, 0.9, 2.0], f, res)
    fit, test = d16.split([tmp_path / "a.npz"], RIG.board, f, res, 1.3, holdout_every=4)
    assert np.allclose(sorted(test["img_z"]), [0.9, 2.0], rtol=1e-3)   # the 4th near hold, and the far one
    assert len(fit["img_t0"]) == 3


def test_neck_grid_is_serpentine_and_clamped():
    ng = importlib.import_module("07_neck_grid")
    poses = ng.grid([-30, 0, 15], [-40, 0, 40], max_pitch=26, max_yaw=34)
    assert poses == [(-26, -34), (-26, 0), (-26, 34), (0, 34), (0, 0), (0, -34), (15, -34), (15, 0), (15, 34)]
    cmd = ng.command(5.0, -13.0, "SpineYaw=-25.40", 1.0, 3.0)
    assert cmd[1] == "NeckPitch=5.00,NeckYaw=-13.00,SpineYaw=-25.40" and "--no-return" in cmd
    out = ("RESULT targets joint=14 target_deg=5 ss_err=-0.2 rms=0.2 max=0.3\n"
           "RESULT targets joint=15 target_deg=-13 ss_err=3.5 rms=1 max=4\n")
    assert ng.errors(out) == {14: -0.2, 15: 3.5}
