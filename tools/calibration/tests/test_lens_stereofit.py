"""stereofit: the KB4T stereo fit from the rig's nominal start, on synthetic corners whose truth is Fit A.
Each test is a check that can fail: the recovery tolerances sit ~2-4x above what 48 holds at 0.3 px give (scratch run
of 20 Oct), and every mutation must make the fit measurably worse."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = importlib.import_module("lens_synth")
sf = S.sf
rm = importlib.import_module("rigmath")


def test_data_layout_and_select_renumber():
    d = S.views([S.I0, S.I1], S.C_TRUE, 6, seed=3, per_hold=2)
    data = sf.Data(d, S.BOARD)
    assert data.n == 12 and data.hold.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    sel = data.select(data.hold % 2 == 1)
    assert sel.n == 6 and sel.hold.tolist() == [1, 1, 3, 3, 5, 5]
    assert sorted(set(sel.pair.tolist())) == list(range(6))
    assert len(sel.P) == sum((data.hold[data.pair] % 2 == 1))


def test_residuals_vanish_at_the_truth_without_noise():
    d = S.views([S.I0, S.I1], S.C_TRUE, 9, seed=4, noise=0.0)
    data = sf.Data(d, S.BOARD)
    poses, ok = sf.pnp_poses(data, "KB4T", [S.I0, S.I1], S.C_TRUE)
    assert ok.all()
    r = sf.solve(data, "KB4T", [S.I0, S.I1], S.C_TRUE, poses, free_intr=False, free_C=False)
    assert np.abs(r["res"]).max() < 1e-6                        # projection, board and pose conventions agree


def test_nominal_start_is_the_rig_file_alone():
    intr, C = sf.nominal_start(S.rig12(), "KB4T")
    assert intr[0].tolist() == [568.0, 568.0, 639.5, 399.5] + [0.0] * 6
    assert np.allclose(C, np.r_[np.c_[np.eye(3), [0.0445, 0, 0]], [[0, 0, 0, 1]]])   # phantom12: camera 0 = left
    mk13 = sf.nominal_start(S.rigcfg.load(S.PKG / "config" / "rig.yaml"), "KB4")[1]
    assert mk13[0, 3] == pytest.approx(-0.079)                   # mk13: kalibr cam0 is the RIGHT eye


def test_kb4t_recovers_fit_a_from_the_nominal_start():
    data, r = S.nominal_fit("KB4T")
    assert r["success"] and data.n == 48
    for c, truth in ((0, S.I0), (1, S.I1)):
        err = np.abs(r["intr"][c] - truth)
        assert err[:4].max() < 0.6, err                          # fx, fy, cx, cy (px)
        assert err[4:8].max() < 0.004, err                       # k1..k4
        assert err[8:].max() < 2e-4, err                         # p1, p2 (truth up to 1.1e-3)
    dR = np.degrees(np.linalg.norm(rm.log_rot(r["C"][:3, :3] @ S.C_TRUE[:3, :3].T)))
    dt = 1e3 * np.abs(r["C"][:3, 3] - S.C_TRUE[:3, 3])              # mm
    assert dR < 0.02, dR
    # the baseline (x) and y to 0.1 mm; the forward offset z is weakly observed (it trades against the two focal
    # lengths) and second order for depth: 0.17 mm off here, so 0.3 mm
    assert dt[0] < 0.1 and dt[1] < 0.1 and dt[2] < 0.3, dt


def test_hold_rms_sees_the_noise_level():
    data, r = S.nominal_fit("KB4T")
    rms = np.array(list(sf.hold_rms(data, r["res"]).values()))
    assert 0.3 < np.median(rms) < 0.5                           # 0.3 px per axis = 0.42 px per corner
    assert rms.max() < 0.6                                       # no clean hold trips the 0.6 px gate


def test_a_21_deg_roll_is_reached_from_the_identity_start():
    """mk13's cameras are rolled 20.74 deg about the optical axis with 5.1 deg of vergence (README "The rig, as
    measured"); the nominal start knows nothing of it"""
    C = S.stereo_pose([0.0, -5.1, 20.74], S.C_TRUE[:3, 3])
    d = S.views([S.I0, S.I1], C, 30, seed=4)
    intr0, C0 = sf.nominal_start(S.rig12(), "KB4T")
    _, r = sf.fit_from_nominal(sf.Data(d, S.BOARD), "KB4T", intr0, C0)
    assert r["success"]
    assert np.degrees(np.linalg.norm(rm.log_rot(r["C"][:3, :3] @ C[:3, :3].T))) < 0.05
    assert 1e3 * np.linalg.norm(r["C"][:3, 3] - C[:3, 3]) < 0.3                  # 30 holds: 0.17 mm (20 Oct)


def test_mutation_a_board_one_percent_off_moves_the_baseline_one_percent():
    """the board is 1 % larger than the target file says: every length the fit returns, the baseline included,
    comes out 1 % short. Warm from the clean fit (the optimum is unique; this keeps the test short)"""
    _, r = S.nominal_fit("KB4T")
    small = S.rigcfg.AprilGrid(7, 5, 0.1 / 1.01, 0.2)
    _, r2 = sf.warm_fit(sf.Data(S.fit_sweep(), small), "KB4T", r["intr"], r["C"])
    ratio = np.linalg.norm(r2["C"][:3, 3]) / np.linalg.norm(S.C_TRUE[:3, 3])
    assert r2["success"] and ratio == pytest.approx(1 / 1.01, abs=5e-4)
