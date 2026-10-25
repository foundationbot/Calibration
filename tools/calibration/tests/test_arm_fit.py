"""Arm session, Task 7: the camera-to-arm fit (armfit.py) on synthetic sessions with known answers: holds found in
/joints, the models M0/M1/M2 fitted from pixels, the held-out residual at the plate, noise floor, plate slip and the
plate's board-scale depth error."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
import armchain  # noqa: E402
import armfit  # noqa: E402
import armplan  # noqa: E402
import neckfk  # noqa: E402
import rigcfg  # noqa: E402
import rigmath as rm  # noqa: E402

URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
GRID = rigcfg.AprilGrid(2, 2, 0.045, 0.2)
INTR = np.array([565.0, 564.0, 640.0, 400.0, -0.008, 0.012, -0.0005, -0.0014, 0.0008, -0.0011])
Q0 = np.radians([-40, -5, 45, -90, -135, 20, 20])
X_TRUE = rm.transform(rm.rot_axis((1, 0, 0), np.pi / 2), (0.03, 0.02, -0.2))


def _rig():
    """A stereo rig in the Head (T_HC) placed 0.5 m in front of the plate at Q0 with the neck at (0, 0.35)."""
    arm, neck = armchain.ArmChain(URDF), neckfk.NeckChain(URDF)
    P, _ = armplan.plate_points(GRID)
    T_TP = arm.T_base_tip(Q0) @ X_TRUE
    flip = np.diag([1.0, -1.0, -1.0])
    T_TC = T_TP @ rm.transform(flip, P.mean(0) - flip @ np.array([0, 0, 0.5]))
    T_HC0 = rm.inv(neck.T_TB(0.0, 0.35)) @ T_TC
    T_C0C1 = rm.transform(rm.rot_axis((0, 1, 0), 0.002), (0.0443, 0, 0))
    lens = [("KB4T", INTR), ("KB4T", INTR + np.r_[2.0, 2.0, -3.0, 1.0, np.zeros(6)])]
    return arm, neck, armfit.Rig(T_HC=[T_HC0, T_HC0 @ T_C0C1], lens=lens, size=(1280, 800)), P


def _session(arm, neck, rig, P, truth, n_poses=16, frames=6, px=0.2, seed=0, spread=0.25):
    """Frames of n_poses holds: arm angles near Q0 (as the encoders read them), the neck at (0, 0.35); the pixels made
    with the TRUE parameters truth = (X, dT, offsets on armfit.INNER) and px noise."""
    rng = np.random.default_rng(seed)
    X, dT, off = truth
    obs = []
    for h in range(n_poses):
        q = Q0 + rng.uniform(-spread, spread, 7)
        for f in range(frames):
            qf = q + rng.normal(0, 1e-4, 7)
            qt = qf.copy()
            qt[list(armfit.INNER)] += off
            T_TP = arm.T_base_tip(qt) @ X
            for c in (0, 1):
                T_TC = dT @ neck.T_TB(0.0, 0.35) @ rig.T_HC[c]
                Pc = (rm.inv(T_TC) @ T_TP @ np.c_[P, np.ones(len(P))].T).T[:, :3]
                uv = armplan.lensmodel.project(rig.lens[c][0], rig.lens[c][1], Pc) + rng.normal(0, px, (len(P), 2))
                obs.append(armfit.Frame(hold=h, cam=c, t=h * 10_000_000_000 + f * 70_000_000, q_arm=qf, q_neck=(0.0, 0.35), P=P, uv=uv))
    return obs


def test_holds_are_the_still_stretches_of_the_arm():
    t = np.arange(0, 20e9, 2e6).astype(np.int64)                   # 500 Hz for 20 s
    q = np.zeros((len(t), 7))
    s = t / 1e9
    q[:, 0] = np.interp(s, [0, 3, 6, 9, 12, 15, 20], [0, 0, 0.5, 0.5, 1.0, 1.0, 1.0])
    holds = armfit.holds_from_joints(t, q, min_s=1.5)
    want = [(0.0, 3.0), (6.0, 9.0), (12.0, 20.0)]
    assert len(holds) == 3                                  # edges within half the speed window (0.15 s)
    assert all(abs(a / 1e9 - wa) <= 0.16 and abs(b / 1e9 - wb) <= 0.16 for (a, b), (wa, wb) in zip(holds, want))


def test_m0_recovers_the_plate_and_leaves_noise_when_the_chain_is_right():
    arm, neck, rig, P = _rig()
    obs = _session(arm, neck, rig, P, (X_TRUE, np.eye(4), np.zeros(5)))
    X0 = X_TRUE @ rm.transform(rm.rot_axis((0, 0, 1), 0.05), (0.004, -0.003, 0.006))     # a wrong start
    fit = armfit.fit(obs, arm, neck, rig, "M0", X0)
    assert fit["rms_px"] < 0.3
    assert np.linalg.norm(fit["X"][:3, 3] - X_TRUE[:3, 3]) < 2e-4
    assert rm.rot_angle_deg(fit["X"][:3, :3].T @ X_TRUE[:3, :3]) < 0.02


def test_m1_finds_a_camera_offset_that_m0_cannot_absorb_and_m2_the_joint_offsets():
    arm, neck, rig, P = _rig()
    dT = rm.transform(rm.rot_axis((1, 1, 0), np.radians(0.4)), (0.006, -0.004, 0.003))
    off = np.radians([0.0, 0.0, 0.6, 0.0, 0.0])                       # the elbow's zero off by 0.6 deg
    obs = _session(arm, neck, rig, P, (X_TRUE, dT, off), n_poses=20)
    m0 = armfit.fit(obs, arm, neck, rig, "M0", X_TRUE)
    m1 = armfit.fit(obs, arm, neck, rig, "M1", X_TRUE)
    m2 = armfit.fit(obs, arm, neck, rig, "M2", X_TRUE)
    assert m0["rms_px"] > 1.0 > m1["rms_px"] > m2["rms_px"]
    assert m2["rms_px"] < 0.3
    assert abs(np.degrees(m2["offsets"][2]) - 0.6) < 0.03
    assert np.linalg.norm(m2["dT"][:3, 3] - dT[:3, 3]) < 1e-3


def test_plate_error_on_held_out_holds_is_zero_for_the_true_model_and_the_mm_of_a_wrong_one():
    arm, neck, rig, P = _rig()
    obs = _session(arm, neck, rig, P, (X_TRUE, np.eye(4), np.zeros(5)), px=0.05, spread=0.12)   # plate 0.4-0.6 m
    good = armfit.plate_errors(obs, arm, neck, rig, {"X": X_TRUE, "dT": np.eye(4), "offsets": np.zeros(5)})
    assert max(good.values()) < 0.2                                  # mm
    shifted = X_TRUE @ rm.transform(np.eye(3), (0.0, 0.0, 0.003))     # the plate 3 mm along its own z
    bad = armfit.plate_errors(obs, arm, neck, rig, {"X": shifted, "dT": np.eye(4), "offsets": np.zeros(5)})
    assert all(abs(v - 3.0) < 0.2 for v in bad.values())


def test_choose_takes_the_smallest_model_within_half_a_millimetre_of_the_best():
    assert armfit.choose({"M0": 4.0, "M1": 1.4, "M2": 1.0}) == "M1"
    assert armfit.choose({"M0": 1.2, "M1": 1.0, "M2": 0.9}) == "M0"
    assert armfit.choose({"M0": 4.0, "M1": 2.0, "M2": 1.0}) == "M2"


def test_board_scale_reads_a_too_long_baseline_as_depth_too_far():
    """Triangulating with a baseline 1 % longer than the rig's real one puts every point 1 % further and the plate
    1 % larger: board scale +1 %."""
    arm, neck, rig, P = _rig()
    obs = _session(arm, neck, rig, P, (X_TRUE, np.eye(4), np.zeros(5)), n_poses=3, px=0.0)
    assert abs(armfit.board_scale_pct(obs, rig, GRID)[0]) < 0.01
    T01 = rm.inv(rig.T_HC[0]) @ rig.T_HC[1]
    long = armfit.Rig(T_HC=[rig.T_HC[0], rig.T_HC[0] @ rm.transform(T01[:3, :3], T01[:3, 3] * 1.01)], lens=rig.lens,
                      size=rig.size)
    s = armfit.board_scale_pct(obs, long, GRID)
    assert all(abs(v - 1.0) < 0.05 for v in s.values())


def test_holds_are_matched_to_the_logged_stops_by_where_the_arm_stood():
    hold_q = np.radians([[0, 0, 0, 0, 0, 0, 0], [10, 0, 0, 0, 0, 0, 0], [10.2, 0, 0, 0, 0, 0, 0],
                         [20, 0, 0, 0, 0, 0, 0], [10, 0, 0, 0, 0, 0, 0]])
    stops = [{"stop": 0, "kind": "fit", "pose": 3, "status": "ok", "reached_deg": [10, 0, 0, 0, 0, 0, 0]},
             {"stop": 1, "kind": "pre", "pose": 3, "status": "ok", "reached_deg": [20, 0, 0, 0, 0, 0, 0]},
             {"stop": 2, "kind": "repeat_same", "pose": 3, "status": "ok", "reached_deg": [10.05, 0, 0, 0, 0, 0, 0]},
             {"stop": 3, "kind": "heldout", "pose": 4, "status": "skipped", "reached_deg": [30, 0, 0, 0, 0, 0, 0]}]
    m = armfit.match_holds(hold_q, stops)
    assert m == {1: stops[0], 4: stops[2]}       # in order; the pre-pose and the skipped stop are not measured


def test_verdict_follows_decision_5():
    ok = {"noise_floor_mm": 0.4, "slip_mm": 0.3, "slip_deg": 0.05}
    assert armfit.verdict({**ok, "heldout_rms_mm": {"M0": 2.5, "M1": 2.2, "M2": 2.1}})["verdict"] == "URDF suffices"
    v = armfit.verdict({**ok, "heldout_rms_mm": {"M0": 8.0, "M1": 2.0, "M2": 1.9}})
    assert v["verdict"] == "calibrated (M1)" and v["model"] == "M1"
    assert armfit.verdict({**ok, "heldout_rms_mm": {"M0": 8.0, "M1": 4.5, "M2": 3.8}})["verdict"] == "acceptable (M2)"
    assert armfit.verdict({**ok, "heldout_rms_mm": {"M0": 9.0, "M1": 7.0, "M2": 6.0}})["verdict"].startswith(
        "model missing something")
    assert armfit.verdict({**ok, "noise_floor_mm": 1.4, "heldout_rms_mm": {"M0": 2.0}})["verdict"] == "INCONCLUSIVE"
    assert armfit.verdict({**ok, "slip_deg": 0.3, "heldout_rms_mm": {"M0": 2.0}})["verdict"] == "VOID (plate slipped)"


def test_holds_need_no_sample_inside_the_speed_window():
    """At 50 Hz the 20 ms window holds no neighbour: the speed must still come from the adjacent samples, and a
    speed that cannot be computed is not 'still'."""
    t = np.arange(0, 20e9, 20e6).astype(np.int64)
    q = np.zeros((len(t), 7))
    q[:, 0] = np.interp(t / 1e9, [0, 3, 6, 9, 12, 20], [0, 0, 0.5, 0.5, 1.0, 1.0])
    holds = armfit.holds_from_joints(t, q, min_s=1.5)
    want = [(0.0, 3.0), (6.0, 9.0), (12.0, 20.0)]
    assert len(holds) == 3
    assert all(abs(a / 1e9 - wa) <= 0.16 and abs(b / 1e9 - wb) <= 0.16 for (a, b), (wa, wb) in zip(holds, want))


def test_holds_survive_encoder_jitter_and_still_split_at_a_slow_correction():
    """Review 2, W3: real /joints jitter by an encoder step (~0.01 deg) from sample to sample; a hold must stay one
    hold. A correction (4 deg in 6 s, the slowest move step 91 sends) must still split two holds."""
    rng = np.random.default_rng(3)
    t = np.arange(0, 30e9, 2e6).astype(np.int64)
    s = t / 1e9
    q = np.zeros((len(t), 7))
    q[:, 3] = np.radians(np.interp(s, [0, 10, 16, 30], [0, 0, 4.0, 4.0]))
    q += np.radians(0.01) * rng.integers(-1, 2, q.shape)
    got = [(round(a / 1e9), round(b / 1e9)) for a, b in armfit.holds_from_joints(t, q, min_s=1.5)]
    assert got == [(0, 10), (16, 30)]


def test_a_missing_noise_floor_or_slip_is_inconclusive():
    h = {"M0": 1.0, "M1": 0.9, "M2": 0.9}
    assert armfit.verdict({"noise_floor_mm": float("nan"), "slip_mm": 0.1, "slip_deg": 0.01,
                           "heldout_rms_mm": h})["verdict"] == "INCONCLUSIVE"
    assert armfit.verdict({"noise_floor_mm": 0.3, "slip_mm": float("nan"), "slip_deg": 0.01,
                           "heldout_rms_mm": h})["verdict"] == "INCONCLUSIVE"


def test_board_scale_pairs_the_cameras_by_the_nearest_stamp():
    """Review 2, W2: the record app stamps each camera with its own capture time; camera 1 here 0.3 ms later."""
    arm, neck, rig, P = _rig()
    obs = _session(arm, neck, rig, P, (X_TRUE, np.eye(4), np.zeros(5)), n_poses=3, px=0.0)
    for f in obs:
        if f.cam == 1:
            f.t += 300_000
    s = armfit.board_scale_pct(obs, rig, GRID)
    assert sorted(s) == [0, 1, 2] and all(abs(v) < 0.01 for v in s.values())


def test_m2_fits_only_the_offsets_the_plan_lists():
    """With wrist roll and gripper yaw frozen, step 90 lists offsets (1, 2, 3, 4); M2 then has 16 parameters and
    plate_offsets applies them to the same joints."""
    arm, neck, rig, P = _rig()
    off = np.radians([0.0, 0.0, 0.6, 0.0, 0.0])
    obs = _session(arm, neck, rig, P, (X_TRUE, np.eye(4), off), n_poses=20, px=0.05, spread=0.15)
    m2 = armfit.fit(obs, arm, neck, rig, "M2", X_TRUE, offsets=(1, 2, 3, 4))
    assert len(m2["offsets"]) == 4 and m2["offset_joints"] == [1, 2, 3, 4]
    assert abs(np.degrees(m2["offsets"][2]) - 0.6) < 0.05 and m2["rms_px"] < 0.3
    errs = armfit.plate_errors(obs, arm, neck, rig, m2, holds=[0, 1])
    assert max(errs.values()) < 0.3
