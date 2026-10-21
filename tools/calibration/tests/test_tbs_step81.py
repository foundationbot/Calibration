"""Step 81: board poses from images, the comparison with the config, the gate, play, takes."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import handeye  # noqa: E402
import neckfk  # noqa: E402
import necktake  # noqa: E402
import okviscfg  # noqa: E402
import rigcfg  # noqa: E402
import rigmath as rm  # noqa: E402
from test_calib_board import APRILGRID  # noqa: E402
from test_calib_validate import K, D, LENS  # noqa: E402

s81 = importlib.import_module("81_tbs_handeye")
URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
REAL = HERE.parent / "results/phantom12-2026-10-15/imucam/phantom12_stereo.yaml"
POSES = [(y, p) for y in (-0.35, -0.175, 0.0, 0.175, 0.35) for p in (-0.3, -0.15, 0.0, 0.15, 0.3)]


def _cfg():
    """phantom12's real T_SC pair and T_BS when this checkout has them, else a stand-in."""
    if REAL.exists():
        return okviscfg.load(REAL)
    T_SC0 = rm.transform(rm.rot_axis([1, 0, 0], -1.5), [-0.067, 0.147, 0.016])
    T_SC1 = T_SC0 @ rm.transform(np.eye(3), [0.0448, 0, 0])
    T_BS = rm.transform(rm.rot_axis([1, 0, 0], -0.6) @ rm.rot_axis([0, 0, 1], 1.57), [0.011, -0.046, 0.083])
    return okviscfg.OkvisRig([T_SC0, T_SC1], [K, K], [D, D], "pinhole-equi", T_BS, 0.0)


def _frames(cfg, T_BS_true, play_yaw_deg=0.0, jitter_deg=0.0, seed=0):
    """Board images of every pose from both cameras, as if the head carried T_BS_true. With play,
    the head sits play/2 short of the encoder's yaw, on the side it came from; with jitter, each
    hold's yaw is off its encoder by a random amount (1-sigma jitter_deg) that both images share."""
    c = neckfk.NeckChain(URDF)
    T_TW = c.T_TB(0, 0) @ T_BS_true @ cfg.T_SC[0] @ rm.transform(np.eye(3), [0, 0, 1.1])
    rng = np.random.default_rng(seed)
    out = []
    for h, (y, p) in enumerate(POSES):
        side = 1 if h % 2 else -1
        y_true = y - side * np.radians(play_yaw_deg) / 2 + np.radians(jitter_deg) * rng.standard_normal()
        for cam in (0, 1):
            T_CW = rm.inv(c.T_TB(y_true, p) @ T_BS_true @ cfg.T_SC[cam]) @ T_TW
            out.append({"hold": h, "cam": cam, "yaw": y, "pitch": p, "T_CW": T_CW,
                        "approach_yaw": side, "approach_pitch": 0})
    return out


def test_pose_from_corners_recovers_the_board_pose():
    board = rigcfg.board_from_target(APRILGRID)
    keys = [(i, k) for i in range(board.n_tags) for k in range(4)]
    X = np.ascontiguousarray(np.array([board.obj_points(i)[k] for i, k in keys]).reshape(-1, 1, 3))
    rv, tv = np.array([[0.2], [-0.1], [0.05]]), np.array([[-0.35], [-0.25], [1.1]])
    uv = cv2.fisheye.projectPoints(X, rv, tv, K, D)[0].reshape(-1, 2)
    T = s81.pose_from_corners({kk: tuple(v) for kk, v in zip(keys, uv)}, board, LENS, K, D,
                              centre=(640, 400), max_radius_px=900)
    assert np.allclose(T[:3, :3], cv2.Rodrigues(rv)[0], atol=1e-6) and np.allclose(T[:3, 3], tv.ravel(), atol=1e-6)


def test_compare_reports_head_axes_mm_and_the_gripper():
    X_ref = rm.transform(rm.rot_axis([0, 1, 0], 0.5), [0.14, 0.02, 0.01])
    X = rm.transform(rm.rot_axis([0, 1, 0], np.radians(0.5)), [0.001, 0.002, 0.002]) @ X_ref
    c = s81.compare(X, X_ref)
    assert np.allclose(c["rot_head_deg"], [0, 0.5, 0], atol=1e-9) and abs(c["angle_deg"] - 0.5) < 1e-9
    assert np.allclose(c["dist_mm"], np.linalg.norm(X[:3, 3] - X_ref[:3, 3]) * 1e3)
    assert abs(c["gripper_mm"] - (2 * 850 * np.sin(np.radians(0.25)) + c["dist_mm"])) < 1e-9
    assert s81.verdict(c) == "REPLACE"
    small = s81.compare(rm.transform(rm.rot_axis([1, 0, 0], np.radians(0.2)), [0.002, 0, 0]) @ X_ref, X_ref)
    assert s81.verdict(small) == "KEEP"


def test_right_config_is_kept():
    cfg = _cfg()
    rep = s81.run(_frames(cfg, cfg.T_BS), neckfk.NeckChain(URDF), cfg)
    assert rep["verdict"] == "KEEP" and rep["vs_config"]["angle_deg"] < 1e-4
    assert rep["spread_after"]["trans_rms_mm"] < 1e-3 and rep["warnings"] == []


def test_a_one_degree_pitch_error_is_measured_and_replaced():
    """The direction in question on phantom12 (30 vs 10 deg camera tilt): an error about Head y.
    The plan's first version (board spread with T_BS held fixed) barely flagged it; the fit names
    it to the hundredth of a degree and gives the new T_BS."""
    cfg = _cfg()
    T_BS_true = rm.transform(rm.rot_axis([0, 1, 0], np.radians(1.0)), [0.002, 0, -0.003]) @ cfg.T_BS
    rep = s81.run(_frames(cfg, T_BS_true), neckfk.NeckChain(URDF), cfg)
    assert rep["verdict"] == "REPLACE"
    assert np.allclose(rep["vs_config"]["rot_head_deg"], [0, 1.0, 0], atol=1e-3)
    assert np.allclose(np.array(rep["T_BS_measured"]), T_BS_true, atol=1e-6)
    assert rep["spread_before"]["rot_rms_deg"] > 0.1 and rep["spread_after"]["rot_rms_deg"] < 1e-3
    assert rep["residual_at_config"]["rot_deg"] > 0.1 and rep["residual_fitted"]["rot_deg"] < 1e-3


def test_play_is_measured_from_the_approach_side():
    cfg = _cfg()
    rep = s81.run(_frames(cfg, cfg.T_BS, play_yaw_deg=0.2), neckfk.NeckChain(URDF), cfg)
    p = rep["play"]                       # the head stops short of the encoder: positive, by the play
    assert abs(p["yaw_deg"] - 0.2) < 0.03 and p["pitch_deg"] is None and p["yaw_holds"] == [12, 13]
    assert p["yaw_se_deg"] is not None and p["yaw_se_deg"] < 0.01


def test_error_bars_are_clustered_by_hold_and_an_unseen_direction_is_named():
    """Joint play is an error a hold's images share: run() clusters X's error bars by hold. A take
    where the pitch never moved leaves part of X unmeasured, and the report says so."""
    cfg, chain = _cfg(), neckfk.NeckChain(URDF)
    frames = _frames(cfg, cfg.T_BS, play_yaw_deg=0.2)
    rep = s81.run(frames, chain, cfg)
    obs, meta = s81.observations(frames, chain, cfg)
    X_cfg = cfg.T_BS @ cfg.T_SC[0]
    clustered = handeye.fit(obs, X_cfg, groups=[m["hold"] for m in meta])["sigma_trans_head_mm"]
    pooled = handeye.fit(obs, X_cfg)["sigma_trans_head_mm"]
    assert np.allclose(rep["fit"]["sigma_trans_head_mm"], clustered) and not np.allclose(clustered, pooled, rtol=0.1)
    flat = [f for f in frames if f["pitch"] == 0.0]
    flat += [dict(f, hold=f["hold"] + 100) for f in flat]           # 10 holds, pitch at 0 in all
    rep = s81.run(flat, chain, cfg)
    assert any("cannot see every direction" in w for w in rep["warnings"]), rep["warnings"]


def test_two_takes_agree_or_not():
    X = rm.transform(np.eye(3), [0.14, 0.02, 0.01])
    rep = {"T_BC_measured": X.tolist()}
    near = {"T_BC_measured": (rm.transform(rm.rot_axis([0, 0, 1], np.radians(0.05)), [0.0005, 0, 0]) @ X).tolist()}
    far = {"T_BC_measured": (rm.transform(rm.rot_axis([0, 0, 1], np.radians(0.2)), [0, 0, 0]) @ X).tolist()}
    assert s81.compare_takes(rep, near)["agree"] and not s81.compare_takes(rep, far)["agree"]


def test_too_few_holds_is_insufficient():
    cfg = _cfg()
    rep = s81.run([f for f in _frames(cfg, cfg.T_BS) if f["hold"] < 4], neckfk.NeckChain(URDF), cfg)
    assert rep["verdict"] == "INSUFFICIENT"


def test_the_verdict_needs_its_error_bars_inside_the_gate():
    near = {"angle_deg": 0.1, "dist_mm": 1.0}
    far = {"angle_deg": 1.0, "dist_mm": 10.0}
    assert s81.verdict(near, 0.05, 0.5) == "KEEP"
    assert s81.verdict(near, 0.12, 0.5) == "REPLACE"               # not shown KEEP, but the fit is good to the gate
    assert s81.verdict(near, 0.16, 0.5) == "INCONCLUSIVE"          # 2 x 0.16 > 0.3: neither shown
    assert s81.verdict({"angle_deg": 0.05, "dist_mm": 1.0}, 0.02, 1.6) == "INCONCLUSIVE"   # millimetres of sigma count
    assert s81.verdict({"angle_deg": 0.2, "dist_mm": 6.0}, 0.05, 2.0) == "INCONCLUSIVE"    # for REPLACE too
    assert s81.verdict(far, 0.1, 1.0) == "REPLACE"
    assert s81.verdict(far, 0.54, 11.3) == "INCONCLUSIVE"          # A3: its own sigma over the gate
    assert s81.verdict(near, 0.0, 0.0, usable=False) == "INCONCLUSIVE"
    assert s81.verdict({"angle_deg": 0.2, "dist_mm": 5.0}) == "REPLACE"   # millimetres count too


def test_a_config_shown_outside_the_gate_is_improved_on_by_a_fit_not_good_to_it():
    """17 Oct, powered neck: the config 10 mm off, the fit good to ~2 mm (1 sigma): 2 sigma is over the
    3 mm gate, so not REPLACE; but 10 - 2 x 2 = 6 mm is still beyond the gate: the config is shown wrong."""
    assert s81.verdict({"angle_deg": 0.39, "dist_mm": 10.1}, 0.06, 2.2) == "IMPROVE"
    assert s81.verdict({"angle_deg": 1.0, "dist_mm": 1.0}, 0.2, 0.5) == "IMPROVE"       # by the angle alone
    assert s81.verdict({"angle_deg": 0.39, "dist_mm": 7.3}, 0.06, 2.2) == "INCONCLUSIVE"  # 7.3 - 4.4 < 3: not shown
    assert s81.verdict({"angle_deg": 4.2, "dist_mm": 64.0}, 2.3, 45.0) == "INCONCLUSIVE"  # A3 by hand: sigma too big
    assert s81.verdict({"angle_deg": 0.39, "dist_mm": 10.1}, 0.06, 2.2, usable=False) == "INCONCLUSIVE"
    # with the cameras' systematic (17 Oct P2: 0.41 deg / 10.3 mm, sigma 0.11 deg / 3.4 mm): outside the gate,
    # but 2 x 3.4 is not inside it and 10.3 is not over 4 sigma: the fit is not shown better than the config
    assert s81.verdict({"angle_deg": 0.41, "dist_mm": 10.31}, 0.109, 3.43) == "INCONCLUSIVE"
    # one part shown outside, the other unknown: not IMPROVE (step 83 would write a rotation unknown to 10 deg)
    assert s81.verdict({"angle_deg": 0.1, "dist_mm": 100.0}, 10.0, 0.5) == "INCONCLUSIVE"
    assert s81.verdict({"angle_deg": 5.0, "dist_mm": 1.0}, 0.05, 50.0) == "INCONCLUSIVE"


def test_takes_agree_within_their_error_bars_only_when_each_is_good_to_the_gate():
    X = rm.transform(np.eye(3), [0.14, 0.02, 0.01])
    off = (rm.transform(rm.rot_axis([0, 0, 1], np.radians(0.05)), [0.0023, 0, 0]) @ X).tolist()
    a = {"T_BC_measured": X.tolist(), "sigma_deg": 0.063, "sigma_mm": 2.3}
    b = {"T_BC_measured": off, "sigma_deg": 0.061, "sigma_mm": 2.1}           # P1 and P2, 17 Oct
    c = s81.compare_takes(a, b)
    assert c["agree"] and c["agree_sigma"] and not c["agree_limits"]
    loose = {"T_BC_measured": off, "sigma_deg": 0.8, "sigma_mm": 22.0}        # A2 by hand: inside 2 sigma, not precise
    c = s81.compare_takes(a, loose)
    assert not c["agree"] and not c["agree_sigma"]
    far = {"T_BC_measured": (rm.transform(np.eye(3), [0.008, 0, 0]) @ X).tolist(), "sigma_deg": 0.06, "sigma_mm": 2.1}
    assert not s81.compare_takes(a, far)["agree"]                            # 8 mm > 2 x 3.1 mm
    near = {"T_BC_measured": (rm.transform(np.eye(3), [0.0005, 0, 0]) @ X).tolist(), "sigma_deg": 0.5, "sigma_mm": 9.0}
    c = s81.compare_takes(a, near)
    assert c["agree"] and c["agree_limits"] and not c["agree_sigma"]          # the fixed limits need no sigma
    at_gate = {"T_BC_measured": X.tolist(), "sigma_deg": 0.3, "sigma_mm": 3.0}
    wide = {"T_BC_measured": (rm.transform(rm.rot_axis([0, 0, 1], np.radians(0.84)), [0.0084, 0, 0]) @ X).tolist(),
            "sigma_deg": 0.3, "sigma_mm": 3.0}
    c = s81.compare_takes(at_gate, wide)                                     # inside 2 combined sigma, 3x the gate apart
    assert not c["agree"] and not c["agree_sigma"]
    assert "further apart than the gate" in c["why_not"]


def test_takes_agree_on_their_own_sigma_the_cameras_systematic_being_common_to_both():
    """17 Oct with the cameras' systematic: P1 and P2 2.1 mm apart, each fit's own sigma ~2.4 mm, each total
    ~3.5 mm (the same 2.6 mm systematic in both: the same cameras). The total is over the gate; the own is not."""
    X = rm.transform(np.eye(3), [0.14, 0.02, 0.01])
    p1 = {"T_BC_measured": X.tolist(), "sigma_deg": 0.107, "sigma_mm": 3.63, "sigma_stat_deg": 0.070, "sigma_stat_mm": 2.54}
    p2 = {"T_BC_measured": (rm.transform(rm.rot_axis([0, 0, 1], np.radians(0.044)), [0.0021, 0, 0]) @ X).tolist(),
          "sigma_deg": 0.109, "sigma_mm": 3.43, "sigma_stat_deg": 0.065, "sigma_stat_mm": 2.26}
    c = s81.compare_takes(p1, p2)
    assert c["agree"] and c["agree_sigma"] and abs(c["combined_sigma_mm"] - np.hypot(2.54, 2.26)) < 1e-9
    loose = {**p2, "sigma_stat_mm": 3.2}
    c = s81.compare_takes(p1, loose)
    assert not c["agree"] and c["why_not"] == ["over the limits 0.1 deg / 1.0 mm", "a take's statistical sigma is over the gate"]


def test_compare_puts_a_translation_on_the_right_head_axis_and_takes_differ_in_mm():
    X_ref = rm.transform(rm.rot_axis([0, 1, 0], 0.5) @ rm.rot_axis([1, 0, 0], 1.2), [0.14, 0.02, 0.01])
    X = rm.transform(np.eye(3), [0.0, 0.004, 0.0]) @ X_ref           # 4 mm along Head y
    assert np.allclose(s81.compare(X, X_ref)["trans_head_mm"], [0, 4, 0], atol=1e-9)
    a = {"T_BC_measured": X_ref.tolist(), "sigma_deg": 0.03, "sigma_mm": 0.3}
    b = {"T_BC_measured": X.tolist(), "sigma_deg": 0.04, "sigma_mm": 0.4}
    c = s81.compare_takes(a, b)
    assert not c["agree"] and abs(c["combined_sigma_mm"] - 0.5) < 1e-9


def test_outliers_are_named_by_rotation_and_by_translation():
    hr = [{"hold": k, "n": 2, "rot_deg": 0.05, "trans_mm": 0.5} for k in range(10)]
    hr[3]["trans_mm"] = 5.0
    hr[7]["rot_deg"] = 0.5
    assert s81.outlier_holds(hr) == [3, 7]


def test_a_neck_unlike_its_urdf_is_inconclusive_with_the_spread_named():
    """The pitch axis 1 deg off the URDF's: the fit leaves a spread, and no verdict is given."""
    cfg, true = _cfg(), neckfk.NeckChain(URDF)
    true.pitch = neckfk.Joint("NeckPitch", true.pitch.parent, true.pitch.child, true.pitch.origin,
                              rm.rot_axis([1, 0, 0], np.radians(1.0)) @ np.array([0, 1, 0.]))
    c = neckfk.NeckChain(URDF)
    T_TW = c.T_TB(0, 0) @ cfg.T_BS @ cfg.T_SC[0] @ rm.transform(np.eye(3), [0, 0, 1.1])
    frames = [{"hold": h, "cam": cam, "yaw": y, "pitch": p, "approach_yaw": 0, "approach_pitch": 0,
               "T_CW": rm.inv(true.T_TB(y, p) @ cfg.T_BS @ cfg.T_SC[cam]) @ T_TW}
              for h, (y, p) in enumerate(POSES) for cam in (0, 1)]
    rep = s81.run(frames, c, cfg)
    assert rep["verdict"] == "INCONCLUSIVE" and "the model does not fit" in rep["inconclusive_because"]
    assert any("large spread" in w for w in rep["warnings"])


def test_few_holds_give_no_verdict_unless_asked():
    cfg = _cfg()
    few = [f for f in _frames(cfg, cfg.T_BS) if f["hold"] < 12]
    rep = s81.run(few, neckfk.NeckChain(URDF), cfg)
    assert rep["verdict"] == "INCONCLUSIVE" and rep["inconclusive_because"] == ["12 holds < 20"]
    assert s81.run(few, neckfk.NeckChain(URDF), cfg, min_verdict_holds=6)["verdict"] == "KEEP"


def test_pose_from_corners_keeps_the_inner_board_and_refuses_bad_fits():
    board = rigcfg.board_from_target(APRILGRID)
    keys = [(i, k) for i in range(board.n_tags) for k in range(4)]
    X = np.ascontiguousarray(np.array([board.obj_points(i)[k] for i, k in keys]).reshape(-1, 1, 3))
    rv, tv = np.array([[0.2], [-0.1], [0.05]]), np.array([[-0.35], [-0.25], [1.1]])
    uv = cv2.fisheye.projectPoints(X, rv, tv, K, D)[0].reshape(-1, 2)
    det = {kk: tuple(v) for kk, v in zip(keys, uv)}
    full = s81.pose_from_corners(det, board, LENS, K, D, centre=(640, 400), max_radius_px=900)
    assert full is not None
    assert s81.pose_from_corners(det, board, LENS, K, D, centre=(640, 400), max_radius_px=900, min_tags=36) is None
    inner = {t for (t, k), (u, v) in det.items() if np.hypot(u - 640, v - 400) <= 90}
    assert 0 < len(inner) < 20                                       # the radius leaves fewer than 20 tags
    assert s81.pose_from_corners(det, board, LENS, K, D, centre=(640, 400), max_radius_px=90) is None
    bad = dict(det)
    for kk in keys[::3]:
        bad[kk] = (bad[kk][0] + 3.0, bad[kk][1] - 3.0)               # a third of the corners 4 px off
    assert s81.pose_from_corners(bad, board, LENS, K, D, centre=(640, 400), max_radius_px=900) is None
    # a board toward the image edge (360-610 px out) with 0.3 px corner noise: the pixel refinement
    # ends below the undistorted-plane PnP's pixel rms (here by ~1e-3 px; more on the real fisheye)
    edge = cv2.fisheye.projectPoints(X, rv, np.array([[0.35], [0.2], [0.55]]), K, D)[0].reshape(-1, 2)
    c = edge + 0.3 * np.random.default_rng(2).standard_normal(edge.shape)
    noisy = {kk: tuple(v) for kk, v in zip(keys, c)}
    obj = np.array([board.obj_points(i)[k] for i, k in keys], np.float64)
    before = importlib.import_module("60_validate_stereo")._rms_reproj(LENS, c, obj, K, D)
    T = s81.pose_from_corners(noisy, board, LENS, K, D, centre=(640, 400), max_radius_px=900)
    pr, _ = LENS.project(obj.reshape(-1, 1, 3), cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))[0], np.ascontiguousarray(T[:3, 3]).reshape(3, 1), K, D)
    after = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - c) ** 2, axis=1))))
    assert after < before - 5e-4, (before, after)


def test_images_in_a_joints_gap_are_found():
    t = np.int64(1_792_000_000_000_000_000) + np.r_[np.arange(0, 1e9, 2e6), np.arange(1.5e9, 2e9, 2e6)].astype(np.int64)
    j = necktake.Joints(t, np.zeros(len(t)), np.zeros(len(t)))
    q = t[0] + np.array([0.5e9, 1.2e9, 1.7e9]).astype(np.int64)
    assert list(s81.in_joint_gap(j, q)) == [False, True, False]
    t2 = np.r_[t[:10], t[9] + 30_000_000, t[9] + 45_000_000]                # 30 ms, then 15 ms apart
    j2 = necktake.Joints(t2, np.zeros(len(t2)), np.zeros(len(t2)))
    assert list(s81.in_joint_gap(j2, [t2[9] + 10_000_000, t2[10] + 5_000_000])) == [True, False]


def test_a_config_just_inside_the_gate_is_not_kept_when_the_error_bar_reaches_out():
    """The config 0.25 deg off, holds jittering 0.08 deg: the difference is inside the 0.3 deg gate
    but the fit's 2 sigma takes it over, so the config is not shown good; the fit itself is good to
    the gate, so it replaces the config (ignoring sigma would say KEEP)."""
    cfg = _cfg()
    T_BS_true = rm.transform(rm.rot_axis([0, 1, 0], np.radians(0.25)), [0, 0, 0]) @ cfg.T_BS
    rep = s81.run(_frames(cfg, T_BS_true, jitter_deg=0.08, seed=3), neckfk.NeckChain(URDF), cfg)
    assert rep["vs_config"]["angle_deg"] < 0.3 and rep["vs_config"]["dist_mm"] < 3.0
    assert rep["inconclusive_because"] == [] and rep["verdict"] == "REPLACE", (rep["sigma_deg"], rep["vs_config"])


BOARD = np.array([[0.12 * i - 0.3, 0.12 * j - 0.18, 0.0] for i in range(6) for j in range(4)])


def _with_corners(cfg, frames, noise_px=0.0, seed=0, C1_render=None):
    """Each frame's corners and pixels as its camera sees the board (BOARD, centred on the board origin);
    C1_render: draw camera 1 at this pose in camera 0 instead of the config's (a wrong stereo pose)."""
    import handeye_px
    rng = np.random.default_rng(seed)
    kind = rigcfg.Lens(cfg.lens).kind
    C1 = rm.inv(cfg.T_SC[0]) @ cfg.T_SC[1]
    out = []
    for f in frames:
        T = f["T_CW"]
        if f["cam"] == 1 and C1_render is not None:
            T = rm.inv(C1_render) @ C1 @ T           # the board as a camera 1 at C1_render sees it
        uv = handeye_px.project(BOARD @ T[:3, :3].T + T[:3, 3], cfg.K[f["cam"]], cfg.D[f["cam"]], kind)
        out.append({**f, "P": BOARD.copy(), "uv": uv + noise_px * rng.standard_normal(uv.shape)})
    return out


def test_pixels_keep_the_right_config():
    cfg = _cfg()
    rep = s81.run_pixels(_with_corners(cfg, _frames(cfg, cfg.T_BS)), neckfk.NeckChain(URDF), cfg)
    assert rep["verdict"] == "KEEP" and rep["vs_config"]["angle_deg"] < 1e-4 and rep["fit"]["rms_px"] < 1e-3


def test_pixels_measure_a_one_degree_error_with_both_cameras_and_each_alone():
    cfg = _cfg()
    T_true = rm.transform(rm.rot_axis([0, 1, 0], np.radians(1.0)), [0, 0, 0]) @ cfg.T_BS
    rep = s81.run_pixels(_with_corners(cfg, _frames(cfg, T_true), noise_px=0.1), neckfk.NeckChain(URDF), cfg)
    X_true = T_true @ cfg.T_SC[0]
    assert rep["verdict"] == "REPLACE" and rep["model"] == "pixels"
    assert rm.rot_angle_deg(np.array(rep["T_BC_measured"])[:3, :3] @ X_true[:3, :3].T) < 0.01
    assert all(abs(pc["vs_config"]["angle_deg"] - 1.0) < 0.02 for pc in rep["per_camera"].values())
    ca = rep["cameras_apart"]
    assert ca["dist_mm"] < 1.0 and abs(rep["sigma_sys_mm"] - ca["dist_mm"] / 2) < 1e-9 and not rep["inconclusive_because"]
    assert abs(rep["sigma_mm"] - np.hypot(rep["sigma_stat_mm"], rep["sigma_sys_mm"])) < 1e-9
    for k in ("T_BC_measured", "T_BS_measured", "sigma_deg", "sigma_mm", "vs_config", "fit", "per_camera"):
        assert k in rep


def test_pixels_say_the_model_does_not_fit_a_head_that_is_not_where_its_encoders_are():
    cfg = _cfg()
    frames = _with_corners(cfg, _frames(cfg, cfg.T_BS, jitter_deg=0.4))     # a hand loading the neck, 16 Oct
    rep = s81.run_pixels(frames, neckfk.NeckChain(URDF), cfg)
    assert rep["fit"]["rms_px"] > s81.MAX_FIT_PX and "the model does not fit" in rep["inconclusive_because"]
    assert rep["verdict"] == "INCONCLUSIVE"


def test_pixels_say_the_model_does_not_fit_when_many_corners_are_cut_however_well_the_rest_fit():
    """Cutting a misfit's corners would hide it (16 Oct by hand: 18-78 % cut): 5 % of the corners 20 px
    off leave a near-perfect fit of the rest, and still the take is not a measurement."""
    cfg = _cfg()
    frames = _with_corners(cfg, _frames(cfg, cfg.T_BS), noise_px=0.1)
    for k, f in enumerate(frames):
        f["uv"][k % 4::20] += 20.0                                         # every 20th corner of every image
    rep = s81.run_pixels(frames, neckfk.NeckChain(URDF), cfg)
    assert rep["fit"]["rms_px"] < 0.5 and rep["fit"]["outlier_fraction"] > s81.MAX_OUTLIER_FRACTION
    assert "the model does not fit" in rep["inconclusive_because"] and rep["verdict"] == "INCONCLUSIVE"


def test_pixels_carry_a_disagreement_between_the_cameras_as_a_systematic():
    """17 Oct: the cameras' own fits 5 mm apart in every half of both takes (their calibrations disagree).
    Here camera 1 really sits 4 mm off its calibrated stereo pose, with per-hold joint jitter: the fits'
    noise cannot see it (the holds are shared), half the distance goes into the error bar, and a right
    config is not called KEEP on the strength of the noise alone."""
    cfg = _cfg()
    C1 = rm.inv(cfg.T_SC[0]) @ cfg.T_SC[1]
    wrong = rm.transform(np.eye(3), [0.004, 0.0, 0.0]) @ C1
    frames = _with_corners(cfg, _frames(cfg, cfg.T_BS, jitter_deg=0.08, seed=3), noise_px=0.2, C1_render=wrong)
    rep = s81.run_pixels(frames, neckfk.NeckChain(URDF), cfg)
    ca = rep["cameras_apart"]
    assert 3.0 < ca["dist_mm"] < 5.0 and abs(rep["sigma_sys_mm"] - ca["dist_mm"] / 2) < 1e-9
    assert rep["sigma_mm"] > rep["sigma_sys_mm"] and any("calibrations" in w for w in rep["warnings"])
    assert rep["verdict"] == "INCONCLUSIVE" and rep["inconclusive_because"] and "KEEP needs" in rep["verdict_reason"]


def test_an_inconclusive_verdict_always_says_why():
    """17 Oct P2: no fault in the fit, INCONCLUSIVE by the arithmetic alone: the report says so, with numbers."""
    rep = s81._decide({}, {"angle_deg": 0.41, "dist_mm": 10.31}, 0.109, 3.43, [], [])
    assert rep["verdict"] == "INCONCLUSIVE" and rep["inconclusive_because"][0].startswith("neither the config nor the fit")
    assert "REPLACE needs 2 sigma 0.218 deg <= 0.3 and 6.86 mm <= 3.0" in rep["verdict_reason"]
    assert "over 4 sigma (0.436 deg, 13.72 mm)" in rep["verdict_reason"]
    ok = s81._decide({}, {"angle_deg": 0.41, "dist_mm": 10.31}, 0.05, 1.0, [], [])
    assert ok["verdict"] == "REPLACE" and ok["inconclusive_because"] == []


def test_pixels_from_one_camera_give_no_verdict():
    cfg = _cfg()
    frames = _with_corners(cfg, [f for f in _frames(cfg, cfg.T_BS) if f["cam"] == 0])
    rep = s81.run_pixels(frames, neckfk.NeckChain(URDF), cfg)
    assert "no cross-check between the cameras" in rep["inconclusive_because"] and rep["verdict"] == "INCONCLUSIVE"
    assert np.isinf(rep["sigma_sys_mm"]) and np.isinf(rep["sigma_mm"])
    import json
    back = json.loads(json.dumps(s81._json_safe(rep), allow_nan=False))      # strict JSON: no Infinity
    assert back["sigma_mm"] is None and back["sigma_sys_deg"] is None
    blind = {**back, "sigma_stat_deg": None, "sigma_stat_mm": None}       # a null sigma reads as infinite
    off = {**back, "T_BC_measured": (rm.transform(np.eye(3), [0.002, 0, 0]) @ np.array(back["T_BC_measured"])).tolist()}
    c = s81.compare_takes(blind, off)                                        # 2 mm apart: only the sigma route could agree
    assert not c["agree"] and not c["agree_sigma"] and "a take's statistical sigma is over the gate" in c["why_not"]


def test_pixels_cross_check_the_cameras_only_on_holds_both_saw_and_only_when_each_fit_sees_everything(monkeypatch):
    """Camera 1 in 5 of 25 holds: its fit alone carries those holds' joint error uncancelled, so the
    distance would not be a calibration disagreement: no cross-check, and no such warning. A pitch that
    never moves: each camera's fit alone is blind in some direction: no cross-check either."""
    cfg = _cfg()
    frames = _with_corners(cfg, [f for f in _frames(cfg, cfg.T_BS, jitter_deg=0.08, seed=1) if f["cam"] == 0 or f["hold"] < 5])
    rep = s81.run_pixels(frames, neckfk.NeckChain(URDF), cfg)
    assert rep["cross_check_holds"] == 5 and "no cross-check between the cameras" in rep["inconclusive_because"]
    assert not any("calibrations" in w for w in rep["warnings"]) and np.isinf(rep["sigma_sys_mm"])
    frames = _with_corners(cfg, [f for f in _frames(cfg, cfg.T_BS, jitter_deg=0.08, seed=1) if f["cam"] == 0 or f["hold"] >= 5])
    rep = s81.run_pixels(frames, neckfk.NeckChain(URDF), cfg)              # 20 shared holds: enough to cross-check
    # fitted on the shared holds the cameras land 0.1-0.3 mm apart; on all of each camera's holds 1.2-7 mm (seeds 1-3)
    assert rep["cross_check_holds"] == 20 and rep["cameras_apart"]["dist_mm"] < 0.6, rep["cameras_apart"]
    monkeypatch.setattr(sys.modules[__name__], "POSES", [(y, 0.0) for y in np.linspace(-0.35, 0.35, 8)])
    rep = s81.run_pixels(_with_corners(cfg, _frames(cfg, cfg.T_BS)), neckfk.NeckChain(URDF), cfg, min_verdict_holds=6)
    assert "no cross-check between the cameras" in rep["inconclusive_because"] and np.isinf(rep["sigma_sys_mm"])
    assert any("cannot see every direction" in w for w in rep["warnings"])


def test_pixels_with_too_few_holds_are_insufficient():
    cfg = _cfg()
    frames = _with_corners(cfg, [f for f in _frames(cfg, cfg.T_BS) if f["hold"] < 4])
    assert s81.run_pixels(frames, neckfk.NeckChain(URDF), cfg)["verdict"] == "INSUFFICIENT"


import dataclasses  # noqa: E402
import json  # noqa: E402

FIT_A_REF = json.loads((HERE / "data" / "kb4t_reference.json").read_text())["intr"]
C_FIT = rm.transform(rm.rot_axis([0, 0, 1], np.radians(0.9)) @ rm.rot_axis([1, 0, 0], np.radians(0.2)),
                     [0.044425, -0.000226, 0.000302])


def _fit_json(tmp_path, C=C_FIT, model="KB4T"):
    p = tmp_path / "fit.json"
    p.write_text(json.dumps({"KB4T@all@800": {"model": model, "intr": FIT_A_REF, "C": C.tolist()}}))
    return p


def test_lens_from_fit_swaps_the_lens_and_the_stereo_pose_and_keeps_the_rest(tmp_path):
    cfg = _cfg()
    out = s81.lens_from_fit(cfg, _fit_json(tmp_path), "KB4T@all@800")
    assert out.lens == "pinhole-kb4t" and rigcfg.Lens(out.lens).kind == "kb4t"
    assert np.allclose(out.K[0], [[FIT_A_REF[0][0], 0, FIT_A_REF[0][2]], [0, FIT_A_REF[0][1], FIT_A_REF[0][3]], [0, 0, 1]])
    assert np.allclose(out.D[1].ravel(), FIT_A_REF[1][4:]) and out.D[1].shape == (6, 1)
    assert np.allclose(out.T_SC[0], cfg.T_SC[0]) and np.allclose(out.T_BS, cfg.T_BS)
    assert np.allclose(rm.inv(out.T_SC[0]) @ out.T_SC[1], C_FIT)              # every C step 81 derives is the fit's


def test_lens_from_fit_refuses_what_it_cannot_use(tmp_path):
    cfg = _cfg()
    with pytest.raises(KeyError):
        s81.lens_from_fit(cfg, _fit_json(tmp_path), "no such key")
    with pytest.raises(ValueError, match="model"):
        s81.lens_from_fit(cfg, _fit_json(tmp_path, model="RT8"), "KB4T@all@800")
    bad = C_FIT.copy()
    bad[:3, :3] *= 1.01
    with pytest.raises(ValueError, match="not a rotation"):
        s81.lens_from_fit(cfg, _fit_json(tmp_path, C=bad), "KB4T@all@800")


def test_lens_from_fit_reads_a_kb4_fit_as_that_lens(tmp_path):
    """The KB4 branch (20 Oct's real-data check 3b went through it): the config written as a KB4 lens fit comes
    back as the config's own lens and stereo pose."""
    cfg = _cfg()
    intr = [[float(v) for v in (k[0, 0], k[1, 1], k[0, 2], k[1, 2], *np.ravel(d))] for k, d in zip(cfg.K, cfg.D)]
    p = tmp_path / "kb4.json"
    p.write_text(json.dumps({"KB4@cfg": {"model": "KB4", "intr": intr,
                                         "C": (rm.inv(cfg.T_SC[0]) @ cfg.T_SC[1]).tolist()}}))
    out = s81.lens_from_fit(cfg, p, "KB4@cfg")
    assert out.lens == "pinhole-equi" and rigcfg.Lens(out.lens).kind == "equi"
    for c in (0, 1):
        assert np.abs(out.K[c] - cfg.K[c]).max() < 1e-12, c
        assert out.D[c].shape == (4, 1) and np.abs(out.D[c].ravel() - np.ravel(cfg.D[c])).max() < 1e-12, c
    assert np.array_equal(out.T_SC[0], cfg.T_SC[0]) and np.abs(out.T_SC[1] - cfg.T_SC[1]).max() < 1e-12
    assert np.array_equal(out.T_BS, cfg.T_BS)


def test_a_kb4t_head_is_measured_through_the_fits_lens_and_misread_through_the_old_one(tmp_path):
    """Pixels made by a KB4T lens and the fit's stereo pose: through lens_from_fit the head reads as it is
    (KEEP, the cameras agree); through the config's KB4 lens and stereo pose the cameras' own fits disagree."""
    cfg = _cfg()
    true = s81.lens_from_fit(cfg, _fit_json(tmp_path), "KB4T@all@800")
    frames = _with_corners(true, _frames(true, true.T_BS), noise_px=0.05)
    rep = s81.run_pixels(frames, neckfk.NeckChain(URDF), true)
    assert rep["verdict"] == "KEEP" and rep["cameras_apart"]["dist_mm"] < 0.2 and rep["fit"]["rms_px"] < 0.1
    old = s81.run_pixels(frames, neckfk.NeckChain(URDF), cfg)
    assert old["fit"]["rms_px"] > 3 * rep["fit"]["rms_px"]


from types import SimpleNamespace  # noqa: E402


def _corner_file(tmp_path, cfg, pairs_per_hold=3):
    """A take as dump_corners.py writes it: every AprilGrid corner both cameras see at each pose, the neck angles
    of the pair, the hold and its approach sides (yaw and pitch sides differ in some holds, so a swap shows). The
    board is centred on camera 0's axis 1.1 m out at zero neck angles."""
    import handeye_px
    board = rigcfg.board_from_target(APRILGRID)
    keys = [(i, k) for i in range(board.n_tags) for k in range(4)]
    P = np.array([board.obj_points(i)[k] for i, k in keys])
    c = neckfk.NeckChain(URDF)
    kind = rigcfg.Lens(cfg.lens).kind
    T_TW = c.T_TB(0, 0) @ cfg.T_BS @ cfg.T_SC[0] @ rm.transform(np.eye(3), [-0.41, -0.29, 1.1])
    img, det = [], []
    for h, (y, p) in enumerate(POSES):
        for j in range(pairs_per_hold):
            n = len(img)
            img.append((h, 10**18 + 10**9 * (10 * h + j), y, p, 1 if h % 2 else -1, 1 if h % 3 else -1))
            for cam in (0, 1):
                T = rm.inv(c.T_TB(y, p) @ cfg.T_BS @ cfg.T_SC[cam]) @ T_TW
                uv = handeye_px.project(P @ T[:3, :3].T + T[:3, 3], cfg.K[cam], cfg.D[cam], kind)
                det += [(n, cam, t, k, u, v) for (t, k), (u, v) in zip(keys, uv)]
    I, D_ = np.array(img, dtype=object), np.array(det, dtype=object)
    path = tmp_path / "take-corners.npz"
    np.savez(path, img_hold=I[:, 0].astype(int), img_t0=I[:, 1].astype(np.int64), img_t1=I[:, 1].astype(np.int64),
             img_yaw=I[:, 2].astype(float), img_pitch=I[:, 3].astype(float), img_ay=I[:, 4].astype(int),
             img_ap=I[:, 5].astype(int), det_img=D_[:, 0].astype(int), det_cam=D_[:, 1].astype(int),
             det_tag=D_[:, 2].astype(int), det_k=D_[:, 3].astype(int), det_u=D_[:, 4].astype(float),
             det_v=D_[:, 5].astype(float))
    return path, SimpleNamespace(board=board, resolution=(1280, 800))


def test_frames_from_a_corner_file_measure_the_head(tmp_path):
    cfg = _cfg()
    path, rig = _corner_file(tmp_path, cfg)
    frames, holds = s81._frames_from_corners(path, rig, cfg, per_hold=2, drop={3}, max_radius_px=900.0)
    assert sorted(holds) == list(range(len(POSES)))                 # every hold listed; the report marks 3 dropped
    assert len(frames) == 2 * 2 * (len(POSES) - 1) and {f["cam"] for f in frames} == {0, 1}
    assert all(f["hold"] != 3 for f in frames) and all(len(f["P"]) == len(f["uv"]) >= 80 for f in frames)
    assert holds[1].approach_yaw == 1 and holds[2].approach_yaw == -1
    assert holds[2].approach_pitch == 1 and holds[3].approach_pitch == -1
    for f in frames:                                                 # every frame carries its own hold's sides
        h = holds[f["hold"]]
        assert (f["approach_yaw"], f["approach_pitch"]) == (h.approach_yaw, h.approach_pitch), f["hold"]
    rep = s81.run_pixels(frames, neckfk.NeckChain(URDF), cfg)
    assert rep["verdict"] == "KEEP" and rep["vs_config"]["dist_mm"] < 1e-3


def test_the_radius_cut_is_the_callers(tmp_path):
    cfg = _cfg()
    path, rig = _corner_file(tmp_path, cfg, pairs_per_hold=1)
    wide, _ = s81._frames_from_corners(path, rig, cfg, per_hold=1, drop=set(), max_radius_px=900.0)
    tight, _ = s81._frames_from_corners(path, rig, cfg, per_hold=1, drop=set(), max_radius_px=300.0)
    assert sum(len(f["uv"]) for f in tight) < sum(len(f["uv"]) for f in wide)
    for f in tight:
        assert np.all(np.hypot(f["uv"][:, 0] - 640, f["uv"][:, 1] - 400) <= 300.0)


RIG = HERE.parent / "config/rig-phantom12.yaml"


def test_main_fits_with_the_lens_and_radius_it_is_given_and_reports_what_it_used(tmp_path, capsys):
    """Step 81's command line on a corner file made through Fit A's lens and stereo pose: --max-radius-px cuts the
    corners the fit uses, --lens-json changes the lens it fits through, and the report records what was used (the
    largest radius of a corner used, the lens and its number of distortion terms), not only the arguments. The
    corner route has no count of images in /joints gaps (dump_corners.py left out an unknown number): null, not 0."""
    cfg = _cfg()
    fit = _fit_json(tmp_path)
    path, _ = _corner_file(tmp_path, s81.lens_from_fit(cfg, fit, "KB4T@all@800"), pairs_per_hold=1)
    base = ["--corners", path, "--config", REAL, "--urdf", URDF, "--rig", RIG]
    lens = ["--lens-json", fit, "--lens-key", "KB4T@all@800"]

    def run(name, *extra):
        out = tmp_path / f"{name}.json"
        code = s81.main([str(v) for v in base + list(extra) + ["--out", out]])
        return code, json.loads(out.read_text()), capsys.readouterr().out

    code, wide, txt = run("wide", *lens, "--max-radius-px", 900)
    assert code == 0 and wide["verdict"] == "KEEP", txt                    # through the lens the corners were made with
    assert wide["lens"]["lens"] == "pinhole-kb4t" and wide["lens"]["n_dist"] == 6 and wide["max_radius_px"] == 900
    d = np.load(path)                                                   # within 900 px every image and corner is used:
    r_file = float(np.hypot(d["det_u"] - 640.0, d["det_v"] - 400.0).max())   # the outermost is the file's (~515 px)
    assert wide["n_frames"] == 2 * len(POSES) and 300 < r_file < 900
    assert abs(wide["max_corner_radius_used_px"] - r_file) < 1e-9
    assert wide["n_images_in_joint_gaps"] is None and "0 images in /joints gaps" not in txt, txt
    _, tight, _ = run("tight", *lens, "--max-radius-px", 300)
    assert tight["max_corner_radius_used_px"] <= 300 and tight["fit"]["n_corners"] < wide["fit"]["n_corners"]
    code, old, _ = run("old", "--max-radius-px", 900)                      # the config's KB4 lens and stereo pose
    assert code == 2 and old["lens"]["lens"] == "pinhole-equi" and old["lens"]["n_dist"] == 4
    assert np.abs(np.array(old["T_BC_measured"]) - np.array(wide["T_BC_measured"])).max() > 1e-3
