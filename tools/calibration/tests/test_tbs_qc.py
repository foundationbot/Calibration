"""Step 75: the quick check's grading of a neck take."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
import necktake  # noqa: E402
import neckfk  # noqa: E402
import rigmath as rm  # noqa: E402

qc = importlib.import_module("75_qc_necktake")
URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
GRID = [(y, p) for p in (-0.3, -0.15, 0.0, 0.15, 0.3) for y in (-0.35, -0.175, 0.0, 0.175, 0.35)]


def _snake():
    """The protocol's path: rows of yaw, alternating direction, then the same path backwards."""
    rows = [GRID[5 * r:5 * r + 5][:: 1 if r % 2 == 0 else -1] for r in range(5)]
    path = [q for row in rows for q in row]
    path = path + path[::-1][1:]
    holds, prev = [], (0.0, 0.0)
    for y, p in path:
        s = lambda d: 0 if abs(d) < 0.02 else (1 if d > 0 else -1)
        holds.append(necktake.Hold(0, 1, y, p, s(y - prev[0]), s(p - prev[1])))
        prev = (y, p)
    return holds


def test_the_protocols_snake_passes_and_a_one_sided_take_does_not():
    g = qc.grade_holds(_snake())
    assert g["ok"] and g["n"] == 49 and g["cells"] == 9 and min(g["sides"].values()) >= 2
    one_way = [h for h in _snake()[:25]]
    assert not qc.grade_holds([necktake.Hold(0, 1, h.yaw, h.pitch, 1, 1) for h in one_way])["ok"]
    assert not qc.grade_holds([necktake.Hold(0, 1, 0.0, p, 0, 0) for p in (-0.1, 0.0, 0.1)])["ok"]


def test_the_17_oct_grid_passes_over_its_own_range_and_a_neck_that_barely_moved_does_not():
    """17 Oct: the board was seen only with the head level or up: yaw -30..30 x pitch -24..0 deg, snaked
    there and back. Graded over that range it fills the 3 x 3 grid; a pitch that moved 2 deg does not count."""
    ys, ps = np.radians([-30, -15, 0, 15, 30]), np.radians([-24, -18, -12, -6, 0])
    rows = [[(y, p) for y in (ys if r % 2 == 0 else ys[::-1])] for r, p in enumerate(ps)]
    path = [q for row in rows for q in row]
    path = path + path[::-1][1:]
    holds, prev = [], (0.0, 0.0)
    for y, p in path:
        s = lambda d: 0 if abs(d) < 0.02 else (1 if d > 0 else -1)  # noqa: E731
        holds.append(necktake.Hold(0, 1, y, p, s(y - prev[0]), s(p - prev[1])))
        prev = (y, p)
    g = qc.grade_holds(holds)
    assert g["ok"] and g["cells"] == 9 and abs(g["yaw_span_deg"] - 60) < 1e-6 and abs(g["pitch_span_deg"] - 24) < 1e-6
    flat = [necktake.Hold(0, 1, h.yaw, h.pitch / 12, h.approach_yaw, h.approach_pitch) for h in holds]
    g = qc.grade_holds(flat)
    assert g["cells"] == 9 and g["pitch_span_deg"] < qc.MIN_PITCH_SPAN_DEG and not g["ok"]


def test_each_hold_criterion_fails_on_its_own():
    snake = _snake()
    few =[necktake.Hold(0, 1, h.yaw, h.pitch, h.approach_yaw, h.approach_pitch) for h in snake[:qc.MIN_HOLDS - 1]]
    assert not qc.grade_holds(few)["ok"]                                         # too few
    corner = [necktake.Hold(0, 1, 0.3 * (k % 2), 0.3 * (k // 2 % 2), (-1) ** k, (-1) ** (k // 2)) for k in range(40)]
    g = qc.grade_holds(corner)
    assert g["n"] >= qc.MIN_HOLDS and g["cells"] < qc.MIN_CELLS and not g["ok"]    # too few cells only
    one_yaw = [necktake.Hold(0, 1, h.yaw, h.pitch, 1, h.approach_pitch) for h in snake]
    assert not qc.grade_holds(one_yaw)["ok"]                                     # yaw from one side only
    one_pitch = [necktake.Hold(0, 1, h.yaw, h.pitch, h.approach_yaw, 1 if h.approach_pitch else 0) for h in snake]
    g = qc.grade_holds(one_pitch)
    assert g["sides"]["pitch-"] == 0 and min(g["sides"]["yaw+"], g["sides"]["yaw-"]) >= qc.MIN_YAW_SIDES and not g["ok"]


def _board_pixels(flip_pitch=1.0, flip_yaw=1.0, scale=1.0):
    """Board centre in a phantom12-like camera 0 (142 mm ahead, 30 deg down, image x = robot right)
    over the grid (angles times scale), the camera seeing the true angles while the encoders report
    flip * angle."""
    c = neckfk.NeckChain(URDF)
    X = rm.transform(rm.rot_axis([0, 1, 0], np.radians(30)) @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.]]),
                     [0.142, 0.021, 0.014])
    p_T = (c.T_TB(0, 0) @ X @ [0, 0, 1.1, 1])[:3]
    rows = []
    for y, p in GRID:
        y, p = scale * y, scale * p
        p_C = (rm.inv(c.T_TB(y, p) @ X) @ [*p_T, 1])[:3]
        rows.append((flip_yaw * y, flip_pitch * p, 640 + 563 * p_C[0] / p_C[2], 400 + 563 * p_C[1] / p_C[2]))
    return np.array(rows).T


def test_encoder_signs_agree_with_the_camera():
    s = qc.encoder_signs(*_board_pixels())
    assert s["yaw_ok"] and s["pitch_ok"] and 350 < s["yaw_px_per_rad"] < 650 and -650 < s["pitch_px_per_rad"] < -350
    assert s["yaw"] == "ok" and s["pitch"] == "ok"


def test_a_reversed_pitch_encoder_is_caught_by_the_camera():
    s = qc.encoder_signs(*_board_pixels(flip_pitch=-1.0))
    assert s["yaw_ok"] and not s["pitch_ok"] and s["pitch"] == "reversed" and s["yaw"] == "ok"


def test_a_reversed_yaw_encoder_is_caught_by_the_camera():
    s = qc.encoder_signs(*_board_pixels(flip_yaw=-1.0))
    assert not s["yaw_ok"] and s["yaw"] == "reversed" and s["pitch"] == "ok"


def test_a_board_that_does_not_follow_the_neck_is_not_called_reversed():
    """The board barely moving in the image (a neck that hardly moved, or a stuck camera): slopes near
    zero read 'too little motion', not 'reversed'."""
    y, p, u, v = _board_pixels()
    s = qc.encoder_signs(y, p, 640 + 0.0 * u + 0.3 * np.sin(np.arange(25)), 400 + 0.3 * np.cos(np.arange(25)))
    assert s["yaw"] == "too little motion" and s["pitch"] == "too little motion"


def _joints(segments, rate=500):
    """/joints from [(seconds, yaw rate, pitch rate)] pieces of constant speed [rad/s]."""
    t, y, p, ty, py = [0.0], [0.0], [0.0], 0.0, 0.0
    for secs, vy, vp in segments:
        for _ in range(int(secs * rate)):
            ty += vy / rate
            py += vp / rate
            t.append(t[-1] + 1 / rate)
            y.append(ty)
            p.append(py)
    return necktake.Joints(np.int64(1_792_000_000_000_000_000) + (np.array(t) * 1e9).astype(np.int64), np.array(y), np.array(p))


def test_sweeps_are_counted_as_step_82_selects_them():
    """2 s of pure pitch, 3 s of yaw with 10 % pitch in it (inside step 82's purity), 2 s of yaw with
    50 % pitch (outside), 1 s too slow: 2 s pitch-only and 3 s yaw-only."""
    j = _joints([(1, 0, 0), (2, 0, 0.5), (3, 0.5, 0.05), (2, 0.5, 0.25), (1, 0.1, 0), (1, 0, 0)])
    s = qc.sweep_seconds(j)
    assert abs(s["pitch_only_s"] - 2.0) < 0.1 and abs(s["yaw_only_s"] - 3.0) < 0.1


def test_streams():
    t = (np.arange(0, 10, 1 / 500) * 1e9).astype(np.int64)
    s = qc.stream(np.delete(t, np.arange(1000, 1010)))
    assert abs(s["rate_hz"] - 499.5) < 1 and abs(s["max_gap_ms"] - 22.0) < 0.1
    assert qc.stream(t[:1])["rate_hz"] == 0.0 and qc.stream([])["max_gap_ms"] == float("inf")


def test_a_camera_that_stopped_is_named_with_the_time():
    t0, t1 = 0, int(300e9)
    t = (np.arange(0, 20, 1 / 14) * 1e9).astype(np.int64)        # 14 fps for 20 s, then nothing
    c = qc.camera_coverage(t, t0, t1)
    assert c["cover"] < 0.1 and abs(c["stopped_at_s"] - 20) < 0.2
    full = qc.camera_coverage((np.arange(0, 300, 1 / 14) * 1e9).astype(np.int64), t0, t1)
    assert full["cover"] > 0.99 and full["stopped_at_s"] is None
    assert qc.camera_coverage([], t0, t1)["cover"] == 0.0


def test_a_stopped_camera_is_a_retake_and_recorder_gaps_only_a_warning():
    t0, t1 = 0, int(300e9)
    stopped = qc.camera_coverage((np.arange(0, 200, 1 / 14) * 1e9).astype(np.int64), t0, t1)
    assert qc.camera_finding("cam1", stopped, 0, 40, 49)["level"] == "RETAKE"
    t = np.arange(0, 300, 1 / 14)
    gappy = qc.camera_coverage((t[(t % 60) > 4] * 1e9).astype(np.int64), t0, t1)    # 4 s lost every minute
    assert gappy["cover"] < qc.MIN_CAM_COVER and gappy["stopped_at_s"] is None
    f = qc.camera_finding("cam1", gappy, 0, 49, 49)
    assert f["level"] == "WARN" and f["what"] == "camera-gaps-cam1"
    full = qc.camera_coverage((t * 1e9).astype(np.int64), t0, t1)
    assert qc.camera_finding("cam1", full, 0, 49, 49) is None
    assert qc.camera_finding("cam1", full, 0, 10, 49)["what"] == "board-cam1"           # lost the board
    assert qc.camera_finding("cam1", stopped, 0, 10, 49)["what"] == "camera-cam1"       # stopping says so first


def test_a_joints_gap_drops_the_hold_only_if_the_neck_moved_across_it():
    j = _joints([(3, 0, 0), (3, 0.002, 0), (3, 0, 0)])           # three holds-worth, the middle one sagging
    T = j.t
    keep = np.ones(len(T), bool)
    keep[(T > T[0] + 1.0e9) & (T < T[0] + 1.5e9)] = False          # a 0.5 s gap while still
    keep[(T > T[0] + 4.0e9) & (T < T[0] + 4.6e9)] = False          # a 0.6 s gap while it sags 0.07 deg
    jg = necktake.Joints(T[keep], j.yaw[keep], j.pitch[keep])
    holds = [necktake.Hold(T[0] + int(k * 3e9), T[0] + int((k + 1) * 3e9 - 1), 0.0, 0.0) for k in range(3)]
    gh = qc.gap_holds(jg, holds)
    assert [k for k, _, _ in gh] == [1] and 550 < gh[0][1] < 650


def test_a_gap_across_a_holds_edge_counts_for_that_hold():
    j = _joints([(3, 0.01, 0), (3, 0, 0)])                           # sagging 0.6 deg/s, then still
    T = j.t
    keep = ~((T > T[0] + 2.7e9) & (T < T[0] + 3.4e9))               # 0.7 s from inside hold 0 into hold 1, 0.17 deg
    jg = necktake.Joints(T[keep], j.yaw[keep], j.pitch[keep])
    holds = [necktake.Hold(T[0], T[0] + int(3e9), 0.0, 0.0), necktake.Hold(T[0] + int(3.2e9), T[0] + int(6e9), 0.0, 0.0)]
    assert sorted({k for k, _, _ in qc.gap_holds(jg, holds)}) == [0, 1]


def _m(tilt=0.0, yaw=0.0, turn=0.0, sigma=0.0, scatter=4e-5, ref=True):
    return {"tilt_deg": tilt, "yaw_deg": yaw, "turn_deg": turn, "yaw_sigma_deg": sigma, "rate_scatter_rad_s": scatter,
            "in_reference": ref}


def test_moved_holds_count_yaw_beyond_its_error_bar_and_a_real_turn_even_when_the_robot_rocked():
    m = [_m(), None, _m(yaw=-0.15), _m(tilt=0.15), _m(turn=0.15), _m(yaw=0.2, sigma=0.1), _m(yaw=0.4, sigma=0.1)]
    r = qc.moved_holds(m)
    assert r["flagged"] == [2, 3, 4, 6] and r["counts"] == {"tilt": 1, "yaw": 2, "turn": 1} and r["yaw_reliable"]
    assert abs(r["max_offset_deg"] - 0.4) < 1e-9 and r["reference_share"] == 1.0
    # rocking: a degree and a half of yaw inside its error bar is the estimate's; twenty degrees is a turn
    noise = qc.moved_holds([_m(yaw=1.5, sigma=1.0, scatter=5e-4), _m(tilt=0.12, scatter=5e-4)])
    assert not noise["yaw_reliable"] and noise["flagged"] == [1] and abs(noise["max_offset_deg"] - 0.12) < 1e-9
    turned = qc.moved_holds([_m(yaw=20.0, sigma=2.6, scatter=3e-4), _m(scatter=3e-4)])
    assert turned["flagged"] == [0] and abs(turned["max_offset_deg"] - 20.0) < 1e-9


def test_the_torso_decision_tells_a_moved_robot_a_shift_and_wander_apart():
    moved = qc.moved_holds([_m(yaw=20.0, sigma=2.6)] + [_m()] * 9)
    f, d = qc.torso_finding(moved, 10)
    assert f["level"] == "STOP" and f["what"] == "robot-moved" and not d
    shifted = qc.moved_holds([_m(tilt=0.15, ref=False)] * 6 + [_m()] * 4)
    f, d = qc.torso_finding(shifted, 10)
    assert f["level"] == "RETAKE" and f["what"] == "torso" and not d
    wander = qc.moved_holds([_m(tilt=0.15)] * 6 + [_m()] * 4)       # many flagged, but most holds stood together
    f, d = qc.torso_finding(wander, 10)
    assert f is None and d == set(range(6))
    turning = qc.moved_holds([_m(turn=0.15, ref=False)] * 6 + [_m()] * 4)   # rotation during holds is no offset
    f, d = qc.torso_finding(turning, 10)
    assert f is None and d == set(range(6))


def test_a_board_that_moved_in_the_torso_frame_stops_the_take():
    assert qc.board_spread_finding({"rot_rms_deg": 6.99, "trans_rms_mm": 161.0})["level"] == "STOP"   # take A
    assert qc.board_spread_finding({"rot_rms_deg": 0.68, "trans_rms_mm": 16.0}) is None              # braced
    assert qc.board_spread_finding({"rot_rms_deg": 0.5, "trans_rms_mm": 45.0})["what"] == "robot-or-board-moved"


def test_the_qc_json_carries_what_run_tbs_reads():
    r = qc.qc_report("RETAKE (holds, torso)", [{"level": "RETAKE", "what": "holds", "action": "x"}], {np.int64(5), 2})
    assert r["verdict"] == "RETAKE" and r["drop_holds"] == [2, 5] and all(type(k) is int for k in r["drop_holds"])


def test_the_worst_finding_decides():
    f = [{"level": "WARN", "what": "sweeps", "action": ""}, {"level": "RETAKE", "what": "holds", "action": ""}]
    assert qc.decide(f) == ("RETAKE (holds)", 1)
    assert qc.decide(f + [{"level": "STOP", "what": "robot-moved", "action": ""}]) == ("STOP (robot-moved)", 2)
    assert qc.decide(f[:1]) == ("GOOD", 0)


def test_too_few_usable_holds_for_step_81_is_said_at_the_robot():
    assert qc.usable_holds_finding(19, 20)["what"] == "usable-holds"
    assert qc.usable_holds_finding(20, 20) is None
