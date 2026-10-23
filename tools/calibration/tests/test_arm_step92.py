"""Arm session, Task 7: step 92 end to end on a synthetic sweep made through phantom12's real Fit A config: /joints
with holds and ramps, corners only during holds, step 91's log, step 90's plan fields. The cameras sit 13 mm / 1.5 deg
off in the torso (ΔT): M0 must miss by more than the 3 mm gate (X absorbs part of it over a small pose spread),
M1 must find it, and the verdict must say so."""
import importlib
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
import armchain  # noqa: E402
import armfit  # noqa: E402
import armplan  # noqa: E402
import neckfk  # noqa: E402
import okviscfg  # noqa: E402
import rigcfg  # noqa: E402
import rigmath as rm  # noqa: E402

step92 = importlib.import_module("92_arm_fit")
URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
CFG = HERE.parent / "results/phantom12-2026-10-21-okvis-kb4t/phantom12_stereo_dense_thor-fita.yaml"
NAMES = ["LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbowPitch", "LeftWristYaw",
         "LeftWristRoll", "LeftGripperYaw", *armchain.RIGHT_ARM, "NeckPitch", "NeckYaw", "SpineYaw",
         "LeftHipRoll", "LeftHipYaw", "LeftHipPitch", "LeftKneePitch", "RightHipRoll", "RightHipYaw", "RightHipPitch",
         "RightKneePitch", "LeftAnkleRoll", "LeftAnklePitch", "RightAnkleRoll", "RightAnklePitch"]
NECK = (np.radians(-0.4), np.radians(20.3))
START = np.radians([-42.6, -4.8, 47.5, -92.3, -135.7, 21.2, 19.8])
GRID = rigcfg.AprilGrid(2, 2, 0.045, 0.2)
KINDS = ["fit", "fit", "fit", "heldout"] * 5 + ["repeat_same", "repeat_same", "repeat_opp"]


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sweep")
    rng = np.random.default_rng(7)
    arm, neck = armchain.ArmChain(URDF), neckfk.NeckChain(URDF)
    cfg = okviscfg.load(CFG)
    rig = armfit.rig_from_config(cfg)
    P, keys = armplan.plate_points(GRID)
    T_TC0 = neck.T_TB(*NECK) @ rig.T_HC[0]
    flip = np.diag([1.0, -1.0, -1.0])
    T_CP = rm.transform(flip, np.array([0.0, 0.05, 0.42]) - flip @ P.mean(0))
    X = rm.inv(arm.T_base_tip(START)) @ T_TC0 @ T_CP                    # the plate 0.42 m in front at START
    dT = rm.transform(rm.rot_axis((0, 1, 1), np.radians(1.5)), (0.012, -0.006, 0.003))
    cams = [armplan.Camera(neck.T_TB(*NECK) @ T, "KB4T", np.r_[l[1]], (1280, 800)) for T, l in zip(rig.T_HC, rig.lens)]
    poses = []
    while len(poses) < 20:
        q = START + rng.uniform(-0.12, 0.12, 7)
        if armplan.visible(armplan.view(cams, armplan.plate_in_torso(arm, X, q[None]), P, GRID),
                           range_m=(0.3, 0.8))[0]:
            poses.append(q)
    visits = [poses[i] for i in range(20)] + [poses[0], poses[4], poses[8]]
    jt, jq, rows, stops = [], [], [], []
    t = 0
    base = np.zeros(len(NAMES))
    base[NAMES.index("NeckYaw")], base[NAMES.index("NeckPitch")] = NECK
    prev = START
    ia = [NAMES.index(n) for n in armchain.RIGHT_ARM]
    for i, (q, kind) in enumerate(zip(visits, KINDS)):
        q = q + rng.normal(0, np.radians(0.05), 7) if i >= 20 else q
        for s in np.linspace(0, 1, 50):                               # a 1 s ramp at 50 Hz
            row = base.copy()
            row[ia] = prev + s * (q - prev)
            jt.append(t)
            jq.append(row)
            t += 20_000_000
        for k in range(150):                                         # a 3 s hold at 50 Hz, images at 10 Hz
            row = base.copy()
            row[ia] = q + np.radians(0.01) * rng.integers(-1, 2, 7)          # encoder steps
            jt.append(t)
            jq.append(row)
            if k % 5 == 2:
                T_TP = arm.T_base_tip(q) @ X
                for c in (0, 1):
                    T = rm.inv(dT @ neck.T_TB(*NECK) @ rig.T_HC[c]) @ T_TP
                    Pc = (T[:3, :3] @ P.T).T + T[:3, 3]
                    uv = armplan.lensmodel.project("KB4T", rig.lens[c][1], Pc) + rng.normal(0, 0.15, (16, 2))
                    for (tag, kk), (u, v) in zip(keys, uv):
                        rows.append((c, t + c * 300_000, tag, kk, u, v))    # camera 1 stamped 0.3 ms later
            t += 20_000_000
        stops.append({"event": "stop", "stop": i, "kind": kind, "pose": i if i < 20 else [0, 4, 8][i - 20],
                      "status": "ok", "corrections": 0, "why": [], "reached_deg": np.degrees(q).tolist()})
        prev = q
    r = np.array(rows, float)
    dump = tmp / "dump.npz"
    np.savez(dump, cam=r[:, 0].astype(int), t=r[:, 1].astype(np.int64), tag=r[:, 2].astype(int),
             k=r[:, 3].astype(int), u=r[:, 4], v=r[:, 5], jt=np.array(jt, np.int64), jq=np.array(jq))
    log = tmp / "log.jsonl"
    dry = [{**s_, "dry_run": True} for s_ in stops]                     # a dry run logged first, as step 91 might
    log.write_text(json.dumps({"event": "start", "dry_run": True}) + "\n"
                   + "".join(json.dumps(s_) + "\n" for s_ in dry)
                   + json.dumps({"event": "start"}) + "\n" + "".join(json.dumps(s_) + "\n" for s_ in stops))
    plan = tmp / "poses.json"
    X0 = X @ rm.transform(rm.rot_axis((1, 0, 0), 0.01), (0.003, 0.002, -0.002))
    plan.write_text(json.dumps({"names": NAMES, "grid": {"cols": 2, "rows": 2, "tag_size": 0.045, "tag_spacing": 0.2},
                                "X0": X0.tolist(), "baseline_deg": dict(zip(NAMES, np.degrees(base).tolist())),
                                "gravity_in_pelvis": [0.207, 0.074, -0.976]}))
    out = tmp / "report.json"
    step92.main(["--dump", str(dump), "--log", str(log), "--poses", str(plan), "--config", str(CFG),
                 "--urdf", str(URDF), "--mjcf", "", "--out", str(out)])
    return json.loads(out.read_text()), X, dT


def test_every_hold_is_found_and_matched(sweep):
    r, _, _ = sweep
    assert r["holds_matched"] == 23 and r["fit_holds"] == 15 and r["heldout_holds"] == 5
    assert r["unmatched_stops"] == []


def test_m0_misses_by_millimetres_and_m1_finds_the_camera_offset(sweep):
    r, X, dT = sweep
    h = r["heldout_rms_mm"]
    assert h["M0"] > 3.0 and h["M1"] < 0.5
    assert r["chosen"] == "M1" and r["verdict"] == "calibrated (M1)"
    got = np.array(r["fits"]["M1"]["dT"])
    assert np.linalg.norm(got[:3, 3] - dT[:3, 3]) < 1.5e-3
    assert rm.rot_angle_deg(got[:3, :3].T @ dT[:3, :3]) < 0.05


def test_repeats_slip_and_scale_are_small_on_a_clean_sweep(sweep):
    r, _, _ = sweep
    assert len(r["same_side_mm"]) == 2 and len(r["opposite_side_mm"]) == 1
    assert r["noise_floor_mm"] < 0.3 and r["slip_mm"] < 0.5 and r["slip_deg"] < 0.1
    assert len(r["board_scale_pct"]) == 23
    assert all(abs(v["median_pct"]) < 0.2 for v in r["board_scale_by_cell"].values())
    assert r["encoder_side_mm_per_Nm"] is None                      # no MJCF given: not measured


def test_only_the_last_real_runs_stops_are_read():
    rows = [{"event": "start", "dry_run": True}, {"event": "stop", "stop": 0, "dry_run": True},
            {"event": "start"}, {"event": "stop", "stop": 0, "tag": "aborted run"}, {"event": "abort"},
            {"event": "start"}, {"event": "move"}, {"event": "stop", "stop": 0, "tag": "real"},
            {"event": "stop", "stop": 1, "dry_run": True}]
    assert step92.real_stops(rows) == [{"event": "stop", "stop": 0, "tag": "real"}]
