"""Arm session, Task 3: the sweep runner (91_arm_session.py) against a fake robot: the commands it sends, how it
reads where the arm stopped, the one correction (T2-5), skips, aborts, the bursts and the log."""
import importlib
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
run91 = importlib.import_module("91_arm_session")
from armchain import RIGHT_ARM  # noqa: E402

JIDX = {n: 7 + i for i, n in enumerate(RIGHT_ARM)}       # /config indices of the right arm


def result_text(targets_deg, reached_deg):
    lines = [f"RESULT targets joint={JIDX[n]} target_deg={t:.3f} ss_err={t - r:.3f} rms=0.1 max=0.2 samples=950"
             for n, t, r in zip(RIGHT_ARM, targets_deg, reached_deg)]
    return "\n".join(lines) + "\nPassed: 0  Failed: 7  Total: 7\n"


def test_parse_result_gives_where_each_joint_stopped():
    t = [-50.0, -5, 45, -100, -135, 21, 20]
    r = [-44.2, -4.9, 47.6, -93.3, -135.6, 21.2, 19.9]
    got = run91.parse_result(result_text(t, r), JIDX)
    assert np.allclose(got, r, atol=1e-3)


def test_parse_result_refuses_a_missing_joint():
    t = [-50.0, -5, 45, -100, -135, 21, 20]
    txt = "\n".join(result_text(t, t).splitlines()[1:])
    with pytest.raises(run91.Abort, match="RightShoulderPitch"):
        run91.parse_result(txt, JIDX)


def test_the_command_lists_every_arm_joint_and_the_gains():
    c = run91.motion_command([-50.0, -5, 45, -100, -135, 21, 20], ramp_s=6.0, hold_s=3.0, kp=100.0)
    assert c.startswith('dma_motor_motion "RightShoulderPitch=-50.00,RightShoulderRoll=-5.00,')
    assert all(f"{n}=" in c for n in RIGHT_ARM)
    assert c.endswith("targets position --ramp 6.0 --hold 3.0 --no-return --kp 100 --tolerance 30")


class FakeRobot:
    """dma_motor_motion stops each joint at target + off_deg (from the stuck_at-th move on, the elbow 20 deg further);
    every other command answers rc 0, except one containing fail_on (an ssh failure)."""

    def __init__(self, off_deg=(4.0, 0, 0, 3.0, 0, 0, 0), stuck_at=None, fail_on=None):
        self.off = np.array(off_deg, float)
        self.calls, self.stuck_at, self.fail_on = [], stuck_at, fail_on

    def __call__(self, cmd: str, timeout_s: float):
        self.calls.append(cmd)
        if self.fail_on and self.fail_on in cmd:
            return 255, "ssh: connect to host phantom12: timed out"
        if cmd.startswith("dma_motor_motion"):
            spec = cmd.split('"')[1]
            t = np.array([float(kv.split("=")[1]) for kv in spec.split(",")])
            r = t + self.off
            if self.stuck_at is not None and len([c for c in self.calls if c.startswith("dma")]) >= self.stuck_at:
                r = r + np.array([0, 0, 0, 20.0, 0, 0, 0])
            return 0, result_text(t, r)
        return 0, "ok"


def _plan(n=3):
    """A small plan: n fit poses near the start, a pre-pose, a repeat and the return; checks always pass."""
    start = [-42.6, -4.8, 47.5, -92.3, -135.7, 21.2, 19.8]
    stops = []
    for k in range(n):
        q = (np.array(start) + np.array([-3.0 * (k + 1), 0, 0, 2.0, 0, 0, 0])).tolist()
        stops.append({"kind": "fit", "pose": k, "q_deg": q, "command_deg": (np.array(q) + [-4, 0, 0, -3, 0, 0, 0]).tolist(),
                      "ramp_s": 6.0})
    stops.append({"kind": "pre", "pose": 0, "q_deg": (np.array(stops[0]["q_deg"]) + [6, 0, 0, 0, 0, 0, 0]).tolist(),
                  "command_deg": (np.array(stops[0]["q_deg"]) + [2, 0, 0, -3, 0, 0, 0]).tolist(), "ramp_s": 6.0})
    stops.append({"kind": "repeat_same", "pose": 0, "q_deg": stops[0]["q_deg"], "command_deg": stops[0]["command_deg"],
                  "ramp_s": 6.0})
    stops.append({"kind": "return", "q_deg": start, "command_deg": (np.array(start) + [-4, 0, 0, -3, 0, 0, 0]).tolist(),
                  "ramp_s": 6.0})
    return {"start_deg": start, "kp": 100.0, "hold_s": 3.0, "stops": stops}


def _run(robot, plan, tmp_path, check=lambda q: (True, [])):
    log = tmp_path / "log.jsonl"
    s = run91.Session(plan, robot, check=check, log_path=log, tag="t", burst_n=10)
    s.sweep()
    return [json.loads(x) for x in log.read_text().splitlines()]


def test_a_pose_reached_within_4_deg_is_kept_and_burst_once(tmp_path):
    robot = FakeRobot(off_deg=(1.0, 0, 0, 1.0, 0, 0, 0))
    log = _run(robot, _plan(), tmp_path)
    moves = [r for r in log if r["event"] == "stop"]
    assert [m["kind"] for m in moves] == ["fit", "fit", "fit", "pre", "repeat_same", "return"]
    assert all(m["corrections"] == 0 and m["status"] == "ok" for m in moves)
    bursts = [c for c in robot.calls if "grab.py --once 10" in c]
    assert len(bursts) == 4                                    # 3 fit + the repeat; none at the pre-pose or return
    assert all("--joints all" in c and "/root/burst-grab/arm-t-" in c for c in bursts)


def test_a_pose_missed_by_more_than_4_deg_is_corrected_once_by_the_miss(tmp_path):
    robot = FakeRobot(off_deg=(-6.0, 0, 0, 0, 0, 0, 0))       # shoulder pitch stops 10 deg off the desired pose
    plan = _plan(1)
    log = _run(robot, plan, tmp_path)
    m = [r for r in log if r["event"] == "stop"][0]
    assert m["corrections"] == 1
    sent = [c for c in robot.calls if c.startswith("dma_motor_motion")]
    first = float(sent[0].split("RightShoulderPitch=")[1].split(",")[0])
    second = float(sent[1].split("RightShoulderPitch=")[1].split(",")[0])
    assert np.isclose(second - first, plan["stops"][0]["q_deg"][0] - m["reached_first_deg"][0], atol=0.02)


def test_a_pose_that_fails_its_checks_after_the_correction_is_skipped_not_burst(tmp_path):
    robot = FakeRobot(off_deg=(1.0, 0, 0, 1.0, 0, 0, 0))
    log = _run(robot, _plan(1), tmp_path, check=lambda q: (False, ["plate out of view"]))
    m = [r for r in log if r["event"] == "stop"][0]
    assert m["status"] == "skipped" and m["corrections"] == 1 and "plate out of view" in m["why"]
    assert not [c for c in robot.calls if "grab.py" in c]


def test_an_arm_far_from_where_it_was_sent_aborts_the_sweep(tmp_path):
    robot = FakeRobot(off_deg=(1.0, 0, 0, 1.0, 0, 0, 0), stuck_at=2)
    with pytest.raises(run91.Abort, match="RightElbowPitch"):
        _run(robot, _plan(3), tmp_path)
    assert len([c for c in robot.calls if c.startswith("dma_motor_motion")]) == 2   # nothing sent after it


def test_a_failed_remote_call_aborts(tmp_path):
    robot = FakeRobot(fail_on="dma_motor_motion")
    with pytest.raises(run91.Abort, match="rc 255"):
        _run(robot, _plan(1), tmp_path)


def test_dry_run_sends_nothing_and_logs_the_commands(tmp_path):
    log = tmp_path / "log.jsonl"
    s = run91.Session(_plan(2), run91.DryRun(), check=lambda q: (True, []), log_path=log, tag="t", burst_n=10)
    s.sweep()
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert all(r.get("dry_run") for r in rows if r["event"] == "stop")
    assert any("dma_motor_motion" in r.get("command", "") for r in rows)


def test_prechecks_name_every_failure():
    snap = {"q_deg": {**{n: 0.0 for n in RIGHT_ARM}, "NeckYaw": -0.4, "NeckPitch": 20.3},
            "joint_torques": {"RightShoulderPitch": 0.2, "RightElbowPitch": 0.1, "LeftKneePitch": 12.0},
            "desired": {"age_s": 5.0, "advancing": True}, "disk_free_gb": 10.0}
    plan = {"start_deg": [-42.6, -4.8, 47.5, -92.3, -135.7, 21.2, 19.8],
            "neck_deg": {"NeckYaw": -0.4, "NeckPitch": 20.3}}
    why = run91.prechecks(snap, plan, need_gb=30.0)
    text = " | ".join(why)
    for word in ("/desired", "start pose", "motors", "LeftKneePitch", "disk"):
        assert word in text
    good = {"q_deg": {**dict(zip(RIGHT_ARM, plan["start_deg"])), "NeckYaw": -0.4, "NeckPitch": 20.3},
            "joint_torques": {"RightShoulderPitch": -11.9, "RightElbowPitch": -15.0, "LeftKneePitch": -5.4},
            "desired": {"age_s": 18000.0, "advancing": False}, "disk_free_gb": 234.0}
    assert run91.prechecks(good, plan, need_gb=30.0) == []


def test_a_plan_that_failed_its_gate_is_refused(tmp_path):
    p = tmp_path / "poses.json"
    p.write_text(json.dumps({"gate": {"pass": False}, "stops": [{"kind": "fit"}]}))
    assert run91.main(["--poses", str(p), "--urdf", "u", "--mjcf", "m", "--tag", "t", "--dry-run"]) == 3
    p.write_text(json.dumps({"gate": {"pass": True}, "stops": []}))
    assert run91.main(["--poses", str(p), "--urdf", "u", "--mjcf", "m", "--tag", "t", "--dry-run"]) == 3
