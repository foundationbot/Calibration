"""handeye_wave.py: the safety rules of the hand-eye arm wave (its docstring's list), on a fake robot."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import handeye_wave as hw  # noqa: E402

NAMES = ["LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbowPitch", "LeftWristYaw", "LeftWristRoll",
         "LeftGripperYaw", "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbowPitch",
         "RightWristYaw", "RightWristRoll", "RightGripperYaw", "NeckPitch", "NeckYaw", "SpineYaw", "LeftHipYaw"]
START = {n: float(i) for i, n in enumerate(NAMES)}
START["NeckPitch"], START["NeckYaw"] = 27.9, 0.3


def plan():
    return {"moves": [{"name": "stage", "deg": [-20, -8, 6, -65, 0, 0, 0], "burst": False},
                      {"name": "p07", "deg": [-58, -21, -3, -43, 46, 23, -40], "burst": True,
                       "necks": [[10, 0], [30, -40]]}]}


def make(tmp_path, robot=None, **kw):
    r = robot or hw.DryRun(NAMES, START)
    return hw.Wave(r, plan(), "Right", tmp_path, log=lambda s: None, **kw), r


def parse(text):
    return dict((kv.split("=")[0], float(kv.split("=")[1])) for kv in text.split(","))


def test_every_joint_listed_others_pinned_at_start(tmp_path):
    w, r = make(tmp_path)
    w.run(hand_pose=None)
    motions = [a for a in r.sent if a[0] == hw.MOTOR_TOOL]
    assert motions
    for argv in motions:
        d = parse(argv[1])
        assert set(d) == set(NAMES)
        for n in NAMES:
            if not n.startswith("Right") and not n.startswith("Neck"):
                assert d[n] == pytest.approx(START[n])


def test_left_arm_moves_left_joints_only(tmp_path):
    r = hw.DryRun(NAMES, START)
    w = hw.Wave(r, plan(), "Left", tmp_path, log=lambda s: None)
    w.run(hand_pose=None)
    d = parse([a for a in r.sent if a[0] == hw.MOTOR_TOOL][0][1])
    assert d["LeftShoulderPitch"] == -20 and d["RightShoulderPitch"] == START["RightShoulderPitch"]


def test_neck_clamped_and_arm_held_during_neck_moves(tmp_path):
    w, r = make(tmp_path)
    w.run(hand_pose=None)
    motions = [parse(a[1]) for a in r.sent if a[0] == hw.MOTOR_TOOL]
    for d in motions:
        assert abs(d["NeckPitch"]) <= 26 and abs(d["NeckYaw"]) <= 34
    assert motions[-1]["RightShoulderPitch"] == -58          # the neck move keeps the arm at p07's target
    assert motions[-1]["NeckPitch"] == 26 and motions[-1]["NeckYaw"] == -34


def test_ramps_scale_with_travel_and_have_a_floor():
    assert hw.arm_ramp([0] * 7, [10] + [0] * 6) == hw.RAMP_MIN_S
    assert hw.arm_ramp([0] * 7, [80] + [0] * 6) == pytest.approx(80 / hw.MAX_RATE_DEG_S)
    assert hw.neck_ramp((0, 0), (5, 0)) == hw.NECK_RAMP_MIN_S


def test_kp_only_and_default_is_the_tools_gains():
    a = hw.motion_argv("t", "x=1", 6, 2)
    assert "--kp" not in a and not any("kd" in s.lower() for s in a)
    b = hw.motion_argv("t", "x=1", 6, 2, kp=60)
    assert b[b.index("--kp") + 1] == "60"
    with pytest.raises(ValueError):
        hw.motion_argv("t", "x=1", 6, 2, kp=0)
    assert "--no-return" in a and a[a.index("--tolerance") + 1] == "30"


def test_missing_pin_refuses():
    with pytest.raises(ValueError):
        hw.spec(NAMES + ["Mystery"], {}, "Right", [0] * 7, (0, 0))


class Failing(hw.DryRun):
    def __init__(self, *a, fail_after=1, why="motors not on", still=True):
        super().__init__(*a)
        self.n, self.fail_after, self.why, self.still = 0, fail_after, why, still

    def master_ok(self):
        self.n += 1
        return self.why if self.n > self.fail_after else None

    def desired_still(self):
        return self.still


def test_motors_off_stops_before_the_next_command(tmp_path):
    r = Failing(NAMES, START, fail_after=1)
    w, _ = make(tmp_path, robot=r)
    with pytest.raises(hw.Stop):
        w.run(hand_pose=None)
    assert len([a for a in r.sent if a[0] == hw.MOTOR_TOOL]) == 1


def test_foreign_writer_stops_everything(tmp_path):
    r = Failing(NAMES, START, fail_after=99, still=False)
    w, _ = make(tmp_path, robot=r)
    with pytest.raises(hw.Stop):
        w.run(hand_pose=None)
    assert r.sent == []


def test_stop_file(tmp_path):
    (tmp_path / "STOP").write_text("")
    w, r = make(tmp_path)
    with pytest.raises(hw.Stop):
        w.run(hand_pose=None)
    assert r.sent == []


def test_tool_failure_stops(tmp_path):
    class Rc(hw.DryRun):
        def run(self, argv, timeout):
            self.sent.append(argv)
            return 1, "RESULT RightShoulderPitch ss_err 35.0 FAIL"
    r = Rc(NAMES, START)
    w, _ = make(tmp_path, robot=r)
    with pytest.raises(hw.Stop):
        w.run(hand_pose=None)
    assert len(r.sent) == 1


def test_bursts_named_and_logged(tmp_path):
    w, r = make(tmp_path)
    w.run(hand_pose="open")
    grabs = [a for a in r.sent if hw.GRAB in a]
    assert [Path(a[a.index("--out") + 1]).name for a in grabs] == ["01-p07-10-0.npz", "02-p07-26-m34.npz"]
    hands = [a for a in r.sent if a[0] == "inspire_hand_poses"]
    assert [a[a.index("--pose") + 1] for a in hands] == ["open", "relax"] and hands[0][2] == "1"
    kinds = [json.loads(l)["kind"] for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert kinds[0] == "start" and kinds[-1] == "done"


def test_pins_move_first_slowly_then_hold(tmp_path):
    w, r = make(tmp_path)
    w.run(hand_pose=None, pins={"LeftHipYaw": START["LeftHipYaw"] - 20})
    motions = [a for a in r.sent if a[0] == hw.MOTOR_TOOL]
    first = parse(motions[0][1])
    assert first["LeftHipYaw"] == START["LeftHipYaw"] - 20
    assert first["RightShoulderPitch"] == pytest.approx(START["RightShoulderPitch"])      # the arm does not move yet
    assert float(motions[0][motions[0].index("--ramp") + 1]) == max(hw.RAMP_MIN_S, 20 / hw.MAX_RATE_DEG_S)
    assert all(parse(a[1])["LeftHipYaw"] == START["LeftHipYaw"] - 20 for a in motions)   # held afterwards


def test_pins_refuse_arm_and_neck(tmp_path):
    w, _ = make(tmp_path)
    for bad in ({"RightShoulderPitch": 0}, {"NeckPitch": 0}, {"Mystery": 0}):
        with pytest.raises(ValueError):
            w.move_pins(bad)


def test_profiles_switch_per_pose_and_restore(tmp_path):
    r = hw.DryRun(NAMES, START)
    r._profile = "1280"
    w = hw.Wave(r, plan(), "Right", tmp_path, log=lambda s: None, profiles=["1280", "768"])
    w.run(hand_pose=None)
    prof = [a[1] for a in r.sent if a[0] == "profile"]
    assert prof == ["768", "1280"]              # 1280 already active: no switch to it; to 768 once; back at the end
    assert r.read_profile() == "1280"
    grabs = [Path(a[a.index("--out") + 1]) for a in r.sent if hw.GRAB in a]
    assert [g.parent.name for g in grabs] == [tmp_path.name] * 2 + ["r768"] * 2
    assert [g.name for g in grabs[:2]] == [g.name for g in grabs[2:]]   # same capture names in both folders


def test_profile_restored_on_stop_and_default_when_unset(tmp_path):
    class Dies(hw.DryRun):
        def run(self, argv, timeout):
            if hw.GRAB in argv:
                raise hw.Stop("burst failed")
            return super().run(argv, timeout)
    r = Dies(NAMES, START)                       # no request file before: restore to the config default
    w = hw.Wave(r, plan(), "Right", tmp_path, log=lambda s: None, profiles=["768"], restore_profile="1280")
    with pytest.raises(hw.Stop):
        w.run(hand_pose=None)
    assert r.read_profile() == "1280"


def test_no_profiles_never_touches_okvis(tmp_path):
    w, r = make(tmp_path)
    w.run(hand_pose=None)
    assert not [a for a in r.sent if a[0] == "profile"]


def test_check_reports_every_problem_and_moves_nothing(tmp_path, monkeypatch):
    class R(hw.DryRun):
        def master_ok(self):
            return "motors not on"

        def desired_still(self):
            return False
    moved = dict(START, LeftHipYaw=START["LeftHipYaw"] + 40)
    r = R(NAMES, moved)
    monkeypatch.setattr(hw.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": "targets", "stderr": "",
                                                                              "returncode": 0})())
    res = hw.check(r, dict(plan(), snapshot={"q_deg": START}), tool="t", grab_test=lambda: (1, "timeout"))
    assert not res["ok"] and r.sent == []
    txt = " ".join(res["problems"])
    assert "motors not on" in txt and "/desired" in txt and "LeftHipYaw +40.0" in txt and "no depth" in txt


def test_check_ok(monkeypatch):
    monkeypatch.setattr(hw.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": "x targets y", "stderr": "",
                                                                              "returncode": 0})())
    res = hw.check(hw.DryRun(NAMES, START), dict(plan(), snapshot={"q_deg": START}), tool="t",
                   grab_test=lambda: (0, "1 frame"))
    assert res["ok"] and res["problems"] == []


def test_check_refuses_tool_without_targets(monkeypatch):
    monkeypatch.setattr(hw.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": "Usage", "stderr": "",
                                                                              "returncode": 0})())
    res = hw.check(hw.DryRun(NAMES, START), dict(plan(), snapshot={"q_deg": START}), tool="t")
    assert not res["ok"] and "targets" in res["problems"][0]



class FlakyGrab(hw.DryRun):
    """the grabber fails `fails` times, then works"""
    def __init__(self, *a, fails=1):
        super().__init__(*a)
        self.fails = fails

    def run(self, argv, timeout):
        self.sent.append(argv)
        if hw.GRAB in argv and self.fails > 0:
            self.fails -= 1
            return 1, "TimeoutError: no 5 consecutive frames in 9.3 s"
        return 0, "RESULT dry-run"


def test_a_failed_burst_is_retried_before_the_wave_stops(tmp_path):
    r = FlakyGrab(NAMES, START, fails=hw.BURST_RETRIES)
    w, _ = make(tmp_path, robot=r)
    w.run(hand_pose=None)                          # every retry used, then fine: the wave completes
    kinds = [json.loads(l)["kind"] for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert kinds.count("burst_retry") == hw.BURST_RETRIES and kinds[-1] == "done"
    assert sum(hw.BURST_RETRY_WAITS_S) >= 30.0     # outlasts the 35 s USB stall with the grab attempts between


def test_three_failures_stop_the_wave(tmp_path):
    r = FlakyGrab(NAMES, START, fails=hw.BURST_RETRIES + 1)
    w, _ = make(tmp_path, robot=r)
    with pytest.raises(hw.Stop):
        w.run(hand_pose=None)


def test_check_ignores_the_neck(monkeypatch):
    monkeypatch.setattr(hw.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": "targets", "stderr": "",
                                                                              "returncode": 0})())
    now = dict(START, NeckPitch=-14.7, NeckYaw=-11.2)
    res = hw.check(hw.DryRun(NAMES, now), dict(plan(), snapshot={"q_deg": START}), tool="t", grab_test=lambda: (0, ""))
    assert res["ok"], res["problems"]
