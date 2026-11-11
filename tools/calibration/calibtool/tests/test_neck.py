"""calibtool.neck: the neck grid through the fixed dma_motor_motion (design plan 7, step 08's rules).

Fakes only: a FakeRobot (scripted /dev/shm/desired head, joint names and angles), a FakeTool runner that answers
like the motor tool (RESULT lines), fake /dev/shm ring files in tmp_path and a fake dma_ipc. Nothing here touches a
robot."""
import inspect
import struct

import pytest

np = pytest.importorskip("numpy")

from calibtool import imusweep, neck  # noqa: E402

NAMES = (["LeftHipPitch", "LeftKneePitch", "RightHipPitch", "RightKneePitch", "LeftAnklePitch", "SpineYaw",
          "LeftShoulderPitch", "LeftElbowPitch", "LeftWristYaw", "LeftGripperYaw", "RightShoulderPitch",
          "RightElbowPitch", "RightWristYaw", "RightGripperYaw", "NeckPitch", "NeckYaw"]
         + [f"ExtraJoint{i:02d}" for i in range(13)])
NECK_IDX = (NAMES.index("NeckPitch"), NAMES.index("NeckYaw"))


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, float(s))


class FakeRobot:
    """desired head (advances when `foreign` says so or while the own tool runs), joints in degrees"""

    def __init__(self, clock, q0=None):
        self.clock = clock
        self.head = 7
        self.q = list(q0 if q0 is not None else [float(i) for i in range(29)])
        self.q[NECK_IDX[0]], self.q[NECK_IDX[1]] = 0.0, 0.0
        self.names = list(NAMES)
        self.foreign_at = []         # clock times at which another writer bumps the head once
        self.reads = 0

    def desired_head(self):
        while self.foreign_at and self.clock() >= self.foreign_at[0]:
            self.foreign_at.pop(0)
            self.head += 5
        return self.head

    def joints(self):
        self.reads += 1
        return list(self.names), list(self.q)


class FakeTool:
    """the motor tool: `--help` text, and per move RESULT lines; the robot's neck follows the command with an offset"""

    def __init__(self, robot, clock, help_text="usage ... targets position ...", err=(0.4, -0.3), rc=1, sag=0.5):
        self.robot, self.clock, self.help, self.err, self.rc, self.sag = robot, clock, help_text, err, rc, sag
        self.calls = []

    def __call__(self, argv, timeout=None):
        self.calls.append(list(argv))
        if "--help" in argv:
            return 0, self.help
        spec = dict(kv.split("=") for kv in argv[1].split(","))
        p, y = float(spec["NeckPitch"]), float(spec["NeckYaw"])
        ramp = float(argv[argv.index("--ramp") + 1])
        hold = float(argv[argv.index("--hold") + 1])
        self.robot.head += 100                       # the own tool writes /dev/shm/desired while it runs
        self.clock.sleep(ramp + hold)
        self.robot.q[NECK_IDX[0]], self.robot.q[NECK_IDX[1]] = p + self.sag, y
        lines = [f"RESULT targets joint={NECK_IDX[0]} target_deg={p:.3f} ss_err={self.err[0]:.3f} rms=0.1",
                 f"RESULT targets joint={NECK_IDX[1]} target_deg={y:.3f} ss_err={self.err[1]:.3f} rms=0.1"]
        return self.rc, "\n".join(lines)


def make(tool_info=None, fast=True, **kw):
    c = Clock()
    r = FakeRobot(c)
    t = FakeTool(r, c, **kw)
    info = tool_info if tool_info is not None else {"path": "/k/dma_motor_motion", "md5": "208231a2",
                                                     "has_targets": True, "error": None}
    n = neck.Neck("/k/dma_motor_motion", t, r, info, clock=c, sleep=c.sleep, fast=fast)
    return n, r, t, c


# ---------------------------------------------------------------- A1: clamps, pins, argv


def test_clamp():
    assert neck.clamp(30, -40) == (26.0, -34.0, True)
    assert neck.clamp(-30, 40) == (-26.0, 34.0, True)
    assert neck.clamp(-8, 11) == (-8.0, 11.0, False)


def test_pins_hold_arms_spine_legs_not_neck_or_ankles():
    q = [float(i) + 0.123 for i in range(len(NAMES))]
    txt = neck.pins(NAMES, q)
    names = [kv.split("=")[0] for kv in txt.split(",")]
    assert "NeckPitch" not in names and "NeckYaw" not in names and "LeftAnklePitch" not in names
    assert "ExtraJoint00" not in names
    for want in ("LeftHipPitch", "RightKneePitch", "SpineYaw", "LeftShoulderPitch", "RightGripperYaw"):
        assert want in names
    assert f"SpineYaw={NAMES.index('SpineYaw') + 0.123:.2f}" in txt


def test_motion_argv_is_step_08s_with_kp_only():
    a = neck.motion_argv("/k/tool", 8.0, -11.0, "SpineYaw=1.00", 1.0, 0.8)
    assert a == ["/k/tool", "NeckPitch=8.0,NeckYaw=-11.0,SpineYaw=1.00", "targets", "position", "--ramp", "1.0",
                 "--hold", "0.8", "--no-return"]
    b = neck.motion_argv("/k/tool", 8.0, -11.0, "SpineYaw=1.00", 1.0, 0.8, kp=60)
    assert b[-2:] == ["--kp", "60"]
    for bad in (0, -5):
        with pytest.raises(ValueError):
            neck.motion_argv("/k/tool", 0, 0, "", 1.0, 0.8, kp=bad)
    assert "kd" not in inspect.signature(neck.motion_argv).parameters
    assert not any("kd" in x for x in a + b)


def test_motion_argv_clamps_whatever_it_is_given():
    a = neck.motion_argv("/k/tool", 40.0, -90.0, "", 1.0, 0.8)
    assert a[1].startswith("NeckPitch=26.0,NeckYaw=-34.0")


def test_no_kd_anywhere_in_neck_source():
    from pathlib import Path
    code = "\n".join(ln.split("#")[0] for ln in Path(neck.__file__).read_text().splitlines())
    assert "--kd" not in code


def test_neck_errors_reads_neck_joints_only():
    out = ("RESULT targets joint=14 target_deg=8.000 ss_err=0.450 rms=0.4\n"
           "RESULT targets joint=15 target_deg=-11.000 ss_err=-0.200 rms=0.2\n"
           "RESULT targets joint=6 target_deg=1.000 ss_err=9.000 rms=9\n")
    assert neck.neck_errors(out, (14, 15)) == {14: 0.45, 15: -0.2}


# ---------------------------------------------------------------- A2: /dev/shm reader


def _ring(path, entries, esz):
    cap = len(entries)
    hdr = bytearray(128)
    hdr[0:8] = b"UEUQ_AMD"
    struct.pack_into("<QQ", hdr, 16, esz, cap)
    struct.pack_into("<Q", hdr, 64, len(entries))
    body = b"".join(e.ljust(esz, b"\0") for e in entries)
    path.write_bytes(bytes(hdr) + body)


def test_shm_robot_reads_rings(tmp_path):
    _ring(tmp_path / "desired", [b"x" * 16, b"y" * 16], 16)
    cfg = b"\0".join(n.encode() for n in NAMES) + b"\0"
    _ring(tmp_path / "config", [cfg], 2048)
    q = np.radians(np.arange(29, dtype=np.float32))
    act = bytes(88) + struct.pack("<29f", *q.tolist())
    _ring(tmp_path / "actuals", [bytes(200), act], 256)
    r = neck.ShmRobot(tmp_path)
    assert r.desired_head() == 2
    names, deg = r.joints()
    assert names[:16] == NAMES[:16]
    assert len(deg) == 29 and abs(deg[14] - 14.0) < 1e-4


# ---------------------------------------------------------------- A3: arm refusals


def test_arm_refuses_without_targets():
    n, r, t, c = make(tool_info={"path": "/usr/bin/dma_motor_motion", "md5": "x", "has_targets": False,
                                 "error": None})
    with pytest.raises(neck.NeckRefused, match="targets"):
        n.arm()
    assert not n.commands and not [a for a in t.calls if "--help" not in a]


def test_arm_refuses_when_desired_advances():
    n, r, t, c = make()
    r.foreign_at = [c() + 0.5]
    with pytest.raises(neck.NeckRefused, match="desired"):
        n.arm()
    assert not n.commands


def test_arm_refuses_without_neck_joints():
    n, r, t, c = make()
    r.names = [x for x in NAMES if x != "NeckYaw"] + ["Other"]
    with pytest.raises(neck.NeckRefused, match="NeckYaw"):
        n.arm()


def test_goto_before_arm_refused():
    n, r, t, c = make()
    with pytest.raises(neck.NeckRefused):
        n.goto(0, 0)
    assert not t.calls


def test_arm_pins_start_angles():
    n, r, t, c = make()
    n.arm()
    pins0 = n.pins
    r.q[NAMES.index("SpineYaw")] = 77.0
    n.goto(8, 11)
    assert n.commands[-1][1].endswith(pins0)
    assert "SpineYaw=77.00" not in n.commands[-1][1]


# ---------------------------------------------------------------- A4: goto


def test_goto_argv_and_posedone():
    n, r, t, c = make(fast=False)                       # step 08's fixed timing
    n.arm()
    d = n.goto(30, -11)
    assert d.command == (26.0, -11.0) and d.clamped
    a = n.commands[-1]
    assert a[0] == "/k/dma_motor_motion" and a[1].startswith("NeckPitch=26.0,NeckYaw=-11.0,")
    assert a[2:] == ["targets", "position", "--ramp", "1.0", "--hold", "0.8", "--no-return"]
    assert d.measured == pytest.approx((26.5, -11.0))
    assert d.errors == {NECK_IDX[0]: 0.4, NECK_IDX[1]: -0.3}


def test_foreign_check_waits_half_a_second_after_own_command():
    n, r, t, c = make()
    n.arm()
    n.goto(0, 0)
    t_end = c()
    n.goto(8, 0)
    # the second command started >= 0.5 s after the first one ended (only the remainder is slept)
    assert n.log[-1]["t_start"] - t_end >= neck.FOREIGN_STILL_S - 1e-9


def test_no_extra_wait_when_capture_already_took_long_enough():
    n, r, t, c = make()
    n.arm()
    n.goto(0, 0)
    c.sleep(2.0)                         # the capture of pose 1
    t0 = c()
    n.goto(8, 0)
    assert n.log[-1]["t_start"] == pytest.approx(t0)


def test_foreign_command_between_poses_refused_nothing_sent():
    n, r, t, c = make()
    n.arm()
    n.goto(0, 0)
    sent = len(n.commands)
    r.foreign_at = [c() + 0.1]
    with pytest.raises(neck.ForeignCommand):
        n.goto(8, 0)
    assert len(n.commands) == sent


def test_missed_pose_raises_and_holds():
    n, r, t, c = make(err=(3.5, 0.1), sag=3.5)
    n.arm()
    with pytest.raises(neck.NeckMissed, match="3.5"):
        n.goto(8, 0)
    assert len(n.commands) == 1


def test_a_slow_neck_that_arrives_after_the_tool_reports_is_not_a_miss():
    """10 Nov, phantom11: the first grid pose (-24/-34 from home) was 21.6 / 13.8 deg short when the tool reported
    (ramp 1 s + hold 0.8 s) and there a moment later; the grid held for nothing"""
    n, r, t, c = make(err=(21.6, 13.8), sag=0.0)
    n.arm()
    real = r.joints

    def lagging():
        names, q = real()
        if c() < t_arrive[0]:
            q[NECK_IDX[0]], q[NECK_IDX[1]] = -2.4, -20.2
        return names, q
    t_arrive = [c() + 1.8 + 1.5]                        # 1.5 s after the tool's ramp + hold
    r.joints = lagging
    d = n.goto(-24, -34)
    assert d.measured == pytest.approx((-24.0, -34.0), abs=0.2)
    assert n.log[-1]["arrived_late_s"] > 1.0


def test_a_neck_that_never_arrives_is_a_miss_after_the_wait():
    n, r, t, c = make(err=(21.6, 13.8), sag=-21.6)
    n.arm()
    t0 = c()
    with pytest.raises(neck.NeckMissed, match="measured"):
        n.goto(-24, 0)
    assert c() - t0 >= neck.ARRIVE_TIMEOUT_S


def test_missing_result_line_is_a_miss():
    n, r, t, c = make()
    n.arm()
    t.err = (0.1, 0.1)
    orig = t.__call__

    def no_result(argv, timeout=None):
        rc, out = orig(argv, timeout)
        return rc, "" if "--help" not in argv else out
    n.run = no_result
    with pytest.raises(neck.NeckMissed):
        n.goto(8, 0)


def test_rc_1_with_good_results_is_fine():
    n, r, t, c = make(rc=1)
    n.arm()
    assert n.goto(0, 11).measured == pytest.approx((0.5, 11.0))


def test_measured_after_settle_waits_until_still():
    n, r, t, c = make()
    n.arm()
    seq = iter([(5.0, 0.0), (5.3, 0.0), (5.31, 0.0), (5.31, 0.0)])

    def joints():
        p, y = next(seq, (5.31, 0.0))
        q = list(r.q)
        q[NECK_IDX[0]], q[NECK_IDX[1]] = p, y
        return list(r.names), q
    r.joints = joints
    n.robot = r
    d = n.goto(5, 0)
    assert d.measured[0] == pytest.approx(5.31, abs=0.02)


# ---------------------------------------------------------------- A5: run_grid


def test_run_grid_hold_ends_at_capture():
    n, r, t, c = make()
    n.arm()
    seen = []

    def on_pose(d):
        seen.append((d.command, len(n.commands)))
        c.sleep(0.3)                     # the capture: no own command runs meanwhile
    out = neck.run_grid(n, [(0, 0), (8, 11), (16, -11)], on_pose, lambda: None)
    assert out.stopped is None and len(out.done) == 3
    assert [s[1] for s in seen] == [1, 2, 3]         # the next command only after on_pose returned


@pytest.mark.parametrize("why", ["abort", "hold"])
def test_run_grid_stops_between_poses(why):
    n, r, t, c = make()
    n.arm()
    calls = {"k": 0}

    def stop():
        calls["k"] += 1
        return why if calls["k"] == 3 else None
    out = neck.run_grid(n, [(0, 0), (8, 0), (16, 0), (24, 0)], lambda d: None, stop)
    assert out.stopped == why and len(out.done) == 2 and len(n.commands) == 2


def test_run_grid_error_stops_without_more_commands():
    n, r, t, c = make(err=(4.0, 0.1), sag=4.0)
    n.arm()
    out = neck.run_grid(n, [(0, 0), (8, 0)], lambda d: None, lambda: None)
    assert out.stopped.startswith("error") and len(n.commands) == 1 and not out.done


# ---------------------------------------------------------------- A6: dry run, SweepRig, IMU


def test_dry_run_records_exact_commands():
    q0 = [float(i) for i in range(29)]
    dr = neck.DryRun(NAMES, q0)
    n = neck.Neck("/root/p11-cal/dma_motor_motion", dr.run, dr.robot, dr.tool_info, clock=dr.clock,
                  sleep=dr.clock.sleep)
    n.arm()
    out = neck.run_grid(n, [(-24, -34), (-24, -23), (30, 0)], lambda d: None, lambda: None)
    assert out.stopped is None
    assert dr.commands == n.commands and len(dr.commands) == 3
    assert dr.commands[2][1].startswith("NeckPitch=26.0,NeckYaw=0.0,")
    assert out.done[0].measured == pytest.approx((-24.0, -34.0))
    assert "dma_motor_motion" in dr.text()


def test_sweep_rig_runs_imusweep_capture():
    n, r, t, c = make()
    n.arm()
    rig = neck.NeckSweepRig(n, imu_accel=lambda s: np.tile([0.0, -9.81, 0.0], (10, 1)),
                            frames=lambda k: [np.zeros((4, 4), np.uint8)] * k)
    assert rig.foreign_seq() == r.head
    res = imusweep.capture_sweep(rig, [(8, -15), (8, 15)], detect=lambda g: None, R_SC=np.eye(3), sleep=c.sleep)
    assert len(res) == 2 and len(n.commands) == 2
    assert res[1].neck == pytest.approx([8.5, 15.0])


def test_dma_imu_against_fake_dma_ipc():
    class S:
        def __init__(self, k):
            self.accel_x, self.accel_y, self.accel_z = 0.1 * k, -9.8, 0.2

    class F:
        def __init__(self, k):
            self.k = k

        def imu(self, slot):
            assert slot == 2
            return S(self.k)

    class Cons:
        def __init__(self):
            self.left = [F(k) for k in range(5)]

        def available(self):
            return bool(self.left)

        def pop(self):
            return self.left.pop(0)

    class Q:
        def __init__(self, name, create, wait):
            assert name == "/raw_imu_actuals" and create is False

        def make_consumer(self):
            return Cons()

    class D:
        RawImuQueue = Q
    slept = []
    imu = neck.DmaImu(D, slot=2, sleep=slept.append)
    a = imu.accel(2.0)
    assert a.shape == (5, 3) and slept == [2.0] and a[4, 0] == pytest.approx(0.4)



def test_fast_timing_scales_the_ramp_with_the_step_and_holds_briefly():
    """10 Nov: 2.53 s per grid pose on phantom11, 1.8 s of it the fixed 1.0 s ramp + 0.8 s hold"""
    n, r, t, c = make()
    n.arm()
    n.goto(8, 11)                                        # from home: 11 deg -> 0.37 s
    a = n.commands[-1]
    assert a[a.index("--ramp") + 1] == "0.37" and a[a.index("--hold") + 1] == str(neck.FAST_SETTLE_S)
    n.goto(8, 34)                                        # the neck at ~8.5/11 now: 23 deg -> 0.77 s
    a = n.commands[-1]
    assert 0.7 <= float(a[a.index("--ramp") + 1]) <= 0.8
    n.goto(8, 35)                                        # a tiny step still gets RAMP_MIN_S
    a = n.commands[-1]
    assert float(a[a.index("--ramp") + 1]) == neck.RAMP_MIN_S
