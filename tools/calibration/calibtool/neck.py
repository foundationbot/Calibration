"""The neck grid through the fixed dma_motor_motion (design plan sections 4, 7, 9.3; step 08's rules, ported without
its robot-only imports).

Safety, each enforced here (tests/test_neck.py pins every one):
  - the motor tool must answer `--help` with 'targets' (the fixed build: 2.1.1's tool leaves unlisted joints at zero
    stiffness); Neck.arm() refuses otherwise, before any command
  - Neck.arm() refuses when /dev/shm/desired advances during START_STILL_S (another program is commanding) or when the
    neck joints are not in /dev/shm/config's names
  - every command pins the arm, spine and leg joints (PIN_PREFIXES) at their angles AT ARM TIME (step 08: the tool's
    own re-hold of unlisted joints at their measured pose made phantom14's body creep over its grids)
  - pitch clamped to +-MAX_PITCH (26: +26 commanded reached +27.7 on phantom15, limit 27.5), yaw to +-MAX_YAW (34: a
    mechanical stop at 35.9 on phantom14), whatever the caller asks
  - kp only: motion_argv takes an optional kp and has no damping parameter at all (gains other than kp are never
    changed); the tool's default gains apply when kp is None (step 08's behaviour)
  - before EVERY own command /dev/shm/desired must have been still for >= FOREIGN_STILL_S since the previous own
    command ended (the tool's own writes move it, so a start-time check is not enough); ForeignCommand otherwise,
    nothing sent
  - a neck joint missing its target by more than MAX_ERR_DEG (the tool's RESULT ss_err), or no RESULT line, stops the
    grid (NeckMissed); the tool's exit code is not the verdict (it exits 1 on its own 1 deg tolerance under gravity sag)
  - safe state = hold: the tool runs with --no-return, so the master keeps the last pose after it exits; stopping
    (abort or hold) means sending nothing more

Hold ends at capture (protocol.HOLD_ENDS_AT_CAPTURE): the tool's --hold is the settle time only (SETTLE_S), so it exits
once the neck has settled; run_grid then reads the measured angles and calls on_pose (the capture) while no own command
runs, and the next pose starts as soon as on_pose returns. The pose's foreign-command window overlaps the capture.

DryRun is a backend that records the exact commands it would send (and answers like the tool); NeckSweepRig is
imusweep.SweepRig over a Neck; DmaImu reads the head IMU slot from a dma_ipc module passed in."""
import math
import mmap
import re
import shlex
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NECK = ("NeckPitch", "NeckYaw")
PIN_PREFIXES = ("LeftShoulder", "LeftElbow", "LeftWrist", "LeftGripper", "RightShoulder", "RightElbow", "RightWrist",
                "RightGripper", "SpineYaw", "LeftHip", "RightHip", "LeftKnee", "RightKnee")
MAX_PITCH = 26.0
MAX_YAW = 34.0
RAMP_S = 1.0
SETTLE_S = 0.8                   # the tool's --hold: step 08 read the angles ramp + 0.8 s after the start
FOREIGN_STILL_S = 0.5            # design 7: desired still this long while no own command runs, before each command
START_STILL_S = 1.0              # step 08: desired still over 1 s before the grid
MAX_ERR_DEG = 3.0                # step 08's --max-err
STILL_DEG = 0.1                  # two measured reads this close: the neck has settled
STILL_POLL_S = 0.1
STILL_TIMEOUT_S = 1.0
# 10 Nov, phantom11: its neck lags the tool's ramp; a pose the tool reports short is polled this long for the neck to
# arrive within MAX_ERR_DEG before it counts as missed
ARRIVE_TIMEOUT_S = 4.0
N_JOINTS = 29
TOOL_TIMEOUT_S = 30.0
RESULT = re.compile(r"RESULT targets joint=(\d+) target_deg=\S+ ss_err=(\S+)")


class NeckRefused(RuntimeError):
    """a refusal before any command (no 'targets', something commanding, no neck joints, not armed)"""


class ForeignCommand(NeckRefused):
    """/dev/shm/desired moved while no own command ran: another program is commanding; nothing was sent"""


class NeckMissed(RuntimeError):
    """the neck did not reach the pose (ss_err above MAX_ERR_DEG or no RESULT line): the grid stops, the pose is held"""


# ---------------------------------------------------------------- pure parts


def clamp(pitch, yaw):
    """(pitch, yaw, clamped) within +-MAX_PITCH / +-MAX_YAW"""
    p = max(-MAX_PITCH, min(MAX_PITCH, float(pitch)))
    y = max(-MAX_YAW, min(MAX_YAW, float(yaw)))
    return p, y, (p, y) != (float(pitch), float(yaw))


def pins(names, q_deg) -> str:
    """'Joint=deg,...' holding every arm/spine/leg joint at q_deg (the neck and ankles are not pinned)"""
    return ",".join(f"{n}={q_deg[i]:.2f}" for i, n in enumerate(names) if n.startswith(PIN_PREFIXES))


def neck_errors(out, idx) -> dict:
    """{joint index: steady-state error} for the neck joints from the tool's RESULT lines"""
    return {int(j): float(e) for j, e in RESULT.findall(out or "") if int(j) in idx}


def motion_argv(tool, pitch, yaw, pins_text, ramp, hold, kp=None) -> list:
    """step 08's command for one neck pose (clamped here too); kp only: --kp when given, no other gain"""
    p, y, _ = clamp(pitch, yaw)
    spec = f"NeckPitch={p},NeckYaw={y}" + (f",{pins_text}" if pins_text else "")
    argv = [str(tool), spec, "targets", "position", "--ramp", str(float(ramp)), "--hold", str(float(hold)),
            "--no-return"]
    if kp is not None:
        if not float(kp) > 0:
            raise ValueError(f"kp {kp}: must be > 0 (Nm/rad)")
        argv += ["--kp", f"{float(kp):g}"]
    return argv


# ---------------------------------------------------------------- /dev/shm


def _latest(mm):
    esz, cap = struct.unpack_from("<QQ", mm, 16)
    head = struct.unpack_from("<Q", mm, 64)[0]
    off = 128 + ((head - 1) % cap) * esz
    return mm[off:off + esz]


def joint_names(entry, n=N_JOINTS) -> list:
    """the first n joint names in a /dev/shm/config entry (CamelCase identifiers, in joint order)"""
    return [m.decode() for m in re.findall(rb"[A-Z][A-Za-z]{3,}", bytes(entry))][:n]


class ShmRobot:
    """the robot's rings, read only: /dev/shm/desired's head counter, joint names (config) and angles (actuals)"""

    def __init__(self, shm="/dev/shm", n=N_JOINTS):
        self.shm, self.n = Path(shm), n

    def _map(self, name):
        with open(self.shm / name, "rb") as f:
            return mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

    def desired_head(self) -> int:
        mm = self._map("desired")
        try:
            return struct.unpack_from("<Q", mm, 64)[0]
        finally:
            mm.close()

    def joints(self):
        """(names, angles in degrees)"""
        cm, am = self._map("config"), self._map("actuals")
        try:
            names = joint_names(_latest(cm), self.n)
            q = struct.unpack_from(f"<{self.n}f", _latest(am), 88)
        finally:
            cm.close()
            am.close()
        return names, [math.degrees(v) for v in q]


# ---------------------------------------------------------------- the neck


@dataclass
class PoseDone:
    command: tuple                  # (pitch, yaw) commanded, after the clamps
    requested: tuple                # (pitch, yaw) asked for
    clamped: bool
    measured: tuple                 # (pitch, yaw) measured after settle
    still: bool                     # the measured angles settled within STILL_TIMEOUT_S
    errors: dict                    # {joint index: ss_err} of the neck joints
    seconds: float                  # the command's duration
    t_end: float

    def as_dict(self) -> dict:
        return {"command": list(self.command), "requested": list(self.requested), "clamped": self.clamped,
                "measured": list(self.measured), "still": self.still,
                "errors": {str(k): v for k, v in self.errors.items()}, "seconds": self.seconds, "t_end": self.t_end}


class Neck:
    """neck poses through the motor tool. run(argv, timeout) -> (rc, text) runs the tool; robot: desired_head() and
    joints() (ShmRobot); tool_info: motors.motor_tool()'s dict (md5, has_targets)"""

    def __init__(self, tool, run, robot, tool_info, clock=time.monotonic, sleep=time.sleep, kp=None, ramp_s=RAMP_S,
                 settle_s=SETTLE_S, max_err_deg=MAX_ERR_DEG):
        self.tool, self.run, self.robot, self.tool_info = str(tool), run, robot, dict(tool_info or {})
        self.clock, self.sleep, self.kp = clock, sleep, kp
        self.ramp_s, self.settle_s, self.max_err = float(ramp_s), float(settle_s), float(max_err_deg)
        if kp is not None:
            motion_argv(self.tool, 0, 0, "", 1, 1, kp=kp)     # validates kp now
        self.armed = False
        self.pins = None
        self.neck_idx = None
        self.commands = []          # every argv sent, in order
        self.log = []               # per command: argv, t_start, t_end, rc, errors
        self._exit = None           # (desired head, clock) when the last own command ended (or arm's check did)
        self.last = None            # the last PoseDone

    def arm(self):
        """step 08's refusals, then the pins at the current angles"""
        info = self.tool_info
        if not info.get("has_targets"):
            raise NeckRefused(f"{info.get('path') or self.tool} has no 'targets' signal: it would leave the spine and "
                              f"legs at zero stiffness during a neck move; use the fixed build"
                              + (f" ({info['error']})" if info.get("error") else ""))
        names, q = self.robot.joints()
        missing = [n for n in NECK if n not in names]
        if missing:
            raise NeckRefused(f"{', '.join(missing)} not in /dev/shm/config's joint names: {names}")
        h0 = self.robot.desired_head()
        self.sleep(START_STILL_S)
        h1 = self.robot.desired_head()
        if h1 != h0:
            raise NeckRefused(f"/dev/shm/desired advanced during {START_STILL_S:g} s ({h0} -> {h1}): something is "
                              "commanding the motors; stop it first (the robot must hang on the gantry)")
        self.neck_idx = (names.index(NECK[0]), names.index(NECK[1]))
        self.pins = pins(names, q)
        self._exit = (h1, self.clock())
        self.armed = True
        return self.pins

    def _foreign_check(self):
        h0, t0 = self._exit
        wait = FOREIGN_STILL_S - (self.clock() - t0)
        if wait > 0:
            self.sleep(wait)
        h = self.robot.desired_head()
        if h != h0:
            raise ForeignCommand(f"/dev/shm/desired moved ({h0} -> {h}) while no own command ran: another program is "
                                 "commanding the motors; nothing sent, the neck holds its pose")

    def _measured(self):
        _, q = self.robot.joints()
        prev = (q[self.neck_idx[0]], q[self.neck_idx[1]])
        t0 = self.clock()
        while self.clock() - t0 < STILL_TIMEOUT_S:
            self.sleep(STILL_POLL_S)
            _, q = self.robot.joints()
            cur = (q[self.neck_idx[0]], q[self.neck_idx[1]])
            if max(abs(cur[0] - prev[0]), abs(cur[1] - prev[1])) <= STILL_DEG:
                return cur, True
            prev = cur
        return prev, False

    def _arrive(self, p, y):
        """(seconds it took, None) once the measured neck is within max_err of (p, y), or (None, its last offsets)
        after ARRIVE_TIMEOUT_S"""
        t0 = self.clock()
        while True:
            mp, my = self.neck_deg()
            off = (mp - p, my - y)
            if max(abs(off[0]), abs(off[1])) <= self.max_err:
                return self.clock() - t0 + 1e-9, off
            if self.clock() - t0 >= ARRIVE_TIMEOUT_S:
                return None, off
            self.sleep(STILL_POLL_S)

    def goto(self, pitch, yaw) -> PoseDone:
        if not self.armed:
            raise NeckRefused("the neck is not armed (Neck.arm(): refusals and pins first)")
        p, y, clamped = clamp(pitch, yaw)
        self._foreign_check()
        argv = motion_argv(self.tool, p, y, self.pins, self.ramp_s, self.settle_s, kp=self.kp)
        t_start = self.clock()
        self.commands.append(argv)
        rc, out = self.run(argv, timeout=self.ramp_s + self.settle_s + TOOL_TIMEOUT_S)
        t_end = self.clock()
        errs = neck_errors(out, self.neck_idx)
        self.log.append({"argv": argv, "t_start": t_start, "t_end": t_end, "rc": rc, "errors": errs})
        self._exit = (self.robot.desired_head(), self.clock())
        if len(errs) != 2:
            raise NeckMissed(f"pose {p:+.1f}/{y:+.1f}: the tool reported no result for both neck joints (rc {rc}): "
                             f"{(out or '').strip()[-300:]}")
        bad = {j: e for j, e in errs.items() if abs(e) > self.max_err}
        if bad:
            late, off = self._arrive(p, y)
            if late is None:
                raise NeckMissed(f"pose {p:+.1f}/{y:+.1f} not reached: " + ", ".join(
                    f"joint {j} off by {e:+.1f} deg" for j, e in sorted(bad.items())) + f" (limit {self.max_err:g} deg)"
                    f"; measured still off by {off[0]:+.1f} / {off[1]:+.1f} deg after {ARRIVE_TIMEOUT_S:g} s")
            self.log[-1]["arrived_late_s"] = late
        meas, still = self._measured()
        self._exit = (self.robot.desired_head(), self.clock())
        self.last = PoseDone((p, y), (float(pitch), float(yaw)), clamped, tuple(meas), still, errs, t_end - t_start,
                             t_end)
        return self.last

    def neck_deg(self) -> tuple:
        """the measured (pitch, yaw) now"""
        _, q = self.robot.joints()
        return q[self.neck_idx[0]], q[self.neck_idx[1]]


@dataclass
class GridOutcome:
    done: list = field(default_factory=list)       # PoseDone per pose reached
    stopped: object = None                         # None (all poses), "abort", "hold" or "error: ..."

    def as_dict(self) -> dict:
        return {"done": [d.as_dict() for d in self.done], "stopped": self.stopped}


def run_grid(neck: Neck, poses, on_pose, should_stop) -> GridOutcome:
    """the poses [(pitch, yaw)] in order. should_stop() -> None | "abort" | "hold" is asked before each pose (stop =
    send nothing more: the master holds the last pose); on_pose(PoseDone) is the capture, run while no own command
    runs: the hold ends when it returns. An error stops the grid with no further command"""
    out = GridOutcome()
    for p, y in poses:
        why = should_stop()
        if why:
            out.stopped = str(why)
            return out
        try:
            d = neck.goto(p, y)
        except (NeckRefused, NeckMissed) as e:
            out.stopped = f"error: {e}"
            return out
        out.done.append(d)
        on_pose(d)
    return out


# ---------------------------------------------------------------- dry run


class _DryClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, float(s))


class _DryRobot:
    def __init__(self, names, q0):
        self.names, self.q, self.head = list(names), list(q0), 1

    def desired_head(self):
        return self.head

    def joints(self):
        return list(self.names), list(self.q)


class DryRun:
    """a backend that records the exact commands the neck would send and answers like the tool (ss_err 0, the neck
    reaching the command); time is simulated. Use: Neck(tool, dr.run, dr.robot, dr.tool_info, clock=dr.clock,
    sleep=dr.clock.sleep)"""

    def __init__(self, names, q0, tool_path="dma_motor_motion"):
        self.clock = _DryClock()
        self.robot = _DryRobot(names, q0)
        self.commands = []
        self.tool_info = {"path": tool_path, "md5": "dry-run", "has_targets": True, "error": None}

    def run(self, argv, timeout=None):
        if "--help" in argv:
            return 0, "dry run: targets"
        self.commands.append(list(argv))
        spec = dict(kv.split("=") for kv in argv[1].split(","))
        names = self.robot.names
        ip, iy = names.index("NeckPitch"), names.index("NeckYaw")
        self.clock.sleep(float(argv[argv.index("--ramp") + 1]) + float(argv[argv.index("--hold") + 1]))
        self.robot.q[ip], self.robot.q[iy] = float(spec["NeckPitch"]), float(spec["NeckYaw"])
        return 0, (f"RESULT targets joint={ip} target_deg={spec['NeckPitch']} ss_err=0.000\n"
                   f"RESULT targets joint={iy} target_deg={spec['NeckYaw']} ss_err=0.000")

    def text(self) -> str:
        return "\n".join(shlex.join(a) for a in self.commands)


# ---------------------------------------------------------------- the IMU sweep's rig


class NeckSweepRig:
    """imusweep.SweepRig over a Neck: foreign_seq = /dev/shm/desired's head, move_neck = Neck.goto (all its checks),
    neck_deg = the last measured pose; imu_accel(seconds) and frames(k) are passed in (DmaImu.accel, camera 0)"""

    def __init__(self, neck: Neck, imu_accel, frames):
        self.neck, self._imu, self._frames = neck, imu_accel, frames

    def foreign_seq(self) -> int:
        return self.neck.robot.desired_head()

    def move_neck(self, pitch_deg, yaw_deg):
        self.neck.goto(pitch_deg, yaw_deg)

    def neck_deg(self) -> tuple:
        return self.neck.last.measured if self.neck.last is not None else self.neck.neck_deg()

    def imu_accel(self, seconds):
        return np.asarray(self._imu(seconds), float).reshape(-1, 3)

    def frames(self, k):
        return list(self._frames(k))


class DmaImu:
    """the head IMU slot's acceleration from /raw_imu_actuals (imucam_fast.py's read); d = the dma_ipc module"""

    def __init__(self, d, slot, sleep=time.sleep, name="/raw_imu_actuals"):
        self.d, self.slot, self.sleep, self.name = d, int(slot), sleep, name
        self._q = None

    def accel(self, seconds) -> np.ndarray:
        if self._q is None:
            self._q = self.d.RawImuQueue(self.name, create=False, wait=True)
        c = self._q.make_consumer()
        self.sleep(seconds)
        out = []
        while c.available():
            s = c.pop().imu(self.slot)
            out.append((s.accel_x, s.accel_y, s.accel_z))
        return np.asarray(out, float).reshape(-1, 3)
