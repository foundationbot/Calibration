#!/usr/bin/env python3
"""Hand-eye arm wave, run ON the robot: one arm through a planned list of poses, several neck angles per pose, a
depth burst at each (docs/plans/2026-11-11-handeye-plan.md section 5; first live test H3).

usage (on the robot, the calib kit on PYTHONPATH with dma_ipc):
  handeye_wave.py <plan.json> --arm Right --out <dir> [--frames 5] [--kp K] [--dry-run] [--only-burst]

plan.json: {"moves": [{"name", "deg": [7 arm joints], "burst": bool, "necks": [[pitch, yaw], ...]}], ...};
the arm joints in armchain order: ShoulderPitch ShoulderRoll ShoulderYaw ElbowPitch WristYaw WristRoll GripperYaw.

Safety, each enforced here (tests/test_handeye_wave.py pins them):
  - the motor tool must answer --help with 'targets' (the fixed build; 2.1.1's own leaves unlisted joints at zero
    stiffness); refused otherwise, before anything moves
  - every command lists EVERY joint of /config: the moving arm and the neck at their targets, every other joint pinned
    at its angle when the session starts (calibtool/neck.py's rule; a re-hold at measured angles made phantom14 creep)
  - before every command: a fresh master ping says motors on, no e-stop; /dev/shm/desired still for >= 0.5 s
    (no other program commands); a STOP file in --out stops before the next command (the robot holds its pose)
  - ramps scale with travel at <= MAX_RATE_DEG_S (5 deg/s, half the 23 Oct EtherCAT-drop move's rate), never below
    RAMP_MIN_S (6 s) for arm moves; neck moves ramp at <= NECK_RATE_DEG_S
  - neck pitch clamped to +-26, yaw to +-34 (neck.py's clamps)
  - kp only: --kp when given (default: the tool's own gains, the user's "don't overdo kp"); no other gain, ever
  - a joint missing its target by more than MAX_MISS_DEG (the tool's --tolerance) stops the session
  - safe state = hold: --no-return; stopping = sending nothing more
Every step is logged with its duration (log.jsonl) so the time per pose is measured, not estimated.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ARM_J = ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "ElbowPitch", "WristYaw", "WristRoll", "GripperYaw")
MOTOR_TOOL = "/root/p11-cal/dma_motor_motion"
GRAB = "/root/burst-grab/repo/tools/burst_pointcloud/grab.py"
GRAB_PYDEPS = "/root/burst-grab/pydeps"
MAX_RATE_DEG_S = 5.0
RAMP_MIN_S = 6.0
NECK_RATE_DEG_S = 15.0
NECK_RAMP_MIN_S = 1.0
ARM_HOLD_S = 2.0
NECK_HOLD_S = 0.8
MAX_MISS_DEG = 30.0
MAX_PITCH, MAX_YAW = 26.0, 34.0
DESIRED_STILL_S = 0.5


def clamp_neck(pitch: float, yaw: float) -> tuple:
    return max(-MAX_PITCH, min(MAX_PITCH, float(pitch))), max(-MAX_YAW, min(MAX_YAW, float(yaw)))


def arm_ramp(a, b) -> float:
    travel = max(abs(float(x) - float(y)) for x, y in zip(a, b))
    return round(max(RAMP_MIN_S, travel / MAX_RATE_DEG_S), 1)


def neck_ramp(a, b) -> float:
    travel = max(abs(float(x) - float(y)) for x, y in zip(a, b))
    return round(max(NECK_RAMP_MIN_S, travel / NECK_RATE_DEG_S), 1)


def spec(names, pins: dict, arm: str, arm_deg, neck) -> str:
    """every /config joint: the arm and the neck at their targets, the rest at its pinned angle"""
    tgt = {arm + j: float(v) for j, v in zip(ARM_J, arm_deg)}
    p, y = clamp_neck(*neck)
    tgt["NeckPitch"], tgt["NeckYaw"] = p, y
    missing = [n for n in names if n not in tgt and n not in pins]
    if missing:
        raise ValueError(f"no pin for {missing}")
    return ",".join(f"{n}={tgt[n] if n in tgt else pins[n]:.2f}" for n in names)


def motion_argv(tool, text, ramp, hold, kp=None) -> list:
    argv = [str(tool), text, "targets", "position", "--ramp", f"{ramp:g}", "--hold", f"{hold:g}", "--no-return",
            "--tolerance", f"{MAX_MISS_DEG:g}"]
    if kp is not None:
        if not float(kp) > 0:
            raise ValueError("kp must be > 0")
        argv += ["--kp", f"{float(kp):g}"]
    return argv


class Robot:
    """the real robot: dma_ipc reads, the calib kit's master ping, subprocess commands"""

    def __init__(self):
        import dma_ipc
        from calibtool import motors
        self.ipc, self.motors = dma_ipc, motors

    def names_and_angles(self):
        cfg = self.ipc.ConfigQueue("/config", wait=False).read_latest()
        names = [cfg.joint_names(i) for i in range(cfg.num_joints)]
        s = self.ipc.SensorQueue("/actuals", wait=False).read_latest()
        return names, {n: math.degrees(float(s.joint_positions[i])) for i, n in enumerate(names)}

    def master_ok(self) -> str | None:
        m = self.motors.master()
        if not m.motors_on:
            return f"motors not on: {m.as_dict()}"
        if m.estop:
            return "e-stop"
        return None

    def desired_still(self) -> bool:
        q = self.ipc.CommandDataQueue("/desired", wait=False)
        a = q.read_latest()
        time.sleep(DESIRED_STILL_S)
        b = q.read_latest()
        return (a.timestamp_ns if a else 0) == (b.timestamp_ns if b else 0)

    def run(self, argv, timeout):
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr


class DryRun(Robot):
    def __init__(self, names, angles):
        self._n, self._a, self.sent = names, angles, []

    def names_and_angles(self):
        return list(self._n), dict(self._a)

    def master_ok(self):
        return None

    def desired_still(self):
        return True

    def run(self, argv, timeout):
        self.sent.append(argv)
        return 0, "RESULT dry-run"


class Stop(Exception):
    pass


class Wave:
    def __init__(self, robot, plan, arm, out: Path, frames=5, kp=None, tool=MOTOR_TOOL, log=print):
        self.r, self.plan, self.arm, self.out, self.frames, self.kp, self.tool = robot, plan, arm, out, frames, kp, tool
        self.say = log
        self.names, start = robot.names_and_angles()
        mine = {arm + j for j in ARM_J} | {"NeckPitch", "NeckYaw"}
        self.pins = {n: round(v, 2) for n, v in start.items() if n not in mine}
        self.arm_now = [start[arm + j] for j in ARM_J]
        self.neck_now = (start["NeckPitch"], start["NeckYaw"])
        self.start = start
        self.events = []

    def _event(self, kind, **kw):
        e = dict(t=time.time(), kind=kind, **kw)
        self.events.append(e)
        with open(self.out / "log.jsonl", "a") as f:
            f.write(json.dumps(e) + "\n")
        self.say(f"{time.strftime('%H:%M:%S')} {kind} " + " ".join(f"{k}={v}" for k, v in kw.items() if k != "out"))

    def _gate(self):
        if (self.out / "STOP").exists():
            raise Stop("STOP file")
        err = self.r.master_ok()
        if err:
            raise Stop(err)
        if not self.r.desired_still():
            raise Stop("another program is writing /desired")

    def _command(self, what, arm_deg, neck, ramp, hold):
        self._gate()
        argv = motion_argv(self.tool, spec(self.names, self.pins, self.arm, arm_deg, neck), ramp, hold, self.kp)
        t0 = time.time()
        rc, out = self.r.run(argv, timeout=ramp + hold + 30)
        res = [ln for ln in out.splitlines() if "RESULT" in ln or "FAIL" in ln or "error" in ln.lower()]
        self._event(what, ramp=ramp, rc=rc, dt=round(time.time() - t0, 2), result=res[-12:])
        if rc != 0:
            raise Stop(f"{what}: motor tool rc {rc}: {res[-3:]}")
        self.arm_now, self.neck_now = list(arm_deg), clamp_neck(*neck)

    def move_pins(self, overrides: dict):
        """move non-arm joints (e.g. a slight leg stretch, the user's OK of 11 Nov) to new pinned angles, alone, at
        <= MAX_RATE_DEG_S with the RAMP_MIN_S floor; the new angles are then held by every later command"""
        unknown = [n for n in overrides if n not in self.pins]
        if unknown:
            raise ValueError(f"not pinnable here (moving arm/neck or unknown): {unknown}")
        new = dict(self.pins, **{n: round(float(v), 2) for n, v in overrides.items()})
        travel = max(abs(new[n] - self.pins[n]) for n in overrides)
        ramp = round(max(RAMP_MIN_S, travel / MAX_RATE_DEG_S), 1)
        self._gate()
        argv = motion_argv(self.tool, spec(self.names, new, self.arm, self.arm_now, self.neck_now), ramp, ARM_HOLD_S,
                           self.kp)
        t0 = time.time()
        rc, out = self.r.run(argv, timeout=ramp + ARM_HOLD_S + 30)
        res = [ln for ln in out.splitlines() if "RESULT" in ln or "FAIL" in ln or "error" in ln.lower()]
        self._event("pins", ramp=ramp, rc=rc, dt=round(time.time() - t0, 2), changed=overrides, result=res[-6:])
        if rc != 0:
            raise Stop(f"pins: motor tool rc {rc}")
        self.pins = new

    def move_arm(self, name, deg):
        self._command(f"arm:{name}", deg, self.neck_now, arm_ramp(self.arm_now, deg), ARM_HOLD_S)

    def move_neck(self, pitch, yaw):
        tgt = clamp_neck(pitch, yaw)
        self._command(f"neck:{tgt[0]:g}/{tgt[1]:g}", self.arm_now, tgt, neck_ramp(self.neck_now, tgt), NECK_HOLD_S)

    def burst(self, name):
        self._gate()
        path = self.out / f"{name}.npz"
        argv = ["nice", sys.executable, GRAB, "--once", str(self.frames), "--joints", "all", "--out", str(path)]
        t0 = time.time()
        env = dict(os.environ, PYTHONPATH=GRAB_PYDEPS)
        if isinstance(self.r, DryRun):
            rc, out = self.r.run(argv, 60)
        else:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env)
            rc, out = p.returncode, p.stdout + p.stderr
        self._event("burst", name=name, rc=rc, dt=round(time.time() - t0, 2), tail=out.strip().splitlines()[-2:])
        if rc != 0:
            raise Stop(f"burst {name} failed")

    def hand(self, pose):
        hand = "1" if self.arm == "Right" else "0"
        argv = ["inspire_hand_poses", "--hand", hand, "--pose", pose, "--no-loop", "--hold", "1"]
        t0 = time.time()
        try:
            rc, out = self.r.run(argv, timeout=20)
        except subprocess.TimeoutExpired:
            rc, out = -1, "timeout"
        self._event("hand", pose=pose, rc=rc, dt=round(time.time() - t0, 2), tail=out.strip().splitlines()[-2:])

    def run(self, only_burst=False, hand_pose="open", pins=None):
        self._event("start", arm=self.arm, frames=self.frames, kp=self.kp, start=self.start)
        if pins:
            self.move_pins(pins)
        if only_burst:
            self.burst("00-rest")
            return
        if hand_pose:
            self.hand(hand_pose)
        k = 0
        for m in self.plan["moves"]:
            self.move_arm(m["name"], m["deg"])
            if m.get("burst"):
                for pitch, yaw in m["necks"]:
                    self.move_neck(pitch, yaw)
                    pitch, yaw = self.neck_now
                    k += 1
                    self.burst(f"{k:02d}-{m['name']}-{int(round(pitch))}-{int(round(yaw))}".replace("--", "-m"))
        if hand_pose:
            self.hand("relax")
        self._event("done", captures=k)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan")
    ap.add_argument("--arm", choices=("Right", "Left"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--kp", type=float, default=None)
    ap.add_argument("--tool", default=MOTOR_TOOL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-burst", action="store_true", help="one burst at the current pose, nothing moves")
    ap.add_argument("--no-hand", action="store_true")
    ap.add_argument("--pins", default="", help="non-arm joints moved first and held there, e.g. "
                    "'RightAnklePitch=-54,RightKneePitch=8' (deg; slow, alone)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.load(open(a.plan))
    if a.dry_run:
        real = Robot()
        robot = DryRun(*real.names_and_angles())
    else:
        robot = Robot()
        h = subprocess.run([a.tool, "--help"], capture_output=True, text=True)
        if "targets" not in h.stdout + h.stderr:
            raise SystemExit(f"{a.tool} has no 'targets' mode (2.1.1 zeroes unlisted joints): refusing")
    w = Wave(robot, plan, a.arm, out, a.frames, a.kp, a.tool)
    try:
        pins = {kv.split("=")[0].strip(): float(kv.split("=")[1]) for kv in a.pins.split(",") if kv.strip()}
        w.run(only_burst=a.only_burst, hand_pose=None if a.no_hand else "open", pins=pins or None)
    except Stop as e:
        w._event("stopped", reason=str(e))
        raise SystemExit(2)
    finally:
        if a.dry_run:
            json.dump(robot.sent, open(out / "dry_run_commands.json", "w"), indent=1)


if __name__ == "__main__":
    main()
