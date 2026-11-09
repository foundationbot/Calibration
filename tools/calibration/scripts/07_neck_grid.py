#!/usr/bin/env python3
"""Step 07 (ON THE ROBOT) — a neck grid for a KB4T lens take with the board on a stand: the robot moves its head
through a pitch x yaw grid and HOLDS still at each pose, while the robot's recorder (record app, step 05 or the
robot's own record path) captures the stereo pair. Step 15's default still-hold mode then takes the holds out of
session.mcap: a still head means a still board in the image, so every pose gives sharp, synced pairs without a
person waving the board. Repeat it with the board at each distance (COLLECT-lens-data.md).

Moves only the neck, through /usr/bin/dma_motor_motion 'targets position' (DMA.ethercat 2.1.0 builds). That tool
re-holds every joint it is not given at its MEASURED pose, so a robot standing on bent knees sags a little with
every command (phantom14, 3 Nov: spine -1.2 -> -25 deg and a knee +24 deg over the neck grids). --also holds named
joints at fixed angles in every command (e.g. the spine and legs at their start values) to stop that creep.

Safe by default: without --go it prints the plan and sends nothing. Pitch is clamped to --max-pitch (default 26:
phantom14's NeckPitch limit is +-27.5 deg), yaw to --max-yaw (default 34: the head met a mechanical stop at 35.9
deg on phantom14). +pitch = head down. Serpentine order keeps each move short. Stops at the first pose a joint
misses by more than --max-err deg (a stop, a dropped motor), leaving the head there.

Usage (on the robot, motors enabled by the operator, recorder already running):
    python3 07_neck_grid.py --pitch=-15,-5,5,15 --yaw=-27,-13,0,13,27                 # plan only
    python3 07_neck_grid.py --pitch=-15,-5,5,15 --yaw=-27,-13,0,13,27 --go [--hold 3]
    python3 07_neck_grid.py ... --also "SpineYaw=-25.4,LeftKneePitch=11.0" --go       # hold the body too
Measured on phantom14 (3 Nov, its predecessor script): 2.9 s for one pose with --ramp 1 --hold 2."""
import argparse
import re
import subprocess
import sys
import time

DMM = "/usr/bin/dma_motor_motion"
RESULT = re.compile(r"RESULT targets joint=(\d+) target_deg=(\S+) ss_err=(\S+)")


def floats(s):
    return [float(v) for v in str(s).split(",") if v.strip()]


def grid(pitches, yaws, max_pitch=26.0, max_yaw=34.0):
    """serpentine (pitch, yaw) poses, each clamped to the limits: yaw sweeps alternate direction per pitch row"""
    out = []
    for i, p in enumerate(pitches):
        row = yaws if i % 2 == 0 else list(reversed(yaws))
        for y in row:
            out.append((max(-max_pitch, min(max_pitch, p)), max(-max_yaw, min(max_yaw, y))))
    return out


def command(pitch, yaw, also, ramp, hold):
    targets = f"NeckPitch={pitch:.2f},NeckYaw={yaw:.2f}" + (f",{also}" if also else "")
    return [DMM, targets, "targets", "position", "--ramp", str(ramp), "--hold", str(hold), "--no-return"]


def errors(out):
    """{joint index: steady-state error deg} from dma_motor_motion's RESULT lines"""
    return {int(j): float(e) for j, _, e in RESULT.findall(out)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pitch", default="-15,-5,5,15", help="deg, + = head down")
    ap.add_argument("--yaw", default="-27,-13,0,13,27", help="deg")
    ap.add_argument("--hold", type=float, default=3.0, help="s still at each pose (step 15 needs >= 3 still frames)")
    ap.add_argument("--ramp", type=float, default=1.0, help="s per move")
    ap.add_argument("--max-err", type=float, default=3.0, help="deg: stop if a joint misses its target by more")
    ap.add_argument("--max-pitch", type=float, default=26.0)
    ap.add_argument("--max-yaw", type=float, default=34.0)
    ap.add_argument("--also", default="", help="extra 'Joint=deg,...' held in every command (e.g. spine, legs)")
    ap.add_argument("--go", action="store_true", help="really move (default: print the plan)")
    a = ap.parse_args(argv)
    poses = grid(floats(a.pitch), floats(a.yaw), a.max_pitch, a.max_yaw)
    n_joints = 2 + len([j for j in a.also.split(",") if j.strip()])
    print(f"{len(poses)} poses, at least {len(poses) * (a.ramp + a.hold):.0f} s of ramps and holds; "
          f"{'MOVING' if a.go else 'plan only (add --go to move)'}")
    for i, (p, y) in enumerate(poses):
        cmd = command(p, y, a.also, a.ramp, a.hold)
        if not a.go:
            print(f"  {i + 1:3d}: pitch {p:+6.1f} yaw {y:+6.1f}   " + " ".join(cmd[1:2]))
            continue
        t0 = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=a.ramp + a.hold + 30).stdout
        errs = errors(out)
        print(f"{time.strftime('%H:%M:%S')} pose {i + 1}/{len(poses)} pitch {p:+.1f} yaw {y:+.1f} "
              + " ".join(f"j{j} {e:+.2f}" for j, e in sorted(errs.items())) + f" ({time.time() - t0:.1f} s)", flush=True)
        if len(errs) < n_joints or any(abs(e) > a.max_err for e in errs.values()):
            sys.exit(f"ABORT at pose {p:+.1f}/{y:+.1f}: a joint missed its target by more than {a.max_err} deg or "
                     f"did not report (head left where it is)\n{out[-1500:]}")
    if a.go:
        print("grid done; the head stays at the last pose")


if __name__ == "__main__":
    main()
