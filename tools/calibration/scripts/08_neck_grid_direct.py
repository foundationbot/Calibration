#!/usr/bin/env python3
"""Step 08 (ON THE ROBOT) — neck-grid lens take with direct camera grabs: no recorder needed.

The robot hangs on the gantry with no locomotion policy commanding. The board stands still at one distance. The
head steps through a pitch x yaw grid, and at each still pose this grabs --per-pose hardware-synced left/right pairs
straight from the DMA camera queues (Y planes). It saves them to <out>/<take>/pose_NNN.npz with the commanded and
measured neck angles. Step 15b (15b_dump_neck_take.py) turns a take folder into corners; run_lens_kb4t.py accepts
take folders directly.

This is how phantom15 and phantom11 were calibrated (4-5 Nov 2026; phantom11: 800 pairs, held-out depth -0.03 %,
rows 0.21 px).

SAFETY, each enforced here:
  - The motor tool must have the 'targets' signal, i.e. DMA.ethercat with a66805d. dma-ethercat 2.1.1's
    dma_motor_motion lacks it, and its commands cover only joints 0..highest listed: a neck move (14/15) leaves the
    spine and legs at position 0 / zero stiffness. --tool points at a fixed build. phantom14's 2.1.0 build worked on
    phantom11/15: their /dev/shm layouts are byte-identical. Refused if `--help` has no 'targets'.
  - No other writer: refused if /dev/shm/desired advances during 1 s (a policy is streaming).
  - Every command pins the arms, spine and legs (names read from /dev/shm/config) at their start angles. The tool
    re-holds unlisted joints at their measured pose, which made phantom14's body creep over its grids.
  - Pitch is clamped to +-26 (NeckPitch limit +-27.5; +26 commanded reached +27.7 on phantom15), yaw to +-34 (the
    head met a mechanical stop at 35.9 on phantom14).
  - Stops at the first pose a neck joint misses by more than --max-err deg, or that yields fewer than --per-pose
    pairs (camera stall). The head stays where it is.
  - `check` moves nothing.

Usage (on the robot, motors enabled by the operator):
    python3 08_neck_grid_direct.py check --left 3 --right 0 --tool /root/kit/dma_motor_motion --pydeps /root/kit/pydeps
    python3 08_neck_grid_direct.py grid --pitch=-24,-16,-8,0,8,16,24 --yaw=-34,-23,-11,0,11,23,34 --take d050 \\
        --left 3 --right 0 --tool ... --pydeps ... --out /root/kit/takes
    python3 08_neck_grid_direct.py poses -- "-5:2" --take test ...     (`--`: a leading minus is not an option)
Measured: about 3.1 s per pose (ramp 1 s, hold 2 s, 5 pairs); a 49-pose grid took about 2.5 min on phantom11.
"""
import argparse
import math
import mmap
import os
import re
import struct
import subprocess
import sys
import time

NECK = ("NeckPitch", "NeckYaw")
PIN_PREFIXES = ("LeftShoulder", "LeftElbow", "LeftWrist", "LeftGripper", "RightShoulder", "RightElbow", "RightWrist",
                "RightGripper", "SpineYaw", "LeftHip", "RightHip", "LeftKnee", "RightKnee")
RESULT = re.compile(r"RESULT targets joint=(\d+) target_deg=\S+ ss_err=(\S+)")


def ring(name):
    f = open("/dev/shm/" + name, "rb")
    return mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)


def latest(mm):
    esz, cap = struct.unpack_from("<QQ", mm, 16)
    head = struct.unpack_from("<Q", mm, 64)[0]
    off = 128 + ((head - 1) % cap) * esz
    return mm[off:off + esz]


def joint_names(entry, n=29):
    """the first n joint names in a /dev/shm/config entry (CamelCase identifiers, in joint order)"""
    return [m.decode() for m in re.findall(rb"[A-Z][A-Za-z]{3,}", bytes(entry))][:n]


def actuals_deg(n=29):
    return [math.degrees(v) for v in struct.unpack_from(f"<{n}f", latest(ring("actuals")), 88)]


def pins(names, q_deg):
    """'Joint=deg,...' holding every arm/spine/leg joint at q_deg (the neck and ankles are not pinned)"""
    return ",".join(f"{n}={q_deg[i]:.2f}" for i, n in enumerate(names) if n.startswith(PIN_PREFIXES))


def grid(pitches, yaws, max_pitch=26.0, max_yaw=34.0):
    """serpentine (pitch, yaw) poses clamped to the limits"""
    out = []
    for k, p in enumerate(pitches):
        for y in (yaws if k % 2 == 0 else list(reversed(yaws))):
            out.append((max(-max_pitch, min(max_pitch, p)), max(-max_yaw, min(max_yaw, y))))
    return out


def neck_errors(out, idx):
    """{joint index: steady-state error} for the neck joints from the tool's RESULT lines"""
    return {int(j): float(e) for j, e in RESULT.findall(out) if int(j) in idx}


class Cams:
    def __init__(self, d, left, right):
        self.d, self.left, self.right = d, left, right
        self.q = {c: d.VideoMetaQueue(d.video_meta_queue_name(c), create=False, wait=True).make_consumer(max_lag=8)
                  for c in (left, right)}
        self.p = {c: d.VideoFramePool(d.video_frame_pool_name(c), create=False) for c in (left, right)}

    def pairs(self, n, timeout=6.0):
        """n hardware-synced pairs (|dt| < 2 ms) -> [(t_left, t_right, Y_left, Y_right)]"""
        import numpy as np
        for q in self.q.values():
            while q.next(0.0) is not None:
                pass
        buf = {self.left: {}, self.right: {}}
        out, t_end = [], time.time() + timeout
        while len(out) < n and time.time() < t_end:
            for c, q in self.q.items():
                m = q.next(0.02)
                if m is None:
                    continue
                img = self.p[c].read(m)
                if img is None:      # after a board crash + reopen the pool's sequence desyncs: restart the producer
                    continue
                w, h = int(m.width), int(m.height)
                buf[c][m.capture_timestamp_ns] = np.frombuffer(bytes(img), np.uint8)[: w * h].reshape(h, w).copy()
            for tl in sorted(buf[self.left]):
                tr = min(buf[self.right], key=lambda t: abs(t - tl), default=None)
                if tr is not None and abs(tr - tl) < 2_000_000:
                    out.append((tl, tr, buf[self.left].pop(tl), buf[self.right].pop(tr)))
            for c in buf:
                for t in sorted(buf[c])[:-6]:
                    buf[c].pop(t)
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["check", "poses", "grid"])
    ap.add_argument("poses", nargs="?", help="poses mode: 'pitch:yaw,pitch:yaw' (deg)")
    ap.add_argument("--left", type=int, required=True, help="DMA camera queue of the LEFT head camera")
    ap.add_argument("--right", type=int, required=True, help="DMA camera queue of the RIGHT head camera")
    ap.add_argument("--tool", default="/usr/bin/dma_motor_motion", help="a dma_motor_motion WITH the 'targets' signal")
    ap.add_argument("--pydeps", default="", help="folder holding dma_ipc if it is not installed")
    ap.add_argument("--out", default="takes")
    ap.add_argument("--take", default="test")
    ap.add_argument("--pitch", default="-24,-16,-8,0,8,16,24", help="deg, + = head down")
    ap.add_argument("--yaw", default="-34,-23,-11,0,11,23,34")
    ap.add_argument("--ramp", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--settle", type=float, default=0.8)
    ap.add_argument("--per-pose", type=int, default=5)
    ap.add_argument("--max-err", type=float, default=3.0)
    a = ap.parse_args(argv)
    if a.pydeps:
        sys.path.insert(0, a.pydeps)
    import numpy as np
    import dma_ipc as d
    names = joint_names(latest(ring("config")))
    if not all(n in names for n in NECK):
        sys.exit(f"ERROR: NeckPitch/NeckYaw not in /dev/shm/config's joint names: {names}")
    neck_idx = (names.index(NECK[0]), names.index(NECK[1]))
    cams = Cams(d, a.left, a.right)
    if a.mode == "check":
        prs = cams.pairs(3)
        q = actuals_deg()
        print(f"neck pitch/yaw {q[neck_idx[0]]:+.2f}/{q[neck_idx[1]]:+.2f} deg | pairs {len(prs)} | dt ms "
              f"{[round((p[1] - p[0]) / 1e6, 3) for p in prs]} | frame {prs[0][2].shape[::-1] if prs else None} | "
              f"brightness {[round(float(p[2].mean()), 1) for p in prs]}")
        return 0 if len(prs) == 3 else 1
    helptext = subprocess.run([a.tool, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    if "targets" not in helptext:
        sys.exit(f"ERROR: {a.tool} has no 'targets' signal: it predates a66805d and would leave the spine and legs at "
                 "zero stiffness during a neck move. Point --tool at a fixed build.")
    mm = ring("desired")
    h0 = struct.unpack_from("<Q", mm, 64)[0]
    time.sleep(1.0)
    if struct.unpack_from("<Q", mm, 64)[0] != h0:
        sys.exit("ERROR: /dev/shm/desired is advancing: something (a locomotion policy?) is commanding the motors. "
                 "Stop it first; the robot must hang on the gantry.")
    hold = pins(names, actuals_deg())
    print("pinned:", hold, flush=True)
    if a.mode == "poses":
        poses = [grid([float(s.split(":")[0])], [float(s.split(":")[1])])[0] for s in a.poses.split(",")]
    else:
        poses = grid([float(v) for v in a.pitch.split(",")], [float(v) for v in a.yaw.split(",")])
    outdir = os.path.join(a.out, a.take)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "neckgrid.log"), "a") as log:
        log.write(f"=== {time.strftime('%F %T')} {a.mode} {len(poses)} poses ramp {a.ramp} hold {a.hold} "
                  f"left {a.left} right {a.right}\n")
        for i, (p, y) in enumerate(poses):
            t0 = time.time()
            cmd = [a.tool, f"NeckPitch={p},NeckYaw={y}," + hold, "targets", "position",
                   "--ramp", str(a.ramp), "--hold", str(a.hold), "--no-return"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            time.sleep(a.ramp + a.settle)
            q = actuals_deg()
            meas = (q[neck_idx[0]], q[neck_idx[1]])
            prs = cams.pairs(a.per_pose)
            out = proc.communicate(timeout=a.ramp + a.hold + 30)[0]
            errs = neck_errors(out, neck_idx)
            if prs:
                np.savez(os.path.join(outdir, f"pose_{i:03d}.npz"), cmd=[p, y], neck=meas, tl=[r[0] for r in prs],
                         tr=[r[1] for r in prs], L=np.stack([r[2] for r in prs]), R=np.stack([r[3] for r in prs]),
                         queues=[a.left, a.right])
            line = (f"{time.strftime('%H:%M:%S')} pose {i + 1}/{len(poses)} cmd {p:+.0f}/{y:+.0f} meas "
                    f"{meas[0]:+.1f}/{meas[1]:+.1f} pairs {len(prs)} "
                    + " ".join(f"j{j} ss_err {e:+.2f}" for j, e in sorted(errs.items())) + f" ({time.time() - t0:.1f} s)")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            if len(errs) != 2 or any(abs(e) > a.max_err for e in errs.values()):
                print(f"ABORT: pose {p}/{y} not reached within {a.max_err} deg\n{out[-1500:]}", flush=True)
                return 1
            if len(prs) < a.per_pose:
                print(f"ABORT: only {len(prs)} camera pairs at pose {p}/{y} (camera stalled? a full no-overlap "
                      "producer restart fixes a desynced pool)", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
