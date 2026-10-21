#!/usr/bin/env python3
"""Quick check of a camera-IMU take on phantom12, run between takes so a bad take is retaken
while the robot and the board are still set up. It judges the TAKE (frames, IMU stream, board
placement, motion), not the fit. Thresholds marked "heuristic" are judgement calls, not package
gates; "useful" means what filter_edges.py 450 keeps: board detected, every corner within 450 px.

usage (on Thor):
  docker run --rm --cpus=2 --cpu-shares=2 --memory=4g -v /root/calib-work:/w \
    -v /data/okvis-recordings/<take>-euroc:/e:ro calib-py:1 python /w/qc_imucam.py /e [stride]
"""
import csv
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, "/w/scripts")
import rigcfg  # noqa: E402

E = sys.argv[1]
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LIMIT_PX, F_PX, CX, CY = 450.0, 563.0, 640.0, 400.0   # filter_edges limit; calib_stereo3noedge f
LEFT, RIGHT = "cam1", "cam2"                           # DMA queue 1 = left, 2 = right, as exported
board = rigcfg.load("/w/config/rig-phantom12.yaml").board
results = []


def verdict(name, ok, detail):
    tag = {True: "PASS", False: "FAIL", None: "WARN"}[ok]
    results.append(tag)
    print(f"  [{tag}] {name}: {detail}")


def grade(x, good, warn):
    return True if x >= good else (None if x >= warn else False)


def stamps(cam):
    with open(f"{E}/{cam}/data.csv") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    return np.array([int(r[0]) for r in rows], np.int64), [r[1] for r in rows]


# ---------------------------------------------------------------- frames
print("== frames")
tl, fl = stamps(LEFT)
tr, _ = stamps(RIGHT)
dur = (tl[-1] - tl[0]) / 1e9
fps = (len(tl) - 1) / dur
idx = np.clip(np.searchsorted(tr, tl), 1, len(tr) - 1)
near_ms = np.minimum(np.abs(tr[idx] - tl), np.abs(tr[idx - 1] - tl)) / 1e6
gaps = np.diff(tl) / 1e9
verdict("frame rate", grade(fps, 11.0, 9.0),
        f"left {len(tl)} / right {len(tr)} frames over {dur:.1f} s = {fps:.1f} fps; "
        f"{int(np.sum(near_ms <= 10))} left frames have a right frame within 10 ms")
verdict("frame gaps", True if gaps.max() <= 0.5 else (None if gaps.max() <= 2 else False),
        f"longest {gaps.max() * 1e3:.0f} ms, {int(np.sum(gaps > 0.25))} gaps over 250 ms "
        "(the record app drops whole frames under CPU load; heuristic: <= 0.5 s)")

# ---------------------------------------------------------------- IMU
print("== IMU (NeckPitch, /imu0)")
ti = np.loadtxt(f"{E}/imu0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
d = np.loadtxt(f"{E}/imu0/data.csv", delimiter=",", comments="#", usecols=range(1, 7))
t = (ti - tl[0]) / 1e9                                 # seconds from the first left frame
w, a = d[:, 0:3], d[:, 3:6]
rate = (len(t) - 1) / (t[-1] - t[0])
info = open(f"{E}/run_info.txt").read() if os.path.exists(f"{E}/run_info.txt") else ""
m = re.search(r"imu_dropped:\s*(\d+)", info)
dropped = int(m.group(1)) if m else -1
gap_ms = float(np.max(np.diff(ti)) / 1e6)
late, early = (ti[0] - tl[0]) / 1e9, (tl[-1] - ti[-1]) / 1e9   # > 0: frames without IMU
ok = rate > 490 and dropped == 0 and gap_ms < 10
verdict("IMU stream", (True if max(late, early) <= 0.1 else None) if ok else False,
        f"{len(t)} samples, {rate:.1f} Hz, dropped {dropped}, longest gap {gap_ms:.1f} ms; "
        f"starts {late:+.2f} s after the first frame, ends {early:+.2f} s before the last")

mid = (t > 5) & (t < dur - 5)                          # leave out the still ends
still = lambda s: float(np.sqrt(np.mean(np.sum((w[s] - np.median(w[s], axis=0)) ** 2, axis=1))))
s0, s1 = still(t < 4), still(t > dur - 4)
verdict("still at start and end", True if max(s0, s1) < 0.03 else None,
        f"gyro motion first 4 s {s0:.3f} rad/s, last 4 s {s1:.3f} rad/s (heuristic: < 0.03)")
g_rms = np.sqrt(np.mean((w[mid] - np.median(w[mid], axis=0)) ** 2, axis=0))
for k, ax in enumerate("xyz"):
    verdict(f"rotation about IMU {ax}", grade(g_rms[k], 0.30, 0.15),
            f"rms {g_rms[k]:.2f} rad/s = {np.degrees(g_rms[k]):.0f} deg/s (heuristic: >= 0.30 good, >= 0.15 weak)")
an = np.linalg.norm(a, axis=1)
lin = float(np.std(an[mid]))                           # |a| changes only with linear acceleration
verdict("translation (std of |a|)", grade(lin, 0.5, 0.25),
        f"{lin:.2f} m/s^2 (heuristic: >= 0.5 good, >= 0.25 weak); per-axis accel std "
        + " / ".join(f"{s:.2f}" for s in np.std(a[mid], axis=0)))
verdict("no saturation", True if (an.max() < 70 and np.abs(w).max() < 30) else False,
        f"max |a| {an.max():.1f} m/s^2 (limit ~78), max |w| {np.abs(w).max():.1f} rad/s")

# ---------------------------------------------------------------- board
print(f"== board (1 in {STRIDE} left frames)")
n = det_n = 0
useful_t, tags_u, dist, bright = [], [], [], []
for i in range(0, len(fl), STRIDE):
    g = cv2.imread(f"{E}/{LEFT}/data/{fl[i]}", cv2.IMREAD_GRAYSCALE)
    if g is None:
        continue
    n += 1
    bright.append(float(g.mean()))
    det = board.detect(g)
    if len({tid for tid, _ in det}) < 2:
        continue
    det_n += 1
    pts = np.array(list(det.values()), float)
    pitch_px = [np.hypot(*np.subtract(det[(tid + 1, k)], det[(tid, k)])) for tid, k in det
                if (tid + 1, k) in det and tid % board.cols != board.cols - 1]
    if pitch_px:
        dist.append(F_PX * board.pitch / float(np.median(pitch_px)))
    if np.hypot(pts[:, 0] - CX, pts[:, 1] - CY).max() <= LIMIT_PX:
        useful_t.append((tl[i] - tl[0]) / 1e9)
        tags_u.append(len({tid for tid, _ in det}))

useful = len(useful_t) / max(n, 1)
verdict("board detected", grade(det_n / max(n, 1), 0.90, 0.75), f"{det_n}/{n} sampled frames")
verdict("useful frames (inside 450 px)", grade(useful, 0.60, 0.40),
        f"{len(useful_t)}/{n} = {useful * 100:.0f} %, median {np.median(tags_u) if tags_u else 0:.0f} "
        f"of {board.n_tags} tags in them")
ut = np.array([0.0] + useful_t + [dur])
ug = np.diff(ut)
verdict("gaps without useful frames", True if ug.max() <= 2 else (None if ug.max() <= 5 else False),
        f"longest {ug.max():.1f} s, {int(np.sum(ug > 1))} gaps over 1 s (heuristic: <= 2 s)")
if dist:
    dm = float(np.median(dist))
    verdict("board distance", True if 0.9 <= dm <= 1.3 else None,
            f"median ~{dm:.2f} m, 10-90 % {np.percentile(dist, 10):.2f}-{np.percentile(dist, 90):.2f} m "
            "(pinhole estimate; aim 1.0-1.2 m)")
bm = float(np.median(bright))
verdict("brightness", True if 60 <= bm <= 200 else None, f"median image mean {bm:.0f} of 255 (heuristic: 60-200)")

# ---------------------------------------------------------------- per 15 s
print("== per 15 s: useful frames | gyro rms rad/s (x y z) | std |a| m/s^2")
ua = np.array(useful_t)
per_bin = max(1, round(15 * fps / STRIDE))
for b0 in np.arange(0, dur, 15):
    s = (t >= b0) & (t < b0 + 15)
    if s.sum() < 100:
        continue
    gr = np.sqrt(np.mean((w[s] - np.median(w[mid], axis=0)) ** 2, axis=0))
    nu = int(np.sum((ua >= b0) & (ua < b0 + 15)))
    print(f"  {b0:5.0f}-{b0 + 15:3.0f} s  {nu:3d}/{per_bin:<3d} | {gr[0]:.2f} {gr[1]:.2f} {gr[2]:.2f} | {np.std(an[s]):.2f}")

overall = "RETAKE" if "FAIL" in results else ("USABLE, see WARN lines" if "WARN" in results else "GOOD")
print(f"QC VERDICT: {overall}")
