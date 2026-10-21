"""Read-only: how far the torso moved during a neck take, from the dma-recorder torso IMU.
Usage (in calib-py:2, cwd = scripts): python - <take> <torso index> < torso_report.py"""
import sys
from pathlib import Path

import numpy as np

import dmarec_imu
import necktake

take, idx = sys.argv[1], int(sys.argv[2])
W = Path("/w/takes")
j = necktake.read_joints(W / take / "session.mcap")
holds = necktake.find_holds(j)
segs = sorted((W / f"{take}-dmarec").glob("recording_*.mcap"))
t, g, a, err = dmarec_imu.read_raw_imu(segs, idx, int(j.t[0]), int(j.t[-1]))
print(f"take {take}: {len(holds)} holds, torso samples {len(t)}, errors {err}")
ts = (t - t[0]) / 1e9
# 1 s means of the accelerometer direction: tilt against the take's median direction
nb = int(ts[-1])
dirs, gyr = [], []
for k in range(nb):
    m = (ts >= k) & (ts < k + 1)
    v = a[m].mean(0)
    dirs.append(v / np.linalg.norm(v))
    gyr.append(np.abs(g[m] - np.median(g, 0)).max(0))
dirs = np.array(dirs)
up = np.median(dirs, 0)
up /= np.linalg.norm(up)
tilt = np.degrees(np.arccos(np.clip(dirs @ up, -1, 1)))
print(f"torso tilt vs the take's median, 1 s means: median {np.median(tilt):.2f} deg, 90th pct {np.percentile(tilt, 90):.2f}, max {tilt.max():.2f}")
dt = np.minimum(np.diff(t, prepend=t[0]) / 1e9, 0.006)
rate_up = (g - np.median(g, 0)) @ up
yaw = np.degrees(np.cumsum(rate_up * dt))
print(f"torso yaw (gyro about vertical, median bias removed): range {yaw.min():+.2f}..{yaw.max():+.2f} deg")
print(f"torso gyro |rate| peaks: 99th pct {np.degrees(np.percentile(np.linalg.norm(g - np.median(g, 0), axis=1), 99)):.2f} deg/s, max {np.degrees(np.linalg.norm(g - np.median(g, 0), axis=1).max()):.2f} deg/s")
m = dmarec_imu.torso_motion(t, g, a, holds)
print("per hold: neck yaw/pitch [deg] | torso tilt, yaw, turn [deg]")
for i, (h, x) in enumerate(zip(holds, m)):
    if x is None:
        print(f"  {i:2d} {np.degrees(h.yaw):+6.1f} {np.degrees(h.pitch):+6.1f} | (too short)")
    else:
        print(f"  {i:2d} {np.degrees(h.yaw):+6.1f} {np.degrees(h.pitch):+6.1f} | {x['tilt_deg']:.2f} {x['yaw_deg']:+.2f} {x['turn_deg']:.2f}")
# does the torso tilt follow the neck? correlate per-hold tilt direction with neck pitch / yaw
td = np.array([[x["tilt_deg"] for x in m if x is not None]])
print(f"joint gaps > 50 ms: {[(round((j.t[k] - j.t[0]) / 1e9, 1), round((j.t[k + 1] - j.t[k]) / 1e6)) for k in np.flatnonzero(np.diff(j.t) > 50e6)]}")
