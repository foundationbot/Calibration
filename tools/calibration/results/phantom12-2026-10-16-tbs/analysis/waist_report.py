"""Read-only: did the torso turn on the waist? Pelvis vs torso IMU yaw, and every joint that moved."""
import json, sys
from pathlib import Path
import numpy as np
import dmarec_imu
from mcap.reader import make_reader
take = sys.argv[1]
W = Path("/w/takes")
segs = sorted((W / f"{take}-dmarec").glob("recording_*.mcap"))
with open(W / take / "session.mcap", "rb") as f:
    r = make_reader(f)
    names = None
    for rec in r.iter_metadata():
        if rec.name == "robot":
            names = rec.metadata.get("joint_names", "").split(",")
    T, P = [], []
    for sch, ch, msg in r.iter_messages(topics=["/joints"]):
        d = json.loads(msg.data)
        T.append(d["timestamp_ns"]); P.append(d["positions"])
T, P = np.array(T, np.int64), np.array(P, float)
t0, t1 = int(T[0]), int(T[-1])
for idx, nm in ((9, "Torso"), (10, "Pelvis"), (2, "NeckPitch")):
    t, g, a, err = dmarec_imu.read_raw_imu(segs, idx, t0, t1)
    dt = np.minimum(np.diff(t, prepend=t[0]) / 1e9, 0.006)
    up = np.median(a, 0); up /= np.linalg.norm(up)
    yaw = np.degrees(np.cumsum(((g - np.median(g, 0)) @ up) * dt))
    print(f"IMU {idx} ({nm}): yaw about its own vertical {yaw.min():+.2f}..{yaw.max():+.2f} deg, end {yaw[-1]:+.2f}")
rng = np.degrees(P.max(0) - P.min(0))
print("joints that moved > 0.5 deg over the take (name: min..max deg, range):")
for k in np.argsort(-rng):
    if rng[k] > 0.5:
        nm = names[k] if names and k < len(names) else f"slot{k}"
        print(f"  {nm:20s} {np.degrees(P[:, k].min()):+8.2f}..{np.degrees(P[:, k].max()):+8.2f}  ({rng[k]:.2f})")
