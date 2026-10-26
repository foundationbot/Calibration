"""Search a right-arm park pose (shoulder + elbow only, wrist unchanged) whose path from the current pose clears the body
by >= 3 cm and with which the left route (left-moves.json waypoints, coarse 3-deg paths) clears by >= 3 cm.
usage: park_search.py repo snap.json mjcf left-moves.json out.json"""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(sys.argv[1]) / "tools/calibration/scripts"))
import armsafety

J = ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "ElbowPitch", "WristYaw", "WristRoll", "GripperYaw")
R, L = tuple("Right" + j for j in J), tuple("Left" + j for j in J)
snap = json.load(open(sys.argv[2])); xml = Path(sys.argv[3]); lm = json.load(open(sys.argv[4]))["moves"]
q0 = np.array([snap["q_deg"][n] for n in R])
rb = armsafety.Body(xml, root="RightShoulderPitchLink", joints=R, skip_surface=("RightShoulderPitchLink",), surface_step=0.008)
rb.set_baseline({n: np.radians(v) for n, v in snap["q_deg"].items()})
lb = armsafety.Body(xml, root="LeftShoulderPitchLink", joints=L, skip_surface=("LeftShoulderPitchLink",), surface_step=0.008)
lq = [np.array(m["deg"]) for m in lm]
LQ = np.radians(np.vstack([a + np.linspace(0, 1, max(2, int(np.abs(b - a).max() / 3) + 1))[:, None] * (b - a)
                           for a, b in zip(lq[:-1], lq[1:])]))
best = []
for sp, sr, sy, ep in itertools.product((-30, -20, -10), (-15, -5, 5), (0, 10, 20), (-60, -40, -20)):
    q1 = np.r_[sp, sr, sy, ep, q0[4:]]
    a, b = np.radians(q0), np.radians(q1)
    n = int(np.ceil(np.degrees(np.abs(b - a)).max())) + 1
    cr = rb.clearance(a + np.linspace(0, 1, n)[:, None] * (b - a)).min()
    if cr < 0.03:
        continue
    base = dict(snap["q_deg"]); base.update(dict(zip(R, q1)))
    lb.set_baseline({k: np.radians(v) for k, v in base.items()})
    cl = lb.clearance(LQ).min()
    travel = float(np.abs(q1 - q0).max())
    print(f"{[sp, sr, sy, ep]} right path {cr * 100:.1f} cm, left route {cl * 100:.1f} cm, travel {travel:.0f}", flush=True)
    if cl >= 0.03:
        best.append((travel, -min(cr, cl), q1.tolist(), cr, cl))
best.sort()
json.dump([{"deg": b[2], "travel": b[0], "right_path_m": b[3], "left_route_m": b[4]} for b in best], open(sys.argv[5], "w"), indent=1)
print("best:", best[:3])
