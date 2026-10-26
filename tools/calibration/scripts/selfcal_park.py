#!/usr/bin/env python3
"""Park one arm out of the other arm's route (plan 2026-10-25-tagfree-selfcal.md, the left-arm run): one move to a
hanging pose with the wrist joints left where they are (no wrist twist: the right arm's cable faults came on large
wrist-roll / shoulder-yaw twists), its path cleared in MuJoCo against the body at the robot's pose (armsafety, as
left_clear.py / right_clear.py). Writes the move as a selfcal route (moves.json, one move, no burst) and the snapshot
with the arm at the parked pose, for clearing the other arm's route next.

usage: selfcal_park.py repo_dir snap.json mjcf.xml arm out_moves.json out_snap.json [sp,sr,sy,ep]
  the optional pose (deg) replaces the hang; 25 Oct: a hang hit the right thigh at SpineYaw -30 deg, so the park pose
  came from a MuJoCo search (park_search.py in the session folder)
"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
sys.path.insert(0, str(D / "tools/calibration/scripts"))
import armsafety  # noqa: E402

J = ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "ElbowPitch", "WristYaw", "WristRoll", "GripperYaw")
HANG = {"Right": (-11.2, 0.2, -7.5, -6.5), "Left": (-11.2, -0.2, 7.5, -6.5)}   # the left arm's measured rest, mirrored

snap_p, xml, arm, out_moves, out_snap = sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
snap = json.load(open(snap_p))
names = tuple(arm + j for j in J)
q0 = [snap["q_deg"][n] for n in names]
pose = [float(x) for x in sys.argv[7].split(",")] if len(sys.argv) > 7 else list(HANG[arm])
q1 = pose + q0[4:]                                                    # shoulder + elbow only; wrist unchanged
body = armsafety.Body(xml, root=f"{arm}ShoulderPitchLink", joints=names, skip_surface=(f"{arm}ShoulderPitchLink",),
                      surface_step=0.008)
body.set_baseline({n: np.radians(v) for n, v in snap["q_deg"].items()})
a, b = np.radians(q0), np.radians(q1)
n = int(np.ceil(np.degrees(np.abs(b - a)).max())) + 1
Q = a + np.linspace(0, 1, n)[:, None] * (b - a)
c = body.clearance(Q)
lo, hi = body.ranges()
move = {"k": 1, "name": f"park-{arm}", "deg": [round(x, 2) for x in q1], "burst": False,
        "path_min_clear_m": round(float(c.min()), 4), "end_clear_m": round(float(c[-1]), 4),
        "in_mjcf_range": bool(np.all(b >= lo) and np.all(b <= hi)),
        "max_travel_deg": round(float(np.degrees(np.abs(b - a)).max()), 1)}
print(f"park {arm}: {[round(x, 1) for x in q0]} -> {move['deg']} travel {move['max_travel_deg']} deg, path min "
      f"{c.min() * 100:.1f} cm, end {c[-1] * 100:.1f} cm, range {move['in_mjcf_range']}"
      + ("" if c.min() >= 0.025 else f"  <-- CHECK {body.closest(Q[int(c.argmin())])}"))
neck = {k: snap["q_deg"][k] for k in ("NeckYaw", "NeckPitch")}
json.dump({"neck_deg": neck, "moves": [move]}, open(out_moves, "w"), indent=1)
parked = json.loads(json.dumps(snap))
parked["q_deg"].update(dict(zip(names, q1)))
json.dump(parked, open(out_snap, "w"))
