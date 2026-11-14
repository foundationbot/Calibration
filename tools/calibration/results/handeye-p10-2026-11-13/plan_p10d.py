"""phantom10 hand-eye re-plan from a fresh snapshot (plan_p10c + 13 Nov 19:40): the P10.json poses (right wave 2, left
mirrored wave 2), staged from the current rest, MuJoCo clearance on every path, commanded and with the measured sag
x1.3. New: when the rest itself is closer than MIN (13 Nov 19:40: the right fingers 8 mm from the right thigh), a
"lift" (shoulder pitch -10 deg: forward) goes first and last; the rest -> lift path may not come closer than the rest
itself (minus 1 mm), lift -> stage needs MIN. usage: plan_p10d.py <snap.json> <P10.json> <out P10.json>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/data/kaustabh/p21arm/route/scripts")
import p21mj  # noqa: E402
MJCF = Path("/data/kaustabh/p21arm/mjcf/phantom_mk1.xml")
J = ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "ElbowPitch", "WristYaw", "WristRoll", "GripperYaw")
NECKS = [[10, -12], [18, 12], [26, -12], [22, 0]]
SAG = np.array([23.5, 12.7, 0, 11.1, 0, 0, 0]) * 1.3
MIN = 0.03
snap = json.load(open(sys.argv[1])); cfg = json.load(open(sys.argv[2]))
ok_all, arms = True, {}
for arm, s in (("Right", 1), ("Left", -1)):
    rest = np.array([snap["q_deg"][arm + j] for j in J])
    poses = [(m["name"], np.array(m["deg"], float)) for m in cfg["arms"][arm]["moves"] if m.get("burst")]
    stage = rest.copy(); stage[:4] = [-20, -8 * s, 6 * s, -65]
    lift = rest.copy(); lift[0] -= 10.0
    b0 = p21mj.make_body(MJCF, snap, snap["q_deg"]["NeckPitch"], snap["q_deg"]["NeckYaw"], arm=arm)
    rest_c = float(np.min(b0.clearance(np.radians(rest[None]))))
    use_lift = rest_c < MIN
    seq = ([("lift", lift)] if use_lift else []) + [("stage", stage)] + poses + [("stage_back", stage)] \
        + ([("lift_back", lift)] if use_lift else []) + [("rest", rest)]
    worst = {}
    for pitch, yaw in NECKS + [[26, 0]]:
        b = p21mj.make_body(MJCF, snap, pitch, yaw, arm=arm)
        prev = rest
        for name, q in seq:
            c = float(np.min(b.clearance(np.radians(np.linspace(prev, q, 25)))))
            if name not in ("lift", "stage", "stage_back", "lift_back", "rest"):   # moves without a burst: as plan_p10c
                sag = SAG * np.array([1, s, 1, 1, 1, 1, 1])
                c = min(c, float(np.min(b.clearance(np.radians(np.linspace(prev + sag, q + sag, 25))))))
            worst[name] = min(worst.get(name, 9), c); prev = q
    print("==", arm, "rest", np.round(rest, 1).tolist(), f"rest clearance {rest_c:.3f} m", "(lift first)" if use_lift else "")
    for name, q in seq:
        need = rest_c - 0.001 if name in ("lift", "rest") and use_lift else MIN
        flag = "OK" if worst[name] >= need or (name == "rest" and not use_lift) else "TOO CLOSE"
        ok_all &= flag == "OK"
        print(f"  {name:10s} min clear {worst[name]:.3f} m  {flag}")
    lifts = [dict(name="lift", deg=lift.round(2).tolist(), burst=False)] if use_lift else []
    arms[arm] = {"moves": lifts + [dict(name="stage", deg=stage.round(2).tolist(), burst=False)]
                 + [dict(name=n, deg=q.round(2).tolist(), burst=True, necks=NECKS) for n, q in poses]
                 + [dict(name="stage_back", deg=stage.round(2).tolist(), burst=False)]
                 + [dict(m, name="lift_back") for m in lifts]
                 + [dict(name="rest", deg=rest.round(2).tolist(), burst=False)],
                 "source": f"plan_p10d.py from the snapshot of {snap.get('t_ns')}; clearance >= {MIN} m incl. sag x1.3"}
import time as _t
cfg = dict(cfg, arms=arms, snapshot={"t": _t.strftime("%Y-%m-%d %H:%M", _t.localtime(snap["t_ns"] / 1e9)),
                                      "q_deg": snap["q_deg"]})
json.dump(cfg, open(sys.argv[3], "w"), indent=1)
print("ALL OK" if ok_all else "NOT OK")
