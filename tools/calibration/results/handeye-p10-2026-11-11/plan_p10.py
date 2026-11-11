"""phantom10 hand-eye wave, first live test: phantom14 wave-2's five arm poses (measured angles, deg), mirrored for the
left arm, each with several neck angles. MuJoCo clearance (armsafety.Body via p21mj.make_body) along every move from
phantom10's snapshot, with the rest of the robot (other arm, spine, legs) at the snapshot pose.
usage: plan_p10.py <snap.json> <out.json>"""
import json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/data/kaustabh/p21arm/route/scripts")
sys.path.insert(0, "/data/kaustabh/p14arm/urdfx")
import p21mj  # noqa: E402

MJCF = Path("/data/kaustabh/p21arm/mjcf/phantom_mk1.xml")
J = ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "ElbowPitch", "WristYaw", "WristRoll", "GripperYaw")
MIRROR = np.array([1, -1, -1, 1, -1, -1, -1])
ORDER = ("p07", "p08", "p09", "p10", "p04")
NECKS = [(10, 0), (22, -12), (22, 12), (26, 0)]       # pitch, yaw (deg); neck.py clamps pitch to 26
MIN_CLEAR = 0.03                                      # m, as the phantom12/21 routes
STEPS = 25


def wave2_right_deg():
    d = json.load(open("/data/kaustabh/p14arm/urdfx/base_w2.json"))["res"]
    P = {}
    for r in d:
        P.setdefault(r["name"].split("-")[1], []).append([np.degrees(r["q"]["Right" + j]) for j in J])
    return {p: np.round(np.mean(P[p], 0), 2) for p in ORDER}


def legs(arm, snap, poses):
    rest = np.array([snap["q_deg"][arm + j] for j in J])
    stage = rest.copy(); stage[:4] = [-20, -8 if arm == "Right" else 8, 6 if arm == "Right" else -6, -65]
    # an arm resting with a large shoulder yaw (phantom10's left: +62) first lifts forward and outward with its yaw
    # unchanged (unwinding it at rest swings the bent forearm into the hip), then goes to the stage pose in the air
    lift = rest.copy(); lift[0] = -35.0; lift[1] = 10.0 if arm == "Left" else -10.0
    big = abs(rest[2]) > 30
    seq = ([("lift", lift)] if big else []) + [("stage", stage)] + [(p, poses[p]) for p in ORDER] \
        + [("stage_back", stage)] + ([("relift", lift)] if big else []) + [("rest", rest)]
    out, prev = [], rest
    for name, q in seq:
        out.append((name, prev, q)); prev = q
    return out


def check(args):
    arm, snap, pitch, yaw, seq = args
    body = p21mj.make_body(MJCF, snap, pitch, yaw, arm=arm)
    res = []
    for name, a, b in seq:
        Q = np.radians(np.linspace(a, b, STEPS))
        c = body.clearance(Q)
        res.append(dict(move=name, min_clear_m=round(float(np.min(c)), 4), end_clear_m=round(float(c[-1]), 4),
                        max_travel_deg=round(float(np.max(np.abs(b - a))), 1)))
    return arm, pitch, yaw, res


def main():
    snap = json.load(open(sys.argv[1]))
    R = wave2_right_deg()
    poses = {"Right": R, "Left": {p: np.round(R[p] * MIRROR, 2) for p in ORDER}}
    jobs = []
    for arm in ("Right", "Left"):
        seq = legs(arm, snap, poses[arm])
        for pitch, yaw in NECKS:
            jobs.append((arm, snap, pitch, yaw, seq))
    with Pool(8) as pool:
        out = pool.map(check, jobs)
    worst = {}
    for arm, pitch, yaw, res in out:
        for r in res:
            k = (arm, r["move"])
            if k not in worst or r["min_clear_m"] < worst[k]["min_clear_m"]:
                worst[k] = dict(r, neck=(pitch, yaw))
    ok = True
    for arm in ("Right", "Left"):
        print("==", arm, "rest", [round(snap["q_deg"][arm + j], 1) for j in J])
        for name, a, b in legs(arm, snap, poses[arm]):
            w = worst[(arm, name)]
            flag = "OK" if w["min_clear_m"] >= MIN_CLEAR else "TOO CLOSE"
            ok &= flag == "OK"
            print("  %-10s -> %s  travel %5.1f  min clear %.3f m (neck %s)  %s" % (
                name, np.round(b, 1).tolist(), w["max_travel_deg"], w["min_clear_m"], w["neck"], flag))
    json.dump(dict(poses={a: {p: v.tolist() for p, v in poses[a].items()} for a in poses}, necks=NECKS,
                   checks=[dict(arm=a, neck=[p, y], res=r) for a, p, y, r in out], all_ok=bool(ok)),
              open(sys.argv[2], "w"), indent=1)
    print("ALL OK" if ok else "NOT OK")


if __name__ == "__main__":
    main()
