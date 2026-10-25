"""Refit 1023c on still frames only: per hold, keep images whose corners sit within STILL_PX of the hold's median
corners (each corner, both cameras). Then M0/M1/M2 on the fit holds, leave-one-out over the fit holds, board scale."""
import sys, json, numpy as np
sys.path.insert(0, "wt2/tools/calibration/scripts")
import importlib, armchain, armfit as af, armplan as ap, neckfk, okviscfg, rigcfg, rigmath as rm
s92 = importlib.import_module("92_arm_fit")
STILL_PX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
plan = json.load(open("poses-1023c.json")); names = plan["names"]; g = plan["grid"]
grid = rigcfg.AprilGrid(g["cols"], g["rows"], g["tag_size"], g["tag_spacing"])
d = dict(np.load("sweep_1023c_corners.npz")); ia = [names.index(n) for n in armchain.RIGHT_ARM]
holds = af.holds_from_joints(d["jt"], d["jq"][:, ia])
hq = np.array([d["jq"][(d["jt"] >= a) & (d["jt"] <= b)][:, ia].mean(0) for a, b in holds])
stops = s92.real_stops([json.loads(x) for x in open("arm-session-1023c.jsonl")])
matched = af.match_holds(hq, stops); keep = sorted(matched)
obs = af.frames_from_dump(d, [holds[h] for h in keep], names, armchain.RIGHT_ARM, grid)
for f in obs: f.hold = keep[f.hold]
# stillness: per hold and camera, corner deviation from the hold's per-corner median
still = []
for h in keep:
    for c in (0, 1):
        fs = [f for f in obs if f.hold == h and f.cam == c and len(f.P) == 16]
        if not fs: continue
        key = lambda f: np.lexsort((f.P[:, 1], f.P[:, 0]))
        U = np.array([f.uv[key(f)] for f in fs]); med = np.median(U, axis=0)
        dev = np.linalg.norm(U - med, axis=2).max(axis=1)
        still += [f for f, e in zip(fs, dev) if e <= STILL_PX]
kept = {h: sum(f.hold == h for f in still) for h in keep}
print("still frames per hold", kept, "of", len(obs))
cfg = okviscfg.load("phantom12_stereo_dense_thor-fitc.yaml"); rig = af.rig_from_config(cfg)
arm = armchain.ArmChain("wt2/config/foundation_rig/20250206_phantom_mk1.urdf"); neck = neckfk.NeckChain("wt2/config/foundation_rig/20250206_phantom_mk1.urdf")
kind = {h: matched[h]["kind"] for h in keep}; fit_h = [h for h in keep if kind[h] == "fit" and kept[h]]
ho_h = [h for h in keep if kind[h] == "heldout" and kept[h]]
X0 = np.array(plan["X0"]); offs = tuple(plan["offsets"]); c = np.mean(ap.plate_points(grid)[0], axis=0)
for m in af.ORDER:
    f = af.fit(still, arm, neck, rig, m, X0, holds=set(fit_h), offsets=offs)
    loo = []
    for h in fit_h:
        g_ = af.fit(still, arm, neck, rig, m, X0, holds=set(fit_h) - {h}, offsets=offs)
        loo.append(af.plate_errors(still, arm, neck, rig, g_, holds=[h], centre=c)[h])
    ho = af.plate_errors(still, arm, neck, rig, f, holds=ho_h, centre=c)
    print(m, "rms px %.2f" % f["rms_px"], "within1px %.2f" % f["within_1px"], "dT mm %.1f deg %.3f" % (np.linalg.norm(f["dT"][:3, 3]) * 1e3, rm.rot_angle_deg(f["dT"][:3, :3])),
          "dT xyz mm", np.round(f["dT"][:3, 3] * 1e3, 1).tolist(), "| leave-one-out RMS %.2f max %.2f mm" % (np.sqrt(np.mean(np.square(loo))), max(loo)), "| held-out", {k: round(v, 2) for k, v in ho.items()})
    if m == "M1":
        P1 = f
offs_ = af.plate_offsets(still, arm, neck, rig, P1, centre=c)
pairs = s92.pairs(matched, "repeat_same"); same = [float(np.linalg.norm(offs_[x] - offs_[y])) for x, y in pairs if x in offs_ and y in offs_]
print("same-side pairs mm", np.round(same, 2).tolist(), "floor %.2f" % (np.sqrt(np.mean(np.square(same))) / np.sqrt(2)) if same else "")
half = len(fit_h) // 2; fx = {"dT": P1["dT"], "offsets": P1["offsets"]}
Xa = af.fit(still, arm, neck, rig, "M0", P1["X"], holds=set(fit_h[:half]), fixed=fx, offsets=offs)["X"]
Xb = af.fit(still, arm, neck, rig, "M0", P1["X"], holds=set(fit_h[half:]), fixed=fx, offsets=offs)["X"]
print("slip mm %.2f deg %.3f" % (np.linalg.norm((Xa[:3, :3] @ c + Xa[:3, 3]) - (Xb[:3, :3] @ c + Xb[:3, 3])) * 1e3, rm.rot_angle_deg(Xa[:3, :3].T @ Xb[:3, :3])))
sc = af.board_scale_pct(still, rig, grid); print("board scale % per hold", {k: round(v, 2) for k, v in sc.items()})
print("plate centre in RightGripperYawLink (m):", np.round(P1["X"][:3, :3] @ c + P1["X"][:3, 3], 4).tolist())
print("M1 dT (4x4):", np.round(P1["dT"], 6).tolist())
