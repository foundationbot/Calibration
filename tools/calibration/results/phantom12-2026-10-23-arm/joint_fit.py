"""1023c + 1023d together (same plate, neck pitch ~21 and ~28 deg): per-session and joint M1, leave-one-out over fit holds."""
import sys, json, importlib, numpy as np
sys.path.insert(0, "wt2/tools/calibration/scripts")
import armchain, armfit as af, armplan as ap, neckfk, okviscfg, rigcfg, rigmath as rm
s92 = importlib.import_module("92_arm_fit")
URDF = "wt2/config/foundation_rig/20250206_phantom_mk1.urdf"
arm, neck = armchain.ArmChain(URDF), neckfk.NeckChain(URDF)
rig = af.rig_from_config(okviscfg.load("phantom12_stereo_dense_thor-fitc.yaml"))
def load(tag, base):
    plan = json.load(open(f"poses-{tag}.json")); names = plan["names"]; g = plan["grid"]
    grid = rigcfg.AprilGrid(g["cols"], g["rows"], g["tag_size"], g["tag_spacing"])
    d = dict(np.load(f"sweep_{tag}_corners.npz")); ia = [names.index(n) for n in armchain.RIGHT_ARM]
    holds = af.holds_from_joints(d["jt"], d["jq"][:, ia])
    hq = np.array([d["jq"][(d["jt"] >= a) & (d["jt"] <= b)][:, ia].mean(0) for a, b in holds])
    m = af.match_holds(hq, s92.real_stops([json.loads(x) for x in open(f"arm-session-{tag}.jsonl")]))
    keep = sorted(m); obs = af.frames_from_dump(d, [holds[h] for h in keep], names, armchain.RIGHT_ARM, grid)
    for f in obs: f.hold = base + keep[f.hold]
    kinds = {base + h: m[h]["kind"] for h in keep}
    return obs, kinds, plan, grid
oc, kc, pc, grid = load("1023c", 0); od, kd, pd, _ = load("1023d", 1000)
c = np.mean(ap.plate_points(grid)[0], axis=0); X0 = np.array(pc["X0"]); offs = tuple(pc["offsets"])
print("neck pitch deg: 1023c %.1f, 1023d %.1f" % (np.degrees(np.median([f.q_neck[1] for f in oc])), np.degrees(np.median([f.q_neck[1] for f in od]))))
def run(label, obs, kinds):
    fit_h = sorted(h for h, k in kinds.items() if k == "fit")
    f = af.fit(obs, arm, neck, rig, "M1", X0, holds=set(fit_h), offsets=offs)
    loo = []
    for h in fit_h:
        g = af.fit(obs, arm, neck, rig, "M1", X0, holds=set(fit_h) - {h}, offsets=offs)
        loo.append(af.plate_errors(obs, arm, neck, rig, g, holds=[h], centre=c)[h])
    others = [h for h, k in kinds.items() if k != "fit"]
    oth = af.plate_errors(obs, arm, neck, rig, f, holds=others, centre=c)
    print(f"{label}: {len(fit_h)} fit holds | rms {f['rms_px']:.2f} px | dT xyz mm {np.round(f['dT'][:3, 3] * 1e3, 1).tolist()} {rm.rot_angle_deg(f['dT'][:3, :3]):.2f} deg"
          f" | leave-one-out RMS {np.sqrt(np.mean(np.square(loo))):.2f} max {max(loo):.2f} mm | repeats/held-out (unfitted) {np.round(list(oth.values()), 2).tolist()}")
    return f
f_c = run("1023c alone", oc, kc); f_d = run("1023d alone", od, kd); f_j = run("joint", oc + od, {**kc, **kd})
m0 = af.fit(oc + od, arm, neck, rig, "M0", X0, holds={h for h, k in {**kc, **kd}.items() if k == "fit"}, offsets=offs)
print("joint M0 (no dT) rms %.2f px" % m0["rms_px"])
print("dT difference between sessions: %.1f mm, %.2f deg" % (np.linalg.norm(f_c["dT"][:3, 3] - f_d["dT"][:3, 3]) * 1e3, rm.rot_angle_deg(f_c["dT"][:3, :3].T @ f_d["dT"][:3, :3])))
print("joint dT 4x4:", np.round(f_j["dT"], 6).tolist())

# the URDF with the MJCF's shoulder / neck origins (NeckPitch y kept at the URDF's 0.01, as the review found)
s92.MJCF_ORIGINS["NeckPitch"] = "0.025 0.01 0.1927"
alt = s92.urdf_with_mjcf_origins(__import__("pathlib").Path(URDF))
arm, neck = armchain.ArmChain(alt), neckfk.NeckChain(alt)
print("--- with the MJCF shoulder/neck origins")
oc, kc, _, _ = load("1023c", 0); od, kd, _, _ = load("1023d", 1000)
m0 = af.fit(oc + od, arm, neck, rig, "M0", X0, holds={h for h, k in {**kc, **kd}.items() if k == "fit"}, offsets=offs)
print("joint M0 (no dT) rms %.2f px" % m0["rms_px"])
fc2 = run("1023c alone", oc, kc); fd2 = run("1023d alone", od, kd); fj2 = run("joint", oc + od, {**kc, **kd})
print("dT difference between sessions: %.1f mm, %.2f deg" % (np.linalg.norm(fc2["dT"][:3, 3] - fd2["dT"][:3, 3]) * 1e3, rm.rot_angle_deg(fc2["dT"][:3, :3].T @ fd2["dT"][:3, :3])))
