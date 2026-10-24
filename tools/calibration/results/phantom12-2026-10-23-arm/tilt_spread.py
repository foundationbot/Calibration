import sys, numpy as np
exec(open("joint_fit.py").read().split("f_c = run(")[0])
s92.MJCF_ORIGINS["NeckPitch"] = "0.025 0.01 0.1927"
alt = s92.urdf_with_mjcf_origins(__import__("pathlib").Path(URDF)); A, N = armchain.ArmChain(alt), neckfk.NeckChain(alt)
globals()["arm"], globals()["neck"] = A, N
oc, kc, _, _ = load("1023c", 0); od, kd, _, _ = load("1023d", 1000)
Xs = {}
for label, obs, kinds in (("1023c", oc, kc), ("1023d", od, kd)):
    fit_h = sorted(h for h, k in kinds.items() if k == "fit")
    full = af.fit(obs, A, N, rig, "M1", X0, holds=set(fit_h), offsets=offs)["X"]; Xs[label] = full
    devs, pos = [], []
    for h in fit_h:
        X = af.fit(obs, A, N, rig, "M1", X0, holds=set(fit_h) - {h}, offsets=offs)["X"]
        devs.append(rm.rot_angle_deg(full[:3, :3].T @ X[:3, :3])); pos.append(np.linalg.norm((X - full)[:3, :3] @ c + (X - full)[:3, 3]) * 1e3)
    print(f"{label}: plate tilt moves {np.round(devs, 2).tolist()} deg when one pose is left out (max {max(devs):.2f}); centre moves max {max(pos):.2f} mm")
    # M0 (no camera correction, fixed origins): is the tilt better pinned without dT?
    f0 = af.fit(obs, A, N, rig, "M0", X0, holds=set(fit_h), offsets=offs)["X"]; Xs[label + "-M0"] = f0
print("between sweeps: M1 tilt %.2f deg, M0 tilt %.2f deg" % (rm.rot_angle_deg(Xs["1023c"][:3, :3].T @ Xs["1023d"][:3, :3]), rm.rot_angle_deg(Xs["1023c-M0"][:3, :3].T @ Xs["1023d-M0"][:3, :3])))
