import sys, numpy as np
exec(open("joint_fit.py").read().split("f_c = run(")[0])
s92.MJCF_ORIGINS["NeckPitch"] = "0.025 0.01 0.1927"
alt = s92.urdf_with_mjcf_origins(__import__("pathlib").Path(URDF))
for label, (A, N) in (("URDF as is", (armchain.ArmChain(URDF), neckfk.NeckChain(URDF))), ("URDF with the fixed origins", (armchain.ArmChain(alt), neckfk.NeckChain(alt)))):
    globals()["arm"], globals()["neck"] = A, N
    oc, kc, _, _ = load("1023c", 0); od, kd, _, _ = load("1023d", 1000)
    obs, kinds = oc + od, {**kc, **kd}
    fit_h = sorted(h for h, k in kinds.items() if k == "fit")
    loo = []
    for h in fit_h:
        g = af.fit(obs, A, N, rig, "M0", X0, holds=set(fit_h) - {h}, offsets=offs)
        loo.append(af.plate_errors(obs, A, N, rig, g, holds=[h], centre=c)[h])
    f = af.fit(obs, A, N, rig, "M0", X0, holds=set(fit_h), offsets=offs)
    oth = af.plate_errors(obs, A, N, rig, f, holds=[h for h, k in kinds.items() if k != "fit"], centre=c)
    allv = loo + list(oth.values())
    print(f"{label}, no extra correction, both head angles: leave-one-out RMS {np.sqrt(np.mean(np.square(loo))):.1f} max {max(loo):.1f} mm | unfitted repeats {np.round(list(oth.values()), 1).tolist()} | all {len(allv)}: RMS {np.sqrt(np.mean(np.square(allv))):.1f}, within 5 mm {sum(v <= 5 for v in allv)}/{len(allv)}")
