"""Both sweeps, a separate plate placement X per sweep (the plate moved between them), one shared camera correction
dT (or none), leave-one-out over all fit holds. URDF as is, and with the fixed shoulder/neck origins."""
import sys, numpy as np
from scipy.optimize import least_squares
exec(open("joint_fit.py").read().split("f_c = run(")[0])
def fit2(sets, A, N, with_dT, drop=None):
    S = [af._Stack([f for f in o if f.hold in h and f.hold != drop], N, rig) for o, h in sets]
    k = 12 + (6 if with_dT else 0)
    def res(p):
        dT = rm.transform(rm.exp_rot(p[12:15]), p[15:18]) if with_dT else np.eye(4)
        out = []
        for i, s in enumerate(S):
            X = X0 @ rm.transform(rm.exp_rot(p[6 * i:6 * i + 3]), p[6 * i + 3:6 * i + 6])
            out.append((s.pixels(A, X, dT, np.zeros(0), ()) - s.uv).ravel())
        return np.concatenate(out)
    r = least_squares(res, np.zeros(k), loss="soft_l1", f_scale=1.0, x_scale=0.01)
    p = r.x; dT = rm.transform(rm.exp_rot(p[12:15]), p[15:18]) if with_dT else np.eye(4)
    Xs = [X0 @ rm.transform(rm.exp_rot(p[6 * i:6 * i + 3]), p[6 * i + 3:6 * i + 6]) for i in range(2)]
    e = r.fun.reshape(-1, 2); return Xs, dT, float(np.sqrt(np.mean(np.sum(e ** 2, 1))))
s92.MJCF_ORIGINS["NeckPitch"] = "0.025 0.01 0.1927"
alt = s92.urdf_with_mjcf_origins(__import__("pathlib").Path(URDF))
for label, (A, N) in (("URDF as is", (armchain.ArmChain(URDF), neckfk.NeckChain(URDF))), ("fixed origins", (armchain.ArmChain(alt), neckfk.NeckChain(alt)))):
    globals()["arm"], globals()["neck"] = A, N
    oc, kc, _, _ = load("1023c", 0); od, kd, _, _ = load("1023d", 1000)
    fc = {h for h, k in kc.items() if k == "fit"}; fd = {h for h, k in kd.items() if k == "fit"}
    for with_dT in (False, True):
        Xs, dT, rms = fit2([(oc, fc), (od, fd)], A, N, with_dT)
        loo = []
        for h in sorted(fc | fd):
            Xl, dTl, _ = fit2([(oc, fc), (od, fd)], A, N, with_dT, drop=h)
            obs, i = (oc, 0) if h < 1000 else (od, 1)
            loo.append(af.plate_errors(obs, A, N, rig, {"X": Xl[i], "dT": dTl, "offsets": np.zeros(0), "offset_joints": []}, holds=[h], centre=c)[h])
        oth = []
        for obs, kinds, i in ((oc, kc, 0), (od, kd, 1)):
            oth += list(af.plate_errors(obs, A, N, rig, {"X": Xs[i], "dT": dT, "offsets": np.zeros(0), "offset_joints": []}, holds=[h for h, k in kinds.items() if k != "fit"], centre=c).values())
        allv = loo + oth
        print(f"{label}, {'shared dT' if with_dT else 'no dT'}: rms {rms:.2f} px | dT {np.round(dT[:3, 3] * 1e3, 1).tolist()} mm {rm.rot_angle_deg(dT[:3, :3]):.2f} deg | unseen poses (n={len(allv)}): RMS {np.sqrt(np.mean(np.square(allv))):.1f} max {max(allv):.1f} mm, within 5 mm {sum(v <= 5 for v in allv)}/{len(allv)}")
