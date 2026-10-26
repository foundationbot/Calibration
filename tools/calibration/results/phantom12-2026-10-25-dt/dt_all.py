"""Phase B of docs/superpowers/plans/2026-10-25-tagfree-selfcal.md (stereo-depth-eval branch): the grasp camera
correction dT refitted on all four right-arm plate sessions (1023c, 1023d, rv1024, rl1024), #816 URDF, one plate X per
session (armfit_multi). Decisions fixed before data in the plan (Phase B + the 23:33 amendment).

Holds per session as each session's own plate script made them (plate_sessions.py).

usage: dt_all.py scripts_dir arm_task2_dir rv1024_dir rl1024_dir config.yaml urdf out.json [workers]
"""
import dataclasses
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SCRIPTS, TASK2, RV, RL, CFG, URDF, OUT = sys.argv[1:8]
WORKERS = int(sys.argv[8]) if len(sys.argv) > 8 else 8
sys.path.insert(0, SCRIPTS)
import armchain, armfit as af, armfit_multi as am, neckfk, okviscfg, plate_sessions, rigmath as rm  # noqa: E402,E401

arm, neck = armchain.ArmChain(URDF), neckfk.NeckChain(URDF)
rig = af.rig_from_config(okviscfg.load(CFG))
N_BOOT, SEED = 200, 20261025


NAMES = ("1023c", "1023d", "rv1024", "rl1024")
L = plate_sessions.load_all(TASK2, RV, RL)
S = [x.obs for x in L]; FIT = [x.fit for x in L]; UNSEEN = [x.unseen for x in L]; X0 = L[0].X0


def summ(T):
    return {"t_mm": np.round(T[:3, 3] * 1e3, 2).tolist(), "deg": round(rm.rot_angle_deg(T[:3, :3]), 3),
            "rotvec_deg": np.round(np.degrees(rm.log_rot(T[:3, :3])), 3).tolist()}


def fit(sessions=S, holds=FIT, **kw):
    return am.fit_all(sessions, arm, neck, rig, X0, holds=holds, **kw)


def loo_one(args):
    """Refit without hold h of session s; the error at h (mm)."""
    s, h, kw = args
    hl = [set(f) - ({h} if i == s else set()) for i, f in enumerate(FIT)]
    f = fit(holds=hl, **kw)
    return s, h, am.unseen_errors(S, arm, neck, rig, f, holds=[[h] if i == s else [] for i in range(len(S))])[s][h]


def boot_one(args):
    """One bootstrap draw: each session's fit holds resampled with replacement (relabelled) -> the fitted dTs."""
    k, kw = args
    rng = np.random.default_rng(SEED + k)
    ss, hs = [], []
    for obs, fh in zip(S, FIT):
        fh = sorted(fh); new, ids = [], set()
        for j, h in enumerate(rng.choice(fh, len(fh), replace=True)):
            new += [dataclasses.replace(x, hold=j) for x in obs if x.hold == h]; ids.add(j)
        ss.append(new); hs.append(ids)
    f = fit(ss, hs, **kw)
    return [(b, f["dT"][g]) for g, b in enumerate(f["bands"])]


def ci(Ts, ref):
    """95 % intervals of the dT components (mm) and of the angle to ref (deg) over bootstrap draws."""
    t = np.array([T[:3, 3] for T in Ts]) * 1e3
    a = [rm.rot_angle_deg(T[:3, :3].T @ ref[:3, :3]) for T in Ts]
    d = [np.linalg.norm(T[:3, 3] - ref[:3, 3]) * 1e3 for T in Ts]
    return {"t_mm_2.5": np.percentile(t, 2.5, 0).round(2).tolist(), "t_mm_97.5": np.percentile(t, 97.5, 0).round(2).tolist(),
            "dist_mm_95": round(float(np.percentile(d, 95)), 2), "angle_deg_95": round(float(np.percentile(a, 95)), 3)}


def rms(v):
    v = np.asarray(list(v), float)
    return round(float(np.sqrt(np.mean(v ** 2))), 2)


if __name__ == "__main__":
    out = {"sessions": {n: {"frames": len(o), "fit_holds": len(f), "unseen_holds": len(u)}
                        for n, o, f, u in zip(NAMES, S, FIT, UNSEEN)}}
    print("sessions", out["sessions"], flush=True)
    today = fit(S[:2], FIT[:2])["dT"][0]                     # 1023c + 1023d: the dT in use since 23 Oct
    out["today_dT_refit_cd"] = summ(today)
    print("today's dT (1023c+d refit):", summ(today), flush=True)
    variants = {"B1": {}, "B2": {"bands": True}, "today_fixed": {"fixed_dT": today}}
    with ProcessPoolExecutor(WORKERS) as ex:
        for name, kw in variants.items():
            f = fit(**kw)
            un = am.unseen_errors(S, arm, neck, rig, f, holds=UNSEEN)
            loo = list(ex.map(loo_one, [(s, h, kw) for s in range(len(S)) for h in sorted(FIT[s])]))
            allv = [e for d in un for e in d.values()] + [e for _, _, e in loo]
            out[name] = {"rms_px": round(f["rms_px"], 3), "dT": {str(b): summ(T) for b, T in zip(f["bands"], f["dT"])},
                         "unseen_fixed_mm": {n: {str(h): round(e, 2) for h, e in d.items()} for n, d in zip(NAMES, un)},
                         "loo_mm": {n: {str(h): round(e, 2) for s, h, e in loo if s == i} for i, n in enumerate(NAMES)},
                         "unseen_rms_mm": rms(allv), "unseen_max_mm": round(max(allv), 2),
                         "unseen_rms_mm_per_session": {n: rms([e for e in un[i].values()] + [e for s, _, e in loo if s == i])
                                                       for i, n in enumerate(NAMES)},
                         "within_5mm": f"{sum(e <= 5 for e in allv)}/{len(allv)}"}
            print(name, {k: v for k, v in out[name].items() if k not in ("unseen_fixed_mm", "loo_mm")}, flush=True)
        b1 = fit()["dT"][0]
        lso = [fit([x for j, x in enumerate(S) if j != i], [x for j, x in enumerate(FIT) if j != i])["dT"][0]
               for i in range(len(S))]
        out["B1"]["leave_session_out"] = {n: {**summ(T), "vs_all_mm": round(float(np.linalg.norm(T[:3, 3] - b1[:3, 3]) * 1e3), 2),
                                              "vs_all_deg": round(rm.rot_angle_deg(T[:3, :3].T @ b1[:3, :3]), 3)}
                                          for n, T in zip(NAMES, lso)}
        for name, kw in (("B1", {}), ("B2", {"bands": True})):
            draws = list(ex.map(boot_one, [(k, kw) for k in range(N_BOOT)]))
            ref = fit(**kw)
            out[name]["bootstrap"] = {str(b): {**ci([dict(d)[b] for d in draws if b in dict(d)], T),     # a draw can
                                               "draws": sum(b in dict(d) for d in draws)}                 # lose a band
                                      for b, T in zip(ref["bands"], ref["dT"])}
            print(name, "bootstrap", out[name]["bootstrap"], flush=True)
    out["B1"]["moved_from_today"] = {"mm": round(float(np.linalg.norm(b1[:3, 3] - today[:3, 3]) * 1e3), 2),
                                     "deg": round(rm.rot_angle_deg(b1[:3, :3].T @ today[:3, :3]), 3)}
    Path(OUT).write_text(json.dumps(out, indent=1))
    print("moved from today:", out["B1"]["moved_from_today"])
