"""Do the neck encoders tell the truth about the head's tilt? Per hold, the head IMU's mean
accelerometer direction (gravity, in the IMU frame) against the model
    u_S,i = R_SB · R_TB(q_i)^T · g_T
with q_i the encoders' mean over the hold, R_SB (3) and g_T (gravity in the torso, 2) fitted.
A gravity residual sees the head's pitch and roll, not its heading. Residual at the accelerometer's
level (~0.05 deg) = the encoders and URDF explain the head's tilt; much more = they do not.
Usage: python gravity_fit.py <signals.npz>... (the package, config and URDF from this checkout)"""
import sys
from pathlib import Path

import numpy as np

CAL = Path(__file__).resolve().parents[3]          # tools/calibration of this checkout
sys.path.insert(0, str(CAL / "scripts"))
import neckfk  # noqa: E402
import okviscfg  # noqa: E402
import rigmath as rm  # noqa: E402

CFG = CAL / "results/phantom12-2026-10-15/imucam/phantom12_stereo.yaml"
URDF = CAL.parents[1] / "config/foundation_rig/20250206_phantom_mk1.urdf"


def hold_means(s, trim=0.2):
    """Per hold: mean head accel direction, mean encoder angles, torso accel direction, approach sides."""
    rows = []
    for t0, t1, y, p, ay, ap, drift in s["holds"]:
        a, b = t0 + trim * (t1 - t0), t1 - trim * (t1 - t0)
        mh = (s["ht"] >= a) & (s["ht"] <= b)
        mj = (s["jt"] >= a) & (s["jt"] <= b)
        mt = (s["torso_t"] >= a) & (s["torso_t"] <= b)
        if mh.sum() < 100 or mj.sum() < 100:
            continue
        u = s["ha"][mh].mean(0)
        ut = s["torso_a"][mt].mean(0)
        rows.append(dict(u=u / np.linalg.norm(u), yaw=s["yaw"][mj].mean(), pitch=s["pitch"][mj].mean(),
                         ut=ut / np.linalg.norm(ut), ay=ay, ap=ap, drift=np.degrees(drift), t=(t0 + t1) / 2,
                         n=int(mh.sum())))
    return rows


def model(params, rows, chain, R_SB0):
    R_SB = R_SB0 @ rm.exp_rot(params[:3])
    th, ph = params[3], params[4]
    g = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return [R_SB @ chain.T_TB(r["yaw"], r["pitch"])[:3, :3].T @ g for r in rows]


def resid(params, rows, chain, R_SB0):
    out = []
    for r, u in zip(rows, model(params, rows, chain, R_SB0)):
        out.append(np.cross(u, r["u"]))           # small-angle direction error, rad
    return np.array(out).ravel()


def lm(f, p, iters=60):
    r = f(p)
    lam = 1e-3
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for i in range(p.size):
            d = np.zeros(p.size)
            d[i] = 1e-7
            J[:, i] = (f(p + d) - f(p - d)) / 2e-7
        H, g = J.T @ J, J.T @ r
        while True:
            dp = -np.linalg.solve(H + lam * np.diag(np.diag(H)), g)
            rn = f(p + dp)
            if rn @ rn < r @ r:
                lam = max(lam / 10, 1e-12)
                break
            lam *= 10
            if lam > 1e9:
                return p, r
        p, r = p + dp, rn
        if np.linalg.norm(dp) < 1e-12:
            break
    return p, r


def main():
    cfg, chain = okviscfg.load(CFG), neckfk.NeckChain(URDF)
    R_SB0 = cfg.T_BS[:3, :3].T
    for path in sys.argv[1:]:
        s = dict(np.load(path))
        rows = hold_means(s)
        f = lambda p: resid(p, rows, chain, R_SB0)  # noqa: E731
        u0 = R_SB0.T @ np.mean([r["u"] for r in rows], 0)     # rough gravity in the Head, start for g_T
        p0 = np.r_[0, 0, 0, np.arccos(np.clip(-u0[2] / np.linalg.norm(u0), -1, 1)), 0.0]
        best = None
        for ph in np.linspace(-np.pi, np.pi, 8, endpoint=False):    # g_T's azimuth: several starts
            for th in (0.05, 0.3, 2.8, 3.1):
                p, r = lm(f, np.r_[0, 0, 0, th, ph])
                if best is None or r @ r < best[1] @ best[1]:
                    best = (p, r)
        p, r = best
        res = np.degrees(np.linalg.norm(r.reshape(-1, 3), axis=1))
        rot = np.degrees(np.linalg.norm(p[:3]))
        ut = np.array([q["ut"] for q in rows])
        tilt = np.degrees(np.arccos(np.clip(ut @ (np.median(ut, 0) / np.linalg.norm(np.median(ut, 0))), -1, 1)))
        print(f"{Path(path).stem}: {len(rows)} holds; fitted R_SB moved {rot:.3f} deg from the config "
              f"({np.degrees(p[:3]).round(3)}); gravity residual rms {np.sqrt(np.mean(res ** 2)):.3f} deg, "
              f"max {res.max():.3f} deg")
        print("   hold  yaw  pitch  ay ap  drift | residual [deg] | torso tilt vs median [deg]")
        for k, (q, e, tt) in enumerate(zip(rows, res, tilt)):
            print(f"   {k:3d} {np.degrees(q['yaw']):+5.0f} {np.degrees(q['pitch']):+5.0f}  {int(q['ay']):+d} {int(q['ap']):+d} "
                  f"{q['drift']:5.2f} | {e:.3f} | {tt:.3f}")


if __name__ == "__main__":
    main()
