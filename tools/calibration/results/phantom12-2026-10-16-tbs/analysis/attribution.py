"""Whose error is it? Per hold, the camera's residual (hand-eye fit, URDF neck, board pose per take)
and the head IMU's gravity residual (torso tilt corrected), both turned into Head axes. The camera
and the IMU are independent sensors; the encoders are the one input both share. If the holds'
residuals agree (same sign, same size, about Head x and Head y, which gravity sees), the error is
in the encoders/neck; if they do not, each sensor's own errors dominate.
Usage: python attribution.py <take> [<data dir>] (<take>-frames.npz and <take>-signals.npz; default: next to this
script; the 16 Oct data is in ~/calib/phantom12/results/tbs-2026-10-16/analysis)"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import fusedfit as ff  # noqa: E402
from gravity_fit import CFG, URDF, hold_means, lm  # noqa: E402
from gravity_fit2 import predict  # noqa: E402
import neckfk  # noqa: E402
import okviscfg  # noqa: E402
import rigmath as rm  # noqa: E402


def main():
    take = sys.argv[1]
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE
    cfg, chain = okviscfg.load(CFG), neckfk.NeckChain(URDF)
    fr = ff.load([data / f"{take}-frames.npz"])[0]
    m = ff.Model(cfg, chain, [fr], False)
    p, res, cov, cond = ff.fit(m)
    X, Z, _ = m.unpack(p)
    # camera residual rotation (camera c axes, E = (ZB)^-1 A X C) -> Head axes: R_X R_C w
    cam_head = {}
    for h in np.unique(fr["hold"]):
        ws = []
        for k in np.flatnonzero(fr["hold"] == h):
            Rc = (X @ m.C[fr["cam"][k]])[:3, :3]
            ws.append(Rc @ res[k, :3])
        cam_head[int(h)] = np.degrees(np.mean(ws, 0))
    s = dict(np.load(data / f"{take}-signals.npz"))
    rows = hold_means(s)
    R_SB0 = cfg.T_BS[:3, :3].T
    f = lambda q: np.array([np.cross(u, r["u"]) for u, r in zip(predict(q, rows, chain, R_SB0, "torso")[0], rows)]).ravel()  # noqa: E731
    best = None
    for R in [rm.rot_axis(ax, ang) for ax in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]) for ang in (0, 1.5, 3.1, -1.5)]:
        q, r = lm(f, np.r_[np.zeros(3), rm.log_rot(R)])
        if best is None or r @ r < best[1] @ best[1]:
            best = (q, r)
    q, r = best
    _, R_SB = predict(q, rows, chain, R_SB0, "torso")
    # gravity residual cross(u_model, u_meas) is in IMU axes: a rotation of the head by w_S -> Head axes R_SB^T w_S
    grav_head = [np.degrees(R_SB.T @ e) for e in r.reshape(-1, 3)]
    # holds in signals (hold_means order = holds order, dropped ones skipped) vs camera hold ids: match by index
    hold_ids = [k for k in range(len(s["holds"]))]
    kept = []
    for k, (t0, t1, *_rest) in enumerate(s["holds"]):
        a, b = t0 + 0.2 * (t1 - t0), t1 - 0.2 * (t1 - t0)
        if ((s["ht"] >= a) & (s["ht"] <= b)).sum() >= 100 and ((s["jt"] >= a) & (s["jt"] <= b)).sum() >= 100:
            kept.append(k)
    g_by = dict(zip(kept, grav_head))
    common = [h for h in sorted(cam_head) if h in g_by]
    C = np.array([cam_head[h] for h in common])
    G = np.array([g_by[h] for h in common])
    C -= C.mean(0)
    G -= G.mean(0)                     # constant offsets belong to X / R_SB; compare the variations
    print(f"{take}: {len(common)} holds in both. Residual rotation about Head x, y [deg], mean removed:")
    print("   hold | camera x   y   | gravity x   y")
    for h, c, g in zip(common, C, G):
        print(f"   {h:3d}  | {c[0]:+.2f} {c[1]:+.2f} | {g[0]:+.2f} {g[1]:+.2f}")
    for i, nm in ((0, "Head x"), (1, "Head y")):
        cc = np.corrcoef(C[:, i], G[:, i])[0, 1]
        k = np.polyfit(C[:, i], G[:, i], 1)[0]
        print(f"   about {nm}: correlation {cc:+.2f}, slope {k:+.2f}; rms camera {np.sqrt(np.mean(C[:, i] ** 2)):.2f}, "
              f"gravity {np.sqrt(np.mean(G[:, i] ** 2)):.2f} deg")


if __name__ == "__main__":
    main()
