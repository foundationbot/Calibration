"""Where does step 81's spread after the fit come from on a powered-neck take? Fits camera-in-Head X and
the board Z on the per-image board poses (dump_frames.py), then splits the residual per hold into Head
axes (rotation and translation), regresses it on the neck angles, and refits on subsets: each camera
alone (each has its own lens model: a lens bias moves the two apart), without the image edge (|yaw| > 20),
and with the neck geometry free (angle between the axes, axis offset).
Usage: python split.py <take> <data dir> [<drop holds, comma-separated>]  (run from the branch's
results/phantom12-2026-10-16-tbs/analysis folder: it imports fusedfit and gravity_fit from there)"""
import sys
from pathlib import Path

import numpy as np

AN = Path.cwd()
sys.path.insert(0, str(AN))
import fusedfit as ff  # noqa: E402
from gravity_fit import CFG, URDF  # noqa: E402
import neckfk  # noqa: E402
import okviscfg  # noqa: E402


def subset(fr, keep):
    return {k: v[keep] for k, v in fr.items()}


def per_hold(m, p, res, fr):
    X, _, _ = m.unpack(p)
    rows = []
    for h in np.unique(fr["hold"]):
        ks = np.flatnonzero(fr["hold"] == h)
        R = [(X @ m.C[fr["cam"][k]])[:3, :3] for k in ks]
        rot = np.degrees(np.mean([Rk @ res[k, :3] for Rk, k in zip(R, ks)], 0))
        tr = 1e3 * np.mean([Rk @ res[k, 3:] for Rk, k in zip(R, ks)], 0)
        rows.append((h, np.degrees(np.mean(fr["yaw"][ks])), np.degrees(np.mean(fr["pitch"][ks])), *rot, *tr))
    return np.array(rows)


def main():
    take, data = sys.argv[1], Path(sys.argv[2])
    drop = {int(x) for x in sys.argv[3].split(",")} if len(sys.argv) > 3 and sys.argv[3] else set()
    cfg, chain = okviscfg.load(CFG), neckfk.NeckChain(URDF)
    fr = ff.load([data / f"{take}-frames.npz"])[0]
    fr = subset(fr, np.array([h not in drop for h in fr["hold"]]))
    print(f"{take}: {len(fr['hold'])} images, {len(np.unique(fr['hold']))} holds (dropped {sorted(drop)})")

    m = ff.Model(cfg, chain, [fr], False)
    p, res, cov, cond = ff.fit(m)
    ff.report(m, p, res, cov, cond, "all images")
    H = per_hold(m, p, res, fr)
    names = ["rot x", "rot y", "rot z", "tr x", "tr y", "tr z"]
    print("  per-hold mean residual in Head axes (deg, mm): rms, and its linear dependence on yaw / pitch (per 10 deg)")
    A = np.c_[np.ones(len(H)), H[:, 1] / 10, H[:, 2] / 10]
    for i, nm in enumerate(names):
        v = H[:, 3 + i]
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        r2 = 1 - np.sum((v - A @ c) ** 2) / max(np.sum((v - v.mean()) ** 2), 1e-12)
        print(f"    {nm}: rms {np.sqrt(np.mean(v ** 2)):.3f}; per 10 deg yaw {c[1]:+.3f}, pitch {c[2]:+.3f}; R2 {r2:.2f}")
    # within-hold scatter (image noise) vs between holds
    wr = [np.degrees(np.linalg.norm(res[k, :3] - res[fr["hold"] == fr["hold"][k]][:, :3].mean(0)))
          for k in range(len(res))]
    print(f"  within-hold image scatter: {np.sqrt(np.mean(np.square(wr))):.3f} deg rms")

    for label, keep in (("camera 0 only", fr["cam"] == 0), ("camera 1 only", fr["cam"] == 1),
                        ("|yaw| <= 20 deg", np.abs(np.degrees(fr["yaw"])) <= 20.5)):
        s = subset(fr, keep)
        ms = ff.Model(cfg, chain, [s], False)
        ps, rs, cs, cn = ff.fit(ms)
        ff.report(ms, ps, rs, cs, cn, f"{label} ({len(s['hold'])} images)")
    mn = ff.Model(cfg, chain, [fr], True)
    pn, rn, cvn, cnn = ff.fit(mn)
    ff.report(mn, pn, rn, cvn, cnn, "neck geometry free")


if __name__ == "__main__":
    main()
