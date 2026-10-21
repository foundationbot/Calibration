"""Fused hand-eye fit over several neck takes, optionally with the neck geometry free.

Unknowns: X = camera 0 in the Head link (shared), Z_k = board in the torso for take k (the torso was
re-locked between takes), and with --neck: the pitch joint's origin in NeckYawLink tilted by delta
about x (the angle between the two neck axes becomes 90 deg - delta) and shifted by dx (the offset
between the axes, perpendicular to both). What a neck-only test cannot see is left fixed: the yaw axis's direction
and position in the torso and both encoder zeros (absorbed by Z_k and X), the pitch origin's shift
along the pitch axis (absorbed by X) and along the yaw axis (commutes with the yaw: absorbed by Z_k).
Usage: python fusedfit.py <config> <urdf> <take.npz>... [--neck] [--synthetic]"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))   # tools/calibration/scripts of this checkout
import handeye  # noqa: E402
import neckfk  # noqa: E402
import okviscfg  # noqa: E402
import rigmath as rm  # noqa: E402


def step(d):
    return rm.transform(rm.exp_rot(d[:3]), d[3:])


class Model:
    def __init__(self, cfg, chain, takes, neck):
        self.cfg, self.chain, self.takes, self.neck = cfg, chain, takes, neck
        self.C = [rm.inv(cfg.T_SC[0]) @ T for T in cfg.T_SC]
        self.X0 = cfg.T_BS @ cfg.T_SC[0]
        self.Z0 = []
        for t in takes:                                   # each take's board from its first image
            A = chain.T_TB(t["yaw"][0], t["pitch"][0])
            self.Z0.append(A @ self.X0 @ self.C[t["cam"][0]] @ rm.inv(rm.inv(t["T_CW"][0])))
        self.n = 6 + 6 * len(takes) + (2 if neck else 0)

    def unpack(self, p):
        X = self.X0 @ step(p[:6])
        Z = [Z0 @ step(p[6 + 6 * k:12 + 6 * k]) for k, Z0 in enumerate(self.Z0)]
        nk = p[6 + 6 * len(self.takes):] if self.neck else np.zeros(2)
        return X, Z, nk

    def T_TB(self, y, pch, nk):
        c = self.chain
        Op = c.pitch.origin
        Opm = rm.transform(rm.rot_axis([1, 0, 0], nk[0]) @ Op[:3, :3], Op[:3, 3] + np.array([nk[1], 0.0, 0.0]))
        return (c.yaw.origin @ rm.transform(rm.rot_axis(c.yaw.axis, y), np.zeros(3))
                @ Opm @ rm.transform(rm.rot_axis(c.pitch.axis, pch), np.zeros(3)))

    def residuals(self, p):
        X, Z, nk = self.unpack(p)
        out = []
        for k, t in enumerate(self.takes):
            for y, pch, cam, T_CW in zip(t["yaw"], t["pitch"], t["cam"], t["T_CW"]):
                E = rm.inv(Z[k] @ rm.inv(T_CW)) @ (self.T_TB(y, pch, nk) @ X @ self.C[cam])
                out.append(np.r_[rm.log_rot(E[:3, :3]), E[:3, 3]])
        return np.array(out).ravel()

    def groups(self):
        return np.concatenate([[f"{k}-{h}" for h in t["hold"]] for k, t in enumerate(self.takes)])


def jac(m, p, eps=1e-6):
    J = np.empty((m.residuals(p).size, m.n))
    for i in range(m.n):
        d = np.zeros(m.n)
        d[i] = eps
        J[:, i] = (m.residuals(p + d) - m.residuals(p - d)) / (2 * eps)
    return J


def fit(m, iters=40):
    p = np.zeros(m.n)
    r = m.residuals(p)
    lam = 1e-3
    for _ in range(iters):
        J = jac(m, p)
        H, g = J.T @ J, J.T @ r
        while True:
            d = -np.linalg.solve(H + lam * np.diag(np.diag(H)), g)
            rn = m.residuals(p + d)
            if rn @ rn < r @ r:
                lam = max(lam / 10, 1e-9)
                break
            lam *= 10
            if lam > 1e9:
                break
        if lam > 1e9:
            break
        p, r = p + d, rn
        if np.linalg.norm(d) < 1e-10:
            break
    J = jac(m, p)
    Hi = np.linalg.inv(J.T @ J)
    grp = np.repeat(m.groups(), 6)
    M = np.zeros((m.n, m.n))
    labels = np.unique(grp)
    for h in labels:
        s = J[grp == h].T @ r[grp == h]
        M += np.outer(s, s)
    cov = len(labels) / (len(labels) - 1) * Hi @ M @ Hi
    sv = np.linalg.svd(J, compute_uv=False)
    return p, r.reshape(-1, 6), cov, sv[0] / sv[-1]


def report(m, p, res, cov, cond, label):
    X, Z, nk = m.unpack(p)
    R = X[:3, :3]
    sr = np.degrees(np.sqrt(np.diag(R @ cov[:3, :3] @ R.T)))
    st = 1e3 * np.sqrt(np.diag(R @ cov[3:6, 3:6] @ R.T))
    w = np.degrees(rm.log_rot(X[:3, :3] @ m.X0[:3, :3].T))
    dt = 1e3 * (X[:3, 3] - m.X0[:3, 3])
    rr = np.degrees(np.sqrt(np.mean(np.sum(res[:, :3] ** 2, 1))))
    rt = 1e3 * np.sqrt(np.mean(np.sum(res[:, 3:] ** 2, 1)))
    print(f"{label}: X vs config {np.linalg.norm(w):.3f} deg ({w[0]:+.3f} {w[1]:+.3f} {w[2]:+.3f}) +- ({sr[0]:.3f} {sr[1]:.3f} {sr[2]:.3f}), "
          f"{np.linalg.norm(dt):.1f} mm ({dt[0]:+.1f} {dt[1]:+.1f} {dt[2]:+.1f}) +- ({st[0]:.1f} {st[1]:.1f} {st[2]:.1f}); "
          f"residual rms {rr:.3f} deg / {rt:.2f} mm; cond {cond:.0f}")
    if m.neck:
        i = 6 + 6 * len(m.takes)
        sn = np.sqrt(np.diag(cov))[i:]
        print(f"    neck: angle between axes {90 - np.degrees(nk[0]):.3f} deg (+- {np.degrees(sn[0]):.3f}), "
              f"pitch axis offset dx {1e3 * nk[1]:+.1f} +- {1e3 * sn[1]:.1f} mm")
    return X


def load(paths):
    out = []
    for q in paths:
        d = np.load(q)
        out.append({k: d[k] for k in d.files})
    return out


def synthetic(cfg, chain, takes, delta_deg, dx, rot_err_deg, noise_rad, noise_m, seed=0):
    """Real poses, images made by a neck with the given geometry and X off the config by rot_err_deg."""
    rng = np.random.default_rng(seed)
    m = Model(cfg, chain, takes, neck=True)
    X_true = m.X0 @ rm.transform(rm.rot_axis([1, 1, 0], np.radians(rot_err_deg)), [0.002, -0.001, 0.003])
    nk = np.array([np.radians(delta_deg), dx])
    out = []
    for k, t in enumerate(takes):
        Z = m.T_TB(0, 0, nk) @ X_true @ rm.transform(np.eye(3), [0.1 * k, 0, 1.1])
        T_CW = []
        for y, pch, cam in zip(t["yaw"], t["pitch"], t["cam"]):
            B = rm.inv(Z) @ m.T_TB(y, pch, nk) @ X_true @ m.C[cam]
            B = B @ rm.transform(rm.exp_rot(noise_rad * rng.standard_normal(3)), noise_m * rng.standard_normal(3))
            T_CW.append(rm.inv(B))
        out.append(dict(t, T_CW=np.array(T_CW)))
    return out, X_true


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg, chain = okviscfg.load(Path(args[0])), neckfk.NeckChain(Path(args[1]))
    takes = load(args[2:])
    names = [Path(a).stem.replace("-frames", "") for a in args[2:]]
    if "--synthetic" in sys.argv:
        takes, X_true = synthetic(cfg, chain, takes, 1.2, 0.003, 0.4, 3e-4, 3e-4)
        w = np.degrees(rm.log_rot(X_true[:3, :3] @ (cfg.T_BS @ cfg.T_SC[0])[:3, :3].T))
        print(f"SYNTHETIC truth: X vs config {np.linalg.norm(w):.3f} deg ({w[0]:+.3f} {w[1]:+.3f} {w[2]:+.3f}); "
              "angle between axes 88.800 deg, dx +3.0 mm")
    Xs = {}
    for neck in (False, True):
        tag = "neck free" if neck else "URDF neck"
        for k, n in enumerate(names):
            m = Model(cfg, chain, [takes[k]], neck)
            p, res, cov, cond = fit(m)
            Xs[(neck, n)] = report(m, p, res, cov, cond, f"[{tag}] {n} alone")
        if len(takes) > 1:
            m = Model(cfg, chain, takes, neck)
            p, res, cov, cond = fit(m)
            report(m, p, res, cov, cond, f"[{tag}] fused {'+'.join(names)}")
            a, b = Xs[(neck, names[0])], Xs[(neck, names[1])]
            ang = rm.rot_angle_deg(a[:3, :3] @ b[:3, :3].T)
            print(f"    {names[0]} vs {names[1]} ({tag}): {ang:.3f} deg, {1e3 * np.linalg.norm(a[:3, 3] - b[:3, 3]):.1f} mm "
                  "(plan: agree within 0.1 deg / 1 mm)")


if __name__ == "__main__":
    main()
