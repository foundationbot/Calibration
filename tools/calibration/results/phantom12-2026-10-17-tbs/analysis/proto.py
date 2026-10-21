"""Prototype: camera-in-Head (X) from the powered-neck takes by four measurement models, on the corner
dumps (dump_corners.py). mono = each camera's own board pose (step 81 today); stereo-pose = corners
triangulated with both cameras, board pose by a rigid (Kabsch) fit; stereo-points = X and the board fitted
directly to the triangulated corners in 3D (residual in mm at the board, what stereo grasping sees); pixels
= X and the board fitted to every corner pixel of both cameras. Prints X vs the config per model, take and
camera, and each model's residual.
Usage (from the branch's results/phantom12-2026-10-16-tbs/analysis folder):
python proto.py <data dir> <take>... """
import importlib
import json
import sys
from pathlib import Path

import numpy as np

AN = Path.cwd()
sys.path.insert(0, str(AN))
import fusedfit as ff  # noqa: E402
from gravity_fit import CFG, URDF  # noqa: E402
import neckfk  # noqa: E402
import okviscfg  # noqa: E402
import rigcfg  # noqa: E402
import rigmath as rm  # noqa: E402

s81 = importlib.import_module("81_tbs_handeye")
RIG = AN.parents[2] / "config/rig-phantom12.yaml"
RADIUS = 450.0
MIN_TAGS = 20
DROP = {"tbs-neck-P1": {4, 33, 48}, "tbs-neck-P2": {49}}       # step 75's drop lists


def project_equi(Pc, K, D):
    """OpenCV fisheye (equidistant) projection of camera-frame points (N, 3) -> pixels (N, 2)."""
    a, b = Pc[:, 0] / Pc[:, 2], Pc[:, 1] / Pc[:, 2]
    r = np.hypot(a, b)
    th = np.arctan(r)
    t2 = th * th
    thd = th * (1 + D[0] * t2 + D[1] * t2 ** 2 + D[2] * t2 ** 3 + D[3] * t2 ** 4)
    s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
    x, y = s * a, s * b
    return np.c_[K[0, 0] * x + K[0, 1] * y + K[0, 2], K[1, 1] * y + K[1, 2]]


def lm(f, p0, iters=60):
    p, r = p0.copy(), f(p0)
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
            if lam > 1e12:
                return p, r, J
        p, r = p + dp, rn
        if np.linalg.norm(dp) < 1e-12:
            break
    return p, r, J


def clustered_cov(J, r, groups):
    Hi = np.linalg.inv(J.T @ J)
    M = np.zeros_like(Hi)
    labels = np.unique(groups)
    for h in labels:
        s = J[groups == h].T @ r[groups == h]
        M += np.outer(s, s)
    return len(labels) / (len(labels) - 1) * Hi @ M @ Hi


def kabsch(Pw, Pc, scale=False):
    mw, mc = Pw.mean(0), Pc.mean(0)
    U, S, Vt = np.linalg.svd((Pc - mc).T @ (Pw - mw))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    s = (S * [1, 1, d]).sum() / np.sum((Pw - mw) ** 2) if scale else 1.0
    return R, mc - s * R @ mw, s


class Take:
    def __init__(self, path, rig, cfg):
        d = dict(np.load(path))
        self.name = path.name.replace("-corners.npz", "")
        qc = path.parent / f"{self.name}-qc.json"
        drop = set(json.loads(qc.read_text())["drop_holds"]) if qc.exists() else DROP.get(self.name, set())
        self.drop = drop
        self.yaw, self.pitch, self.hold = d["img_yaw"], d["img_pitch"], d["img_hold"]
        centre = np.array(rig.resolution, float) / 2
        self.det = {}                                    # (pair, cam) -> (keys (n, 2), uv (n, 2))
        order = np.lexsort((d["det_cam"], d["det_img"]))
        keys = np.c_[d["det_img"], d["det_cam"]][order]
        cut = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
        for blk in np.split(order, cut):
            n, c = int(d["det_img"][blk[0]]), int(d["det_cam"][blk[0]])
            uv = np.c_[d["det_u"][blk], d["det_v"][blk]]
            ok = np.hypot(*(uv - centre).T) <= RADIUS
            self.det[(n, c)] = (np.c_[d["det_tag"][blk], d["det_k"][blk]][ok], uv[ok])
        self.pairs = [n for n in range(len(self.yaw)) if self.hold[n] not in drop]


def obj(rig, keys):
    return np.array([rig.board.obj_points(int(t))[int(k)] for t, k in keys])


def mono_frames(take, rig, cfg, lens, cams):
    out = {k: [] for k in ("hold", "cam", "yaw", "pitch", "T_CW")}
    centre = tuple(np.array(rig.resolution, float) / 2)
    for n in take.pairs:
        for c in cams:
            if (n, c) not in take.det:
                continue
            keys, uv = take.det[(n, c)]
            det = {(int(t), int(k)): (u, v) for (t, k), (u, v) in zip(keys, uv)}
            T = s81.pose_from_corners(det, rig.board, lens, cfg.K[c], cfg.D[c], centre)
            if T is None:
                continue
            for k, v in (("hold", take.hold[n]), ("cam", c), ("yaw", take.yaw[n]), ("pitch", take.pitch[n]), ("T_CW", T)):
                out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


def triangulate(take, rig, cfg, lens):
    """-> per pair: (pair, board points (n, 3), camera-0 points (n, 3)) for corners seen by both cameras."""
    import cv2
    T01 = rm.inv(cfg.T_SC[0]) @ cfg.T_SC[1]              # camera 1 in camera 0
    P1 = rm.inv(T01)[:3]
    out = []
    for n in take.pairs:
        if (n, 0) not in take.det or (n, 1) not in take.det:
            continue
        (k0, uv0), (k1, uv1) = take.det[(n, 0)], take.det[(n, 1)]
        i1 = {(int(t), int(k)): i for i, (t, k) in enumerate(k1)}
        both = [(i, i1[(int(t), int(k))]) for i, (t, k) in enumerate(k0) if (int(t), int(k)) in i1]
        if len({int(k0[i][0]) for i, _ in both}) < MIN_TAGS:
            continue
        a, b = np.array([i for i, _ in both]), np.array([j for _, j in both])
        x0 = lens.undistort(uv0[a].reshape(-1, 1, 2), cfg.K[0], cfg.D[0]).reshape(-1, 2)
        x1 = lens.undistort(uv1[b].reshape(-1, 1, 2), cfg.K[1], cfg.D[1]).reshape(-1, 2)
        X = cv2.triangulatePoints(np.eye(4)[:3], P1, x0.T, x1.T)
        out.append((n, obj(rig, k0[a]), (X[:3] / X[3]).T))
    return out


def report(label, X, X0, cov=None, res_txt=""):
    w = np.degrees(rm.log_rot(X[:3, :3] @ X0[:3, :3].T))
    dt = 1e3 * (X[:3, 3] - X0[:3, 3])
    s = ""
    if cov is not None:
        R = X[:3, :3]
        sr = np.degrees(np.sqrt(np.diag(R @ cov[:3, :3] @ R.T)))
        st = 1e3 * np.sqrt(np.diag(R @ cov[3:6, 3:6] @ R.T))
        s = f" +- {np.linalg.norm(sr):.3f} deg / {np.linalg.norm(st):.2f} mm"
    print(f"  {label:34s} {np.linalg.norm(w):.3f} deg ({w[0]:+.3f} {w[1]:+.3f} {w[2]:+.3f}), "
          f"{np.linalg.norm(dt):5.1f} mm ({dt[0]:+5.1f} {dt[1]:+5.1f} {dt[2]:+5.1f}){s}  {res_txt}")
    return w, dt


def pose_fit(frames_list, cfg, chain, label, X0):
    m = ff.Model(cfg, chain, frames_list, False)
    p, res, cov, cond = ff.fit(m)
    X, _, _ = m.unpack(p)
    rr = np.degrees(np.sqrt(np.mean(np.sum(res[:, :3] ** 2, 1))))
    rt = 1e3 * np.sqrt(np.mean(np.sum(res[:, 3:] ** 2, 1)))
    report(label, X, X0, cov, f"residual {rr:.3f} deg / {rt:.2f} mm")
    return X


def point_fit(tri_list, takes, chain, X0, label):
    """X, Z_k minimising |A X p_C0 - Z_k P_W| over every triangulated corner (torso frame, metres)."""
    Z0 = []
    for tri, tk in zip(tri_list, takes):
        n, Pw, Pc = tri[0]
        R, t, _ = kabsch(Pw, Pc)
        Z0.append(chain.T_TB(tk.yaw[n], tk.pitch[n]) @ X0 @ rm.transform(R, t))
    A = [[chain.T_TB(tk.yaw[n], tk.pitch[n]) for n, _, _ in tri] for tri, tk in zip(tri_list, takes)]
    groups = np.concatenate([np.repeat([f"{k}-{tk.hold[n]}" for n, _, Pc in tri for _ in range(1)],
                                       [3 * len(Pc) for _, _, Pc in tri]) for k, (tri, tk) in enumerate(zip(tri_list, takes))])

    def f(p):
        X = X0 @ ff.step(p[:6])
        out = []
        for k, tri in enumerate(tri_list):
            Z = Z0[k] @ ff.step(p[6 + 6 * k:12 + 6 * k])
            for Ai, (_, Pw, Pc) in zip(A[k], tri):
                M = Ai @ X
                out.append(((M[:3, :3] @ Pc.T).T + M[:3, 3]) - ((Z[:3, :3] @ Pw.T).T + Z[:3, 3]))
        return np.concatenate(out).ravel()

    p, r, J = lm(f, np.zeros(6 + 6 * len(tri_list)))
    keep = np.abs(r) < 5 * 1.4826 * np.median(np.abs(r))
    cov = clustered_cov(J[keep], r[keep], groups[keep])
    X = X0 @ ff.step(p[:6])
    per_pt = np.linalg.norm(r.reshape(-1, 3), axis=1) * 1e3
    report(label, X, X0, cov, f"3D residual rms {np.sqrt(np.mean(per_pt ** 2)):.2f} mm, median {np.median(per_pt):.2f}")
    return X, r


def pixel_fit(takes, rig, cfg, chain, X0, cams, label):
    """X, Z_k minimising the pixel reprojection of every corner in the given cameras."""
    C = [rm.inv(cfg.T_SC[0]) @ T for T in cfg.T_SC]
    items, Z0 = [], []
    for k, tk in enumerate(takes):
        tri = triangulate(tk, rig, cfg, rigcfg.Lens(cfg.lens))
        n0, Pw, Pc = tri[0]
        R, t, _ = kabsch(Pw, Pc)
        Z0.append(chain.T_TB(tk.yaw[n0], tk.pitch[n0]) @ X0 @ rm.transform(R, t))
        for n in tk.pairs:
            for c in cams:
                if (n, c) in tk.det and len({int(t) for t, _ in tk.det[(n, c)][0]}) >= MIN_TAGS:
                    keys, uv = tk.det[(n, c)]
                    items.append((k, c, chain.T_TB(tk.yaw[n], tk.pitch[n]), obj(rig, keys), uv, f"{k}-{tk.hold[n]}"))
    groups = np.concatenate([np.repeat(g, 2 * len(uv)) for *_, uv, g in items])

    def f(p):
        X = X0 @ ff.step(p[:6])
        out = []
        for k, c, Ai, Pw, uv in ((it[0], it[1], it[2], it[3], it[4]) for it in items):
            Z = Z0[k] @ ff.step(p[6 + 6 * k:12 + 6 * k])
            T = rm.inv(Ai @ X @ C[c]) @ Z
            out.append((project_equi((T[:3, :3] @ Pw.T).T + T[:3, 3], cfg.K[c], cfg.D[c]) - uv).ravel())
        return np.concatenate(out)

    p, r, J = lm(f, np.zeros(6 + 6 * len(takes)), iters=30)
    keep = np.abs(r) < 5 * 1.4826 * np.median(np.abs(r))
    cov = clustered_cov(J[keep], r[keep], groups[keep])
    X = X0 @ ff.step(p[:6])
    rr = np.sqrt(np.mean(np.sum(r.reshape(-1, 2) ** 2, 1)))
    report(label, X, X0, cov, f"pixel rms {rr:.3f} px")
    return X


def main():
    import cv2
    data = Path(sys.argv[1])
    rig, cfg, chain = rigcfg.load(RIG), okviscfg.load(CFG), neckfk.NeckChain(URDF)
    lens = rigcfg.Lens(cfg.lens)
    X0 = cfg.T_BS @ cfg.T_SC[0]
    # the numpy projection must match OpenCV's
    P = np.random.default_rng(0).uniform([-0.4, -0.3, 0.8], [0.4, 0.3, 1.4], (50, 3))
    ref, _ = cv2.fisheye.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cfg.K[0], cfg.D[0])
    assert np.abs(project_equi(P, cfg.K[0], cfg.D[0]) - ref.reshape(-1, 2)).max() < 1e-6
    takes = [Take(data / f"{t}-corners.npz", rig, cfg) for t in sys.argv[2:]]
    print(f"X = camera 0 in the Head link, vs the config (T_BS T_SC0): angle (Head x y z), mm (Head x y z) +- 1 sigma")
    tris = []
    for tk in takes:
        print(f"{tk.name}: {len(tk.pairs)} stereo pairs (holds left out: {sorted(tk.drop)})")
        m0, m1 = mono_frames(tk, rig, cfg, lens, [0]), mono_frames(tk, rig, cfg, lens, [1])
        mb = {k: np.concatenate([m0[k], m1[k]]) for k in m0}
        pose_fit([mb], cfg, chain, "mono, both cameras", X0)
        pose_fit([m0], cfg, chain, "mono, camera 0", X0)
        pose_fit([m1], cfg, chain, "mono, camera 1", X0)
        tri = triangulate(tk, rig, cfg, lens)
        tris.append(tri)
        sc, rs = [], []
        st = {k: [] for k in ("hold", "cam", "yaw", "pitch", "T_CW")}
        for n, Pw, Pc in tri:
            R, t, s = kabsch(Pw, Pc, scale=True)
            sc.append(s)
            R, t, _ = kabsch(Pw, Pc)
            rs.append(np.sqrt(np.mean(np.sum(((R @ Pw.T).T + t - Pc) ** 2, 1))) * 1e3)
            for k, v in (("hold", tk.hold[n]), ("cam", 0), ("yaw", tk.yaw[n]), ("pitch", tk.pitch[n]), ("T_CW", rm.transform(R, t))):
                st[k].append(v)
        print(f"  triangulated board: scale vs the board {np.median(sc):.5f} (IQR {np.subtract(*np.percentile(sc, [75, 25])):.5f}), "
              f"rigid-fit residual median {np.median(rs):.2f} mm over {len(tri)} pairs")
        pose_fit([{k: np.array(v) for k, v in st.items()}], cfg, chain, "stereo-pose", X0)
        point_fit([tri], [tk], chain, X0, "stereo-points")
        pixel_fit([tk], rig, cfg, chain, X0, [0, 1], "pixels, both cameras")
        pixel_fit([tk], rig, cfg, chain, X0, [0], "pixels, camera 0")
        pixel_fit([tk], rig, cfg, chain, X0, [1], "pixels, camera 1")
    if len(takes) > 1:
        print("both takes together (one X, a board pose per take):")
        point_fit(tris, takes, chain, X0, "stereo-points")
        pixel_fit(takes, rig, cfg, chain, X0, [0, 1], "pixels, both cameras")


if __name__ == "__main__":
    main()
