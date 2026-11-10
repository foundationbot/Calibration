"""The fast stereo lens solve (M3): stereofit.solve's problem and contract, solved by Levenberg-Marquardt with an exact
Jacobian and the per-pair board poses eliminated by Schur complement.

Same model and parameters as stereofit (per camera fx fy cx cy k1..k4 [p1 p2]; the stereo pose C, camera-1 points into
camera 0; one board pose per stereo pair; per pair a log board scale when scale=True), the same residuals (pixels,
1e3 for a non-finite component or a corner at z <= 1e-6, with a zero Jacobian there as finite differences see it) and
the same trim rounds. Only the minimiser differs: scipy's TRF with a finite-difference Jacobian is replaced by

- the exact Jacobian: KB4/KB4T derivatives in closed form (project_jac); any other lens model lensmodel knows (RADTAN
  registered by calibtool.livecfg) only with the camera fixed, its 2x3 point derivative by central differences;
- local updates: C <- C [Exp(dr) | dt] (stereofit's C0 @ [rotmat(dC[:3]) | dC[3:]] at dC = 0), board R <- Exp(dr) R,
  t <- t + dt, log scale += d;
- Marquardt damping on the Jacobi-scaled normal equations (the x_scale="jac" equivalent) with Nielsen's update;
- the 6 (7 with scale) numbers of each pair eliminated per iteration: the linear system is the global block only
  (2 x 10 intrinsics + 6 stereo pose = 26 for KB4T). With the camera fixed there is no global block: the pairs are
  independent and solved as a vector of small LMs, each with its own damping and stop.

A solve "succeeds" as scipy's does: a convergence test met before max_nfev residual evaluations (one evaluation per
trial step). The stopping tests are tighter than scipy's defaults (relative cost decrease < 1e-12 on a good step,
scaled gradient or step at rounding level), so a converged fast solve sits on the minimum scipy approaches. A solve (a
pair, with the camera fixed) that runs to max_nfev, or stalls at full damping, without meeting them still succeeds if
the solve met scipy's own ftol test on the way (a good step, ratio > 0.25, relative decrease of the whole problem's
cost < 1e-8 -- with the camera fixed: summed over the pairs, as scipy solves them as one problem): where scipy would
have stopped and reported success; it ran on past that point only for accuracy. Without this a pair whose minimum
runs off (a handful of corners left by trimming, a board scale running away with the cost still falling) fails the
fast solve where scipy reports success (measured 9 Nov).

Switched on by CALIB_SOLVER=fast (stereofit.solve and lensscore.score read it at each call); default scipy."""
from __future__ import annotations

import numpy as np

from calibtool import paths  # noqa: F401  (puts scripts/ on sys.path)
import lensmodel  # noqa: E402

ENV = "CALIB_SOLVER"
KB_MODELS = ("KB4", "KB4T")
TRIM_PX, TRIM_ROUNDS = 3.0, 4                      # stereofit.TRIM_PX, TRIM_ROUNDS
FTOL = 1e-12                                       # relative cost decrease on a good step
SCIPY_FTOL = 1e-8                                  # scipy least_squares' default ftol: its success test (see above)
GTOL = 1e-10                                       # scaled gradient (px), relative to the residual norm
XTOL = 1e-12                                       # scaled step (px), relative to the residual norm
LAM0, LAM_MAX = 1e-3, 1e16
FD_REL = 1e-6                                      # central-difference step of the point derivative (relative)


# ---------------------------------------------------------------- projection with derivatives


def project_jac(model, intr, P):
    """Camera-frame points (..., 3) -> (uv (..., 2), d uv / d P (..., 2, 3), d uv / d intr (..., 2, ni) or None).
    KB4/KB4T: closed form, uv computed as lensmodel.project computes it. Other models (registered in lensmodel): uv by
    lensmodel.project, d uv / d P by central differences, no intrinsic derivative."""
    P = np.asarray(P, float)
    if model not in KB_MODELS:
        return _project_jac_generic(model, intr, P)
    intr = np.asarray(intr, float)
    fx, fy, cx, cy = intr[..., 0], intr[..., 1], intr[..., 2], intr[..., 3]
    k1, k2, k3, k4 = intr[..., 4], intr[..., 5], intr[..., 6], intr[..., 7]
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho2 = X * X + Y * Y
        rho = np.hypot(X, Y)
        th = np.arctan2(rho, Z)
        t2 = th * th
        thd = th * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
        s = np.where(rho > 1e-12, thd / np.maximum(rho, 1e-12), 1.0 / Z)
        mx, my = s * X, s * Y
        r2t = rho2 + Z * Z
        dthd = 1 + 3 * k1 * t2 + 5 * k2 * t2 ** 2 + 7 * k3 * t2 ** 3 + 9 * k4 * t2 ** 4
        on = rho > 1e-12
        rs = np.where(on, rho, 1.0)
        a = np.where(on, dthd * Z / (rs * rs * r2t) - thd / (rs * rs * rs), 0.0)      # ds/dX = a X, ds/dY = a Y
        sZ = -dthd / r2t                                                              # ds/dZ * rho / rho
        dm = np.empty(P.shape[:-1] + (2, 3))
        dm[..., 0, 0], dm[..., 0, 1], dm[..., 0, 2] = s + a * X * X, a * X * Y, X * sZ
        dm[..., 1, 0], dm[..., 1, 1], dm[..., 1, 2] = a * X * Y, s + a * Y * Y, Y * sZ
        # d thd / d k_i = th t2^i; mx = (X / rho) thd
        ux, uy = np.where(on, X / rs, 0.0), np.where(on, Y / rs, 0.0)
        dk = np.stack([th * t2, th * t2 ** 2, th * t2 ** 3, th * t2 ** 4], axis=-1)    # (..., 4)
        dmk = np.stack([ux[..., None] * dk, uy[..., None] * dk], axis=-2)              # (..., 2, 4)
    ni = intr.shape[-1]
    if model == "KB4T":
        p1, p2 = intr[..., 8], intr[..., 9]
        r2 = mx * mx + my * my
        mxt = mx + 2 * p1 * mx * my + p2 * (r2 + 2 * mx * mx)
        myt = my + p1 * (r2 + 2 * my * my) + 2 * p2 * mx * my
        Jt = np.empty(P.shape[:-1] + (2, 2))
        Jt[..., 0, 0] = 1 + 2 * p1 * my + 6 * p2 * mx
        Jt[..., 0, 1] = Jt[..., 1, 0] = 2 * p1 * mx + 2 * p2 * my
        Jt[..., 1, 1] = 1 + 6 * p1 * my + 2 * p2 * mx
        dm = Jt @ dm
        dmk = Jt @ dmk
        dmp = np.empty(P.shape[:-1] + (2, 2))
        dmp[..., 0, 0], dmp[..., 1, 0] = 2 * mx * my, r2 + 2 * my * my              # d / d p1
        dmp[..., 0, 1], dmp[..., 1, 1] = r2 + 2 * mx * mx, 2 * mx * my              # d / d p2
        mx, my = mxt, myt
    f = np.stack([fx * np.ones_like(X), fy * np.ones_like(X)], axis=-1)[..., None]    # (..., 2, 1)
    uv = np.stack([fx * mx + cx, fy * my + cy], axis=-1)
    dI = np.zeros(P.shape[:-1] + (2, ni))
    dI[..., 0, 0], dI[..., 1, 1] = mx, my
    dI[..., 0, 2] = dI[..., 1, 3] = 1.0
    dI[..., 4:8] = f * dmk
    if model == "KB4T":
        dI[..., 8:10] = f * dmp
    return uv, f * dm, dI


def _project_jac_generic(model, intr, P):
    uv = lensmodel.project(model, intr, P.reshape(-1, 3)).reshape(P.shape[:-1] + (2,))
    dP = np.empty(P.shape[:-1] + (2, 3))
    h = FD_REL * np.maximum(np.linalg.norm(P, axis=-1), 1e-9)
    for j in range(3):
        e = np.zeros_like(P)
        e[..., j] = h
        up = lensmodel.project(model, intr, (P + e).reshape(-1, 3)).reshape(uv.shape)
        um = lensmodel.project(model, intr, (P - e).reshape(-1, 3)).reshape(uv.shape)
        dP[..., j] = (up - um) / (2 * h[..., None])
    return uv, dP, None


# ---------------------------------------------------------------- rotations


def expmap(r):
    """rotation vectors (..., 3) -> matrices (..., 3, 3) (Rodrigues, stereofit.rotmat's formula)"""
    r = np.asarray(r, float)
    th = np.linalg.norm(r, axis=-1)
    k = r / np.where(th > 1e-12, th, 1.0)[..., None]
    K = np.zeros(r.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    s, c = np.sin(th)[..., None, None], np.cos(th)[..., None, None]
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    R[th < 1e-12] = np.eye(3)
    return R


def rotvec(R):
    """matrices (n, 3, 3) -> rotation vectors (n, 3): rigmath.log_rot vectorised (no angle threshold: the
    antisymmetric part under 90 deg, the symmetric part over it)"""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    w = 0.5 * np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    s = np.linalg.norm(w, axis=1)
    c = (np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0
    a = np.arctan2(s, c)
    out = w * np.where(s > 0, a / np.where(s > 0, s, 1.0), 1.0)[:, None]
    big = c <= 0.0
    if big.any():
        Rb = R[big]
        B = (Rb + np.transpose(Rb, (0, 2, 1))) / 2.0 - c[big, None, None] * np.eye(3)
        k = np.argmax(np.diagonal(B, axis1=1, axis2=2), axis=1)
        n = B[np.arange(len(k)), :, k]
        n /= np.linalg.norm(n, axis=1)[:, None]
        sgn = np.where(np.einsum("ij,ij->i", n, w[big]) >= 0.0, 1.0, -1.0)
        out[big] = (a[big] * sgn)[:, None] * n
    return out


def _skew(v):
    """vectors (..., 3) -> cross-product matrices (..., 3, 3)"""
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = v[..., 2], -v[..., 1], v[..., 0]
    return K


# ---------------------------------------------------------------- the problem


class State:
    """the solve's parameters: intr (2, ni), C (4, 4), board rotations R (n, 3, 3), translations t (n, 3), log scales
    ls (n,)"""

    def __init__(self, intr, C, R, t, ls):
        self.intr, self.C, self.R, self.t, self.ls = intr, C, R, t, ls

    @classmethod
    def start(cls, intr, C, poses, ls=None):
        poses = np.asarray(poses, float).reshape(-1, 6)
        n = len(poses)
        return cls(np.array([np.asarray(v, float) for v in intr]), np.array(C, float), expmap(poses[:, :3]),
                   poses[:, 3:].copy(), np.zeros(n) if ls is None else np.asarray(ls, float).copy())

    def poses(self):
        return np.c_[rotvec(self.R), self.t]

    def step(self, dg, dp, free_intr, free_C, scale):
        """the state moved by the global step dg (intrinsics of both cameras, then the stereo pose's rotation and
        translation) and the per-pair steps dp (n, 6|7: rotation, translation, log scale)"""
        intr, C, i = self.intr, self.C, 0
        if free_intr:
            ni = intr.shape[1]
            intr = intr + np.asarray(dg[:2 * ni]).reshape(2, ni)
            i = 2 * ni
        if free_C:
            D = np.eye(4)
            D[:3, :3], D[:3, 3] = expmap(dg[i:i + 3]), dg[i + 3:i + 6]
            C = C @ D
        R = expmap(dp[:, :3]) @ self.R
        ls = self.ls + dp[:, 6] if scale else self.ls
        return State(intr, C, R, self.t + dp[:, 3:6], ls)


class Problem:
    """the corners of a stereofit.Data packed per pair (n pairs x M slots, padded): residuals and Jacobian blocks"""

    def __init__(self, data, model):
        self.model, self.n, self.N = model, int(data.n), len(data.P)
        order = np.argsort(data.pair, kind="stable")
        pair = data.pair[order]
        counts = np.bincount(pair, minlength=self.n)
        self.M = M = max(int(counts.max()) if len(counts) else 1, 1)
        start = np.r_[0, np.cumsum(counts)[:-1]]
        idx = np.full((self.n, M), -1)
        idx[pair, np.arange(len(order)) - start[pair]] = order
        self.idx, self.valid = idx, idx >= 0
        safe = np.where(self.valid, idx, 0)
        self.P = np.where(self.valid[..., None], data.P[safe], 0.0)
        self.uv = data.uv[safe]
        self.cam1 = self.valid & (data.cam[safe] == 1)
        self.cam0 = self.valid & (data.cam[safe] == 0)

    def padded(self, per_corner):
        """a per-corner bool (N,) -> (n, M), False in the padding"""
        return self.valid & np.asarray(per_corner, bool)[np.where(self.valid, self.idx, 0)]

    def residuals(self, st):
        """residuals (N, 2) in the data's corner order, clamps included (stereofit.residuals)"""
        r = self.linearise(st, False, False, False, self.valid, jac=False).reshape(self.n, self.M, 2)
        out = np.empty((self.N, 2))
        out[self.idx[self.valid]] = r[self.valid]
        return out

    def linearise(self, st, free_intr, free_C, scale, w, jac=True, pairs=None):
        """residuals (n, 2M) times the weight w (n, M) and, with jac, the Jacobian: global block (n, 2M, G) (both
        cameras' intrinsics if free_intr, then the stereo pose if free_C) and per-pair block (n, 2M, 6|7). pairs:
        only these pairs (index array)"""
        sel = slice(None) if pairs is None else pairs
        P, uv, c0, c1 = self.P[sel], self.uv[sel], self.cam0[sel], self.cam1[sel]
        R, t, ls, w = st.R[sel], st.t[sel], st.ls[sel], w[sel]
        n, M = P.shape[:2]
        Q = (P * np.exp(ls)[:, None, None]) @ np.transpose(R, (0, 2, 1))       # R (P e^ls)
        Pc0 = Q + t[:, None, :]
        Rc, tc = st.C[:3, :3], st.C[:3, 3]
        Ci = np.linalg.inv(st.C)                                                # stereofit's camera-1 transform
        Pc1 = Pc0 @ Ci[:3, :3].T + Ci[:3, 3]
        Pc = np.where(c1[..., None], Pc1, Pc0)
        ni = st.intr.shape[1]
        r = np.zeros((n, M, 2))
        dP = np.zeros((n, M, 2, 3)) if jac else None
        dI = np.zeros((n, M, 2, ni)) if jac and free_intr else None
        for c, m in ((0, c0), (1, c1)):
            if not m.any():
                continue
            if jac:
                u, dPc, dIc = project_jac(self.model, st.intr[c], Pc[m])
                dP[m] = dPc
                if free_intr:
                    if dIc is None:
                        raise NotImplementedError(f"free intrinsics need {'/'.join(KB_MODELS)}, not {self.model}")
                    dI[m] = dIc
            else:
                u = lensmodel.project(self.model, st.intr[c], Pc[m])
            r[m] = u - uv[m]
        bad = ~np.isfinite(r)
        r[bad] = 1e3
        behind = Pc[..., 2] <= 1e-6
        r[behind] = 1e3
        ok = w[..., None] & ~bad & ~behind[..., None]                           # (n, M, 2): rows that count
        r = np.where(w[..., None], r, 0.0)
        if not jac:
            return r.reshape(n, 2 * M)
        dP = np.where(ok[..., None], dP, 0.0)
        A = np.where(c1[..., None, None], Rc.T, np.eye(3))                      # d Pc / d Pc0
        dPA = dP @ A                                                            # d uv / d Pc0
        k = 6 + bool(scale)
        Jp = np.empty((n, M, 2, k))
        Jp[..., :3] = -dPA @ _skew(Q)                                           # R <- Exp(dr) R
        Jp[..., 3:6] = dPA
        if scale:
            Jp[..., 6] = (dPA @ Q[..., None])[..., 0]
        G = 2 * ni * bool(free_intr) + 6 * bool(free_C)
        Jg = np.zeros((n, M, 2, G))
        if free_intr:
            dI = np.where(ok[..., None], dI, 0.0)
            Jg[..., :ni] = np.where(c0[..., None, None], dI, 0.0)
            Jg[..., ni:2 * ni] = np.where(c1[..., None, None], dI, 0.0)
        if free_C:
            i = 2 * ni * bool(free_intr)
            JC = np.concatenate([dP @ _skew(Pc1), -dP], axis=-1)                # C <- C [Exp(dr) | dt]
            Jg[..., i:i + 6] = np.where(c1[..., None, None], JC, 0.0)
        return r.reshape(n, 2 * M), Jg.reshape(n, 2 * M, G), Jp.reshape(n, 2 * M, k)


# ---------------------------------------------------------------- Levenberg-Marquardt


def _diag_scale(H):
    """Jacobi scale sqrt(diag H) (..., k), 1 where a column is empty (a pair with every corner trimmed)"""
    d = np.sqrt(np.diagonal(H, axis1=-2, axis2=-1))
    return np.where(d > 0, d, 1.0)


def _lm_global(prob, st, w, free_intr, free_C, scale, max_nfev):
    """LM over the global block and every pair's block at once, the pairs eliminated by Schur complement.
    -> (state, success, evaluations)"""
    lin = prob.linearise(st, free_intr, free_C, scale, w)
    nfev, lam, nu = 1, LAM0, 2.0
    loose = False                                                           # scipy's ftol test met on the way
    while True:
        r, Jg, Jp = lin
        cost = 0.5 * float(r.ravel() @ r.ravel())
        G, k = Jg.shape[-1], Jp.shape[-1]
        Jg2 = Jg.reshape(-1, G)
        JpT = np.transpose(Jp, (0, 2, 1))
        Hgg, gg = Jg2.T @ Jg2, Jg2.T @ r.ravel()
        Hgp = np.transpose(Jg, (0, 2, 1)) @ Jp                                 # (n, G, k)
        Hpp, gp = JpT @ Jp, (JpT @ r[..., None])[..., 0]                       # (n, k, k), (n, k)
        dg, dp = _diag_scale(Hgg), _diag_scale(Hpp)
        Hgg, gg = Hgg / np.outer(dg, dg), gg / dg
        Hgp = Hgp / (dg[None, :, None] * dp[:, None, :])
        Hpp, gp = Hpp / (dp[:, :, None] * dp[:, None, :]), gp / dp
        rn = max(np.sqrt(2 * cost), 1.0)
        if max(np.abs(gg).max(initial=0), np.abs(gp).max(initial=0)) <= GTOL * rn:
            return st, True, nfev
        HpgB = np.concatenate([np.transpose(Hgp, (0, 2, 1)), gp[..., None]], axis=-1)   # (n, k, G + 1)
        while True:
            if nfev >= max_nfev:
                return st, loose, nfev
            X = np.linalg.solve(Hpp + lam * np.eye(k), HpgB)
            S = Hgg + lam * np.eye(G) - np.einsum("pgk,pkh->gh", Hgp, X[..., :G])
            b = -gg + np.einsum("pgk,pk->g", Hgp, X[..., G])
            dgs = np.linalg.solve(S, b) if G else np.zeros(0)
            dps = -X[..., G] - X[..., :G] @ dgs
            dot = float(gg @ dgs + np.sum(gp * dps))
            sq = float(dgs @ dgs + np.sum(dps * dps))
            pred = 0.5 * (lam * sq - dot)
            new = st.step(dgs / dg, dps / dp, free_intr, free_C, scale)
            nlin = prob.linearise(new, free_intr, free_C, scale, w)
            nfev += 1
            rn_ = nlin[0].ravel()
            ncost = 0.5 * float(rn_ @ rn_)
            rho = (cost - ncost) / pred if pred > 0 and np.isfinite(ncost) else -1.0
            if rho > 0:
                loose = loose or ((cost - ncost) < SCIPY_FTOL * cost and rho > 0.25)
                st, lin = new, nlin
                lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
                nu = 2.0
                if ((cost - ncost) < FTOL * cost and rho > 0.25) or np.sqrt(sq) <= XTOL * rn:
                    return st, True, nfev
                break
            lam *= nu
            nu *= 2
            if lam > LAM_MAX:
                return st, loose or max(np.abs(gg).max(initial=0), np.abs(gp).max(initial=0)) <= 1e-6 * rn, nfev


def _lm_pairs(prob, st, w, scale, max_nfev):
    """the camera fixed: every pair its own LM (damping, cost, stop), run as one vector; only pairs still running are
    evaluated. -> (state, success = every pair converged (or met scipy's ftol test, see the module), evaluations)"""
    n = prob.n
    st = State(st.intr, st.C, st.R.copy(), st.t.copy(), st.ls.copy())
    r, _, Jp = prob.linearise(st, False, False, scale, w)
    nfev = 1
    k = Jp.shape[-1]
    lam, nu = np.full(n, LAM0), np.full(n, 2.0)
    fresh = np.ones(n, bool)                               # the normal equations need recomputing
    done, good = np.zeros(n, bool), np.zeros(n, bool)
    loose = False                                          # scipy's ftol test met on the way (on the sum, as scipy)
    cost = 0.5 * np.einsum("pi,pi->p", r, r)
    Hpp, gp, dp = np.zeros((n, k, k)), np.zeros((n, k)), np.ones((n, k))
    eye = np.eye(k)
    while True:
        f = np.flatnonzero(fresh & ~done)
        if len(f):
            JpT = np.transpose(Jp[f], (0, 2, 1))
            H, g = JpT @ Jp[f], (JpT @ r[f][..., None])[..., 0]
            d = _diag_scale(H)
            Hpp[f], gp[f], dp[f] = H / (d[:, :, None] * d[:, None, :]), g / d, d
            rn = np.maximum(np.sqrt(2 * cost[f]), 1.0)
            conv = np.abs(gp[f]).max(axis=1) <= GTOL * rn
            done[f[conv]] = good[f[conv]] = True
            fresh[f] = False
        a = np.flatnonzero(~done)
        if not len(a):
            return st, bool(good.all()), nfev
        if nfev >= max_nfev:
            return st, bool(good[done].all() and loose), nfev
        dps = np.linalg.solve(Hpp[a] + lam[a, None, None] * eye, -gp[a][..., None])[..., 0]
        sq = np.einsum("pk,pk->p", dps, dps)
        pred = 0.5 * (lam[a] * sq - np.einsum("pk,pk->p", gp[a], dps))
        step = np.zeros((n, k))
        step[a] = dps / dp[a]
        new = st.step(np.zeros(0), step, False, False, scale)
        nr, _, nJ = prob.linearise(new, False, False, scale, w, pairs=a)
        nfev += 1
        ncost = 0.5 * np.einsum("pi,pi->p", nr, nr)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where((pred > 0) & np.isfinite(ncost), (cost[a] - ncost) / pred, -1.0)
        acc = rho > 0
        ai, rj = a[acc], a[~acc]
        rn = np.maximum(np.sqrt(2 * cost[ai]), 1.0)
        conv = (((cost[ai] - ncost[acc]) < FTOL * cost[ai]) & (rho[acc] > 0.25)) | (np.sqrt(sq[acc]) <= XTOL * rn)
        dec = float(np.sum(cost[ai] - ncost[acc]))                  # scipy solves the pairs as one problem
        loose = loose or (len(ai) > 0 and dec < SCIPY_FTOL * float(cost.sum()) and dec > 0.25 * float(pred[acc].sum()))
        st.R[ai], st.t[ai], st.ls[ai] = new.R[ai], new.t[ai], new.ls[ai]
        r[ai], Jp[ai], cost[ai] = nr[acc], nJ[acc], ncost[acc]
        lam[ai] *= np.maximum(1 / 3, 1 - (2 * rho[acc] - 1) ** 3)
        nu[ai] = 2.0
        fresh[ai] = True
        done[ai[conv]] = good[ai[conv]] = True
        lam[rj] *= nu[rj]
        nu[rj] *= 2
        stall = rj[lam[rj] > LAM_MAX]
        if len(stall):
            rn = np.maximum(np.sqrt(2 * cost[stall]), 1.0)
            good[stall] = loose | (np.abs(gp[stall]).max(axis=1) <= 1e-6 * rn)
            done[stall] = True


def solve(data, model, intr0, C0, poses0, free_intr=True, free_C=True, scale=False, trim_px=TRIM_PX,
          rounds=TRIM_ROUNDS, max_nfev=300, ls0=None):
    """stereofit.solve's contract, solved fast: least squares on the kept corners, then corners beyond trim_px of the
    fit cut and the solve repeated until the kept set repeats (at most `rounds` rounds); success only if every round
    converged within max_nfev evaluations. Free intrinsics need KB4/KB4T (NotImplementedError otherwise: keep scipy).
    -> dict(intr, C, poses, log_scale, res (all corners, N x 2), kept (mask), success, nfev)"""
    if free_intr and model not in KB_MODELS:
        raise NotImplementedError(f"free intrinsics need {'/'.join(KB_MODELS)}, not {model}")
    prob = Problem(data, model)
    st = State.start(intr0, C0, poses0, ls0 if scale else None)
    keep = np.ones(prob.N, bool)
    nfev, ok = 0, True
    for _ in range(rounds):
        w = prob.padded(keep)
        if free_intr or free_C:
            st, success, nf = _lm_global(prob, st, w, free_intr, free_C, scale, max_nfev)
        else:
            st, success, nf = _lm_pairs(prob, st, w, scale, max_nfev)
        nfev += nf
        ok = ok and success
        res = prob.residuals(st)
        new = np.linalg.norm(res, axis=1) <= trim_px
        if np.array_equal(new, keep):
            break
        keep = new
    intr = [np.array(v) for v in st.intr] if free_intr else [np.array(v, float) for v in intr0]
    return {"intr": intr, "C": st.C, "poses": st.poses(), "log_scale": st.ls if scale else np.zeros(prob.n),
            "res": res, "kept": keep, "success": bool(ok), "nfev": int(nfev)}
