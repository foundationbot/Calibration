"""Pixel-level robot-world hand-eye fit for the head camera (step 81's default model).

The loop is handeye.py's, A_i · X · C_c = Z_k · B_ic, closed on the board's corners in the images
instead of on per-image board poses. Corner j of image i (camera c, take k) is predicted at

    u_ij = project_c( (A_i · X · C_c)^-1 · Z_k · P_j )

(P_j the corner on the board, project_c camera c's lens model), and X (camera 0 in the Head link) and
one board pose Z_k per take are fitted to the measured pixels by Levenberg-Marquardt. Why pixels: a
per-image PnP pose is weakest along the optical axis, and handeye.py weighs its rotation and
translation 1 rad : 1 m; on the 17 Oct powered-neck takes the two cameras' own pose-level fits came out
12 mm apart and their pixel-level fits 5 mm apart, from the same images (that 5 mm is the same in every
half of both takes: a disagreement between the two cameras' calibrations, which step 81 carries as a
systematic error). Corners further than OUTLIER_PX from the current fit's prediction are left
out and the fit repeated until the set left out stops changing (a cut corner comes back once the fit no
longer leans on gross ones; at most TRIM_ROUNDS); the share left out and the rms over all corners are reported, since
cutting a misfit's corners would hide it. X's error bars are clustered by hold (a hold's images share its joint error) with
handeye.py's small-sample inflation G / (G - 7); a direction the data cannot see gets an infinite
error bar. numpy only."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigmath import exp_rot, inv, transform  # noqa: E402

OUTLIER_PX = 3.0         # a corner this far from its prediction after the first fit is left out
MAX_COND = 1e6           # condition number of the Jacobian beyond which a direction counts as unseen
BEHIND_PX = 1e3          # the residual of a corner that lands behind its camera (a wild LM step)
TRIM_ROUNDS = 5          # refits of the outlier cut at most


@dataclass
class Image:
    take: int            # which board pose Z_k (one per take: the board and torso were still within it)
    cam: int             # camera c: C[c] = camera c in camera 0
    A: np.ndarray        # T_TB(q): the Head link in the torso at this image's neck angles
    P: np.ndarray        # (n, 3) the detected corners on the board [m]
    uv: np.ndarray       # (n, 2) their pixels
    group: str           # the hold it came from: the error bars are clustered by it


def project(P, K, D, kind: str) -> np.ndarray:
    """Camera-frame points (N, 3) -> pixels (N, 2). kind 'equi': OpenCV's fisheye model (k1..k4);
    'radtan': the pinhole model with k1, k2, p1, p2 (the four numbers OKVIS keeps); 'kb4t': the fisheye
    model plus tangential p1, p2 (lensmodel.py; six numbers, Fit A of 20 Oct)."""
    P = np.asarray(P, float)
    D = np.asarray(D, float).ravel()
    if kind == "kb4t":
        import lensmodel
        if abs(K[0, 1]) > 1e-12:                 # KB4T has no skew term (as rigcfg.Lens._kb4t_intr): never drop it
            raise ValueError(f"kb4t lens: no skew (K[0, 1] = {K[0, 1]})")
        return lensmodel.project("KB4T", [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D], P)
    with np.errstate(divide="ignore", invalid="ignore"):     # z = 0: residuals() replaces it (BEHIND_PX)
        x, y = P[:, 0] / P[:, 2], P[:, 1] / P[:, 2]
    if kind == "equi":
        r = np.hypot(x, y)
        th = np.arctan(r)
        t2 = th * th
        thd = th * (1 + D[0] * t2 + D[1] * t2 ** 2 + D[2] * t2 ** 3 + D[3] * t2 ** 4)
        s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
        xd, yd = s * x, s * y
    elif kind == "radtan":
        r2 = x * x + y * y
        rad = 1 + D[0] * r2 + D[1] * r2 * r2
        xd = x * rad + 2 * D[2] * x * y + D[3] * (r2 + 2 * x * x)
        yd = y * rad + D[2] * (r2 + 2 * y * y) + 2 * D[3] * x * y
    else:
        raise ValueError(f"lens kind {kind!r}: 'equi', 'radtan' or 'kb4t'")
    return np.c_[K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]]


def residuals(X, Zs, images, C, K, D, kind) -> np.ndarray:
    """-> (total corners, 2): predicted minus measured pixels, image by image."""
    out = []
    for im in images:
        T = inv(im.A @ X @ C[im.cam]) @ Zs[im.take]          # the board in camera c
        Pc = im.P @ T[:3, :3].T + T[:3, 3]
        r = project(Pc, K[im.cam], D[im.cam], kind) - im.uv
        r[Pc[:, 2] <= 1e-6] = BEHIND_PX
        out.append(r)
    return np.concatenate(out) if out else np.empty((0, 2))


def _apply(X0, Zs0, p):
    return X0 @ transform(exp_rot(p[:3]), p[3:6]), [Z @ transform(exp_rot(p[6 + 6 * k:9 + 6 * k]), p[9 + 6 * k:12 + 6 * k])
                                                     for k, Z in enumerate(Zs0)]


def _jacobian(f, p, eps=1e-7):
    r0 = f(p)
    J = np.empty((r0.size, p.size))
    for i in range(p.size):
        d = np.zeros(p.size)
        d[i] = eps
        J[:, i] = (f(p + d) - f(p - d)) / (2 * eps)
    return J


def _lm(f, p, iters, step_tol):
    r = f(p)
    lam, converged, it = 1e-3, False, 0
    for it in range(iters):
        J = _jacobian(f, p)
        H, g = J.T @ J, J.T @ r
        while True:
            d = -np.linalg.lstsq(H + lam * np.diag(np.diag(H)), g, rcond=None)[0]   # an unseen direction: no crash
            rn = f(p + d)
            if rn @ rn < r @ r:
                lam = max(lam / 10, 1e-12)
                break
            lam *= 10
            if lam > 1e12:
                break
        if lam > 1e12:
            converged = True          # no step lowers the cost: at the minimum
            break
        p, r = p + d, rn
        if np.linalg.norm(d) < step_tol:
            converged = True
            break
    return p, r, converged, it + 1


def _trim(images, err, outlier_px):
    """-> (images with their corners within outlier_px, per-corner mask of the ones left out). An image
    left with fewer than 4 corners goes, all of its corners counted out."""
    kept, out, i0 = [], np.zeros(len(err), bool), 0
    for im in images:
        ok = err[i0:i0 + len(im.P)] <= outlier_px
        if ok.sum() >= 4:
            kept.append(Image(im.take, im.cam, im.A, im.P[ok], im.uv[ok], im.group))
            out[i0:i0 + len(im.P)] = ~ok
        else:
            out[i0:i0 + len(im.P)] = True
        i0 += len(im.P)
    return kept, out


def fit(images, X0, Zs0, C, K, D, kind, iters: int = 40, step_tol: float = 1e-10,
        outlier_px: float = OUTLIER_PX) -> dict:
    """X0: starting camera-in-Head (the config's); Zs0: a starting board-in-torso per take (index =
    Image.take). C, K, D: per camera. -> X, Zs, rms_px (the corners kept), rms_all_px (every corner, at
    the same solution), the corners used and left out and the share left out, the cut's rounds and
    whether it settled (trim_converged), X's 1-sigma in Head axes
    (inf when cond > MAX_COND), cond, convergence, iterations, per-group rms [px]."""
    if iters < 1:
        raise ValueError("iters must be at least 1")
    if not images:
        raise ValueError("no images")
    n_takes = max(im.take for im in images) + 1
    if len(Zs0) != n_takes:
        raise ValueError(f"{len(Zs0)} starting board poses for {n_takes} takes")
    for k, im in enumerate(images):
        if len(im.P) != len(im.uv) or len(im.P) < 4:
            raise ValueError(f"image {k}: {len(im.P)} board points for {len(im.uv)} pixels (need >= 4 matched)")
        if not (np.all(np.isfinite(im.A)) and np.all(np.isfinite(im.P)) and np.all(np.isfinite(im.uv))):
            raise ValueError(f"image {k}: NaN or inf")
    X0, Zs0 = np.array(X0, float), [np.array(Z, float) for Z in Zs0]
    n_par = 6 + 6 * n_takes

    def f_of(ims):
        return lambda p: residuals(*_apply(X0, Zs0, p), ims, C, K, D, kind).ravel()

    if np.mean(residuals(X0, Zs0, images, C, K, D, kind) == BEHIND_PX) > 0.5:
        raise ValueError("the starting poses put most corners behind their cameras: Zs0 is not the board in the "
                         "torso (T_TW = A · X · C · T_CW)")
    f_all = f_of(images)
    p, r, conv, its = _lm(f_all, np.zeros(n_par), iters, step_tol)
    out_prev = np.zeros(sum(len(im.P) for im in images), bool)
    kept, rounds, stable = images, 0, False
    for rounds in range(1, TRIM_ROUNDS + 1):   # leave out corners far from the current fit (a mis-refined one), refit
        kept, out = _trim(images, np.linalg.norm(f_all(p).reshape(-1, 2), axis=1), outlier_px)
        if not kept:
            raise ValueError(f"every image lost its corners to the outlier cut ({outlier_px} px): the model does "
                             "not fit at all")
        if np.array_equal(out, out_prev):
            stable = True
            break
        out_prev = out
        p, r, conv, its2 = _lm(f_of(kept), p, iters, step_tol)
        its += its2
    n_out = int(out_prev.sum())
    f = f_of(kept)
    J = _jacobian(f, p)
    X, Zs = _apply(X0, Zs0, p)
    sv = np.linalg.svd(J, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    groups = np.repeat([im.group for im in kept], [2 * len(im.P) for im in kept])
    labels = np.unique(groups)
    if cond > MAX_COND:
        sr = st = np.full(3, np.inf)
    else:
        Hi = np.linalg.inv(J.T @ J)
        M = np.zeros((n_par, n_par))
        for h in labels:
            s = J[groups == h].T @ r[groups == h]
            M += np.outer(s, s)
        cov = len(labels) / max(len(labels) - 7, 1) * Hi @ M @ Hi     # handeye.py's G / (G - 7)
        R = X[:3, :3]                         # X's right perturbation, turned into Head axes
        sr = np.degrees(np.sqrt(np.diag(R @ cov[:3, :3] @ R.T)))
        st = 1e3 * np.sqrt(np.diag(R @ cov[3:6, 3:6] @ R.T))
    rr = r.reshape(-1, 2)
    per = {str(h): float(np.sqrt(np.mean(np.sum(rr[groups[::2] == h] ** 2, 1)))) for h in labels}
    ra = f_all(p).reshape(-1, 2)
    return {"X": X, "Zs": Zs, "rms_px": float(np.sqrt(np.mean(np.sum(rr ** 2, 1)))),
            "rms_all_px": float(np.sqrt(np.mean(np.sum(ra ** 2, 1)))), "outlier_fraction": n_out / len(ra),
            "trim_rounds": rounds, "trim_converged": stable,
            "n_corners": int(len(rr)), "n_outliers": n_out, "n_images": len(kept), "n_groups": int(len(labels)),
            "sigma_rot_head_deg": sr, "sigma_trans_head_mm": st, "cond": cond, "converged": conv,
            "iterations": its, "group_rms_px": per}
