"""Robot-world hand-eye fit for the head camera (step 81).

Every image i of camera c closes one loop from the torso to that camera:

    A_i · X · C_c  =  Z · B_ic

A_i = T_TB(q_i)  the Head link in the torso frame, from the neck encoders and the URDF;
X   = T_BC0      camera 0 in the Head link: the unknown, = T_BS · T_SC0;
C_c = T_C0Cc     camera c in camera 0, the calibrated stereo pose (identity for c = 0);
Z   = T_TW       the board in the torso frame: unknown, constant while torso and board are still;
B_ic = T_WCc     camera c in the board frame, the inverse of the board pose PnP returns.

Solved by Levenberg-Marquardt over X and Z (12 numbers) on pose residuals: the rotation of
(Z B)^-1 (A X C) in radians times W_ROT_M, and its translation in metres, so 1 mrad weighs like
1 mm. X's error bars come from the Jacobian at the solution: clustered by hold when the caller
says which hold each image came from (a hold's images share its joint error, the URDF's and the
lens model's, so they are not independent: counted as independent, the error bars came out up to
3x too small in simulation), otherwise (J^T J)^-1 scaled by the residual variance. A direction of
X the data cannot see (a joint that never moved) gets an infinite error bar, not pinv's silence.
numpy only: OpenCV 5.0.0's Python wheel has no calibrateRobotWorldHandEye."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigmath import exp_rot, inv, log_rot, transform  # noqa: E402

W_ROT_M = 1.0
MAX_COND = 1e6           # condition number of the Jacobian beyond which a direction counts as unseen


def _step(d) -> np.ndarray:
    return transform(exp_rot(d[:3]), d[3:])


def _jacobian(X, Z, obs, eps: float = 1e-6) -> np.ndarray:
    J = np.empty((6 * len(obs), 12))
    for k in range(12):                       # central differences, right perturbations of X and Z
        d = np.zeros(12)
        d[k] = eps
        J[:, k] = (residuals(X @ _step(d[:6]), Z @ _step(d[6:]), obs).ravel()
                   - residuals(X @ _step(-d[:6]), Z @ _step(-d[6:]), obs).ravel()) / (2 * eps)
    return J


def residuals(X, Z, obs) -> np.ndarray:
    """obs: [(A, C, B)] -> N x 6 (weighted rotation [rad], translation [m]) in camera c's axes."""
    out = np.empty((len(obs), 6))
    for k, (A, C, B) in enumerate(obs):
        E = inv(Z @ B) @ (A @ X @ C)
        out[k, :3] = W_ROT_M * log_rot(E[:3, :3])
        out[k, 3:] = E[:3, 3]
    return out


def fit(obs, X0, Z0=None, iters: int = 50, step_tol: float = 1e-10, groups=None) -> dict:
    """X0: starting camera-in-Head (the config's). Z0: starting board-in-torso (default: from the
    first observation). groups: the hold each observation came from (X's covariance is then
    clustered by hold); None treats every observation as independent. -> X, Z, residuals (N x 6),
    rms, X's 1-sigma in Head axes (inf when cond > MAX_COND), cond (the Jacobian's condition
    number), convergence."""
    if iters < 1:
        raise ValueError("iters must be at least 1")
    bad = [k for k, o in enumerate(obs) if not all(np.all(np.isfinite(m)) for m in o)]
    if bad:
        raise ValueError(f"observations with NaN or inf: {bad[:10]}")
    if groups is not None and len(groups) != len(obs):
        raise ValueError(f"{len(groups)} groups for {len(obs)} observations")
    A, C, B = obs[0]
    X = np.array(X0, float)
    Z = A @ X @ C @ inv(B) if Z0 is None else np.array(Z0, float)
    r = residuals(X, Z, obs).ravel()
    lam, converged = 1e-3, False
    for it in range(iters):
        J = _jacobian(X, Z, obs)
        H, g = J.T @ J, J.T @ r
        while True:
            d = -np.linalg.solve(H + lam * np.diag(np.diag(H)), g)
            Xn, Zn = X @ _step(d[:6]), Z @ _step(d[6:])
            rn = residuals(Xn, Zn, obs).ravel()
            if rn @ rn < r @ r:
                lam = max(lam / 10, 1e-9)
                break
            lam *= 10
            if lam > 1e9:
                break
        if lam > 1e9:
            converged = True       # no step lowers the cost: at the minimum
            break
        done = np.linalg.norm(d) < step_tol
        X, Z, r = Xn, Zn, rn
        if done:
            converged = True
            break
    res = r.reshape(-1, 6)
    J = _jacobian(X, Z, obs)                  # at the solution (r is already there)
    sv = np.linalg.svd(J, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    if cond > MAX_COND:
        sr = st = np.full(3, np.inf)
    else:
        Hi = np.linalg.inv(J.T @ J)
        if groups is None:
            cov = (r @ r / max(r.size - 12, 1)) * Hi
        else:
            g = np.repeat(np.asarray(groups), 6)
            labels = np.unique(g)
            M = np.zeros((12, 12))
            for h in labels:
                s = J[g == h].T @ r[g == h]
                M += np.outer(s, s)
            # small-sample inflation G / (G - 7) (X's 6 numbers + 1): with G = 9-16 holds the plain
            # G / (G - 1) left the sigma 1.3-1.6x below the scatter in simulation (Task 6 re-review)
            # (below 9 holds the factor is not monotonic, max(G - 7, 1); step 81 gives no verdict below 20)
            cov = len(labels) / max(len(labels) - 7, 1) * Hi @ M @ Hi
        R = X[:3, :3]                         # X's right perturbation, turned into Head axes
        sr = np.degrees(np.sqrt(np.diag(R @ cov[:3, :3] @ R.T)))     # parameters: plain radians
        st = 1e3 * np.sqrt(np.diag(R @ cov[3:6, 3:6] @ R.T))
    return {"X": X, "Z": Z, "residuals": res, "n": len(obs), "iterations": it + 1, "converged": converged,
            "rms_rot_deg": float(np.degrees(np.sqrt(np.mean(np.sum(res[:, :3] ** 2, 1)))) / W_ROT_M),
            "rms_trans_mm": float(1e3 * np.sqrt(np.mean(np.sum(res[:, 3:] ** 2, 1)))),
            "sigma_rot_head_deg": sr, "sigma_trans_head_mm": st, "cond": cond}
