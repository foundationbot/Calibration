#!/usr/bin/env python3
"""Step 82 — independent check: the rotation of T_BS from the gyro and the neck joint rates.

While only the neck moves, the gyro (bias removed) reads the Head's turn in IMU axes:
omega_S = R_SB · omega_B, with omega_B from the joint rates (neckfk.omega_B). Fitting R_SB over all
moving samples (Kabsch, over a grid of joint/IMU clock lags) gives the rotation of T_BS with no
camera and no URDF lengths, only the two joint axes. It is a check, not the measurement (step 81 is):
the gyro's own scale and axis errors (~1 %, ~0.5 deg) move it by ~0.3 deg, so the gate is ROT_DEG.
The verdict carries a jackknife error bar over time segments of the take (within-take: it does not
see how the head is loaded differently from one take to the next; two hand takes on 16 Oct sat ~3
sigma apart): PASS only when the angle plus 2 sigma is inside the gate,
FAIL when the angle minus 2 sigma is outside it, INCONCLUSIVE otherwise, and INCONCLUSIVE when one
joint barely moved (the fit cannot see every direction).

Also reported, from the pitch-only and the yaw-only sweeps separately: each joint axis measured in
IMU axes against where the config puts it, and the angle between the two measured axes against the
URDF's (90 deg): an error there is the URDF's neck geometry, not T_BS. A hand never moves one joint
alone, so before averaging a sweep's samples the other joint's rate (predicted from the encoders
and the fitted R_SB) is subtracted: 2 % of the other joint left in shifts an axis by ~1 deg. The gyro
cannot see an encoder's sign: a reversed encoder fits as well with R_SB turned half a turn about one
Head axis, which flip_diagnosis names (an IMU mounted half a turn round looks the same; step 75's
camera sign check tells which). See calibration.md section 6b.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import necktake  # noqa: E402
from rigmath import log_rot, rot_angle_deg  # noqa: E402

ROT_DEG = 1.0          # gate between the measured and the config's R_BS
SWEEP_RATE, PURITY = 0.3, 0.2       # a sweep: one joint above SWEEP_RATE [rad/s], the other below PURITY of it
MIN_EIG_RATIO = 0.1    # second eigenvalue of sum omega_B omega_B^T over the first: below, a joint barely moved
N_SEG = 10             # jackknife segments
GAP_NS = 10_000_000    # a /joints gap longer than this: IMU samples whose rate window spans it are not used
INTER_AXIS_WARN_DEG = 0.3


def kabsch(a, b) -> np.ndarray:
    """R minimising sum ||b_k - R a_k||^2 (rows are samples)."""
    U, _, Vt = np.linalg.svd(np.asarray(a).T @ np.asarray(b))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def sweep_masks(qdy, qdp):
    """(pitch-only, yaw-only) samples by the encoders: one joint faster than SWEEP_RATE, the other
    slower than PURITY times it. Step 75 counts its sweeps with the same masks."""
    qdy, qdp = np.abs(np.asarray(qdy)), np.abs(np.asarray(qdp))
    return (qdp > SWEEP_RATE) & (qdy < PURITY * qdp), (qdy > SWEEP_RATE) & (qdp < PURITY * qdy)


def gap_mask(j, t_ns, half_window_s: float = 0.02) -> np.ndarray:
    """False for stamps whose +-half_window_s of /joints spans a gap longer than GAP_NS: rates there
    come from a straight line across the gap (a CPU-starved recorder left such gaps of up to 2.6 s)."""
    t_ns = np.asarray(t_ns, np.int64)
    ok = np.ones(len(t_ns), bool)
    w = int(half_window_s * 1e9)
    for g in np.flatnonzero(np.diff(j.t) > GAP_NS):
        lo, hi = np.searchsorted(t_ns, [j.t[g] - w, j.t[g + 1] + w])
        ok[lo:hi] = False
    return ok


def head_rates(chain, j, t_ns, lag_s=0.0):
    """omega_B at IMU stamps, from the joints shifted by lag_s (positive: the IMU lags)."""
    t = np.asarray(t_ns, np.int64) - int(lag_s * 1e9)
    _, qp = necktake.at(j, t)
    qdy, qdp = necktake.rates_at(j, t)
    return chain.omega_B(qp, qdy, qdp)


def _search(chain, j, t_imu, wS, ok, min_rate, max_lag_s, lag_step_s):
    best = None
    for lag in np.arange(-max_lag_s, max_lag_s + 1e-9, lag_step_s):
        wB = head_rates(chain, j, t_imu, lag)
        m = ok & (np.linalg.norm(wB, axis=1) > min_rate)
        if m.sum() < 50:
            continue
        R = kabsch(wB[m], wS[m])
        res = float(np.sqrt(np.mean(np.sum((wS[m] - wB[m] @ R.T) ** 2, axis=1))))
        if best is None or res < best["residual_rad_s"]:
            best = {"R_SB": R, "residual_rad_s": res, "lag_s": float(lag), "n_moving": int(m.sum()), "_m": m, "_wB": wB}
    return best


def fit(chain, j, t_imu, w_imu, min_rate=0.05, max_lag_s=0.05, lag_step_s=0.002) -> dict:
    """Gyro bias from the held poses: the median over holds of each hold's mean of what the encoders
    do not explain (the head sags and trembles while held; means, not a median of samples: the gyro
    is quantised, 0.53 mrad/s steps on phantom12). Then, for each joint/IMU lag on a grid, the Kabsch
    R_SB over the samples turning faster than min_rate [rad/s] (Kabsch weights by |omega|^2, so slow
    hand motion counts little but is not thrown away); keeps the lag with the smallest residual. Two
    passes: the second bias removes the first fit's encoder-explained rate. Also: the eigenvalue ratio
    of the motion (observability) and a jackknife sigma of R_SB over N_SEG time segments."""
    t_imu, w_imu = np.asarray(t_imu, np.int64), np.asarray(w_imu, float)
    ok = gap_mask(j, t_imu)
    hm = [m for m in ((t_imu >= h.t0) & (t_imu <= h.t1) & ok for h in necktake.find_holds(j)) if m.sum() >= 50]
    if not hm:
        raise ValueError("no held poses: the gyro bias cannot be measured")
    bias = np.median([w_imu[m].mean(axis=0) for m in hm], axis=0)
    best = _search(chain, j, t_imu, w_imu - bias, ok, min_rate, max_lag_s, lag_step_s)
    if best is None:
        raise ValueError("no moving samples above min_rate: the take has no neck motion")
    explained = best["_wB"] @ best["R_SB"].T
    bias = np.median([(w_imu[m] - explained[m]).mean(axis=0) for m in hm], axis=0)
    best = _search(chain, j, t_imu, w_imu - bias, ok, min_rate, max_lag_s, lag_step_s)
    m, wB, wS = best.pop("_m"), best.pop("_wB"), w_imu - bias
    ev = np.sort(np.linalg.eigvalsh(wB[m].T @ wB[m]))[::-1]
    idx = np.flatnonzero(m)
    d = []
    for seg in np.array_split(idx, N_SEG):
        keep = np.setdiff1d(idx, seg)
        d.append(log_rot(kabsch(wB[keep], wS[keep]) @ best["R_SB"].T))
    d = np.array(d)
    cov = (N_SEG - 1) / N_SEG * (d - d.mean(0)).T @ (d - d.mean(0))
    best.update(bias=bias, n_still=int(np.any(hm, axis=0).sum()), n_gap_masked=int((~ok).sum()),
                lag_at_edge=abs(abs(best["lag_s"]) - max_lag_s) < lag_step_s / 2,
                eig_ratio=float(ev[1] / ev[0]), sigma_deg=float(np.degrees(np.sqrt(np.trace(cov)))))
    return best


def _ang(u, v) -> np.ndarray:
    """Angle [deg] between rows of u and v (or a vector), sign included."""
    u = np.atleast_2d(u)
    v = np.broadcast_to(v, u.shape)
    c = np.sum(u * v, 1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _unit_mean(v) -> np.ndarray:
    u = v / np.linalg.norm(v, axis=1)[:, None]
    m = u.mean(axis=0)
    return m / np.linalg.norm(m)


MIN_STROKE = 50        # samples (0.1 s at 500 Hz): shorter strokes swung the 16 Oct angle by degrees
N_BOOT = 200


def _strokes(mask, t_ns, gap_ns=50_000_000, min_n=MIN_STROKE):
    """Index arrays of the runs of mask (a run ends where consecutive selected samples are gap_ns
    apart), at least min_n samples each."""
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    cut = np.flatnonzero(np.diff(np.asarray(t_ns)[idx]) > gap_ns) + 1
    return [s for s in np.split(idx, cut) if s.size >= min_n]


def _inter_axis(p_strokes, y_strokes, rng=None):
    """Angle [deg] between the pitch axis (mean direction over the pitch strokes' unit samples) and
    the yaw strokes' axes, weighted by stroke length. p_strokes / y_strokes: lists of N x 3 arrays
    of signed samples. With rng: one bootstrap resample of both stroke sets."""
    if rng is not None:
        p_strokes = [p_strokes[i] for i in rng.integers(0, len(p_strokes), len(p_strokes))]
        y_strokes = [y_strokes[i] for i in rng.integers(0, len(y_strokes), len(y_strokes))]
    a_p = _unit_mean(np.vstack(p_strokes))
    angs = np.array([_ang(_unit_mean(s), a_p)[0] for s in y_strokes])
    w = np.array([len(s) for s in y_strokes], float)
    return float(np.sum(w * angs) / np.sum(w))


def sweep_axes(chain, j, t_imu, w_imu, bias, lag_s, R_SB_cfg, R_SB_fit) -> dict:
    """Each joint's axis from its own sweep, sign following the encoder (omega / joint rate), as the
    mean direction of the samples (angles of single samples are biased upward by gyro noise), after
    subtracting the other joint's rate predicted with the fitted R_SB:
    pitch_axis_deg  angle to the config's R_SB e_pitch;
    yaw_axis_deg    each yaw sample turned into NeckYawLink axes (R_YB(q_pitch) R_SB_cfg^T), so a
                    sweep at any pitch counts, against the URDF's yaw axis there;
    inter_axis_deg  the angle between each yaw stroke's axis and the measured pitch axis (it does
                    not depend on q_pitch), weighted by stroke length over strokes of at least
                    MIN_STROKE samples; inter_axis_sigma_deg from a bootstrap over the pitch and the
                    yaw strokes together; inter_axis_dir_diff_deg = strokes turning + minus - (an
                    error that depends on the direction: lag, play). The URDF's angle is
                    urdf_inter_axis_deg. Not corrected for the torso turning in step with the head
                    (the dma-recorder torso gyro is not read here): on 16 Oct's hand takes ~1 % of
                    the head's rate, which tilts the axes by up to ~0.7 deg, so a hand take tests the
                    URDF's 90 deg only to ~2 deg."""
    t_imu = np.asarray(t_imu, np.int64)
    t = t_imu - int(lag_s * 1e9)
    _, qp = necktake.at(j, t)
    qdy, qdp = necktake.rates_at(j, t)
    wS = np.asarray(w_imu) - bias
    turning = (np.linalg.norm(wS, axis=1) > 0.5 * SWEEP_RATE) & gap_mask(j, t_imu)
    pm, ym = sweep_masks(qdy, qdp)
    pm, ym = pm & turning, ym & turning
    a_pB = chain.pitch.axis
    wP = wS - qdy[:, None] * (chain.yaw_axis_in_B(qp) @ R_SB_fit.T)
    wY = wS - qdp[:, None] * (R_SB_fit @ a_pB)
    out = {"n_pitch": int(pm.sum()), "n_yaw": int(ym.sum()),
           "pitch_coupling": float(np.median(np.abs(qdy[pm] / qdp[pm]))) if pm.any() else None,
           "yaw_coupling": float(np.median(np.abs(qdp[ym] / qdy[ym]))) if ym.any() else None,
           "urdf_inter_axis_deg": float(_ang(a_pB, chain.yaw_axis_in_B(0.0))[0])}
    a_p = None
    if pm.sum() >= 20:
        a_p = _unit_mean(wP[pm] * np.sign(qdp[pm])[:, None])
        out["pitch_axis_S"] = a_p.tolist()
        out["pitch_axis_deg"] = float(_ang(a_p, R_SB_cfg @ a_pB)[0])
    if ym.sum() >= 20:
        u = np.array([chain.R_YB(float(q)) @ R_SB_cfg.T @ w for q, w in zip(qp[ym], wY[ym] * np.sign(qdy[ym])[:, None])])
        a_y = _unit_mean(u)
        out["yaw_axis_link"] = a_y.tolist()
        out["yaw_axis_deg"] = float(_ang(a_y, chain.yaw.axis)[0])
    ps = [wP[s] * np.sign(qdp[s])[:, None] for s in _strokes(pm, t_imu)]
    ys = _strokes(ym, t_imu)
    out["n_pitch_strokes"], out["n_yaw_strokes"] = len(ps), len(ys)
    if ps and ys:
        yv = [wY[s] * np.sign(qdy[s])[:, None] for s in ys]
        out["inter_axis_deg"] = _inter_axis(ps, yv)
        if len(ps) > 1 and len(ys) > 1:
            rng = np.random.default_rng(0)
            out["inter_axis_sigma_deg"] = float(np.std([_inter_axis(ps, yv, rng) for _ in range(N_BOOT)], ddof=1))
        up = [k for k, s in enumerate(ys) if np.mean(qdy[s]) > 0]
        dn = [k for k, s in enumerate(ys) if np.mean(qdy[s]) < 0]
        if up and dn:
            out["inter_axis_dir_diff_deg"] = _inter_axis(ps, [yv[k] for k in up]) - _inter_axis(ps, [yv[k] for k in dn])
    return out


def flip_diagnosis(R_SB_meas, R_SB_cfg):
    """None, or which encoder sign would explain a half turn between the measured and configured
    R_SB. For this URDF's neck (pitch about Head y, yaw about NeckYawLink z, zero rpy) a reversed
    pitch encoder fits the gyro exactly with R_SB turned 180 deg about Head z, a reversed yaw
    encoder about Head y, both about Head x. An IMU mounted half a turn round looks the same."""
    rel = np.asarray(R_SB_cfg).T @ np.asarray(R_SB_meas)     # in Head axes
    if rot_angle_deg(rel) < 170.0:
        return None
    S = rel + np.eye(3)                                       # = 2 a a^T for a half turn about a
    a = S[:, int(np.argmax(np.linalg.norm(S, axis=0)))]
    a = a / np.linalg.norm(a)
    k = int(np.argmax(np.abs(a)))
    return {0: "yaw and pitch", 1: "yaw", 2: "pitch"}[k] if abs(a[k]) > 0.95 else "unclear"


def verdict(angle_deg, sigma_deg, eig_ratio) -> str:
    if eig_ratio < MIN_EIG_RATIO:
        return "INCONCLUSIVE"
    if angle_deg + 2 * sigma_deg <= ROT_DEG:
        return "PASS"
    return "FAIL" if angle_deg - 2 * sigma_deg > ROT_DEG else "INCONCLUSIVE"


def run(chain, j, t_imu, w_imu, R_SB_cfg) -> dict:
    r = fit(chain, j, t_imu, w_imu)
    ax = sweep_axes(chain, j, t_imu, w_imu, r["bias"], r["lag_s"], R_SB_cfg, r["R_SB"])
    ang = rot_angle_deg(r["R_SB"].T @ R_SB_cfg)
    rep = {"verdict": verdict(ang, r["sigma_deg"], r["eig_ratio"]), "R_BS_angle_deg": ang,
           "sigma_deg": r["sigma_deg"], "gate_deg": ROT_DEG, "eig_ratio": r["eig_ratio"],
           "R_BS_measured": r["R_SB"].T.tolist(), "half_turn_explained_by": flip_diagnosis(r["R_SB"], R_SB_cfg),
           "bias_rad_s": r["bias"].tolist(), "lag_s": r["lag_s"], "lag_at_edge": r["lag_at_edge"],
           "residual_rad_s": r["residual_rad_s"], "n_moving": r["n_moving"], "n_still": r["n_still"],
           "n_gap_masked": r["n_gap_masked"], "axes": ax, "warnings": []}
    if r["eig_ratio"] < MIN_EIG_RATIO:
        rep["warnings"].append(f"one neck joint barely moved (motion eigenvalue ratio {r['eig_ratio']:.3f} < "
                               f"{MIN_EIG_RATIO}): R_SB is not determined in every direction")
    if r["lag_at_edge"]:
        rep["warnings"].append("the best joint/IMU lag is at the edge of the search: clocks disagree")
    if "inter_axis_deg" in ax:
        th, tu = ax["inter_axis_deg"], ax["urdf_inter_axis_deg"]
        diff = min(abs(th - tu), abs(180.0 - th - tu))
        sig = ax.get("inter_axis_sigma_deg")
        if sig is not None and diff > max(INTER_AXIS_WARN_DEG, 3 * sig):
            rep["warnings"].append(f"the measured angle between the neck axes differs from the URDF's by {diff:.2f} deg "
                                   f"(> max({INTER_AXIS_WARN_DEG}, 3 sigma = {3 * sig:.2f})): the URDF's neck geometry is "
                                   "off, not (only) T_BS")
    return rep


def main() -> int:
    import argparse
    import neckfk
    import okviscfg
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rig", type=Path, help="accepted for a uniform runner; unused")
    a = ap.parse_args()
    chain, cfg = neckfk.NeckChain(a.urdf), okviscfg.load(a.config)
    j = necktake.read_joints(a.session)
    t, w, _ = necktake.read_imu(a.export)
    rep = run(chain, j, t, w, cfg.T_BS[:3, :3].T)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    ax = rep["axes"]
    nan = float("nan")
    print(f"step 82: {rep['n_moving']} moving samples ({rep['n_gap_masked']} masked at /joints gaps), residual "
          f"{rep['residual_rad_s']:.4f} rad/s, joint lag {rep['lag_s'] * 1e3:+.0f} ms, motion eigenvalue ratio {rep['eig_ratio']:.2f}")
    print(f"  measured vs config R_BS: {rep['R_BS_angle_deg']:.3f} +- {rep['sigma_deg']:.3f} deg (within-take sigma; gate "
          f"{ROT_DEG}, PASS needs angle + 2 sigma inside it; compare takes for the take-to-take part)")
    print(f"  sweeps: pitch {ax['n_pitch']} samples (other joint {ax['pitch_coupling'] or nan:.3f} of it), yaw "
          f"{ax['n_yaw']} ({ax['yaw_coupling'] or nan:.3f}); pitch axis {ax.get('pitch_axis_deg', nan):.3f} deg, yaw axis "
          f"{ax.get('yaw_axis_deg', nan):.3f} deg off the config; angle between the axes "
          f"{ax.get('inter_axis_deg', nan):.3f} +- {ax.get('inter_axis_sigma_deg') or nan:.3f} over "
          f"{ax['n_pitch_strokes']} pitch / {ax['n_yaw_strokes']} yaw strokes (+ minus - strokes "
          f"{ax.get('inter_axis_dir_diff_deg', nan):+.2f}; URDF {ax['urdf_inter_axis_deg']:.3f}; torso co-rotation not "
          "removed)")
    if rep["half_turn_explained_by"]:
        print(f"  half a turn from the config: the {rep['half_turn_explained_by']} encoder sign is reversed "
              "against its URDF axis, or the IMU is mounted half a turn round; step 75's camera sign check "
              "tells which")
    for warn in rep["warnings"]:
        print(f"  WARNING: {warn}")
    print(f"VERDICT {rep['verdict']}  -> {a.out}")
    return {"PASS": 0, "FAIL": 1}.get(rep["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
