#!/usr/bin/env python3
"""Step 81 — does phantom12 need its own camera-in-Head? The hand-eye fit on one neck take.

With the torso and the board still and only the neck moving, every image of a held pose closes
the loop  T_TB(q) · X · C_c = Z · B  (handeye.py). Fitting X (camera 0 in the Head link) and Z
(the board in the torso) over all held poses measures where the camera really sits in the head.
--model pixels (the default, handeye_px.py) fits X and Z to every board corner's pixels in both
cameras; --model poses (handeye.py) fits them to per-image board poses, which on the 17 Oct
powered-neck takes put the two cameras' own fits 12 mm apart where the pixel-level fits put them 5 mm
apart, from the same images. Those 5 mm (and 0.15 deg) are the same in every half of both takes: the
two cameras' calibrations (intrinsics or stereo pose) disagree, by far more than the fits' noise
(fitted on the same holds, their shared joint error cancels in the difference). So with pixels X's
sigma is the fit's statistical sigma (clustered by hold) and, in quadrature, a systematic of half the
distance between each camera's fit alone on the holds both cameras saw, both reported. Compared with
the config's X = T_BS · T_SC0, with X's sigma:

    KEEP          X within GATE_DEG and GATE_MM of the config even at 2 sigma (about 5 mm at the
                  gripper at 0.85 m): the config stays;
    REPLACE       not shown KEEP, and X's 2 sigma inside the gate: the measurement is
                  good to the gate, so T_BS_measured = X · T_SC0^-1 replaces the config (step 83),
                  whether the config is outside the gate or only not shown inside it;
    IMPROVE       the config shown outside the gate (its difference minus 2 sigma beyond the gate, in
                  rotation or translation) while X's 2 sigma is not inside it, and in each of
                  rotation and translation the fit either good to the gate (2 sigma inside it) or shown
                  better than the config (the difference over 4 sigma): the measurement is better than
                  the config but not good to the gate, and step 83 writes it with its error bar (the
                  user's rule of 16 Oct: the config stays only within the gate, else the measurement
                  replaces it);
    INCONCLUSIVE  anything else, or fewer than MIN_VERDICT_HOLDS holds, a direction the data cannot
                  see, no convergence, or the model not fitting (pixels: an rms over MAX_FIT_PX or more
                  than MAX_OUTLIER_FRACTION of the corners cut; poses: a large spread after the fit),
                  or (pixels) no cross-check between the cameras (fewer than the verdict's holds seen
                  by both, or a camera's fit alone blind in some direction): neither is shown good.
                  Every report carries the arithmetic behind its verdict (verdict_reason).

Also reported: X's error bars; with --model pixels each camera's own fit, their distance and each
hold's pixel rms; with --model poses the fit's residual at the config's X against the fitted one,
the board spread in the torso frame before (config X) and after (fitted X), joint play from the side
each hold was reached from and holds whose residual stands out; and with --compare, the agreement
with another take's fit: within AGREE_DEG / AGREE_MM, or within 2 combined statistical sigma when each
fit's statistical sigma is within the gate and the takes are within the gate of each other (the
cameras' systematic is the same in both takes and cancels in their difference; each take's
statistical sigma, ~2.4 mm on 17 Oct, is over AGREE_MM, which the fixed limits alone cannot be met within). A NeckPitch encoder zero error is part of the measured
X (it holds for the current zero only); a NeckYaw zero error goes into Z and is invisible here. Images
whose /joints samples are more than GAP_MS apart are not used (their angles would be interpolated
across a gap). See calibration.md section 6b.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handeye  # noqa: E402
import handeye_px  # noqa: E402
import rigmath as rm  # noqa: E402

GATE_DEG, GATE_MM, REACH_M = 0.3, 3.0, 0.85     # the user's budget for manipulation, 2026-10-16
AGREE_DEG, AGREE_MM = 0.1, 1.0                  # two takes must agree this well
MIN_HOLDS = 6                                   # below: no fit at all
MIN_VERDICT_HOLDS = 20                          # below: the clustered error bars are too optimistic for a verdict
MIN_TAGS, MAX_RMS_PX = 20, 1.0                  # a board pose needs this many tags and a pixel fit this good
GAP_MS = 20.0                                   # an image whose /joints samples are further apart is not used
SPREAD_MM, SPREAD_DEG = 2.0, 0.1                # poses: spread after the fit beyond these: the model does not fit
MAX_FIT_PX = 1.5                                # pixels: rms after the fit beyond this: the model does not fit
                                                # (17 Oct: powered neck 0.74-0.77 px; by hand 3.0-6.7 px over every corner)
MAX_OUTLIER_FRACTION = 0.02                     # pixels: a larger share of corners cut: the model does not fit
                                                # (17 Oct: powered 0.4-0.5 %, by hand 25-60 %; conservative: a
                                                # powered take at ~1.2 px rms can cut 1-3 % and trip it)
MAX_RADIUS_PX = 450.0                           # corners further from the image centre: the lens model's outer ring


def pose_from_corners(det: dict, board, lens, K, D, centre, max_radius_px=MAX_RADIUS_PX, min_tags=MIN_TAGS,
                      max_rms_px=MAX_RMS_PX):
    """Board pose in the camera (T_CW) from detected corners {(tag, k): (u, v)}: corners beyond
    max_radius_px from the image centre are dropped (the lens model's outer ring, section 8.2); the
    PnP on undistorted points (which weighs edge corners ~4x) is refined on the pixel reprojection
    error; None with fewer than min_tags tags or a pixel rms above max_rms_px."""
    import cv2
    validate = importlib.import_module("60_validate_stereo")
    keys = [k for k, (u, v) in det.items() if np.hypot(u - centre[0], v - centre[1]) <= max_radius_px]
    if len({t for t, _ in keys}) < min_tags:
        return None
    obj = np.array([board.obj_points(t)[k] for t, k in keys], np.float64)
    c = np.array([det[k] for k in keys], np.float64)
    R, t = validate._pose(lens, c, obj, K, D)
    if R is None:
        return None
    o3 = obj.reshape(-1, 1, 3)

    def res(p):
        pr, _ = lens.project(o3, p[:3].reshape(3, 1), p[3:].reshape(3, 1), K, D)
        return (pr.reshape(-1, 2) - c).ravel()

    p = np.r_[cv2.Rodrigues(R)[0].ravel(), t]
    for _ in range(10):
        r = res(p)
        J = np.empty((r.size, 6))
        for i in range(6):
            d = np.zeros(6)
            d[i] = 1e-7
            J[:, i] = (res(p + d) - res(p - d)) / 2e-7
        dp = -np.linalg.lstsq(J, r, rcond=None)[0]
        p = p + dp
        if np.linalg.norm(dp) < 1e-10:
            break
    rms = float(np.sqrt(np.mean(np.sum(res(p).reshape(-1, 2) ** 2, axis=1))))
    if rms > max_rms_px:
        return None
    return rm.transform(cv2.Rodrigues(p[:3])[0], p[3:])


LENS_OF_MODEL = {"KB4T": "pinhole-kb4t", "KB4": "pinhole-equi"}


def lens_from_fit(cfg, path, key=None, okvis_order=(0, 1)):
    """The OKVIS config with another calibration's lens and stereo pose: K, D and the lens name from a lens fit's
    JSON entry (model, intr per camera [fx, fy, cx, cy, *distortion], C = camera 1 in camera 0, mapping camera-1
    points into camera 0), and camera 1's T_SC = T_SC0 · C so that every C this step derives is the fit's.
    T_SC0 and T_BS stay the config's: X is still compared with T_BS · T_SC0, and step 83 writes
    T_BS = X · T_SC0^-1 into a config with that T_SC0. (Fit A of 20 Oct: results/phantom12-2026-10-20-lens.)
    The fit's cameras are Kalibr's (step 45's camera c is the rig's kalibr[c]); OKVIS camera i is Kalibr camera
    okvis_order[i], so a rig in order [1, 0] (mk13) gets the two lenses swapped and C inverted. key None: the file
    is one entry (step 45's output)."""
    where = f"{path}:{key}" if key else str(path)
    e = json.loads(Path(path).read_text())
    e = e[key] if key else e
    if e.get("model") not in LENS_OF_MODEL:
        raise ValueError(f"{where}: model {e.get('model')!r}; this step fits with {sorted(LENS_OF_MODEL)}")
    if "intr" not in e or "C" not in e:
        raise ValueError(f"{where}: no calibration in it (verdict {e.get('verdict')!r})")
    order = [int(k) for k in okvis_order]
    if sorted(order) != [0, 1]:
        raise ValueError(f"okvis_order {order}: two cameras, [0, 1] or [1, 0]")
    C = np.asarray(e["C"], float)
    if np.abs(C[:3, :3] @ C[:3, :3].T - np.eye(3)).max() > 1e-9 or not np.allclose(C[3], [0, 0, 0, 1]):
        raise ValueError(f"{where}: C is not a rotation and translation")
    if order == [1, 0]:
        C = rm.inv(C)
    intr = [e["intr"][k] for k in order]
    K = [np.array([[v[0], 0.0, v[2]], [0.0, v[1], v[3]], [0.0, 0.0, 1.0]]) for v in intr]
    D = [np.asarray(v[4:], float).reshape(-1, 1) for v in intr]
    return dataclasses.replace(cfg, K=K, D=D, lens=LENS_OF_MODEL[e["model"]], T_SC=[cfg.T_SC[0], cfg.T_SC[0] @ C])


def compare(X, X_ref, reach_m: float = REACH_M) -> dict:
    """How far X is from X_ref: rotation in Head axes [deg], camera position in Head axes [mm], and
    the worst-case error at a point reach_m from the camera, 2 r sin(angle / 2) + |dt| [mm]."""
    w = np.degrees(rm.log_rot(X[:3, :3] @ X_ref[:3, :3].T))
    dt = (X[:3, 3] - X_ref[:3, 3]) * 1e3
    ang = rm.rot_angle_deg(X[:3, :3] @ X_ref[:3, :3].T)
    return {"rot_head_deg": w.tolist(), "angle_deg": ang, "trans_head_mm": dt.tolist(),
            "dist_mm": float(np.linalg.norm(dt)),
            "gripper_mm": float(2 * reach_m * np.sin(np.radians(ang) / 2) * 1e3 + np.linalg.norm(dt))}


def verdict(cmp: dict, sigma_deg: float = 0.0, sigma_mm: float = 0.0, usable: bool = True,
            gate_deg: float = GATE_DEG, gate_mm: float = GATE_MM) -> str:
    """KEEP / REPLACE / IMPROVE / INCONCLUSIVE from the difference to the config and the fit's sigma
    (total rotation [deg] and translation [mm]); see the module docstring."""
    if not usable:
        return "INCONCLUSIVE"
    if cmp["angle_deg"] + 2 * sigma_deg <= gate_deg and cmp["dist_mm"] + 2 * sigma_mm <= gate_mm:
        return "KEEP"
    if 2 * sigma_deg <= gate_deg and 2 * sigma_mm <= gate_mm:
        return "REPLACE"        # the fit is good to the gate: use it, whether or not the config is outside
    outside = cmp["angle_deg"] - 2 * sigma_deg > gate_deg or cmp["dist_mm"] - 2 * sigma_mm > gate_mm
    rot_ok = 2 * sigma_deg <= gate_deg or cmp["angle_deg"] > 4 * sigma_deg      # good to the gate, or better
    trans_ok = 2 * sigma_mm <= gate_mm or cmp["dist_mm"] > 4 * sigma_mm         # than the config there
    if outside and rot_ok and trans_ok:
        return "IMPROVE"        # the config is shown outside the gate; the fit is better, if not good to it
    return "INCONCLUSIVE"


def observations(frames, chain, cfg):
    """frames: [{hold, cam, yaw, pitch, T_CW, approach_yaw, approach_pitch}] -> (obs, meta) for
    handeye.fit, where obs = (T_TB(q), camera cam in camera 0, camera cam in the board frame)."""
    C = [rm.inv(cfg.T_SC[0]) @ T for T in cfg.T_SC]
    obs = [(chain.T_TB(f["yaw"], f["pitch"]), C[f["cam"]], rm.inv(f["T_CW"])) for f in frames]
    return obs, frames


def board_in_torso(obs, X) -> list:
    """Z_i = A_i X C B_i^-1: where each image puts the board in the torso frame, given X."""
    return [A @ X @ C @ rm.inv(B) for A, C, B in obs]


def _rms(res) -> dict:
    res = np.asarray(res)
    return {"rot_deg": float(np.degrees(np.sqrt(np.mean(np.sum(res[:, :3] ** 2, 1)))) / handeye.W_ROT_M),
            "trans_mm": float(1e3 * np.sqrt(np.mean(np.sum(res[:, 3:] ** 2, 1))))}


def play(obs, meta, X, Z, chain) -> dict:
    """Joint play [deg]: per hold, the mean residual rotation of its images about each joint's axis
    (torso frame); then the mean over holds reached with the joint last moving in + minus those
    reached moving in -. Positive: the head stops short of the encoder in the direction it came
    from (backlash); negative: it runs ahead (as with a hand loading an unpowered neck, 16 Oct).
    With its standard error and the hold counts [+, -]; None without both sides."""
    out = {}
    for joint, key in (("yaw", "approach_yaw"), ("pitch", "approach_pitch")):
        per = {}
        for (A, C, B), m in zip(obs, meta):
            s = m.get(key, 0)
            if s == 0:
                continue
            E = (A @ X @ C) @ rm.inv(Z @ B)                      # torso-frame discrepancy
            axis = chain.axes_in_T(m["yaw"], m["pitch"])[0 if joint == "yaw" else 1]
            per.setdefault((m["hold"], s), []).append(np.degrees(rm.log_rot(E[:3, :3]) @ axis))
        vals = {1: [np.mean(v) for (h, s), v in per.items() if s == 1], -1: [np.mean(v) for (h, s), v in per.items() if s == -1]}
        n = [len(vals[1]), len(vals[-1])]
        if n[0] and n[1]:
            d = float(np.mean(vals[1]) - np.mean(vals[-1]))
            se = (float(np.sqrt(np.var(vals[1], ddof=1) / n[0] + np.var(vals[-1], ddof=1) / n[1]))
                  if min(n) > 1 else None)
        else:
            d, se = None, None
        out[f"{joint}_deg"], out[f"{joint}_se_deg"], out[f"{joint}_holds"] = d, se, n
    return out


def hold_residuals(res, meta) -> list:
    """Per hold: rms residual rotation [deg] and translation [mm] of its images after the fit."""
    by = {}
    for r, m in zip(res, meta):
        by.setdefault(m["hold"], []).append(r)
    out = []
    for h, rs in sorted(by.items()):
        out.append({"hold": int(h), "n": len(rs), **_rms(rs)})
    return out


def outlier_holds(hr, factor: float = 3.0) -> list:
    """Holds whose residual rotation or translation is over factor times the median (floors 0.01 deg,
    0.2 mm)."""
    if not hr:
        return []
    mr = max(np.median([h["rot_deg"] for h in hr]), 0.01)
    mt = max(np.median([h["trans_mm"] for h in hr]), 0.2)
    return [h["hold"] for h in hr if max(h["rot_deg"] / mr, h["trans_mm"] / mt) > factor]


def run(frames, chain, cfg, min_verdict_holds: int = MIN_VERDICT_HOLDS) -> dict:
    holds = {f["hold"] for f in frames}
    if len(holds) < MIN_HOLDS:
        return {"verdict": "INSUFFICIENT", "n_holds": len(holds), "n_frames": len(frames)}
    obs, meta = observations(frames, chain, cfg)
    X_cfg = cfg.T_BS @ cfg.T_SC[0]
    r = handeye.fit(obs, X_cfg, groups=[m["hold"] for m in meta])   # error bars clustered by hold
    X, Z = r["X"], r["Z"]
    cmp = compare(X, X_cfg)
    hr = hold_residuals(r["residuals"], meta)
    per_cam = {}
    for c in sorted({f["cam"] for f in frames}):
        sel = [o for o, m in zip(obs, meta) if m["cam"] == c]
        per_cam[str(c)] = rm.spread(board_in_torso(sel, X))
    Z_cfg = rm.mean_pose(board_in_torso(obs, X_cfg))
    sigma_deg = float(np.linalg.norm(r["sigma_rot_head_deg"]))
    sigma_mm = float(np.linalg.norm(r["sigma_trans_head_mm"]))
    rep = {"n_holds": len(holds), "n_frames": len(frames),
           "vs_config": cmp, "sigma_deg": sigma_deg, "sigma_mm": sigma_mm,
           "gate": {"deg": GATE_DEG, "mm": GATE_MM, "reach_m": REACH_M},
           "fit": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()
                   if k not in ("X", "Z", "residuals")},
           "residual_at_config": _rms(handeye.residuals(X_cfg, Z_cfg, obs)), "residual_fitted": _rms(r["residuals"]),
           "T_BC_measured": X.tolist(), "T_BS_measured": (X @ rm.inv(cfg.T_SC[0])).tolist(),
           "T_TW": Z.tolist(),
           "spread_before": rm.spread(board_in_torso(obs, X_cfg)), "spread_after": rm.spread(board_in_torso(obs, X)),
           "spread_after_per_camera": per_cam, "play": play(obs, meta, X, Z, chain), "hold_residuals": hr,
           "outlier_holds": outlier_holds(hr)}
    warn, why = [], []
    if not r["converged"]:
        warn.append("the fit did not converge")
        why.append("no convergence")
    if r["cond"] > handeye.MAX_COND:
        warn.append(f"the data cannot see every direction of X (Jacobian condition {r['cond']:.1e}: a joint that "
                    "barely moved?); along it X stays at the config, so this verdict does not measure it")
        why.append("an unseen direction")
    if rep["spread_after"]["trans_rms_mm"] > SPREAD_MM or rep["spread_after"]["rot_rms_deg"] > SPREAD_DEG:
        warn.append("large spread after the fit: the head is not where the encoders put it (a hand loading an "
                    "unpowered neck, 16 Oct), the joints have play, the neck does not move as the URDF says, or the "
                    "torso or board moved (see play, outlier_holds)")
        why.append("the model does not fit")
    if len(holds) < min_verdict_holds:
        why.append(f"{len(holds)} holds < {min_verdict_holds}")
    return _decide(rep, cmp, sigma_deg, sigma_mm, why, warn)


def run_pixels(frames, chain, cfg, min_verdict_holds: int = MIN_VERDICT_HOLDS) -> dict:
    """The pixel-level fit (handeye_px.py) of one take, over both cameras and over each alone. frames:
    _frames' output (each with its corners P and pixels uv). The report has run()'s keys that step 83
    and --compare read (verdict, T_BC_measured, sigma_deg, sigma_mm, vs_config)."""
    import rigcfg
    holds = {f["hold"] for f in frames}
    if len(holds) < MIN_HOLDS:
        return {"model": "pixels", "verdict": "INSUFFICIENT", "n_holds": len(holds), "n_frames": len(frames)}
    kind = rigcfg.Lens(cfg.lens).kind
    C = [rm.inv(cfg.T_SC[0]) @ T for T in cfg.T_SC]
    X_cfg = cfg.T_BS @ cfg.T_SC[0]
    f0 = frames[0]
    Z0 = chain.T_TB(f0["yaw"], f0["pitch"]) @ X_cfg @ C[f0["cam"]] @ f0["T_CW"]     # T_TW = T_TC · T_CW

    def fit(sel):
        ims = [handeye_px.Image(0, f["cam"], chain.T_TB(f["yaw"], f["pitch"]), f["P"], f["uv"], str(f["hold"]))
               for f in sel]
        r = handeye_px.fit(ims, X_cfg, [Z0], C, cfg.K, cfg.D, kind)
        return r, float(np.linalg.norm(r["sigma_rot_head_deg"])), float(np.linalg.norm(r["sigma_trans_head_mm"]))

    r, stat_deg, stat_mm = fit(frames)
    X = r["X"]
    cmp = compare(X, X_cfg)
    # the cross-check: each camera's fit alone on the holds both cameras saw (fitted on the same holds,
    # their shared joint error cancels in the difference, so half the distance is the cameras' systematic)
    both = sorted({f["hold"] for f in frames if f["cam"] == 0} & {f["hold"] for f in frames if f["cam"] == 1})
    per_cam, no_cross = {}, None
    sys_deg = sys_mm = float("inf")
    if len(both) >= min_verdict_holds:
        for c in (0, 1):
            rc, sd, sm = fit([f for f in frames if f["cam"] == c and f["hold"] in both])
            per_cam[str(c)] = {"vs_config": compare(rc["X"], X_cfg), "sigma_deg": sd, "sigma_mm": sm,
                               "rms_px": rc["rms_px"], "T_BC_measured": rc["X"].tolist()}
        if all(np.isfinite([pc["sigma_deg"], pc["sigma_mm"]]).all() for pc in per_cam.values()):
            d = compare(np.array(per_cam["0"]["T_BC_measured"]), np.array(per_cam["1"]["T_BC_measured"]))
            sys_deg, sys_mm = d["angle_deg"] / 2, d["dist_mm"] / 2
        else:
            no_cross = "a camera's fit alone cannot see every direction of X"
    else:
        no_cross = f"{len(both)} holds seen by both cameras < {min_verdict_holds}"
    sigma_deg, sigma_mm = float(np.hypot(stat_deg, sys_deg)), float(np.hypot(stat_mm, sys_mm))
    rep = {"model": "pixels", "n_holds": len(holds), "n_frames": len(frames), "vs_config": cmp,
           "sigma_deg": sigma_deg, "sigma_mm": sigma_mm, "sigma_stat_deg": stat_deg, "sigma_stat_mm": stat_mm,
           "sigma_sys_deg": sys_deg, "sigma_sys_mm": sys_mm,
           "gate": {"deg": GATE_DEG, "mm": GATE_MM, "reach_m": REACH_M},
           "fit": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if k not in ("X", "Zs")},
           "T_BC_measured": X.tolist(), "T_BS_measured": (X @ rm.inv(cfg.T_SC[0])).tolist(),
           "T_TW": r["Zs"][0].tolist(), "per_camera": per_cam, "cross_check_holds": len(both)}
    warn, why = [], []
    if not r["converged"]:
        warn.append("the fit did not converge")
        why.append("no convergence")
    if r["cond"] > handeye_px.MAX_COND:
        warn.append(f"the data cannot see every direction of X (Jacobian condition {r['cond']:.1e}: a joint that "
                    "barely moved?); along it X stays at the config, so this verdict does not measure it")
        why.append("an unseen direction")
    if r["rms_px"] > MAX_FIT_PX or r["outlier_fraction"] > MAX_OUTLIER_FRACTION:
        warn.append(f"pixel rms {r['rms_px']:.2f} after the fit ({r['rms_all_px']:.2f} over every corner), "
                    f"{r['outlier_fraction']:.1%} of the corners cut (limits {MAX_FIT_PX} px, "
                    f"{MAX_OUTLIER_FRACTION:.0%}): the head is not where the encoders put it (a hand loading an "
                    "unpowered neck: the 16 Oct takes, rerun 18 Oct, 3.0-6.7 px over every corner, 25-60 % cut), the neck "
                    "does not move as the URDF says, or "
                    "the torso or board moved (see fit.group_rms_px, per hold)")
        why.append("the model does not fit")
    if no_cross is None:
        rep["cameras_apart"] = {"angle_deg": d["angle_deg"], "dist_mm": d["dist_mm"],
                                "trans_head_mm": d["trans_head_mm"], "rot_head_deg": d["rot_head_deg"]}
        if d["angle_deg"] > GATE_DEG or d["dist_mm"] > GATE_MM:
            warn.append(f"each camera's fit alone ({len(both)} shared holds): {d['angle_deg']:.3f} deg / "
                        f"{d['dist_mm']:.2f} mm apart, beyond the gate: the two cameras' calibrations (intrinsics or "
                        "stereo pose) disagree; half of it is in X's sigma as a systematic")
    else:
        warn.append(f"no cross-check between the cameras ({no_cross}): X's systematic error is unknown")
        why.append("no cross-check between the cameras")
    if len(holds) < min_verdict_holds:
        why.append(f"{len(holds)} holds < {min_verdict_holds}")
    return _decide(rep, cmp, sigma_deg, sigma_mm, why, warn)

def verdict_reason(cmp: dict, sigma_deg: float, sigma_mm: float, gate_deg: float = GATE_DEG,
                   gate_mm: float = GATE_MM) -> str:
    """The arithmetic behind verdict(): each rule with this fit's numbers."""
    a, d, sd, sm = cmp["angle_deg"], cmp["dist_mm"], sigma_deg, sigma_mm
    return (f"X - config {a:.3f} deg / {d:.2f} mm, X's 1 sigma {sd:.3f} deg / {sm:.2f} mm. KEEP needs "
            f"{a + 2 * sd:.3f} deg <= {gate_deg} and {d + 2 * sm:.2f} mm <= {gate_mm}; REPLACE needs 2 sigma "
            f"{2 * sd:.3f} deg <= {gate_deg} and {2 * sm:.2f} mm <= {gate_mm}; IMPROVE needs the config outside "
            f"({a - 2 * sd:.3f} deg > {gate_deg} or {d - 2 * sm:.2f} mm > {gate_mm}) and each part good to the gate "
            f"or over 4 sigma ({4 * sd:.3f} deg, {4 * sm:.2f} mm)")


def _decide(rep: dict, cmp: dict, sigma_deg: float, sigma_mm: float, why: list, warn: list) -> dict:
    """The verdict, its arithmetic, and a reason whenever it is INCONCLUSIVE."""
    rep["verdict"] = verdict(cmp, sigma_deg, sigma_mm, usable=not why)
    rep["verdict_reason"] = verdict_reason(cmp, sigma_deg, sigma_mm)
    if rep["verdict"] == "INCONCLUSIVE" and not why:
        why.append("neither the config nor the fit shown good to the gate at 2 sigma, and the fit not shown "
                   "better than the config (see verdict_reason)")
    rep["inconclusive_because"] = why
    rep["warnings"] = warn
    return rep


def _json_safe(o):
    """The report as strict JSON: numpy types as Python ones, non-finite numbers (an infinite sigma: no
    cross-check, an unseen direction) as null; compare_takes reads a null sigma back as infinite."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _sigma(r: dict, key: str) -> float:
    v = r.get(key)
    return float("inf") if v is None else float(v)


def compare_takes(rep: dict, other: dict) -> dict:
    """How far two takes' fits are apart; they agree within AGREE_DEG / AGREE_MM (agree_limits), or
    (agree_sigma) within 2 combined sigma when each fit's sigma is within the gate and the takes are
    within the gate of each other: with a take's own sigma over AGREE_MM, the fixed limits alone would
    refuse two takes that agree within their errors. The sigma is each fit's statistical sigma
    (sigma_stat_*, pixels; sigma_* otherwise): the cameras' systematic is the same in both takes and
    cancels in their difference. why_not says which condition failed."""
    c = compare(np.array(rep["T_BC_measured"]), np.array(other["T_BC_measured"]))
    lim = c["angle_deg"] <= AGREE_DEG and c["dist_mm"] <= AGREE_MM
    out = {"angle_deg": c["angle_deg"], "dist_mm": c["dist_mm"], "agree_limits": lim, "agree_sigma": False}
    why = [] if lim else [f"over the limits {AGREE_DEG} deg / {AGREE_MM} mm"]
    if "sigma_deg" in rep and "sigma_deg" in other:
        own = [(_sigma(r, "sigma_stat_deg" if "sigma_stat_deg" in r else "sigma_deg"),
                _sigma(r, "sigma_stat_mm" if "sigma_stat_mm" in r else "sigma_mm")) for r in (rep, other)]
        sd = float(np.hypot(own[0][0], own[1][0]))
        sm = float(np.hypot(own[0][1], own[1][1]))
        out["combined_sigma_deg"], out["combined_sigma_mm"] = sd, sm
        precise = all(d <= GATE_DEG and m <= GATE_MM for d, m in own)
        close = c["angle_deg"] <= GATE_DEG and c["dist_mm"] <= GATE_MM
        within = c["angle_deg"] <= 2 * sd and c["dist_mm"] <= 2 * sm
        out["agree_sigma"] = bool(precise and close and within)
        why += ([] if precise else ["a take's statistical sigma is over the gate"]) + \
               ([] if close else ["further apart than the gate"]) + ([] if within else ["outside 2 combined sigma"])
    else:
        why.append("no sigma to compare with")
    out["agree"] = bool(lim or out["agree_sigma"])
    out["why_not"] = [] if out["agree"] else why
    return out


def in_joint_gap(j, t_ns, max_ms: float = GAP_MS) -> np.ndarray:
    """True for stamps whose bracketing /joints samples are more than max_ms apart."""
    t_ns = np.asarray(t_ns, np.int64)
    i = np.clip(np.searchsorted(j.t, t_ns), 1, len(j.t) - 1)
    return (j.t[i] - j.t[i - 1]) > max_ms * 1e6


def _frames(a, rig, cfg, joints, holds, drop, max_radius_px=MAX_RADIUS_PX):
    """-> (frames, number of images left out because they fall in a /joints gap). Each frame carries its
    board pose (T_CW) and the corners it was found from: P (n x 3, on the board) and uv (n x 2, pixels)."""
    import cv2
    import necktake
    import rigcfg
    lens = rigcfg.Lens(cfg.lens)
    centre = (rig.resolution[0] / 2.0, rig.resolution[1] / 2.0)
    out, n_gap = [], 0
    for cam, k in enumerate(rig.okvis_order):
        t, paths = necktake.read_frames(a.export, rig.kalibr[k]["record_dir"])
        for hi, h in enumerate(holds):
            if hi in drop:
                continue
            idx = np.flatnonzero((t >= h.t0) & (t <= h.t1))
            if idx.size:
                gap = in_joint_gap(joints, t[idx])
                n_gap += int(gap.sum())
                idx = idx[~gap]
            for i in idx[np.linspace(0, idx.size - 1, min(a.per_hold, idx.size)).astype(int)] if idx.size else []:
                im = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
                det = rig.board.detect(im)
                T = pose_from_corners(det, rig.board, lens, cfg.K[cam], cfg.D[cam], centre, max_radius_px=max_radius_px)
                if T is None:
                    continue
                qy, qp = necktake.at(joints, np.array([t[i]]))
                keys = [kk for kk, (u, v) in det.items() if np.hypot(u - centre[0], v - centre[1]) <= max_radius_px]
                out.append({"hold": hi, "cam": cam, "yaw": float(qy[0]), "pitch": float(qp[0]), "T_CW": T,
                            "approach_yaw": h.approach_yaw, "approach_pitch": h.approach_pitch,
                            "P": np.array([rig.board.obj_points(tg)[cn] for tg, cn in keys]),
                            "uv": np.array([det[kk] for kk in keys], float)})
    return out, n_gap


def _frames_from_corners(path, rig, cfg, per_hold, drop, max_radius_px=MAX_RADIUS_PX):
    """-> (frames, {hold: necktake.Hold} of every hold, dropped ones too): step 81's frames from a take's corner file
    instead of its export
    (results/phantom12-2026-10-17-tbs/analysis/dump_corners.py layout: per stereo pair img_hold, img_t0,
    img_yaw, img_pitch (rad, the neck at the left image's stamp), img_ay, img_ap (the hold's approach sides);
    per corner det_img, det_cam (OKVIS camera), det_tag, det_k, det_u, det_v), so a take whose PNG export is
    gone (P1, P2: deleted 19 Oct) can be fitted. Up to per_hold pairs per hold, spread as _frames spreads its
    images, both cameras of each; the file has no images in /joints gaps (dump_corners.py left them out)."""
    import necktake
    import rigcfg
    d = dict(np.load(path))
    lens = rigcfg.Lens(cfg.lens)
    centre = (rig.resolution[0] / 2.0, rig.resolution[1] / 2.0)
    holds, out = {}, []
    for h in sorted(set(d["img_hold"].tolist())):
        idx = np.flatnonzero(d["img_hold"] == h)
        holds[h] = necktake.Hold(int(d["img_t0"][idx].min()), int(d["img_t0"][idx].max()),
                                 float(np.median(d["img_yaw"][idx])), float(np.median(d["img_pitch"][idx])),
                                 int(d["img_ay"][idx[0]]), int(d["img_ap"][idx[0]]))
        if h in drop:
            continue
        for i in idx[np.linspace(0, idx.size - 1, min(per_hold, idx.size)).astype(int)]:
            for cam in (0, 1):
                m = (d["det_img"] == i) & (d["det_cam"] == cam)
                det = {(int(t), int(k)): (float(u), float(v))
                       for t, k, u, v in zip(d["det_tag"][m], d["det_k"][m], d["det_u"][m], d["det_v"][m])}
                T = pose_from_corners(det, rig.board, lens, cfg.K[cam], cfg.D[cam], centre, max_radius_px=max_radius_px)
                if T is None:
                    continue
                keys = [kk for kk, (u, v) in det.items() if np.hypot(u - centre[0], v - centre[1]) <= max_radius_px]
                out.append({"hold": h, "cam": cam, "yaw": float(d["img_yaw"][i]), "pitch": float(d["img_pitch"][i]),
                            "T_CW": T, "approach_yaw": holds[h].approach_yaw, "approach_pitch": holds[h].approach_pitch,
                            "P": np.array([rig.board.obj_points(tg)[cn] for tg, cn in keys]),
                            "uv": np.array([det[kk] for kk in keys], float)})
    return out, holds


def main(argv: list | None = None) -> int:
    """The command line (argv: sys.argv[1:] when None). -> 0 KEEP, 1 REPLACE or IMPROVE, 2 anything else."""
    import argparse
    import necktake
    import neckfk
    import okviscfg
    import rigcfg
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, required=False, help="with --export")
    ap.add_argument("--export", type=Path, required=False, help="with --session")
    ap.add_argument("--corners", type=Path, help="instead of --session and --export: the take's corner file "
                                                 "(dump_corners.py layout; P1, P2 on Thor /root/calib-proto/data)")
    ap.add_argument("--max-radius-px", type=float, default=MAX_RADIUS_PX,
                    help=f"corners further from the image centre are not used (default {MAX_RADIUS_PX:g}: the old "
                         "lens model's outer ring)")
    ap.add_argument("--config", type=Path, required=True, help="OKVIS config with T_SC and T_BS")
    ap.add_argument("--lens-json", type=Path, help="a lens fit's JSON (model, intr, C): its lens and stereo pose "
                                                   "replace the config's; T_SC0 and T_BS stay the config's")
    ap.add_argument("--lens-key", help="the entry of --lens-json to use (Fit A: KB4T@all@800); without it the file "
                                       "is one entry (step 45's output)")
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--model", choices=("pixels", "poses"), default="pixels",
                    help="fit X to every board corner's pixels (default, handeye_px.py) or to per-image board poses "
                         "(handeye.py)")
    ap.add_argument("--per-hold", type=int, default=5, help="images used per hold and camera")
    ap.add_argument("--drop-holds", default="", help="comma-separated hold numbers to leave out (step 75 names them)")
    ap.add_argument("--min-verdict-holds", type=int, default=MIN_VERDICT_HOLDS,
                    help="fewer holds: INCONCLUSIVE (the clustered error bars are optimistic below ~20)")
    ap.add_argument("--compare", type=Path, help="another take's step-81 report")
    ap.add_argument("--qc", type=Path, help="step 75's --qc-out of this take: its verdict is recorded (step 83 "
                                            "refuses reports from takes that did not pass)")
    ap.add_argument("--out", type=Path, required=True)
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    if a.corners is None and (a.session is None or a.export is None):
        ap.error("give --corners, or --session and --export")
    if a.lens_key is not None and a.lens_json is None:
        ap.error("--lens-key needs --lens-json")
    rig, cfg, chain = rigcfg.load(a.rig), okviscfg.load(a.config), neckfk.NeckChain(a.urdf)
    if a.lens_json:
        cfg = lens_from_fit(cfg, a.lens_json, a.lens_key, rig.okvis_order)
    other, notes = None, []
    if a.compare:                                    # read first: a bad --compare must not cost the fit
        other = json.loads(a.compare.read_text()) if a.compare.exists() else None
        if other is None or "T_BC_measured" not in other:
            notes.append(f"--compare {a.compare.name}: no measured report there, not compared")
            other = None
    drop = {int(v) for v in a.drop_holds.split(",") if v.strip()}
    if a.corners:
        frames, holds = _frames_from_corners(a.corners, rig, cfg, a.per_hold, drop, a.max_radius_px)
        # dump_corners.py left out the images in /joints gaps and did not record how many: None, not 0
        n_gap, take = None, a.corners.name.replace("-corners.npz", "")
    else:
        joints = necktake.read_joints(a.session)
        found = necktake.find_holds(joints)
        frames, n_gap = _frames(a, rig, cfg, joints, found, drop, a.max_radius_px)
        holds, take = dict(enumerate(found)), a.session.parent.name
    rep = (run_pixels if a.model == "pixels" else run)(frames, chain, cfg, a.min_verdict_holds)
    rep["model"] = a.model
    rep["take"] = take
    rep["max_radius_px"] = a.max_radius_px
    centre = np.array(rig.resolution, float) / 2.0   # what was used, not only the arguments: the outermost corner
    rep["max_corner_radius_used_px"] = (float(max(np.hypot(*(f["uv"] - centre).T).max() for f in frames))
                                        if frames else None)
    rep["T_SC0"] = cfg.T_SC[0].tolist()              # step 83 refuses a target config with another camera pose
    n_dist = int(np.asarray(cfg.D[0]).size)          # the lens fitted through: 4 (KB4), 6 (KB4T)
    rep["lens"] = ({"json": str(a.lens_json), "key": a.lens_key, "lens": cfg.lens, "n_dist": n_dist} if a.lens_json
                   else {"config": str(a.config), "lens": cfg.lens, "n_dist": n_dist})
    rep["n_images_in_joint_gaps"] = n_gap
    rep["holds"] = [{"hold": i, "yaw": h.yaw, "pitch": h.pitch, "approach": [h.approach_yaw, h.approach_pitch],
                     "dropped": i in drop} for i, h in sorted(holds.items())]
    if other is not None and "T_BC_measured" in rep:
        rep["vs_other_take"] = compare_takes(rep, other)
    if a.qc:
        rep["qc_verdict"] = json.loads(a.qc.read_text())["verdict"] if a.qc.exists() else "MISSING"
        if rep["qc_verdict"] != "GOOD":
            notes.append(f"step 75 said {rep['qc_verdict']} for this take: this verdict is a diagnostic, not the decision")
    rep.setdefault("warnings", []).extend(notes)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(_json_safe(rep), indent=1, allow_nan=False))
    gaps = (f"{n_gap} images in /joints gaps left out" if n_gap is not None
            else "images in /joints gaps left out by dump_corners.py, how many not recorded")
    print(f"step 81: {len(holds)} holds found, {rep['n_holds']} used, {rep['n_frames']} board images ({gaps})")
    if "vs_config" in rep:
        c, f = rep["vs_config"], rep["fit"]
        print(f"  camera-in-Head vs config: {c['angle_deg']:.3f} deg (Head x y z: "
              + " ".join(f"{v:+.3f}" for v in c["rot_head_deg"]) + f"), {c['dist_mm']:.2f} mm ("
              + " ".join(f"{v:+.2f}" for v in c["trans_head_mm"]) + f"); ~{c['gripper_mm']:.1f} mm at "
              f"{REACH_M} m. Gate {GATE_DEG} deg / {GATE_MM} mm, at 2 sigma")
        print("  the fit's 1-sigma (clustered by hold): " + " ".join(f"{v:.3f}" for v in f["sigma_rot_head_deg"]) + " deg, "
              + " ".join(f"{v:.2f}" for v in f["sigma_trans_head_mm"]) + " mm (total "
              f"{rep.get('sigma_stat_deg', rep['sigma_deg']):.3f} deg, {rep.get('sigma_stat_mm', rep['sigma_mm']):.2f} mm); "
              f"Jacobian condition {f['cond']:.0f}")
        if a.model == "pixels":
            print(f"  X's sigma (1 sigma) {rep['sigma_deg']:.3f} deg / {rep['sigma_mm']:.2f} mm: the fit's statistical "
                  f"{rep['sigma_stat_deg']:.3f} deg / {rep['sigma_stat_mm']:.2f} mm and the cameras' systematic "
                  f"{rep['sigma_sys_deg']:.3f} deg / {rep['sigma_sys_mm']:.2f} mm (half their distance) in quadrature")
            print(f"  pixel rms after the fit {f['rms_px']:.3f} px over {f['n_corners']} corners of {f['n_images']} "
                  f"images, {f['rms_all_px']:.3f} px over every corner ({f['n_outliers']} = "
                  f"{f['outlier_fraction']:.1%} left out beyond {handeye_px.OUTLIER_PX} px; the model fits up to "
                  f"{MAX_FIT_PX} px and {MAX_OUTLIER_FRACTION:.0%})")
            for c, pc in rep["per_camera"].items():
                v = pc["vs_config"]
                print(f"  camera {c} alone: {v['angle_deg']:.3f} deg, {v['dist_mm']:.2f} mm ("
                      + " ".join(f"{x:+.2f}" for x in v["trans_head_mm"]) + f") +- {pc['sigma_deg']:.3f} deg / "
                      f"{pc['sigma_mm']:.2f} mm, {pc['rms_px']:.3f} px")
            if "cameras_apart" in rep:
                ca = rep["cameras_apart"]
                print(f"  each camera's fit alone ({rep['cross_check_holds']} shared holds) {ca['angle_deg']:.3f} deg / "
                      f"{ca['dist_mm']:.2f} mm apart (Head x y z " + " ".join(f"{x:+.2f}" for x in ca["trans_head_mm"])
                      + " mm): half of it is the systematic")
        else:
            b, s, rc, rf = rep["spread_before"], rep["spread_after"], rep["residual_at_config"], rep["residual_fitted"]
            print(f"  fit residual: config X {rc['trans_mm']:.2f} mm / {rc['rot_deg']:.3f} deg -> fitted {rf['trans_mm']:.2f} mm / "
                  f"{rf['rot_deg']:.3f} deg; board spread in the torso: config {b['trans_rms_mm']:.2f} mm / "
                  f"{b['rot_rms_deg']:.3f} deg -> fitted {s['trans_rms_mm']:.2f} mm / {s['rot_rms_deg']:.3f} deg")
            p = rep["play"]
            fmt = lambda v: "n/a" if v is None else f"{v:+.2f}"  # noqa: E731
            print(f"  play (+ = the head stops short of the encoder): yaw {fmt(p['yaw_deg'])} +- {fmt(p['yaw_se_deg'])} deg "
                  f"over {p['yaw_holds']} holds, pitch {fmt(p['pitch_deg'])} +- {fmt(p['pitch_se_deg'])} over "
                  f"{p['pitch_holds']}; outlier holds {rep['outlier_holds']}")
        for w in rep["warnings"]:
            print(f"  WARNING: {w}")
        print(f"  verdict arithmetic: {rep['verdict_reason']}")
        if rep["inconclusive_because"]:
            print("  no verdict possible: " + "; ".join(rep["inconclusive_because"]))
    else:
        for w in rep["warnings"]:
            print(f"  WARNING: {w}")
    if "vs_other_take" in rep:
        o = rep["vs_other_take"]
        comb = (f"; the fits' statistical combined 1-sigma {o['combined_sigma_deg']:.3f} deg / {o['combined_sigma_mm']:.2f} mm"
                if "combined_sigma_deg" in o else "")
        how = ("within the limits" if o["agree_limits"] else "within 2 combined sigma" if o["agree_sigma"]
               else "; ".join(o["why_not"]))
        print(f"  vs {a.compare.name}: {o['angle_deg']:.3f} deg, {o['dist_mm']:.2f} mm -> "
              f"{'agree' if o['agree'] else 'DISAGREE'}, {how} (limits {AGREE_DEG} deg / {AGREE_MM} mm{comb})")
    print(f"VERDICT {rep['verdict']}  -> {a.out}")
    return {"KEEP": 0, "REPLACE": 1, "IMPROVE": 1}.get(rep["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
