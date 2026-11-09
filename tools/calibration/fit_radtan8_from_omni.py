#!/usr/bin/env python3
"""Fit a pinhole + radialtangential8 camera to each camera of a kalibr omni-radtan camchain.

  python3 tools/calibration/fit_radtan8_from_omni.py --camchain stereo-camchain.yaml --out fit.json [--scale 0.5]

Why: on a lens kalibr's omni-radtan fits and pinhole-radtan / pinhole-equi do not (phantom15's ego
pair, 2026-11-04: omni 0.30 px and the CAD baseline, radtan 1.6 px, equi a 39 mm baseline), OKVIS
cannot load omni -- but its RadialTangentialDistortion8 IS OpenCV's rational model, coefficient for
coefficient (k1 k2 p1 p2 k3 k4 k5 k6). So: resample the omni projection at every --step pixel of the
frame, unproject with the omni model, and least-squares fit radtan8 intrinsics so the same rays land
on the same pixels. The JSON then carries, per camera, the OKVIS block (focal_length,
principal_point, distortion_coefficients) and the residual against omni by distance from the image
centre -- overall, within r_u <= 3 (RadialTangentialDistortion8::distort refuses r_u^2 > 9, so
beyond that OKVIS cannot evaluate the model whatever the fit says), and a 4-coefficient radtan fit of
the same rays for comparison. The stereo extrinsic (T_cn_cnm1) is copied through unchanged.

--scale resizes the omni intrinsics to the frame OKVIS will see (0.5: 1280x800 kalibr takes, 640x400
on the robot). The focal is scaled; the principal point follows ViParametersReader's resize_factor
convention, (c + 0.5) * s - 0.5; xi and the distortion are scale-free.

Fit every pixel (the default --fit-radius covers the corners): a cap that leaves the far corners out
let the rational denominator put a pole INSIDE the frame on phantom15's cam1 (35000 px residual at
the corner). The script checks for that -- the JSON says whether the radial profile is monotonic and
where the denominator bottoms out to 1.15x the corner radius -- and the OpenCV projection of the
result is cross-checked against the formula it was fit with.

Stereo depth is the harder customer. It reads the DIFFERENCE between the two cameras' residuals, and
on phantom15 an unweighted fit left cam1 (xi 2.76) with a 0.35 px bias at the centre that cam0 (xi
1.99, 0.06 px) did not share: 0.5 px of disparity bias, 7% of depth at 2 m, a floor sinking 20 cm.
So samples inside --inner-ru (undistorted radius; 1.4 covers the rectified frame cv::stereoRectify
alpha=0 hands the depth network) weigh 1 and the rest --outer-weight, which buys the inner region at
the expense of the far corners, and the JSON's "stereo" block reports the disparity bias the pair
would carry: the two rect->raw maps rebuilt from the fit vs from omni with one rectifying rotation,
dx(cam1) - dx(cam0), as a 3x3 zone table. Aim for well under 0.2 px. Needs numpy, scipy and a cv2
with the omnidir contrib module (the kalibr image has all three).
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares


def radtan8_project(theta, rays):
    """theta = fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6; rays Nx3. OpenCV's rational model == OKVIS's."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = theta
    x, y = rays[:, 0] / rays[:, 2], rays[:, 1] / rays[:, 2]
    r2 = x * x + y * y
    rad = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * xd + cx, fy * yd + cy], 1)


def omni_unproject(px, K, D, xi):
    """Pixel -> pinhole-normalized ray (x/z, y/z) under kalibr's omni-radtan (MEI) model: K^-1,
    fixed-point radtan undistort on the xi-plane, lift to the unit sphere, divide by z.
    cv2.omnidir.undistortPoints is not used because its output convention differs between builds;
    the forward cv2.omnidir.projectPoints round trip in fit_camera is what validates this."""
    k1, k2, p1, p2 = D
    m = (px - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
    u = m.copy()
    for _ in range(50):
        x, y = u[:, 0], u[:, 1]
        r2 = x * x + y * y
        rad = 1 + r2 * (k1 + r2 * k2)
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        u = (m - np.stack([dx, dy], 1)) / rad[:, None]
    x, y = u[:, 0], u[:, 1]
    r2 = x * x + y * y
    lam = (xi + np.sqrt(1 + (1 - xi * xi) * r2)) / (r2 + 1)
    z = lam - xi
    return np.stack([lam * x / z, lam * y / z], 1)


def zone_stats(err, px, width, height):
    """RMS / max pixel error overall and by distance from the image centre."""
    r = np.hypot(px[:, 0] - width / 2, px[:, 1] - height / 2)
    zones = [("all", 0, np.inf), ("centre_r<100", 0, 100), ("mid_100-200", 100, 200), ("edge_200-300", 200, 300),
             ("corner_r>300", 300, np.inf)]
    out = {}
    for name, lo, hi in zones:
        m = (r >= lo) & (r < hi)
        if m.any():
            out[name] = {"n": int(m.sum()), "rms_px": round(float(np.sqrt((err[m] ** 2).mean())), 4),
                         "max_px": round(float(err[m].max()), 4)}
    return out


def fit_camera(cam, width, height, scale, step, fit_radius, loss, f_scale, inner_ru, outer_weight):
    xi, fx, fy, cx, cy = cam["intrinsics"]
    K = np.array([[fx * scale, 0, (cx + 0.5) * scale - 0.5], [0, fy * scale, (cy + 0.5) * scale - 0.5], [0, 0, 1.0]])
    D = np.array(cam["distortion_coeffs"], float)
    vv, uu = np.mgrid[0:height:step, 0:width:step]
    px = np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float64)
    und = omni_unproject(px, K, D, xi)
    rays = np.hstack([und, np.ones((len(und), 1))])
    back, _ = cv2.omnidir.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, xi, D.reshape(1, 4))
    rt = np.linalg.norm(back.reshape(-1, 2) - px, axis=1)
    keep = rt < 1e-3
    if keep.mean() < 0.98:
        raise SystemExit("omni inverse failed on %.1f%% of samples" % (100 * (1 - keep.mean())))
    px, rays, und = px[keep], rays[keep], und[keep]
    rpx = np.hypot(px[:, 0] - width / 2, px[:, 1] - height / 2)
    ru = np.hypot(und[:, 0], und[:, 1])
    fit = rpx <= fit_radius
    weight = np.sqrt(np.where(ru[fit] <= inner_ru, 1.0, outer_weight))[:, None]
    theta0 = np.array([K[0, 0] / (1 + xi), K[1, 1] / (1 + xi), K[0, 2], K[1, 2]] + [0.0] * 8)
    results = {}
    for name, mask in (("radtan4", np.array([1] * 8 + [0] * 4, bool)), ("radtan8", np.ones(12, bool))):

        def resid(active):
            th = theta0.copy()
            th[mask] = active
            return ((radtan8_project(th, rays[fit]) - px[fit]) * weight).ravel()

        sol = least_squares(resid, theta0[mask], method="trf", loss=loss, f_scale=f_scale, x_scale="jac",
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000)
        theta = theta0.copy()
        theta[mask] = sol.x
        err = np.linalg.norm(radtan8_project(theta, rays) - px, axis=1)
        results[name] = {
            "focal_length": theta[:2].tolist(), "principal_point": theta[2:4].tolist(),
            "distortion_coefficients": theta[4:].tolist() if name == "radtan8" else theta[4:8].tolist(),
            "distortion_type": "radialtangential8" if name == "radtan8" else "radialtangential",
            "error": zone_stats(err, px, width, height),
            "error_ru<=3": zone_stats(err[ru <= 3], px[ru <= 3], width, height)["all"],
            "pixel_radius_at_ru3": float(rpx[ru <= 3].max()) if (ru > 3).any() else None,
        }
    theta = np.array(results["radtan8"]["focal_length"] + results["radtan8"]["principal_point"]
                     + results["radtan8"]["distortion_coefficients"])
    Kf = np.array([[theta[0], 0, theta[2]], [0, theta[1], theta[3]], [0, 0, 1]])
    cvp, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kf, theta[4:])
    results["radtan8"]["opencv_vs_formula_max_px"] = float(np.abs(cvp.reshape(-1, 2) - radtan8_project(theta, rays)).max())
    r_line = np.linspace(0, 1.15 * ru.max(), 2000)
    r2 = r_line * r_line
    k1, k2, p1, p2, k3, k4, k5, k6 = theta[4:]
    den = 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
    rd = r_line * (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / den
    fold = np.nonzero(np.diff(rd) <= 0)[0]
    results["radtan8"]["radial_monotonic_to_1.15x_corner"] = not fold.size
    results["radtan8"]["first_fold_undistorted_radius"] = float(r_line[fold[0]]) if fold.size else None
    results["radtan8"]["denominator_min_to_1.15x_corner"] = float(den.min())
    results["radtan8"]["max_undistorted_radius_sampled"] = float(ru.max())
    results["omni_scaled"] = {"xi": xi, "focal_length": [K[0, 0], K[1, 1]], "principal_point": [K[0, 2], K[1, 2]],
                              "distortion_coefficients": D.tolist(), "samples": int(keep.sum()),
                              "round_trip_max_px": float(rt[keep].max())}
    return results


def disparity_bias(cams, T_cn_cnm1, width, height, f_new=300.0):
    """Rectify the pair from the radtan8 fit and from omni with ONE rectifying rotation (cv::stereoRectify
    reads only R and T for it) and compare the rect->raw maps: dx(cam1) - dx(cam0) is the disparity
    bias the fit would put on the depth network, in px of a f_new rectified frame. 3x3 zone medians."""
    T = np.array(T_cn_cnm1, float)
    R = np.ascontiguousarray(T[:3, :3])
    t = np.ascontiguousarray(T[:3, 3]).reshape(3, 1)
    K_new = np.array([[f_new, 0, width / 2], [0, f_new, height / 2], [0, 0, 1.0]])
    Ks, Ds = [], []
    for c in cams:
        r = c["radtan8"]
        Ks.append(np.array([[r["focal_length"][0], 0, r["principal_point"][0]],
                            [0, r["focal_length"][1], r["principal_point"][1]], [0, 0, 1.0]]))
        Ds.append(np.array(r["distortion_coefficients"]))
    R1, R2 = cv2.stereoRectify(Ks[0], Ds[0], Ks[1], Ds[1], (width, height), R, t, flags=cv2.CALIB_ZERO_DISPARITY)[:2]
    dxs, per_camera = [], []
    for c, K, D, Rr in zip(cams, Ks, Ds, (R1, R2)):
        o = c["omni_scaled"]
        Ko = np.array([[o["focal_length"][0], 0, o["principal_point"][0]], [0, o["focal_length"][1], o["principal_point"][1]], [0, 0, 1.0]])
        m8 = cv2.initUndistortRectifyMap(K, D, Rr, K_new, (width, height), cv2.CV_32FC1)
        mo = cv2.omnidir.initUndistortRectifyMap(Ko, np.array(o["distortion_coefficients"]).reshape(1, 4), np.array([[o["xi"]]]),
                                                 Rr, K_new, (width, height), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE)
        d = np.hypot(m8[0] - mo[0], m8[1] - mo[1])
        per_camera.append({"median_px": float(np.median(d)), "p95_px": float(np.percentile(d, 95)), "max_px": float(d.max())})
        dxs.append(m8[0] - mo[0])
    bias = dxs[1] - dxs[0]
    zones = [[round(float(np.median(bias[i * height // 3:(i + 1) * height // 3, j * width // 3:(j + 1) * width // 3])), 3)
              for j in range(3)] for i in range(3)]
    return {"f_new": f_new, "rectified_map_error_vs_omni": per_camera, "bias_px_zones_3x3": zones,
            "bias_abs_p50_px": float(np.median(np.abs(bias))), "bias_abs_max_px": float(np.abs(bias).max())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--camchain", required=True, help="kalibr camchain with camera_model: omni, distortion_model: radtan")
    ap.add_argument("--out", required=True, help="JSON to write")
    ap.add_argument("--scale", type=float, default=0.5, help="frame scale vs the camchain's resolution (default 0.5)")
    ap.add_argument("--step", type=int, default=2, help="sample every Nth pixel (default 2)")
    ap.add_argument("--fit-radius", type=float, default=None,
                    help="fit only samples within this many px of the centre (default: the whole frame)")
    ap.add_argument("--loss", default="linear", help="scipy least_squares loss (default linear)")
    ap.add_argument("--f-scale", type=float, default=1.0)
    ap.add_argument("--inner-ru", type=float, default=1.4,
                    help="undistorted radius of the region weighted 1 (default 1.4, the rectified depth frame)")
    ap.add_argument("--outer-weight", type=float, default=0.01,
                    help="weight of samples beyond --inner-ru (default 0.01; 1 = plain least squares)")
    a = ap.parse_args(argv)
    cc = yaml.safe_load(open(a.camchain))
    out = {"source": a.camchain, "scale": a.scale, "sample_step_px": a.step,
           "fit": {"radius_px": a.fit_radius, "loss": a.loss, "f_scale": a.f_scale, "inner_ru": a.inner_ru,
                   "outer_weight": a.outer_weight},
           "coefficient_order": "k1 k2 p1 p2 k3 k4 k5 k6 (OpenCV rational == OKVIS RadialTangentialDistortion8)",
           "cameras": {}}
    for name in sorted(k for k in cc if k.startswith("cam")):
        cam = cc[name]
        if cam.get("camera_model") != "omni" or cam.get("distortion_model") != "radtan":
            raise SystemExit("%s: expected camera_model omni + distortion_model radtan, got %s-%s"
                             % (name, cam.get("camera_model"), cam.get("distortion_model")))
        w, h = (int(round(v * a.scale)) for v in cam["resolution"])
        res = fit_camera(cam, w, h, a.scale, a.step, a.fit_radius if a.fit_radius else np.hypot(w, h),
                         a.loss, a.f_scale, a.inner_ru, a.outer_weight)
        res["image_dimension"] = [w, h]
        if "T_cn_cnm1" in cam:
            res["T_cn_cnm1"] = cam["T_cn_cnm1"]
        out["cameras"][name] = res
        for m in ("radtan4", "radtan8"):
            e = res[m]["error"]
            print(name, m, " ".join("%s rms %.3f max %.3f" % (z, e[z]["rms_px"], e[z]["max_px"]) for z in e),
                  "| ru<=3 rms %.3f max %.3f" % (res[m]["error_ru<=3"]["rms_px"], res[m]["error_ru<=3"]["max_px"]))
        r8 = res["radtan8"]
        print(name, "radtan8: opencv-vs-formula %.1e" % r8["opencv_vs_formula_max_px"],
              "monotonic", r8["radial_monotonic_to_1.15x_corner"], "fold_at", r8["first_fold_undistorted_radius"],
              "sampled_max_ru %.3f" % r8["max_undistorted_radius_sampled"],
              "den_min %.3f" % r8["denominator_min_to_1.15x_corner"],
              "ru=3 at %s px from centre" % (round(r8["pixel_radius_at_ru3"], 1) if r8["pixel_radius_at_ru3"] else "-"))
    names = sorted(out["cameras"])
    if len(names) == 2 and "T_cn_cnm1" in out["cameras"][names[1]]:
        w, h = out["cameras"][names[0]]["image_dimension"]
        st = disparity_bias([out["cameras"][n] for n in names], out["cameras"][names[1]]["T_cn_cnm1"], w, h)
        out["stereo"] = st
        print("stereo disparity bias vs omni (px, f=%g rectified; rows top->bottom, cols left->right):" % st["f_new"])
        for row in st["bias_px_zones_3x3"]:
            print("    " + "  ".join("%+.3f" % v for v in row))
        print("    |bias| p50 %.3f max %.3f" % (st["bias_abs_p50_px"], st["bias_abs_max_px"]))
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
