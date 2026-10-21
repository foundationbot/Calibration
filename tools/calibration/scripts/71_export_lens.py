#!/usr/bin/env python3
"""Step 71 — export a validated KB4T calibration:
  (a) <rig>-kb4t.json: the calibration itself (both cameras' fx, fy, cx, cy, k1..k4, p1, p2; C = camera-1 points into
      camera 0), with step 66's verdict and the image cells it could not validate or found unconstrained
  (b) camchain-kb4t-equi4.yaml: a Kalibr camchain (pinhole-equi with k1..k4 ONLY; T_cn_cnm1 = inverse of C) for
      Kalibr's camera-IMU fit, which cannot take p1, p2 (there the cost is pixels, not depth)
  (c) only with --okvis-template AND --allow-4term-okvis: an OKVIS config by step 70's text surgery (equidistant
      k1..k4; T_SC of OKVIS cam0 kept from the template, the other camera's T_SC = T_SC0 * C in OKVIS block order;
      everything else unchanged), marked an APPROXIMATION, NOT FOR STEREO DEPTH in its header. Without the flag no
      OKVIS file is written and the depth cost is printed: dropping p1, p2 is not an approximation for depth (Fit A,
      21 Oct: depth at 1 m off by 0.58 % in the centre, 9.30 % within 450 px, 25.2 % within 560 px).
  (d) only with --okvis-template AND --okvis-kb4t: the same text surgery with the whole lens, distortion_type kb4t and
      distortion_coefficients [k1, k2, k3, k4, p1, p2], for an OKVIS image with the kb4t change
      (OKVIS2-X #796, in AI-1184 since fc255aaa): the stereo network's rectification
      (depth) uses all six numbers, VIO k1..k4. Older OKVIS images refuse the file. The export for depth.
(b) and (c) drop p1, p2 (and VIO in (d)): what that costs in pixels and in stereo depth at 1 m is computed for this
calibration, printed, written into (a) and as comments into (b), (c) and (d).
Refuses a fit whose step-45 verdict is not "converged" (an unconverged fit has numbers too, and the report's sha256
cannot tell), a calibration step 66 did not PASS, or a report of another calibration (sha256 of the numbers).
Usage: python 71_export_lens.py <fit.json> <report.json> <out-dir>
           [--okvis-template T.yaml [--okvis-kb4t | --allow-4term-okvis]] [--rig PATH]"""
import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depthmap  # noqa: E402
import lensmodel  # noqa: E402
import okviscfg  # noqa: E402
import rigcfg  # noqa: E402
import rigmath as rm  # noqa: E402
import stereofit as sf  # noqa: E402

m70 = importlib.import_module("70_make_mk13_yaml")
NAMES = ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2"]
DEPTH_RADII_PX = (350, 450, 560)     # 560 px: the depth image's extent on phantom12


RAY_TOL_PX = 1e-3                    # a grid point's ray must project back to it within this (unproject converged)


def p_drop_cost(intr, resolution, step=20):
    """pixel shift from dropping p1, p2 (KB4T -> its KB4 part) for one camera: largest within 350 px of the principal
    point, and over the whole image (Fit A camera 0: 0.90 px and 4.2 px, the latter in the image's far corners).
    Grid points where the lens has no ray are left out and counted (points_without_ray): a lens whose KB4 polynomial
    turns over inside the image has none beyond the turn, and unproject returns NaN or, more often, a finite wrong ray
    there; a ray counts only if it projects back to its pixel within RAY_TOL_PX"""
    W, H = resolution
    g = np.array([(u, v) for v in np.arange(0, H, step) for u in np.arange(0, W, step)], float)
    with np.errstate(all="ignore"):
        ray = lensmodel.unproject("KB4T", intr, g)
        back = np.linalg.norm(lensmodel.project("KB4T", intr, ray) - g, axis=1)
        d = np.linalg.norm(lensmodel.project("KB4", intr[:8], ray) - g, axis=1)
    has_ray = np.isfinite(back) & (back <= RAY_TOL_PX) & np.isfinite(d)
    d = np.where(has_ray, d, np.nan)
    r = np.hypot(g[:, 0] - intr[2], g[:, 1] - intr[3])

    def largest(x):
        return float(np.nanmax(x)) if np.isfinite(x).any() else float("nan")
    return {"within_350px": largest(d[r <= 350]), "whole_image": largest(d),
            "points_without_ray": int((~has_ray).sum())}


def p_drop_depth_cost(intr, C, resolution, z=1.0):
    """stereo depth from dropping p1, p2 in both cameras (KB4T -> its KB4 part, no refit): depthmap.depth_diff_map of
    the KB4T calibration read by its truncation at z m; the largest |difference| (%) in the centre cell and within
    350, 450 and 560 px of camera 0's principal point (Fit A, 21 Oct: 0.58, 4.54, 9.30, 25.2 %)"""
    uv, pct = depthmap.depth_diff_map(("KB4T", intr, C), ("KB4", [np.asarray(v, float)[:8] for v in intr], C),
                                      resolution, z=z)
    pp = np.asarray(intr[0], float)[2:4]
    out = {"z_m": float(z), "centre_cell": depthmap.by_cell(uv, pct, pp)["centre"]["max_abs_pct"]}
    for r in DEPTH_RADII_PX:
        out[f"within_{r}px"] = float(np.abs(depthmap.limit_radius(uv, pct, pp, r)[1]).max())
    return out


def depth_cost_text(dc):
    return (f"stereo depth at {dc['z_m']:g} m off by up to {dc['centre_cell']:.2f} % in the centre cell, "
            + ", ".join(f"{dc[f'within_{r}px']:.2f} % within {r} px" for r in DEPTH_RADII_PX)
            + " of the principal point")


def camchain(intr, C, resolution):
    """Kalibr camchain dict: cam0 / cam1 = kalibr order = step 15's cameras 0 / 1"""
    def cam(i):
        return {"camera_model": "pinhole", "distortion_model": "equidistant",
                "distortion_coeffs": [float(x) for x in intr[i][4:8]], "intrinsics": [float(x) for x in intr[i][:4]],
                "resolution": [int(x) for x in resolution], "rostopic": f"/cam{i}/image_raw", "cam_overlaps": [1 - i]}
    return {"cam0": cam(0), "cam1": {**cam(1), "T_cn_cnm1": rm.inv(C).tolist()}}


def okvis_text(template: Path, rig, intr, C, notes, kb4t=False):
    """the template with both cameras' T_SC, focal_length, principal_point, distortion_coefficients and
    distortion_type replaced, in OKVIS block order (OKVIS cam i <- kalibr cam okvis_order[i]), and notes inserted as
    comments after %YAML:1.0. The lens: equidistant k1..k4, or with kb4t the whole KB4T lens (distortion_type kb4t,
    k1..k4, p1, p2)"""
    text = template.read_text()
    q = [int(v) for v in re.findall(r"dma_queue: (\d+)", text)]
    if q != rig.okvis_queues():
        sys.exit(f"ERROR: template dma_queue values {q} != {rig.okvis_queues()} that {rig.path.name} routes to OKVIS "
                 "cam0/cam1: a block would get another camera's calibration (as step 70)")
    if "%YAML:1.0\n" not in text:
        sys.exit(f"ERROR: {template} has no '%YAML:1.0' line")
    order = rig.okvis_order
    T_SC0 = okviscfg.load(template).T_SC[0]
    C_okvis = C if order == [0, 1] else rm.inv(C)            # OKVIS cam1 in OKVIS cam0
    text = m70.replace_tsc(text, [T_SC0, T_SC0 @ C_okvis])
    text = m70.repl_seq(text, "focal_length", [intr[k][:2] for k in order])
    text = m70.repl_seq(text, "principal_point", [intr[k][2:4] for k in order])
    text = m70.repl_seq(text, "distortion_coefficients", [intr[k][4:10] if kb4t else intr[k][4:8] for k in order])
    text = m70.replace_distortion_type(text, "kb4t" if kb4t else "equidistant")
    return text.replace("%YAML:1.0\n", "%YAML:1.0\n" + "".join(f"# {n}\n" for n in notes), 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fit", type=Path, help="step 45's JSON")
    ap.add_argument("report", type=Path, help="step 66's report.json for this calibration")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--okvis-template", type=Path, help="an OKVIS config whose T_SC of OKVIS cam0, T_BS, "
                                                        "image_delay and tuning are kept")
    ap.add_argument("--allow-4term-okvis", action="store_true",
                    help="write the OKVIS config although its 4-term lens is unusable for stereo depth (the depth "
                         "cost is printed); only for a run that uses no stereo depth")
    ap.add_argument("--okvis-kb4t", action="store_true",
                    help="write the OKVIS config with the whole lens (distortion_type kb4t, 6 coefficients): depth "
                         "uses KB4T, VIO k1..k4; needs an OKVIS image with the kb4t change (older images refuse it)")
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    if a.okvis_kb4t and a.allow_4term_okvis:
        sys.exit("ERROR: --okvis-kb4t (the whole KB4T lens) and --allow-4term-okvis (its 4-term part): one OKVIS lens "
                 "per file, not both")
    if a.okvis_kb4t and not a.okvis_template:
        sys.exit("ERROR: --okvis-kb4t needs --okvis-template (the OKVIS config whose T_SC0, T_BS and tuning are kept)")
    rig = rigcfg.load(a.rig)
    fit = sf.load_entry(a.fit)
    fit_verdict = fit.get("verdict")
    if not str(fit_verdict).startswith("converged"):
        sys.exit(f"ERROR: {a.fit}: step 45's verdict {fit_verdict!r}: only a converged fit is exported")
    model, intr, C = sf.load_calibration(a.fit)
    rep = json.loads(a.report.read_text())
    if rep.get("verdict") != "PASS":
        sys.exit(f"ERROR: {a.report}: verdict {rep.get('verdict')!r}; only a PASS is exported")
    if rep["calibration"]["sha256"] != sf.calibration_digest(model, intr, C):
        sys.exit(f"ERROR: {a.report} judged another calibration than {a.fit}")
    if model != "KB4T":
        sys.exit(f"ERROR: {a.fit} is {model}; this step exports KB4T")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cost = [p_drop_cost(intr[c], rig.resolution) for c in (0, 1)]
    dcost = p_drop_depth_cost(intr, C, rig.resolution)
    warn = ("p1, p2 DROPPED (Kalibr, and OKVIS without distortion_type kb4t, take k1..k4 only): pixels move by up to "
            + "; ".join(f"camera {c} {v['within_350px']:.2f} px within 350 px, {v['whole_image']:.2f} px at the edge"
                        + (f" ({v['points_without_ray']} grid points where the lens has no ray left out: its "
                           "polynomial turns over in the image)" if v["points_without_ray"] else "")
                        for c, v in enumerate(cost))
            + "; " + depth_cost_text(dcost))
    print(f"WARNING: {warn}")
    unc = sorted(set(rep.get("uncertainty", {}).get("unconstrained_cells", [])))
    canon = {"format": "kb4t-stereo/1", "model": model, "rig": rig.name, "resolution": list(rig.resolution),
             "intr_names": NAMES,
             "cameras": [{"kalibr": f"cam{c}", "side": rig.kalibr[c]["side"], "dma_queue": int(rig.kalibr[c]["dma_queue"]),
                          "intr": intr[c].tolist()} for c in (0, 1)],
             "C_cam1_to_cam0": C.tolist(), "baseline_mm": float(1e3 * np.linalg.norm(C[:3, 3])),
             "validation": {"verdict": rep["verdict"], "report": str(a.report),
                            "validated_cells": rep["validated_cells"], "not_validated_cells": rep["not_validated_cells"],
                            "unconstrained_cells": unc},
             "p_drop_cost_px": cost, "p_drop_depth_pct": dcost, "sha256": rep["calibration"]["sha256"],
             "fit": str(a.fit), "fit_verdict": fit_verdict, "git_sha": sf.git_sha()}
    sf.write_json(a.out_dir / f"{rig.name}-kb4t.json", canon)
    cc = camchain(intr, C, rig.resolution)
    (a.out_dir / "camchain-kb4t-equi4.yaml").write_text(f"# KB4T fit {a.fit}, sha256 {canon['sha256'][:12]}\n# {warn}\n"
                                                         + yaml.safe_dump(cc))
    print(f"wrote {a.out_dir / (rig.name + '-kb4t.json')} and {a.out_dir / 'camchain-kb4t-equi4.yaml'}")
    if a.okvis_template and a.okvis_kb4t:
        vio_cost = "; ".join(f"camera {c} {v['within_350px']:.2f} px within 350 px, {v['whole_image']:.2f} px at the "
                             "edge" for c, v in enumerate(cost))
        notes = [f"{rig.okvis_yaml_name}: {rig.name}, KB4T stereo calibration {canon['sha256'][:12]} (step 71 from "
                 f"{a.fit.name}; step 66 PASS).",
                 "distortion_type kb4t, distortion_coefficients [k1, k2, k3, k4, p1, p2]. depth: full KB4T (the stereo "
                 "network's rectification map); VIO: k1..k4 (p1, p2 dropped there, pixel cost " + vio_cost + ").",
                 "Needs an OKVIS image with the kb4t change (OKVIS2-X #796, in AI-1184 since fc255aaa); older images refuse "
                 "this file. Deploy check: OKVIS logs 'KB4T rectification "
                 "(stereo network)' at startup.",
                 "not validated (mask): " + (", ".join(rep["not_validated_cells"]) or "none")
                 + "; unconstrained: " + (", ".join(unc) or "none"),
                 "T_SC of OKVIS cam0 from the template; the other camera's = T_SC0 * stereo pose."]
        out = a.out_dir / rig.okvis_yaml_name
        out.write_text(okvis_text(a.okvis_template, rig, intr, C, notes, kb4t=True))
        print(f"wrote {out} (distortion_type kb4t: needs an OKVIS image with the kb4t change; older images refuse it)")
    elif a.okvis_template and not a.allow_4term_okvis:
        print(f"NOT WRITTEN: the OKVIS config. Its 4-term lens (p1, p2 dropped) puts {depth_cost_text(dcost)}: "
              "unusable for depth. --okvis-kb4t writes the whole lens (distortion_type kb4t) for an OKVIS image with "
              "the kb4t change; --allow-4term-okvis writes the 4-term one anyway, for a run that uses no stereo depth.")
    elif a.okvis_template:
        notes = [f"{rig.okvis_yaml_name}: {rig.name}, KB4T stereo calibration {canon['sha256'][:12]} (step 71 from "
                 f"{a.fit.name}; step 66 PASS).",
                 "APPROXIMATION, NOT FOR STEREO DEPTH: the lens is written as its 4-term equidistant part (written "
                 "because --allow-4term-okvis was given; --okvis-kb4t writes the whole lens).", warn,
                 "KB4T p1, p2 (not read by OKVIS): " + "; ".join(f"camera {c} {intr[c][8]:.6g}, {intr[c][9]:.6g}"
                                                                for c in (0, 1)),
                 "not validated (mask): " + (", ".join(rep["not_validated_cells"]) or "none")
                 + "; unconstrained: " + (", ".join(unc) or "none"),
                 "T_SC of OKVIS cam0 from the template; the other camera's = T_SC0 * stereo pose."]
        out = a.out_dir / rig.okvis_yaml_name
        out.write_text(okvis_text(a.okvis_template, rig, intr, C, notes))
        print(f"wrote {out} (APPROXIMATION, NOT FOR STEREO DEPTH: see its header)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
