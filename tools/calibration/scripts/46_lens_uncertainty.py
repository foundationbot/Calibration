#!/usr/bin/env python3
"""Step 46 — how well the fit sweep pins the calibration down: split-half refits and the repeatability map.

The holds step 45 kept, ranked by hold id and split even / odd; each half refitted warm from step 45's calibration
(board poses by PnP, refitted with the camera fixed, then everything; converged or exit 2). Per number, sigma =
|a - b| / 2: a ONE-SAMPLE estimate (a single split), labelled so wherever it is printed, and the spread of a half-size
fit (the full fit's is ~1/sqrt(2) of it). Repeatability map: points at 1 m depth on a 40 px grid of camera 0's image,
pixels predicted by half a, triangulated with half b (depthmap.py, the method of the 20 Oct A/B review); per image
cell the largest |difference|; a cell beyond gates.lens_repeat_max_pct is flagged "unconstrained: collect more or mask"
(20 Oct: Fit A and Fit B, fitted on nearly the same data, differed by up to 6 % at 1 m in the image corners and agreed
within 0.5 % elsewhere; this map shows that kind of gap before any test data does).
Real data: two KB4T fits on half the data each (a full fit is 5-27 min on Thor); run it there.
Usage: python 46_lens_uncertainty.py <fit-corners.npz> <fit.json> <out.json> [--rig PATH]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depthmap  # noqa: E402
import rigcfg  # noqa: E402
import rigmath as rm  # noqa: E402
import stereofit as sf  # noqa: E402

SIGMA_KIND = "one-sample split-half estimate |a - b| / 2 (a single split; the spread of a half-size fit)"


def sigmas(fa, fb):
    """per number |a - b| / 2: both cameras' intrinsics, the stereo rotation (deg, as a rotation vector of a's
    rotation relative to b's) and translation (mm), the baseline length (mm)"""
    return {"intr": [(np.abs(np.asarray(x) - np.asarray(y)) / 2).tolist() for x, y in zip(fa["intr"], fb["intr"])],
            "C_rotvec_deg": (np.abs(np.degrees(rm.log_rot(fa["C"][:3, :3] @ fb["C"][:3, :3].T))) / 2).tolist(),
            "C_t_mm": (np.abs(1e3 * (fa["C"][:3, 3] - fb["C"][:3, 3])) / 2).tolist(),
            "baseline_mm": float(abs(1e3 * (np.linalg.norm(fa["C"][:3, 3]) - np.linalg.norm(fb["C"][:3, 3]))) / 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("corners", type=Path, help="step 15's npz of the FIT sweep (the one step 45 fitted)")
    ap.add_argument("fit", type=Path, help="step 45's JSON")
    ap.add_argument("out", type=Path)
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    rig = rigcfg.load(a.rig)
    gates = rig.lens_gates()
    fit = json.loads(a.fit.read_text())
    if not str(fit.get("verdict", "")).startswith("converged"):
        sys.exit(f"ERROR: {a.fit}: step 45 did not converge ({fit.get('verdict')}); nothing to split")
    if fit["inputs"]["sha256"] != sf.file_sha256(a.corners):
        sys.exit(f"ERROR: {a.corners} is not the corner file {a.fit} was fitted on")
    model, intr, C = sf.load_calibration(a.fit)
    data = sf.Data(str(a.corners), rig.board)
    kept = np.asarray(fit["kept_holds"])
    halves = []
    for parity in (0, 1):
        holds = kept[parity::2]
        used, r = sf.warm_fit(data.select(np.isin(data.hold, holds)), model, intr, C)
        if not r["success"]:
            print(f"half {parity}: unconverged")
            return 2
        halves.append({"holds": holds.tolist(), "pairs": int(used.n), "intr": r["intr"], "C": r["C"],
                       "nfev": r["nfev"], "rms_px": sf.fit_cost(r)[1]})
    uv, pct = depthmap.depth_diff_map((model, halves[0]["intr"], halves[0]["C"]),
                                      (model, halves[1]["intr"], halves[1]["C"]), rig.resolution, z=1.0)
    cells = depthmap.by_cell(uv, pct, intr[0][2:4])
    flagged = [c for c, v in cells.items() if v["max_abs_pct"] > gates["lens_repeat_max_pct"]]
    out = {"sigma_kind": SIGMA_KIND, "sigma": sigmas(*halves), "fit": str(a.fit), "git_sha": sf.git_sha(),
           "halves": [{**h, "intr": [v.tolist() for v in h["intr"]], "C": h["C"].tolist()} for h in halves],
           "repeat_map": {"z_m": 1.0, "step_px": 40, "points": np.c_[uv, pct].tolist()},
           "repeat_cells": cells, "repeat_gate_pct": gates["lens_repeat_max_pct"], "unconstrained_cells": flagged}
    sf.write_json(a.out, out)
    print(f"sigma ({SIGMA_KIND}): baseline {out['sigma']['baseline_mm']:.3f} mm; camera 0 fx "
          f"{out['sigma']['intr'][0][0]:.2f} px")
    print("repeatability at 1 m, largest |difference| per cell: "
          + ", ".join(f"{c} {v['max_abs_pct']:.2f} %" for c, v in cells.items()))
    print(f"unconstrained (beyond {gates['lens_repeat_max_pct']} %: collect more or mask): {flagged or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
