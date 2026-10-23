#!/usr/bin/env python3
"""Step 66 — validate a lens calibration on the TEST sweep (a recording no fit saw): depth error per image cell and
distance, a fixed pass rule, and a report (Markdown + JSON).

The scorer (lensscore.score, port of sweep_590.evaluate_pp): the calibration fixed, per pair the board pose, then pose
and board scale; board scale - 1 = the depth error at the board. In order:
  1. quality gate on the test sweep, as flags only: holds whose corners fit a rigid board worse than
     gates.lens_hold_rms_max_px with the calibration being judged are listed per cell ("flagged: board or calibration
     bad here") and still judged. Nothing is dropped and the test recording is never rejected: measured with the
     judged calibration, a bad hold can be the calibration's fault, and dropping it would turn that cell's FAIL into
     "not validated" (21 Oct, the KB4 refit's bottom-right corner). Step 45 drops such holds from the FIT sweep.
  2. scorer self-check, before any table: the baseline x 1.01 must read +1.00 % +- 0.10 in every distance bin with
     data (NO RESULT otherwise). Both the scoring's and the self-check's least squares must reach scipy's convergence
     (at stereofit.MAX_NFEV evaluations a round, each solve restarted once from its result; NO RESULT otherwise).
     Pairs whose PnP start failed are left out and counted (test.pnp_failed_pairs)
  3. cells (lenscells, relative to the calibration's principal point) x distance bins; clusters = holds; a 95 %
     cluster-bootstrap interval (2000 resamples) from gates.lens_min_holds holds up
  4. the FIXED pass rule: every entry with >= lens_min_holds holds has |median| <= lens_pass_pct (corner cells
     lens_pass_corner_pct): PASS, else FAIL; entries with fewer holds are "not validated" and listed (step 71 marks
     those image regions); no validated entry at all = NO RESULT
The calibration can be any JSON entry with model, intr, C (step 45's output, or file.json:key); a step-45 file whose
verdict is not "converged" (stopped at its cap, rejected, wrong sides) is not judged: NO RESULT. --fit-report and
--uncertainty add step 45's dropped holds and step 46's sigmas and repeatability map to the report.
Exit 0 PASS, 1 FAIL, 2 NO RESULT (an unconverged fit, a scoring solve that did not converge, the self-check failed,
or no entry reached lens_min_holds holds). Real data: scoring took 4-17 min per calibration and set on Thor (19-20 Oct).
Usage: python 66_validate_lens.py <calibration.json[:key]> <test-corners.npz> <out-dir>
           [--fit-report fit.json] [--uncertainty uncertainty.json] [--rig PATH]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lenscells  # noqa: E402
import lensscore as ls  # noqa: E402
import rigcfg  # noqa: E402
import stereofit as sf  # noqa: E402

NAMES = ("fx (px)", "fy (px)", "cx (px)", "cy (px)", "k1", "k2", "k3", "k4", "p1", "p2")


def _bins(lo_hi):
    return f"{lo_hi[0]:g}-{lo_hi[1]:g}"


def _rms(v):
    """a hold's pixel rms; inf (null once in JSON): stereofit.hold_rms found no corner within GROSS_PX"""
    return f"{v:.2f}" if v is not None and np.isfinite(v) else f"no corner within {sf.GROSS_PX:g} px"


def _num(v, spec):
    """a number from step 46's file, which writes a non-finite one as null"""
    return "-" if v is None else format(v, spec)


def hold_places(rec, pp):
    """{hold: {(cell, distance bin index or -1)} of its pairs} (cells relative to the principal point pp)"""
    lab = lenscells.cell_of(np.asarray(rec["u"]) - pp[0], np.asarray(rec["v"]) - pp[1])
    out = {}
    for h, c, i in zip(rec["hold"], lab, lenscells.dist_bin(rec["z"])):
        out.setdefault(int(h), set()).add((str(c), int(i)))
    return out


def markdown(rep):
    """the report in the user's table style: few columns, plain headers with units, a 'how to read it' list above
    every table"""
    L = [f"# Lens validation: {rep['rig']} — {rep['verdict']}", "",
         f"- Calibration: `{rep['calibration']['spec']}` ({rep['calibration']['model']}, sha256 "
         f"{rep['calibration']['sha256'][:12]})",
         f"- Test sweep: `{rep['test']['corners']}`: {rep['test']['pairs']} still pairs in {rep['test']['holds']} holds"
         + (f"; {rep['test']['pnp_failed_pairs']} of them left out of the scoring (their PnP start failed)"
            if "pnp_failed_pairs" in rep["test"] else ""),
         f"- Package commit {rep['git_sha'][:12]}; rig file `{rep['rig_path']}`", "", "## Verdict", "", rep["reason"], ""]
    sc = rep.get("self_check")
    if sc:
        L += ["## Scorer self-check", "", "How to read it:",
              "- the stereo baseline is lengthened by 1 % and the test sweep scored again",
              "- a scorer that works reads +1.00 % in every distance bin (allowed +-0.10); otherwise nothing below "
              "is trusted", "", "| distance (m) | pairs | reads (%) |", "|---|---|---|"]
        L += [f"| {_bins(b['bin_m'])} | {b['pairs']} | "
              + ("-" if b["median_pct"] is None else f"{b['median_pct']:+.3f}") + " |" for b in sc["bins"]]
        L += ["", f"Self-check {'passed' if sc['passed'] else 'FAILED'}.", ""]
    if "table" in rep:
        n = rep["gates"]["lens_min_holds"]
        cov = {(r["cell"], tuple(r["dist_m"])): r["holds"] for r in rep["table"]}
        L += ["## Coverage of the test sweep", "", "How to read it:",
              "- holds of the test sweep per image cell and distance (a hold = one still stretch of the board), "
              "flagged holds included",
              f"- {n} or more holds make an entry validated; fewer and it is not judged", "",
              "| cell | " + " | ".join(f"{_bins(b)} m" for b in lenscells.DIST_BINS) + " |",
              "|---|" + "---|" * len(lenscells.DIST_BINS)]
        L += [f"| {c} | " + " | ".join(str(cov.get((c, b), 0)) for b in lenscells.DIST_BINS) + " |"
              for c in lenscells.CELLS]
        L += ["", "## Depth error by cell and distance", "", "How to read it:",
              "- depth error = how far stereo depth is off at the board, in % of the distance (+ = reads too far)",
              "- median over the cell's pairs; the interval is 95 %, by resampling whole holds, given from "
              f"{n} holds up",
              f"- limit {rep['gates']['lens_pass_pct']} % (corner cells {rep['gates']['lens_pass_corner_pct']} %); "
              f"entries with fewer than {n} holds are listed as not validated",
              f"- flagged holds (rigid-board rms over {rep['gates']['lens_hold_rms_max_px']} px with this "
              "calibration) are judged like the others: where they gather, the board or the calibration is bad", "",
              "| cell | distance (m) | holds | median error (%) | 95 % interval (%) | result |",
              "|---|---|---|---|---|---|"]
        for r in rep["table"]:
            ci = r["ci95_pct"]
            res = ("pass" if r["ok"] else "FAIL") if "ok" in r else "not validated"
            nf = len(r["flagged_holds"])
            L.append(f"| {r['cell']} | {_bins(r['dist_m'])} | {r['holds']}" + (f" ({nf} flagged)" if nf else "")
                     + f" | {r['median_pct']:+.2f} | "
                     + (f"{ci[0]:+.2f} to {ci[1]:+.2f}" if ci else "-") + f" | {res} |")
        L += ["", "Not validated (no or too few holds): " + (", ".join(rep["not_validated_cells"]) or "none")
              + " (no distance validated in these cells).", ""]
    unc = rep.get("uncertainty")
    if unc:
        s = unc["sigma"]
        L += ["## How well each number is pinned down", "", "How to read it:",
              f"- sigma = half the difference between two fits on alternate holds of the fit sweep: a {unc['sigma_kind']}",
              "- treat it as an order of magnitude, not a confidence interval", "",
              "| number | camera 0 | camera 1 |", "|---|---|---|"]
        L += [f"| {nm} | {_num(s['intr'][0][i], '.3g')} | {_num(s['intr'][1][i], '.3g')} |"
              for i, nm in enumerate(NAMES[:len(s['intr'][0])])]
        L += ["", "How to read it:", "- the stereo pose's sigma: its rotation (a rotation vector, deg) and camera 1's position "
              "(mm), per axis of camera 0 (x right, y down, z forward)", "",
              "| stereo pose | x | y | z |", "|---|---|---|---|",
              "| rotation (deg) | " + " | ".join(_num(v, ".4f") for v in s["C_rotvec_deg"]) + " |",
              "| translation (mm) | " + " | ".join(_num(v, ".3f") for v in s["C_t_mm"]) + " |", "",
              f"Baseline length: sigma {_num(s['baseline_mm'], '.3f')} mm.", "",
              "## Repeatability at 1 m", "", "How to read it:",
              "- the two half fits compared point by point: how far one reads the other's scene at 1 m, worst point "
              "per image cell",
              f"- beyond {unc['repeat_gate_pct']} % the cell is unconstrained by the data: collect more there, or mask it",
              "", "| cell | largest difference at 1 m (%) | grid points | flag |", "|---|---|---|---|"]
        L += [f"| {c} | {_num(v['max_abs_pct'], '.2f')} | {v['points']} | "
              + ("unconstrained" if c in unc["unconstrained_cells"] else "") + " |" for c, v in unc["repeat_cells"].items()]
        L.append("")
    if "test_flagged" not in rep:                                        # refused before any scoring
        return "\n".join(L) + "\n"
    L += ["## Holds over the quality gate", "", "How to read it:",
          f"- a hold whose corners fit a rigid board worse than {rep['gates']['lens_hold_rms_max_px']} px rms "
          "(a gross misfit; phantom12, 22 Oct: good held-out neck poses median 0.42, 95th 1.05 px)",
          f"- fit sweep only: a board deeper than {rep['gates'].get('lens_hold_max_z_m', '-')} m is dropped by "
          "step 45 too, whatever its rms (far boards)",
          "- fit sweep: dropped by step 45 before its final fit (the calibration is fitted to every hold, so a bad "
          "hold is the board's)",
          "- test sweep: flagged, not dropped, and judged with its cell: measured with the calibration being judged, "
          "a bad hold means the board or the calibration is bad there", "",
          "| sweep | hold | cell | pixel rms (px) | what happened |", "|---|---|---|---|---|"]
    rows = ([("fit", h, "-", v, "dropped by step 45: rms") for h, v in rep.get("fit_dropped", [])]
            + [("fit", h, "-", rep.get("fit_hold_rms", {}).get(str(h)), f"dropped by step 45: board at {z:.2f} m")
               for h, z in rep.get("fit_dropped_far", [])]
            + [("test", f["hold"], ", ".join(f["cells"]), f["rms_px"], "flagged: board or calibration bad here")
               for f in rep["test_flagged"]])
    L += [f"| {s} | {h} | {c} | {_rms(v)} | {w} |" for s, h, c, v, w in rows] or ["| - | none | - | - | - |"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("calibration", help="JSON with model, intr, C; file.json:key for an entry inside")
    ap.add_argument("corners", type=Path, help="step 15's npz of the TEST sweep")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--fit-report", type=Path, help="step 45's JSON (its dropped holds go in the report)")
    ap.add_argument("--uncertainty", type=Path, help="step 46's JSON (sigmas and repeatability map)")
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    rig = rigcfg.load(a.rig)
    gates = rig.lens_gates()
    entry = sf.load_entry(a.calibration)
    model, intr, C = sf.load_calibration(a.calibration)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    data = sf.Data(str(a.corners), rig.board)
    rep = {"rig": rig.name, "rig_path": str(rig.path), "gates": gates, "git_sha": sf.git_sha(),
           "calibration": {"spec": a.calibration, "model": model, "sha256": sf.calibration_digest(model, intr, C),
                           "fit_verdict": entry.get("verdict")},              # None: not a step-45 file
           "test": {"corners": str(a.corners), "sha256": sf.file_sha256(a.corners), "pairs": int(data.n),
                    "holds": int(len(np.unique(data.hold)))}}
    if a.fit_report:
        fit = json.loads(a.fit_report.read_text())
        rep["fit_dropped"] = fit.get("dropped_holds", [])
        rep["fit_dropped_far"] = fit.get("dropped_far_holds", [])          # step 45 since 22 Oct; older: none
        rep["fit_hold_rms"] = {str(h): v for h, v in fit.get("hold_rms_px", [])}
    if a.uncertainty:
        rep["uncertainty"] = json.loads(a.uncertainty.read_text())
        rep["uncertainty"].pop("repeat_map", None)                       # the points stay in step 46's file

    def finish(verdict, reason, code):
        rep["verdict"], rep["reason"] = verdict, reason
        sf.write_json(a.out_dir / "report.json", rep)
        (a.out_dir / "report.md").write_text(markdown(rep))
        print(f"{verdict}: {reason}\n-> {a.out_dir / 'report.md'}")
        return code

    refused = sf.converged_fit(entry)
    if refused:
        return finish("NO RESULT", f"{a.calibration}: {refused}", 2)
    rec, hrms, solves = ls.score(model, intr, C, data)
    rep["scoring"] = solves
    rep["test"]["pnp_failed_pairs"] = solves["pnp_failed_pairs"]
    if not ls.solved(solves):
        return finish("NO RESULT", "the scorer's least squares did not converge on the test sweep (scipy success: pose "
                      f"solve {solves['pose_success']}, pose-and-scale solve {solves['scale_success']}): no depth "
                      "number of this run is trusted", 2)
    flagged = sorted(h for h, v in hrms.items() if v > gates["lens_hold_rms_max_px"])
    places = hold_places(rec, intr[0][2:4])
    rep["test_flagged"] = [{"hold": h, "rms_px": hrms[h], "cells": sorted({c for c, _ in places[h]},
                                                                           key=lenscells.CELLS.index),
                            "dist_m": sorted(list(lenscells.DIST_BINS[i]) for _, i in places[h] if i >= 0)}
                           for h in flagged]
    rep["test_flagged"].sort(key=lambda f: (lenscells.CELLS.index(f["cells"][0]), f["hold"]))
    rep["self_check"] = ls.self_check(model, intr, C, data, rec)
    if not rep["self_check"]["solved"]:
        return finish("NO RESULT", "the scorer self-check's rescoring did not converge (scipy success false): no depth "
                      "number of this run is trusted", 2)
    if not rep["self_check"]["passed"]:
        return finish("NO RESULT", "the scorer self-check failed: no depth number of this run is trusted", 2)
    table = ls.cell_dist_table(rec, intr[0][2:4], int(gates["lens_min_holds"]))
    for r in table:
        here = (r["cell"], lenscells.DIST_BINS.index(tuple(r["dist_m"])))
        r["flagged_holds"] = [h for h in flagged if here in places[h]]
    verdict, val, notval = ls.verdict(table, gates)
    rep["table"] = val + notval
    rep["table"].sort(key=lambda r: (lenscells.CELLS.index(r["cell"]), r["dist_m"][0]))
    rep["validated_cells"] = {c: [r["dist_m"] for r in val if r["cell"] == c] for c in lenscells.CELLS
                              if any(r["cell"] == c for r in val)}
    rep["not_validated_cells"] = [c for c in lenscells.CELLS if c not in rep["validated_cells"]]
    if verdict == "NO RESULT":
        return finish(verdict, f"no cell x distance entry reached {gates['lens_min_holds']} holds", 2)
    bad = [r for r in val if not r["ok"]]
    reason = (f"all {len(val)} validated entries are within the limits" if not bad else
              f"{len(bad)} of {len(val)} validated entries are outside the limits: "
              + "; ".join(f"{r['cell']} {_bins(r['dist_m'])} m {r['median_pct']:+.2f} %" for r in bad))
    reason += f"; {len(notval)} entries not validated"
    if flagged:
        reason += f"; {len(flagged)} of {len(hrms)} test holds flagged (board or calibration bad there; judged)"
    return finish(verdict, reason, 0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    sys.exit(main())
