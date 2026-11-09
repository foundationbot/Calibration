#!/usr/bin/env python3
"""One command: record-app sessions -> a validated KB4T lens calibration and its OKVIS config.

    15  corners per session (--max-blur: the moving-board mode, for a handheld board and hardware-synced cameras;
        default: still holds); a neck-grid take FOLDER (step 08) goes through step 15b instead
    16  split: every 4th hold (time order, all distances up to --fit-max-z) and anything farther go to held-out TEST
    45  the KB4T fit on the rest (FIT), up to --fit-max-z (default 3.0 m: far boards constrain the lens at range;
        fitting only <= 1.3 m, phantom12's rule, read +1.3 to +2.4 % long beyond 1.7 m on phantom09/15, 4 Nov)
    66  validation on TEST, with the fixed pass rule; with --baseline-okvis also the robot's current config, on the
        same TEST pairs, so the report says whether the new lens is better than what runs now
    71  the OKVIS config (distortion_type kb4t, --okvis-template's T_SC of cam0, T_BS and tuning kept), only on a
        PASS (step 71's own rule); on a FAIL the summary still compares it with the baseline, the call is a human's

Every step's own output stays in --out (corners, fit.json, validate*/report.md, export/); the summary at the end
gives fit rms, baseline, and per calibration the TEST depth error (median, IQR) and row error.
Deploying needs an OKVIS image that reads distortion_type kb4t (okvis_common Kb4tRectify, 774a946a).

phantom09, 4 Nov 2026 (the run this was built from, steps by hand then): three handheld takes, 2 ms exposure:
    run_lens_kb4t.py --rig ../config/rig-phantom09.yaml --out p09 --max-blur 0.5 --exposure-ms 2 \\
        --okvis-template live.yaml --baseline-okvis live.yaml A/session.mcap F/session.mcap E/session.mcap
gave 290 FIT / 414 TEST pairs, rms 0.49 px, baseline 43.73 mm, TEST depth error -0.19 % (kalibr equi: +0.81 %).

Usage: python run_lens_kb4t.py --rig RIG --out DIR [--max-blur PX --exposure-ms MS] [--okvis-template YAML]
           [--baseline-okvis YAML] [--fit-max-z M] [--holdout-every N] [--procs N] [--skip-validate] <session.mcap> ...
Run from this folder (or anywhere: it calls the steps by path) with the repo's interface/ importable.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rigcfg  # noqa: E402


def step(args, log):
    """run one pipeline step (python script in this folder) -> its exit code; output teed into log"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), str(HERE.parents[2] / "interface"), env.get("PYTHONPATH", "")])
    cmd = [sys.executable, str(HERE / args[0])] + [str(x) for x in args[1:]]
    print("==>", " ".join(cmd[1:]), flush=True)
    with open(log, "a") as fh:
        fh.write("==> " + " ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in p.stdout:
            fh.write(line)
            if "RuntimeWarning" not in line:
                sys.stdout.write("    " + line)
        return p.wait()


def run_parallel(jobs, out):
    """[(name, args)] -> {name: exit code}: each step in its own process at once, output to out/<name>-<step>.log"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), str(HERE.parents[2] / "interface"), env.get("PYTHONPATH", "")])
    procs = {}
    for name, args in jobs:
        lg = out / f"{name}-{Path(args[0]).stem}.log"
        print("==>", " ".join(str(x) for x in args), f"(parallel, log {lg.name})", flush=True)
        procs[name] = (subprocess.Popen([sys.executable, str(HERE / args[0])] + [str(x) for x in args[1:]],
                                        stdout=open(lg, "w"), stderr=subprocess.STDOUT, env=env), lg)
    rcs = {}
    for name, (p, lg) in procs.items():
        rcs[name] = p.wait()
        tail = [ln for ln in lg.read_text().splitlines() if "RuntimeWarning" not in ln][-2:]
        print(f"    {name}: exit {rcs[name]}: " + " | ".join(tail), flush=True)
    return rcs


def okvis_calibration(yaml_path, out_json):
    """the lens + stereo pose of an OKVIS config as a calibration entry step 66 can judge (no step-45 verdict)"""
    import okviscfg
    cfg = okviscfg.load(yaml_path)
    intr = [np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], np.ravel(D)] for K, D in zip(cfg.K, cfg.D)]
    model = {8: "KB4", 10: "KB4T"}.get(len(intr[0]))
    if model is None:
        sys.exit(f"ERROR: {yaml_path}: lens with {len(intr[0]) - 4} coefficients: not an equidistant/kb4t config")
    C = np.linalg.inv(cfg.T_SC[0]) @ cfg.T_SC[1]
    out_json.write_text(json.dumps({"model": model, "intr": [v.tolist() for v in intr], "C": C.tolist(),
                                    "source": str(yaml_path)}, indent=1))
    return model


BANDS = ((0.0, 1.0), (1.0, 1.3), (1.3, 1.7), (1.7, 9.9))   # summary only; step 66 judges cells up to 1.3 m


def summary_row(name, calib_json, test_npz, rig):
    """TEST depth error (median, IQR, % ; + = reads too far) and row error (median |dy|, px) of one calibration"""
    import cv2
    import lensmodel
    import lensscore
    import stereofit as sf
    model, intr, C = sf.load_calibration(calib_json)
    data = sf.Data(str(test_npz), rig.board)
    rec, _, _ = lensscore.score(model, intr, C, data)
    sc = np.array(rec["scale"]) - 1
    zz = np.array(rec["z"])
    T10 = np.linalg.inv(C)
    R0, R1, *_ = cv2.stereoRectify(np.eye(3), np.zeros(5), np.eye(3), np.zeros(5), tuple(rig.resolution),
                                   np.ascontiguousarray(T10[:3, :3]), np.ascontiguousarray(T10[:3, 3].reshape(3, 1)),
                                   flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    z = np.load(test_npz)
    dys = []
    for j in range(len(z["img_t0"])):
        m = z["det_img"] == j
        d = [{(t, k): (u, v) for t, k, u, v, c in zip(z["det_tag"][m], z["det_k"][m], z["det_u"][m], z["det_v"][m],
                                                       z["det_cam"][m]) if c == cam} for cam in (0, 1)]
        keys = sorted(set(d[0]) & set(d[1]))
        if keys:
            b0 = (R0 @ lensmodel.unproject(model, intr[0], [d[0][k] for k in keys]).T).T
            b1 = (R1 @ lensmodel.unproject(model, intr[1], [d[1][k] for k in keys]).T).T
            dys.append(rig.nominal_focal_px * (b0[:, 1] / b0[:, 2] - b1[:, 1] / b1[:, 2]))
    dy = np.concatenate(dys) if dys else np.zeros(1)
    bands = []
    for lo, hi in BANDS:
        m = (zz > lo) & (zz <= hi)
        bands.append(f"{lo:.1f}-{hi:.1f} m: {100 * np.median(sc[m]):+.2f} % (n {m.sum()})" if m.sum() >= 5
                     else f"{lo:.1f}-{hi:.1f} m: - (n {m.sum()})")
    return (f"{name:<22} {model:<5} baseline {1e3 * np.linalg.norm(C[:3, 3]):6.2f} mm | TEST {len(sc)} pairs: depth "
            f"{100 * np.median(sc):+6.2f} % (IQR {100 * np.percentile(sc, 25):+.2f} .. {100 * np.percentile(sc, 75):+.2f})"
            f" | rows {np.median(np.abs(dy)):.3f} px\n{'':<29}by distance: " + " | ".join(bands))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", type=Path, nargs="+", help="record-app session.mcap files and/or neck-grid take "
                                                             "folders (step 08: <take>/pose_NNN.npz)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-blur", type=float, help="moving-board mode (step 15): blur limit, px")
    ap.add_argument("--exposure-ms", type=float, help="moving-board mode: the cameras' exposure")
    ap.add_argument("--fit-max-z", type=float, default=3.0, help="steps 16 + 45: fit boards up to this distance (m); "
                                                                  "the rig's lens_hold_max_z_m (1.3) is the old rule")
    ap.add_argument("--okvis-template", type=Path, help="step 71: the robot's current OKVIS config (kept: T_SC0, "
                                                         "T_BS, tuning); without it no OKVIS config is written")
    ap.add_argument("--baseline-okvis", type=Path, help="also validate this OKVIS config on the same TEST pairs")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--skip-validate", action="store_true", help="stop after the fit (no 66, no 71)")
    ap.add_argument("--min-validated", type=int, default=6, help="step 71 only if step 66 judged at least this "
                                                                  "many cell x distance entries (a PASS on 1 means "
                                                                  "nothing)")
    ap.add_argument("--holdout-every", type=int, default=4, help="step 16: every N-th near hold to TEST")
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    if (a.max_blur is None) != (a.exposure_ms is None):
        sys.exit("ERROR: --max-blur and --exposure-ms go together (the moving-board mode)")
    rig = rigcfg.load(a.rig)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    rigarg = ["--rig", Path(a.rig).resolve()] if a.rig else []

    corners = []
    for s in a.sessions:
        if s.is_dir() and any(s.glob("pose_*.npz")):        # a neck-grid take folder (step 08)
            c = out / f"corners-{s.resolve().name}.npz"
            if step(["15b_dump_neck_take.py", s.resolve(), c, "--procs", a.procs] + rigarg, log):
                print(f"WARNING: step 15b found nothing usable in {s}; left out", flush=True)
                continue
            corners.append(c)
            continue
        c = out / f"corners-{s.resolve().parent.name or s.stem}.npz"
        mode = (["--max-blur", a.max_blur, "--exposure-ms", a.exposure_ms, "--procs", a.procs]
                if a.max_blur is not None else [])
        if step(["15_dump_corners.py", s.resolve(), c] + mode + rigarg, log):
            sys.exit(f"ERROR: step 15 failed on {s} (see {log})")
        corners.append(c)
    if not corners:
        sys.exit("ERROR: no corners from any input")

    fit_npz, test_npz = out / "fit.npz", out / "test.npz"
    zarg = ["--max-fit-z", a.fit_max_z, "--holdout-every", a.holdout_every]
    if step(["16_split_distance.py", fit_npz, test_npz] + corners + zarg + rigarg, log):
        sys.exit(f"ERROR: step 16 failed (see {log})")

    fit_json = out / "fit-kb4t.json"
    rc = step(["45_fit_lens.py", fit_npz, fit_json, "--model", "KB4T", "--max-z", a.fit_max_z] + rigarg, log)
    if rc:
        sys.exit(f"ERROR: step 45 exit {rc}: no usable fit (see {log} and {fit_json})")
    if a.skip_validate:
        print(summary_row("new KB4T fit", fit_json, test_npz, rig))
        return

    rows, verdict, names = [], None, {0: "PASS", 1: "FAIL", 2: "NO RESULT"}
    # step 66 for the new fit and for the baseline are independent: run them at the same time
    jobs = [("new", ["66_validate_lens.py", fit_json, test_npz, out / "validate", "--fit-report", fit_json] + rigarg)]
    if a.baseline_okvis:
        base_json = out / "baseline-okvis.json"
        okvis_calibration(a.baseline_okvis, base_json)
        jobs.append(("baseline", ["66_validate_lens.py", base_json, test_npz, out / "validate-baseline"] + rigarg))
    rcs = run_parallel(jobs, out)
    verdict = names.get(rcs["new"], f"exit {rcs['new']}")
    rows.append(summary_row("new KB4T fit", fit_json, test_npz, rig))
    if a.baseline_okvis:
        rows.append(summary_row(f"baseline ({a.baseline_okvis.name})", base_json, test_npz, rig)
                    + f" | step 66 {names.get(rcs['baseline'], rcs['baseline'])}")

    exported, nval = None, 0
    rep_path = out / "validate" / "report.json"
    if rep_path.exists():
        vc = json.loads(rep_path.read_text()).get("validated_cells", {})   # cell -> its judged distance bins
        nval = sum(len(v) for v in vc.values()) if isinstance(vc, dict) else len(vc)
    if verdict == "PASS" and nval < a.min_validated:
        verdict = f"PASS on only {nval} entries (< --min-validated {a.min_validated}): not enough to trust"
    if a.okvis_template and verdict == "PASS":
        if step(["71_export_lens.py", fit_json, out / "validate" / "report.json", out / "export", "--okvis-template",
                 a.okvis_template.resolve(), "--okvis-kb4t"] + rigarg, log) == 0:
            exported = out / "export" / rig.okvis_yaml_name

    print("\n=== summary " + "=" * 100)
    fit = json.loads(fit_json.read_text())
    print(f"step 45: {fit.get('verdict')}   step 66 on TEST: {verdict}, {nval} entries judged   "
          f"({out / 'validate' / 'report.md'})")
    for r in rows:
        print(r)
    why = "" if a.okvis_template else " (no --okvis-template)"
    why = why or ("" if verdict == "PASS" else f" (step 66 {verdict}: only a PASS on >= {a.min_validated} entries "
                                               "is exported)")
    print(f"OKVIS config: {exported if exported else 'not written' + why}")


if __name__ == "__main__":
    main()
