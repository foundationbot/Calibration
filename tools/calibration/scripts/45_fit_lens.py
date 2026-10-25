#!/usr/bin/env python3
"""Step 45 — the KB4T stereo lens fit, from the rig file's nominal camera, on the FIT sweep's corners (step 15).

KB4T = equidistant fisheye k1..k4 plus Brown tangential p1, p2 for both cameras, and the stereo pose C (camera-1
points into camera 0); --model KB4 drops p1, p2. The fit (stereofit.fit_from_nominal): board poses by PnP with the
nominal camera, the stereo pose alone, then everything, converged or nothing: up to 3 attempts, each restarted from
the last; an attempt is up to 4 trim rounds of least squares, each capped at 2000 evaluations (stereofit.MAX_ATTEMPTS,
TRIM_ROUNDS, MAX_NFEV). No scipy success after them: exit 2, "unconverged". Then the quality gate (hold-gate plan,
22 Oct, rule Z): per hold, the board centre's median camera-0 depth and the rigid-board pixel rms with the fitted
camera; holds beyond gates.lens_hold_max_z_m (far boards, unusable whatever their rms) or above
gates.lens_hold_rms_max_px (gross misfits) are dropped, listed, and the fit is repeated without them (warm); more than
gates.lens_bad_hold_max_frac of the holds dropped, both kinds together, rejects the recording (exit 2). Last, camera 1
must sit on the side the rig file says (exit 3 otherwise, step 60's camera-identity rule): the fit finds the true side
even from a start on the wrong one, so a contradiction means the rig file's left/right is wrong.

--start <fit.json>: a warm fit from a saved step-45 fit of the SAME corners (its inputs.sha256) that reached scipy's
convergence, instead of the fit from nominal; then the same gate and checks. For re-deciding a gate on a fit already
made (hold-gate plan D5, 22 Oct) without its 49 min from nominal on a loaded Thor.

Real data: 5-27 min per KB4T fit on a quiet Thor (20 Oct), 49 min on a loaded one (22 Oct). Run it there, in a capped
calib-py:3 container.
Usage: python 45_fit_lens.py <fit-corners.npz> <out.json> [--model KB4T] [--rig PATH] [--start fit.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rigcfg  # noqa: E402
import stereofit as sf  # noqa: E402


def hold_depths(data, poses):
    """{hold: median camera-0 depth z (m) of its pairs' board centres (the centre of the corners seen, as
    lensscore.score)} at the fitted board poses"""
    R = sf.rotmat(poses[:, :3])
    z = np.array([(R[p] @ data.P[data.pair == p].mean(0) + poses[p, 3:])[2] for p in range(data.n)])
    return {int(h): float(np.median(z[data.hold == h])) for h in np.unique(data.hold)}


def quality_gate(data, r, gates):
    """-> ({hold: rms px}, [holds above the rms gate], {hold: z m}, [holds beyond the depth gate])"""
    rms, z = sf.hold_rms(data, r["res"]), hold_depths(data, r["poses"])
    return (rms, sorted(h for h, v in rms.items() if v > gates["lens_hold_rms_max_px"]),
            z, sorted(h for h, v in z.items() if v > gates["lens_hold_max_z_m"]))


def unconverged_text(r):
    """what the fit tried, from the attempts it made and the caps it ran with"""
    n = len(r["attempts"])
    return (f"{n} attempt{'' if n == 1 else 's'} of at most {r['max_attempts']} did not reach scipy's convergence "
            f"({r['nfev']} evaluations; cap {r['max_nfev']} evaluations per solve round, up to {sf.TRIM_ROUNDS} "
            "trim rounds an attempt)")


def start_from(path, corners, model):
    """-> ([intr0, intr1], C) of a saved step-45 fit, refused (SystemExit) unless it is a converged fit of the same
    corners with the same model"""
    try:
        f = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        sys.exit(f"ERROR: --start {path}: {e}")
    if f.get("inputs", {}).get("sha256") != sf.file_sha256(corners):
        sys.exit(f"ERROR: --start {path}: fitted on corners with sha256 {f.get('inputs', {}).get('sha256')}, "
                 f"not {corners}'s {sf.file_sha256(corners)}")
    if not f.get("success") or "intr" not in f:
        sys.exit(f"ERROR: --start {path}: not a converged fit (success {f.get('success')})")
    if f.get("model") != model:
        sys.exit(f"ERROR: --start {path}: model {f.get('model')}, not {model}")
    return [np.asarray(v, float) for v in f["intr"]], np.asarray(f["C"], float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("corners", type=Path, help="step 15's npz of the FIT sweep")
    ap.add_argument("out", type=Path, help="the fit (JSON), written whatever the outcome")
    ap.add_argument("--model", default="KB4T", choices=sorted(sf.lensmodel.MODELS))
    ap.add_argument("--start", type=Path, help="a saved converged step-45 fit of the same corners: warm fit from it")
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    rig = rigcfg.load(a.rig)
    gates = rig.lens_gates()
    try:
        intr0, C0 = sf.nominal_start(rig, a.model)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
    start = start_from(a.start, a.corners, a.model) if a.start else None
    t0 = time.time()
    data = sf.Data(str(a.corners), rig.board)
    print(f"{a.corners}: {data.n} pairs in {len(np.unique(data.hold))} holds with >= 12 corners in both cameras")
    out = {"model": a.model, "rig": {"name": rig.name, "path": str(rig.path), "nominal_focal_px": rig.nominal_focal_px,
                                     "nominal_baseline_m": rig.nominal_baseline_m},
           "inputs": {"corners": str(a.corners), "sha256": sf.file_sha256(a.corners)}, "git_sha": sf.git_sha(),
           "gates": gates}
    if a.start:
        out["start"] = {"path": str(a.start), "sha256": sf.file_sha256(a.start)}

    def finish(verdict, code, r=None, used=None):
        out["verdict"] = verdict
        if r is not None:
            W, H = rig.resolution
            out.update({"intr": [v.tolist() for v in r["intr"]], "C": np.asarray(r["C"]).tolist(),
                        "baseline_mm": float(1e3 * np.linalg.norm(r["C"][:3, 3])), "success": bool(r["success"]),
                        "nfev": r["nfev"], "attempts": r["attempts"], "fit_pairs": int(used.n),
                        "fit_holds": int(len(np.unique(used.hold))), "kept_holds": sorted(set(used.hold.tolist())),
                        "rms_px": sf.fit_cost(r)[1], "cut_fraction": float(1 - r["kept"].mean()),
                        "rms_by_ring": sf.ring_rms(used, r["res"], ((W - 1) / 2, (H - 1) / 2))})
        out["seconds"] = round(time.time() - t0)
        sf.write_json(a.out, out)
        print(f"{verdict} -> {a.out}")
        return code

    used, r = sf.warm_fit(data, a.model, *start) if start else sf.fit_from_nominal(data, a.model, intr0, C0)
    out["pnp_failed_pairs"] = r["pnp_failed_pairs"]                 # left out of the fit: their PnP start failed
    if not r["success"]:
        return finish(f"unconverged: {unconverged_text(r)}", 2, r, used)
    rms, misfit, z, far = quality_gate(used, r, gates)
    out["hold_rms_px"] = [[h, v] for h, v in sorted(rms.items())]
    out["hold_z_m"] = [[h, v] for h, v in sorted(z.items())]
    out["dropped_holds"] = [[h, rms[h]] for h in misfit]
    out["dropped_far_holds"] = [[h, z[h]] for h in far]
    bad = sorted(set(misfit) | set(far))
    print(f"quality gate: {len(misfit)} of {len(rms)} holds above {gates['lens_hold_rms_max_px']} px"
          + (": " + ", ".join(f"{h} ({rms[h]:.2f} px)" for h in misfit) if misfit else "")
          + f"; {len(far)} beyond {gates['lens_hold_max_z_m']} m"
          + (": " + ", ".join(f"{h} ({z[h]:.2f} m)" for h in far) if far else ""))
    if len(bad) > gates["lens_bad_hold_max_frac"] * len(rms):
        return finish(f"rejected: {len(bad)} of {len(rms)} holds fit a rigid board worse than "
                      f"{gates['lens_hold_rms_max_px']} px or sit beyond {gates['lens_hold_max_z_m']} m "
                      f"(limit {gates['lens_bad_hold_max_frac']:.0%})", 2, r, used)
    if bad:
        used, r = sf.warm_fit(used.select(~np.isin(used.hold, bad)), a.model, r["intr"], r["C"])
        out["pnp_failed_pairs"] += r["pnp_failed_pairs"]
        if not r["success"]:
            return finish(f"unconverged after dropping the quality gate's holds: {unconverged_text(r)}", 2, r, used)
    side = rig.expected_dx_sign()
    if np.sign(r["C"][0, 3]) != side:
        return finish(f"camera sides: the fit puts camera 1 at x = {1e3 * r['C'][0, 3]:+.1f} mm in camera 0, but "
                      f"{rig.path.name} says kalibr cam0 is the {rig.kalibr[0]['side'].upper()} eye (camera 1 at "
                      f"{'+' if side > 0 else '-'}x). Fix cameras.kalibr[].side.", 3, r, used)
    return finish(f"converged: rms {sf.fit_cost(r)[1]:.3f} px over {used.n} pairs, baseline "
                  f"{1e3 * np.linalg.norm(r['C'][:3, 3]):.2f} mm", 0, r, used)


if __name__ == "__main__":
    sys.exit(main())
