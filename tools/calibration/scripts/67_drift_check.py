#!/usr/bin/env python3
"""Step 67 — has the head moved since its calibration? A one-minute check from one still board.

From a 30 s recording of a still board (~0.7-1.0 m, centred in the image) it measures camera 1's roll and pitch
against the calibration OKVIS runs (epipolar coplanarity of the corners' rays) and the depth error at the board
(board scale), and says OK, DRIFTED or NO RESULT (driftcheck.py; plan
docs/superpowers/plans/2026-10-22-phantom12-drift-check.md). It would have caught phantom12's 19-21 Oct camera-1 roll
(-0.36 deg) at once.

  DRIFTED    |roll| > 0.10 deg, |pitch| > 0.03 deg, or (board at 0.4-1.3 m) |depth error| > 1.0 %
  NO RESULT  fewer than 10 still pairs, the board too small in the image (< 250 x 150 raw px) or off-centre
             (> 150 / 100 raw px from the principal point; off-centre a lens mismatch reads as pitch)

Exit 0 OK, 1 DRIFTED, 2 NO RESULT, 3 an error (exit 1 never means anything but DRIFTED).
Usage: python 67_drift_check.py (--session session.mcap | --corners corners.npz)
           (--config okvis.yaml | --calibration fit.json[:key]) --rig config/rig-phantom12.yaml [--out report.json]"""
import argparse
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import driftcheck as dc  # noqa: E402
import rigcfg  # noqa: E402
import stereofit as sf  # noqa: E402

PER_HOLD = 30                  # a still board is one long hold: keep 30 of its pairs (step 15 keeps 3 per hold)
EXIT = {"OK": 0, "DRIFTED": 1, "NO RESULT": 2}


def _dump(path, board, out, still_px=1.0, per_hold=PER_HOLD):
    return importlib.import_module("15_dump_corners").dump(path, board, out, still_px=still_px, per_hold=per_hold)


def main(argv=None) -> int:
    try:
        return _main(argv)
    except SystemExit:
        raise
    except Exception as e:                       # noqa: BLE001 -- a crash must not read as DRIFTED (exit 1)
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 3


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--session", type=Path, help="the record app's session.mcap of a still board")
    src.add_argument("--corners", type=Path, help="a step-15 corner file")
    cal = ap.add_mutually_exclusive_group(required=True)
    cal.add_argument("--config", type=Path, help="the OKVIS config OKVIS runs (kb4t or equidistant)")
    cal.add_argument("--calibration", help="a fit JSON (file.json[:key]) with model, intr, C")
    ap.add_argument("--out", type=Path)
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    rig = rigcfg.load(a.rig)
    if not hasattr(rig.board, "n_tags"):
        print(f"ERROR: {rig.path.name}'s board is not an AprilGrid (pass --rig config/rig-phantom12.yaml)", file=sys.stderr)
        return 3
    if a.config:
        model, intr, C = dc.load_okvis(a.config, rig.okvis_order)
        source = str(a.config)
    else:
        model, intr, C = sf.load_calibration(a.calibration)
        source = a.calibration
    with tempfile.TemporaryDirectory() as tmp:
        corners = a.corners
        if a.session:
            corners = Path(tmp) / "corners.npz"
            _dump(a.session, rig.board, corners, per_hold=PER_HOLD)
        rep = dc.check(model, intr, C, dict(np.load(corners)), rig.board)
    rep["calibration"] = {"source": source, "sha256": sf.calibration_digest(model, intr, C)}
    rep["input"] = str(a.session or a.corners)
    if a.out:
        sf.write_json(a.out, rep)
    nums = (f"roll {rep['roll_deg']:+.3f} deg, pitch {rep['pitch_deg']:+.3f} deg" if "roll_deg" in rep else "no fit")
    if "depth_pct" in rep:
        nums += f", depth {rep['depth_pct']:+.2f} % at {rep['z_m']:.2f} m" + ("" if rep["depth_judged"] else " (not judged)")
    print(f"{rep['verdict']}: {nums} ({rep['pairs']} pairs)")
    for r in rep["reasons"]:
        print(f"  - {r}")
    return EXIT[rep["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
