"""The plate yardstick (plan docs/superpowers/plans/2026-10-25-tagfree-selfcal.md, Phase C, on the stereo-depth-eval
branch): phantom12's right-arm plate sessions frozen as an offline test set for any tag-free camera-to-arm method.

A method hands in, per capture (burst), the hand link RightGripperYawLink in Fit C's camera 0 and whether the method
fitted on that capture. Each is judged against the plate's hand pose at the wrist point (RightWristRollLink's origin,
mm) and in angle (deg), the gate of the rl1024 test: 3 mm and 0.3 deg. Only held-out captures count; the verdict
needs >= 80 % of at least 3 of them passing. The set is frozen by a manifest of sha256 sums that every read checks.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rigmath as rm  # noqa: E402

GATE_MM, GATE_DEG = 3.0, 0.3
PASS_SHARE, MIN_HELD = 0.8, 3
MANIFEST = "MANIFEST.json"


def pose_error(T_ref, T, p_hand) -> tuple:
    """(mm, deg): where T puts the hand-frame point p_hand against where T_ref puts it, and the angle between them."""
    T_ref, T = np.asarray(T_ref, float), np.asarray(T, float)
    p = np.r_[np.asarray(p_hand, float), 1.0]
    return float(np.linalg.norm((T @ p - T_ref @ p)[:3]) * 1e3), rm.rot_angle_deg(T_ref[:3, :3].T @ T[:3, :3])


def score(ref: dict, cand: dict) -> dict:
    """ref {capture: {T_C0_hand, wrist_in_hand}} (one reference model); cand {capture: {T_C0_hand, in_sample}}."""
    caps, held, ok = {}, 0, 0
    for n, c in cand.items():
        if n not in ref:
            continue
        if "in_sample" not in c:
            raise ValueError(f"{n}: say whether the method fitted on it (in_sample)")
        mm, deg = pose_error(ref[n]["T_C0_hand"], c["T_C0_hand"], ref[n]["wrist_in_hand"])
        p = bool(mm <= GATE_MM and deg <= GATE_DEG)
        counted = not c["in_sample"]
        held += counted; ok += counted and p
        caps[n] = {"wrist_mm": round(mm, 2), "rot_deg": round(deg, 3), "pass": p, "counted": counted}
    if held < MIN_HELD:
        verdict = "TOO FEW HELD-OUT"
    else:
        verdict = "PASS" if ok >= PASS_SHARE * held else "FAIL"
    return {"captures": caps, "missing": sorted(set(ref) - set(cand)), "held_out": held, "held_out_pass": ok,
            "gate": {"mm": GATE_MM, "deg": GATE_DEG, "share": PASS_SHARE, "min_held_out": MIN_HELD}, "verdict": verdict}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_manifest(root, files) -> dict:
    """sha256 of each file (paths relative to root, or absolute) into root/MANIFEST.json."""
    root = Path(root)
    m = {str(f): _sha(root / f) for f in files}
    (root / MANIFEST).write_text(json.dumps(m, indent=1))
    return m


def check_manifest(root) -> dict:
    root = Path(root)
    m = json.loads((root / MANIFEST).read_text())
    bad = [f for f, s in m.items() if _sha(root / f) != s]
    if bad:
        raise ValueError(f"the frozen set changed: {bad}")
    return m
