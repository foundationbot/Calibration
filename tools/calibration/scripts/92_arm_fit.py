#!/usr/bin/env python3
"""Step 92 — the arm session's analysis (docs/superpowers/plans/2026-10-22-phantom12-arm-session.md, Task 7A).

Inputs: arm_corners.py's dump of the sweep's recording (every image's plate corners, every /joints sample), step 91's
log, step 90's poses.json, the live OKVIS config, the URDF and the MJCF. It finds the holds in /joints, matches them
to the logged stops, fits M0 / M1 / M2 on the fit holds and judges them on the held-out ones at the plate (mm),
chooses the model, measures the noise floor (same-side repeats), the play between approach sides (opposite-side
repeats), the plate slip (X from each half of the tour), the encoder side (M0's residual against the gravity torque),
M0 through the MJCF's shoulder and neck origins (T2-1), and the plate's board-scale depth error by image region.
Writes report.json and prints the numbers with the verdict (decision 5). Numbers go to the user only after an
independent review.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import armchain  # noqa: E402
import armfit as af  # noqa: E402
import armplan as ap  # noqa: E402
import rigmath as rm  # noqa: E402

MJCF_ORIGINS = {"RightShoulderPitch": "-0.015 -0.1034 0.2823", "NeckYaw": "-0.015 0 0.3815",
                "NeckPitch": "0.025 0 0.1927"}      # phantom_mk1.xml's body positions (2026-07-11), same rpy


def urdf_with_mjcf_origins(urdf: Path) -> Path:
    """The URDF with the three joint origins the MJCF places differently (rev 3, 'URDF vs MJCF')."""
    s = urdf.read_text()
    for j, xyz in MJCF_ORIGINS.items():
        pat = re.compile(r'(<joint name="' + j + r'"[^>]*>(?:(?!</joint>).)*?<origin[^>]*?xyz=")([^"]*)(")', re.S)
        s, n = pat.subn(lambda m: m.group(1) + xyz + m.group(3), s, count=1)
        if n != 1:
            raise ValueError(f"joint {j}: no origin to replace in {urdf}")
    f = Path(tempfile.mkdtemp()) / "mjcf-origins.urdf"
    f.write_text(s)
    return f


def real_stops(rows) -> list:
    """The 'stop' rows of the last real run in a step-91 log: dry-run rows and anything before the last real
    'start' are dropped (review 2, W4: dry rows could take the real holds)."""
    rows = [r for r in rows if not r.get("dry_run")]
    starts = [i for i, r in enumerate(rows) if r.get("event") == "start"]
    return [r for r in rows[(starts[-1] + 1 if starts else 0):] if r.get("event") == "stop"]


def rms(v) -> float:
    v = np.asarray(list(v), float)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")


def pairs(matched: dict, kind: str) -> list:
    """[(hold of the first visit, hold of the repeat)] for repeats of a kind."""
    first = {}
    for h, s in sorted(matched.items()):
        if s["kind"] in ("fit", "heldout") and s["pose"] not in first:
            first[s["pose"]] = h
    return [(first[s["pose"]], h) for h, s in sorted(matched.items()) if s["kind"] == kind and s["pose"] in first]


def run(a) -> dict:
    import neckfk
    import okviscfg
    import rigcfg
    plan = json.loads(Path(a.poses).read_text())
    names = plan["names"]
    g = plan["grid"]
    grid = rigcfg.AprilGrid(g["cols"], g["rows"], g["tag_size"], g["tag_spacing"])
    d = dict(np.load(a.dump))
    ia = [names.index(n) for n in armchain.RIGHT_ARM]
    holds = af.holds_from_joints(d["jt"], d["jq"][:, ia], min_s=a.min_hold_s)
    hold_q = np.array([d["jq"][(d["jt"] >= t0) & (d["jt"] <= t1)][:, ia].mean(0) for t0, t1 in holds])
    stops = real_stops([json.loads(x) for x in Path(a.log).read_text().splitlines()])
    matched = af.match_holds(hold_q, stops)
    unmatched = [s["stop"] for s in stops if s["kind"] in af.MEASURED and s["status"] == "ok"
                 and s not in matched.values()]
    keep = sorted(matched)
    sel_holds = [holds[h] for h in keep]
    obs = af.frames_from_dump(d, sel_holds, names, armchain.RIGHT_ARM, grid)
    for f in obs:                                   # hold index -> the index into `holds`
        f.hold = keep[f.hold]
    kind = {h: matched[h]["kind"] for h in keep}
    fit_h = [h for h in keep if kind[h] == "fit"]
    ho_h = [h for h in keep if kind[h] == "heldout"]
    cfg = okviscfg.load(a.config)
    rig = af.rig_from_config(cfg)
    arm, neck = armchain.ArmChain(a.urdf), neckfk.NeckChain(a.urdf)
    X0 = np.array(plan["X0"])

    fits, held = {}, {}
    for m in af.ORDER:
        fits[m] = af.fit(obs, arm, neck, rig, m, X0, holds=set(fit_h))
        held[m] = af.plate_errors(obs, arm, neck, rig, fits[m], holds=ho_h,
                                  centre=np.mean(ap.plate_points(grid)[0], axis=0))
    heldout_rms = {m: rms(held[m].values()) for m in af.ORDER}
    chosen = af.choose(heldout_rms)
    P = fits[chosen]
    c = np.mean(ap.plate_points(grid)[0], axis=0)
    offs = af.plate_offsets(obs, arm, neck, rig, P, centre=c)
    same = [float(np.linalg.norm(offs[x] - offs[y])) for x, y in pairs(matched, "repeat_same") if x in offs and y in offs]
    opp = [float(np.linalg.norm(offs[x] - offs[y])) for x, y in pairs(matched, "repeat_opp") if x in offs and y in offs]
    floor = rms(same) / np.sqrt(2) if len(same) >= 2 else float("nan")       # fewer than 2 pairs: not measured
    half = len(fit_h) // 2
    fixed = {"dT": P["dT"], "offsets": P["offsets"]}
    Xa = af.fit(obs, arm, neck, rig, "M0", P["X"], holds=set(fit_h[:half]), fixed=fixed)["X"]
    Xb = af.fit(obs, arm, neck, rig, "M0", P["X"], holds=set(fit_h[half:]), fixed=fixed)["X"]
    slip_mm = float(np.linalg.norm((Xa[:3, :3] @ c + Xa[:3, 3]) - (Xb[:3, :3] @ c + Xb[:3, 3])) * 1e3)
    slip_deg = rm.rot_angle_deg(Xa[:3, :3].T @ Xb[:3, :3])

    # encoder side: the chosen model's plate offsets on the holds it was not fitted to (held out and repeats), against
    # the gravity torque of shoulder pitch and elbow (rev 2 item 4; review 2, W9: fitted holds absorb it)
    slope = None
    if a.mjcf:
        import armsafety
        body = armsafety.Body(a.mjcf)
        body.set_baseline({n: np.radians(v) for n, v in plan["baseline_deg"].items()})
        o0 = {h: v for h, v in offs.items() if h not in fit_h}
        hs = sorted(o0)
        G = body.gravity(hold_q[hs], plan["gravity_in_pelvis"])[:, [0, 3]]
        A = np.c_[G, np.ones(len(hs))]
        coef = np.linalg.lstsq(A, np.array([o0[h] for h in hs]), rcond=None)[0]     # rows: mm/Nm per torque, offset
        slope = {"RightShoulderPitch": float(np.linalg.norm(coef[0])),
                 "RightElbowPitch": float(np.linalg.norm(coef[1]))}

    # T2-1: M0 through the MJCF's shoulder and neck origins
    alt = urdf_with_mjcf_origins(Path(a.urdf))
    arm2, neck2 = armchain.ArmChain(alt), neckfk.NeckChain(alt)
    f2 = af.fit(obs, arm2, neck2, rig, "M0", X0, holds=set(fit_h))
    m0_mjcf = rms(af.plate_errors(obs, arm2, neck2, rig, f2, holds=ho_h, centre=c).values())

    # the plate's board-scale depth error, by image region of camera 0
    scale = af.board_scale_pct(obs, rig, grid)
    cells = {}
    for h in scale:
        uv = np.concatenate([f.uv for f in obs if f.hold == h and f.cam == 0])
        cells.setdefault(int(ap.image_cell(uv.mean(0)[None])[0]), []).append(scale[h])
    by_cell = {k: {"n": len(v), "median_pct": float(np.median(v))} for k, v in sorted(cells.items())}

    v = af.verdict({"noise_floor_mm": floor, "slip_mm": slip_mm, "slip_deg": slip_deg,
                    "heldout_rms_mm": heldout_rms})
    return {
        "holds_found": len(holds), "holds_matched": len(keep), "unmatched_stops": unmatched,
        "fit_holds": len(fit_h), "heldout_holds": len(ho_h), "same_side_pairs": len(same),
        "frames": len(obs),
        "fits": {m: {"rms_px": f["rms_px"], "within_1px": f["within_1px"], "n_corners": f["n_corners"],
                     "X": f["X"].tolist(), "dT": f["dT"].tolist(),
                     "dT_mm": float(np.linalg.norm(f["dT"][:3, 3]) * 1e3), "dT_deg": rm.rot_angle_deg(f["dT"][:3, :3]),
                     "offsets_deg": dict(zip([armchain.RIGHT_ARM[j] for j in ap.INNER],
                                             np.degrees(f["offsets"]).round(4).tolist()))}
                 for m, f in fits.items()},
        "heldout_mm": {m: {str(h): round(v, 3) for h, v in held[m].items()} for m in af.ORDER},
        "heldout_rms_mm": heldout_rms, "heldout_max_mm": {m: max(held[m].values()) for m in af.ORDER},
        "chosen": chosen, "noise_floor_mm": floor, "same_side_mm": same, "opposite_side_mm": opp,
        "slip_mm": slip_mm, "slip_deg": slip_deg, "encoder_side_mm_per_Nm": slope,
        "M0_through_mjcf_origins_heldout_rms_mm": m0_mjcf,
        "plate_centre_in_gripper_m": (P["X"][:3, :3] @ c + P["X"][:3, 3]).tolist(),
        "board_scale_pct": {str(h): round(v, 3) for h, v in scale.items()}, "board_scale_by_cell": by_cell,
        **v,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("dump", "log", "poses", "config", "urdf", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--mjcf", default="", help="the MJCF for the encoder-side check (gravity torque); empty: skipped")
    p.add_argument("--min-hold-s", type=float, default=1.5)
    a = p.parse_args(argv)
    r = run(a)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ("holds_found", "holds_matched", "unmatched_stops", "fit_holds", "heldout_holds",
                                         "heldout_rms_mm", "chosen", "noise_floor_mm", "slip_mm", "slip_deg",
                                         "encoder_side_mm_per_Nm", "M0_through_mjcf_origins_heldout_rms_mm",
                                         "board_scale_by_cell", "verdict")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
