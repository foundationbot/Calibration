#!/usr/bin/env python3
"""Step 16 — split step-15 corner files by board distance into a FIT set and a held-out TEST set.

Step 45 fits only boards up to gates.lens_hold_max_z_m (far boards are small in the image and, on phantom12, fitted
badly: hold-gate plan, 22 Oct) and rejects a recording with more than gates.lens_bad_hold_max_frac of its holds
dropped. A handheld session mixes distances inside every take, so its corner files go through here first. TEST gets
  - every pair beyond --max-fit-z (default: the rig's lens_hold_max_z_m): depth at range, which no near board shows;
    step 66 bins only to 1.3 m (lenscells.DIST_BINS), so run_lens_kb4t.py reports these per distance band itself
  - every --holdout-every-th hold of the near ones (default 4: a quarter), in time order within each file, so the
    held-out near boards come from all along each take: these are what step 66 judges per image cell and distance
    (4 Nov: a far-only TEST left step 66 one judgeable cell, and a "PASS" on it meant nothing)
FIT gets the remaining near pairs. No pair is in both, so step 71's check that the validation used other data than
the fit passes (the two files differ).

The distance is a rough one: PnP of the left image's corners with a plain equidistant lens (no k1..k4) at the rig's
nominal focal length and the image centre, within a few per cent of step 66's own distance (phantom09: median ratio
1.01); enough to sort pairs, not to measure. A hold is never split between FIT and the near TEST part.

Output: two npz files in step 15's layout plus img_z (the rough distance, m) in each.
Usage: python 16_split_distance.py <fit.npz> <test.npz> <corners.npz> [<corners.npz> ...] [--max-fit-z M]
           [--holdout-every N] [--rig PATH]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rigcfg  # noqa: E402

PAIR_KEYS = ("img_hold", "img_t0", "img_t1")
DET_KEYS = ("det_img", "det_cam", "det_tag", "det_k", "det_u", "det_v")


def rough_z(z, board, focal_px, resolution):
    """per pair: camera-0 depth (m) of the centre of the board's SEEN corners, as steps 45 and 66 measure a hold's
    distance (lensscore: the centre of the corners seen), by PnP of the left-image corners with a plain equidistant
    lens (no k1..k4) at the rig's nominal focal length and the image centre; nan if PnP fails. Checked on phantom09
    (4 Nov) against step 66's own distance: median ratio 1.01 (p10-p90 0.98-1.04). The first version read a pinhole
    at the board origin (a corner tag) and was 1.26-1.34x too far: it sent pairs step 66 puts at 0.9-1.3 m to TEST"""
    import cv2
    K = np.array([[focal_px, 0, resolution[0] / 2.0], [0, focal_px, resolution[1] / 2.0], [0, 0, 1.0]])
    out = np.full(len(z["img_t0"]), np.nan)
    for j in range(len(out)):
        m = (z["det_img"] == j) & (z["det_cam"] == 0)
        if m.sum() < 8:
            continue
        obj = np.array([board.obj_points(int(t))[int(k)] for t, k in zip(z["det_tag"][m], z["det_k"][m])], float)
        img = np.stack([z["det_u"][m], z["det_v"][m]], 1).astype(float)
        # an equidistant lens (r = f theta, the rig's pinhole-equi) read as a pinhole shrinks the board and reads
        # it 1.26-1.34x too far (phantom09, 4 Nov): undo r = f theta to the pinhole plane first, no k1..k4
        norm = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2), K, np.zeros(4)).reshape(-1, 2)
        ok, rv, tv = cv2.solvePnP(obj, norm, np.eye(3), None)
        if ok:
            R, _ = cv2.Rodrigues(rv)
            out[j] = float((R @ obj.mean(0) + np.ravel(tv))[2])
    return out


def take(z, keep, hold_offset=0):
    """the pairs `keep` (bool per pair) of corner file z, renumbered from 0, holds shifted by hold_offset"""
    idx = np.flatnonzero(keep)
    remap = -np.ones(len(z["img_t0"]), np.int64)
    remap[idx] = np.arange(len(idx))
    d = np.isin(z["det_img"], idx)
    out = {k: np.asarray(z[k])[idx] for k in PAIR_KEYS + ("img_z",)}
    out["img_hold"] = out["img_hold"] + hold_offset
    out.update({k: np.asarray(z[k])[d] for k in DET_KEYS})
    out["det_img"] = remap[np.asarray(z["det_img"])[d]]
    return out


def concat(parts):
    """step-15 files joined: pair indices offset per part (holds were offset by take())"""
    out, n = {k: [] for k in PAIR_KEYS + ("img_z",) + DET_KEYS}, 0
    for p in parts:
        for k in out:
            out[k].append(p[k] + n if k == "det_img" else p[k])
        n += len(p["img_t0"])
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


def near_holdout(holds, t0, near, every):
    """bool per pair: near pairs whose hold is every `every`-th near hold in time order (0: none)"""
    out = np.zeros(len(holds), bool)
    if every <= 0:
        return out
    hs = [h for h in sorted(set(holds[near].tolist()), key=lambda h: t0[(holds == h) & near].min())]
    held = set(hs[every - 1::every])
    return near & np.isin(holds, list(held))


def split(files, board, focal_px, resolution, max_fit_z, holdout_every=4):
    """-> (fit dict, test dict): near pairs (rough distance <= max_fit_z) in fit except every holdout_every-th near
    hold, which goes to test with all farther (or unknown-distance) pairs"""
    fit, test, hold_offset = [], [], 0
    for f in files:
        z = dict(np.load(f))
        z["img_z"] = rough_z(z, board, focal_px, resolution)
        near = np.nan_to_num(z["img_z"], nan=np.inf) <= max_fit_z
        held = near_holdout(np.asarray(z["img_hold"]), np.asarray(z["img_t0"]), near, holdout_every)
        fit.append(take(z, near & ~held, hold_offset))
        test.append(take(z, ~near | held, hold_offset))
        hold_offset += int(z["img_hold"].max()) + 1 if len(z["img_hold"]) else 0
    return concat(fit), concat(test)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fit_out", type=Path)
    ap.add_argument("test_out", type=Path)
    ap.add_argument("corners", type=Path, nargs="+", help="step 15's npz files")
    ap.add_argument("--max-fit-z", type=float, help="default: the rig's gates.lens_hold_max_z_m")
    ap.add_argument("--holdout-every", type=int, default=4, help="every N-th near hold to TEST (0: far only)")
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    rig = rigcfg.load(a.rig)
    zmax = a.max_fit_z if a.max_fit_z is not None else float(rig.lens_gates()["lens_hold_max_z_m"])
    fit, test = split(a.corners, rig.board, rig.nominal_focal_px, rig.resolution, zmax, a.holdout_every)
    np.savez_compressed(a.fit_out, **fit)
    np.savez_compressed(a.test_out, **test)
    tz = test["img_z"]
    far = np.nan_to_num(tz, nan=np.inf) > zmax
    print(f"split at {zmax:.2f} m, every {a.holdout_every}th near hold held out: FIT {len(fit['img_t0'])} pairs -> "
          f"{a.fit_out}; TEST {len(tz)} pairs ({int((~far).sum())} near, {int(far.sum())} beyond {zmax:.2f} m) -> "
          f"{a.test_out}")
    if not len(tz):
        sys.exit("ERROR: no TEST pair: nothing to validate the fit on")
    if not far.any():
        print(f"WARNING: no pair beyond {zmax:.2f} m: the fit's depth at range is not tested (record farther boards)")


if __name__ == "__main__":
    main()
