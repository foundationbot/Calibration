#!/usr/bin/env python3
"""Step 15b — corners from a neck-grid take folder (step 08's <take>/pose_NNN.npz), in step 15's npz layout.

Each neck pose is one still hold. In every left/right pair of the pose the board is detected. A pair is kept when its
left corners sit within --still-px of the pose's first detection (median displacement over the corners both
detected: a mean over a varying corner set read moving corners as motion, and passed only 21 of 60 still pairs on
phantom14). Up to --per-hold kept pairs from the middle of the pose go out, with both cameras' corners.

Output: img_hold, img_t0, img_t1, det_img, det_cam, det_tag, det_k, det_u, det_v (camera 0 = left), plus per pair the
measured neck angles img_pitch / img_yaw (rad) and the approach direction of the last move, img_ap / img_ay (-1/0/+1).
Steps 16/45/66 read it like a step-15 file. run_lens_kb4t.py calls this for take folders.
Usage: python 15b_dump_neck_take.py <take dir> <out.npz> [--still-px 1.0] [--per-hold 3] [--min-tags 6] [--procs 4]
           [--rig PATH]
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rigcfg  # noqa: E402


_W = {}   # the pool workers' view (fork): board, still_px, min_tags


def _pose(f):
    """one pose file -> (cmd, neck, tl, tr, [(i, left detection, right detection) of the still pairs])"""
    board, still_px, min_tags = _W["board"], _W["still_px"], _W["min_tags"]
    z = np.load(f)
    kept, d0 = [], None
    for i in range(len(z["tl"])):
        dl = board.detect(z["L"][i])
        if len({t for t, _ in dl}) < min_tags:
            continue
        if d0 is None:
            d0 = dl
        common = set(d0) & set(dl)
        if not common or np.median([np.hypot(*(np.subtract(dl[k], d0[k]))) for k in common]) > still_px:
            continue
        dr = board.detect(z["R"][i])
        if len({t for t, _ in dr}) < min_tags:
            continue
        kept.append((i, dl, dr))
    return np.asarray(z["cmd"], float), np.asarray(z["neck"], float), np.asarray(z["tl"]), np.asarray(z["tr"]), kept


def dump_take(take, board, out, still_px=1.0, per_hold=3, min_tags=6, procs=1):
    """-> (poses read, poses with the board in both cameras, pairs written); procs > 1 detects poses in parallel
    (fork pool: detection is all the work, the poses are independent)"""
    img_hold, t0s, t1s, rows, nposes, nheld = [], [], [], [], 0, 0
    yaws, pitches, ays, aps, prev = [], [], [], [], None
    files = sorted(glob.glob(f"{take}/pose_*.npz"))
    _W.update(board=board, still_px=still_px, min_tags=min_tags)
    if procs > 1 and len(files) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(procs, len(files))) as pool:
            results = pool.map(_pose, files)
    else:
        results = [_pose(f) for f in files]
    for h, (cmd, neck, tl, tr, kept) in enumerate(results):
        nposes += 1
        ap_, ay_ = (0, 0) if prev is None else (int(np.sign(cmd[0] - prev[0])), int(np.sign(cmd[1] - prev[1])))
        prev = cmd
        pit, yaw = np.radians(neck)
        mid = len(kept) // 2
        for i, dl, dr in kept[max(0, mid - per_hold // 2): max(0, mid - per_hold // 2) + per_hold]:
            j = len(img_hold)
            img_hold.append(h)
            t0s.append(int(tl[i]))
            t1s.append(int(tr[i]))
            yaws.append(yaw)
            pitches.append(pit)
            ays.append(ay_)
            aps.append(ap_)
            for cam, det in enumerate((dl, dr)):
                for (tag, k), (u, v) in det.items():
                    rows.append((j, cam, tag, k, u, v))
        nheld += bool(kept)
    r = np.array(rows, float).reshape(-1, 6)
    np.savez_compressed(out, img_hold=np.array(img_hold, np.int64), img_t0=np.array(t0s, np.int64),
                        img_t1=np.array(t1s, np.int64), det_img=r[:, 0].astype(np.int64),
                        det_cam=r[:, 1].astype(np.int64), det_tag=r[:, 2].astype(np.int64),
                        det_k=r[:, 3].astype(np.int64), det_u=r[:, 4], det_v=r[:, 5], img_yaw=np.array(yaws),
                        img_pitch=np.array(pitches), img_ay=np.array(ays, np.int64), img_ap=np.array(aps, np.int64))
    return nposes, nheld, len(img_hold)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("take", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--still-px", type=float, default=1.0)
    ap.add_argument("--per-hold", type=int, default=3)
    ap.add_argument("--min-tags", type=int, default=6)
    ap.add_argument("--procs", type=int, default=4, help="poses detected in parallel")
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    n, held, kept = dump_take(a.take, rigcfg.load(a.rig).board, a.out, a.still_px, a.per_hold, a.min_tags, a.procs)
    print(f"{a.take}: {n} poses, {held} with the board in both cameras, {kept} still pairs -> {a.out}")
    if not kept:
        sys.exit(f"ERROR: {a.take}: no still pair with the board in both cameras (out of range, dark, or edge-on?)")


if __name__ == "__main__":
    main()
