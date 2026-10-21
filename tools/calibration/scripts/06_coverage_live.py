#!/usr/bin/env python3
"""Step 06 — live coverage map for the hand-held KB4T sweeps: where in the image, and how far away, the board has been
held still, while it is being held.

Reads the left camera's live stream (mediamtx RTSP on the robot: compressed frames, fine for locating the board, never
for calibration; the fit's corners come from the lossless recording, step 15), detects the AprilGrid and counts HOLDS:
the board centre staying within --still-px of where a still stretch began, for --still-s. A hold lands in an image
cell (lenscells, relative to the nominal principal point) and a distance bin (board-centre depth from the board's
known size: PnP with the nominal equidistant camera, ~1 %). Every --every s it prints a text map (holds per cell and
bin; a cell is done at --target holds in at least one bin, by default gates.lens_min_holds + 4) and rewrites
<out-dir>/coverage.png and coverage.json. Run it once per sweep (fit, then test): sweeps are counted separately.
Usage (Thor, calib-py:3, --network host): python 06_coverage_live.py <out-dir> [--url rtsp://127.0.0.1:8554/left]
       [--still-px 2] [--still-s 1.0] [--every 5] [--target N] [--rig PATH]
       Ctrl-C ends it; the last map stays in <out-dir>"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lenscells  # noqa: E402
import lensmodel  # noqa: E402
import rigcfg  # noqa: E402

GRID = (("top-left", "top", "top-right"), ("left", "centre", "right"), ("bottom-left", "bottom", "bottom-right"))


class StillTracker:
    """Counts holds in a stream of board centres: a still stretch starts at a frame and lasts while the centre stays
    within still_px of that frame's; it counts once, when it has lasted still_s. A frame without the board ends it."""

    def __init__(self, still_px=2.0, still_s=1.0):
        self.still_px, self.still_s = still_px, still_s
        self.start, self.counted = None, False

    def update(self, t, centre):
        """-> True when this frame completes a hold"""
        if centre is None:
            self.start = None
            return False
        c = np.asarray(centre, float)
        if self.start is None or np.hypot(*(c - self.start[1])) > self.still_px:
            self.start, self.counted = (t, c), False
            return False
        if not self.counted and t - self.start[0] >= self.still_s:
            self.counted = True
            return True
        return False


def observe(det, board, f, pp, sxy=(1.0, 1.0), min_tags=6):
    """detected corners {(tag, k): (u, v)} of a frame scaled by sxy to the rig resolution -> (board centre px, depth z
    m) or None: centre = the corners' mean, z = the depth of the centre of the corners seen, by PnP with the nominal
    equidistant camera (focal f, principal point pp; a ~1 % focal error is ~1 % of distance)"""
    import cv2
    if len({t for t, _ in det}) < min_tags:
        return None
    keys = list(det)
    uv = np.array([det[k] for k in keys], float) * np.asarray(sxy, float)
    obj = np.array([board.obj_points(t)[k] for t, k in keys], float)
    ray = lensmodel.unproject("KB4", [f, f, pp[0], pp[1], 0.0, 0.0, 0.0, 0.0], uv)
    front = ray[:, 2] > 0.1
    if front.sum() < 12:
        return None
    und = np.ascontiguousarray(ray[front, :2] / ray[front, 2:3]).reshape(-1, 1, 2)
    ok, rv, tv = cv2.solvePnP(np.ascontiguousarray(obj[front]).reshape(-1, 1, 3), und, np.eye(3), None)
    if not ok:
        return None
    z = float((cv2.Rodrigues(rv)[0] @ obj.mean(0) + tv.ravel())[2])
    return uv.mean(0), z


def add_hold(counts, centre, z, pp):
    """counts[cell, bin] += 1 for a hold at centre (px) and depth z (m); -> (cell, bin) or None outside every bin"""
    b = int(lenscells.dist_bin(z)[0])
    if b < 0:
        return None
    c = lenscells.cell_of(centre[0] - pp[0], centre[1] - pp[1])[0]
    counts[lenscells.CELLS.index(c), b] += 1
    return c, b


def text_map(counts, target, status=""):
    """the 3 x 3 map: holds per distance bin in each cell, 'ok' or how many more holds it needs"""
    bins = " / ".join(f"{lo:g}-{hi:g}" for lo, hi in lenscells.DIST_BINS)
    lines = [f"coverage {time.strftime('%H:%M:%S')}  holds per cell at {bins} m; a cell is done at {target} holds in "
             f"one distance  {status}"]
    for row in GRID:
        cells = []
        for c in row:
            n = counts[lenscells.CELLS.index(c)]
            need = max(0, target - int(n.max()))
            cells.append(f"{c:>12} {'/'.join(str(int(x)) for x in n):>8} {'ok' if not need else f'need {need}':>7}")
        lines.append("  ".join(cells))
    short = [c for c in lenscells.CELLS if counts[lenscells.CELLS.index(c)].max() < target]
    lines.append("still short: " + (", ".join(short) if short else "none: this sweep is complete"))
    return "\n".join(lines)


def heat_png(counts, target, pp, resolution, path, scale=4):
    """an image of the camera's frame at 1/scale, each pixel coloured by its cell's best distance bin against the
    target (dark = nothing yet), with the holds per bin written in each cell"""
    import cv2
    W, H = resolution
    u, v = np.meshgrid(np.arange(0, W, scale) + scale / 2, np.arange(0, H, scale) + scale / 2)
    lab = lenscells.cell_of((u - pp[0]).ravel(), (v - pp[1]).ravel()).reshape(u.shape)
    frac = np.zeros(u.shape)
    for i, c in enumerate(lenscells.CELLS):
        frac[lab == c] = min(counts[i].max() / target, 1.0)
    img = cv2.applyColorMap((frac * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    for i, c in enumerate(lenscells.CELLS):
        m = lab == c
        if m.any():
            y, x = np.argwhere(m).mean(0).astype(int)
            txt = f"{c} {'/'.join(str(int(n)) for n in counts[i])}"
            cv2.putText(img, txt, (max(0, x - 60), y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(path), img)


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--url", default="rtsp://127.0.0.1:8554/left")
    ap.add_argument("--still-px", type=float, default=2.0)
    ap.add_argument("--still-s", type=float, default=1.0)
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("--target", type=int, default=None,
                    help="holds per cell to call it done (default: the rig's lens_min_holds + 4: step 66 counts "
                         "holds seen by BOTH cameras, placed around the FITTED principal point, so the live "
                         "count needs a margin over the 8 it needs)")
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    rig = rigcfg.load(a.rig)
    W, H = rig.resolution
    f, pp = rig.nominal_focal_px, ((W - 1) / 2.0, (H - 1) / 2.0)
    target = a.target if a.target is not None else int(rig.lens_gates()["lens_min_holds"]) + 4
    a.out_dir.mkdir(parents=True, exist_ok=True)
    counts = np.zeros((len(lenscells.CELLS), len(lenscells.DIST_BINS)), int)
    track, holds = StillTracker(a.still_px, a.still_s), []
    cap = cv2.VideoCapture(a.url)
    frames = boards = 0
    last = time.monotonic()

    def save():
        print(text_map(counts, target, f"({frames} frames, {boards} with the board)"), flush=True)
        heat_png(counts, target, pp, rig.resolution, a.out_dir / "coverage.png")
        (a.out_dir / "coverage.json").write_text(json.dumps({"cells": list(lenscells.CELLS),
                                                             "dist_bins_m": lenscells.DIST_BINS,
                                                             "counts": counts.tolist(), "holds": holds}))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"no frame from {a.url}; retrying in 1 s", flush=True)
                time.sleep(1.0)
                cap = cv2.VideoCapture(a.url)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            obs = observe(rig.board.detect(gray), rig.board, f, pp, (W / gray.shape[1], H / gray.shape[0]))
            now = time.monotonic()
            frames += 1
            boards += obs is not None
            if track.update(now, None if obs is None else obs[0]):
                where = add_hold(counts, obs[0], obs[1], pp)
                holds.append({"t": time.time(), "u": float(obs[0][0]), "v": float(obs[0][1]), "z": obs[1],
                              "cell": where[0] if where else None})
            if now - last >= a.every:
                save()
                last = now
    except KeyboardInterrupt:
        pass
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
