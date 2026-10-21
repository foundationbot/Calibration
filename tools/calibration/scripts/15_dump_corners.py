#!/usr/bin/env python3
"""Step 15 — stereo corners straight from a record-app session.mcap, still pairs only.

WHY THIS EXISTS
---------------
The lens analysis (lensfit.Data) needs, per stereo pair, every detected corner of both images. The export
route (mcap_to_euroc.py, then a dump) writes every PNG a second time: ~20 GB per 8-minute pair session, on a
robot disk that is 79 % full. This reads the MCAP through interface/mcap_reader.py (mmap, linear scan,
truncation-tolerant), decodes only the frames it uses, and keeps only STILL pairs: the board centre moved
less than still_px over the hold, so motion blur and the residual left/right exposure offset cost nothing.

Output: an npz in dump_corners.py's layout (img_hold, img_t0, img_t1, det_img, det_cam, det_tag, det_k, det_u,
det_v; camera k = the rig file's kalibr cam k, read from the record app's channel /cam<its dma_queue>/image:
cam 0 = left = /cam1, cam 1 = right = /cam2 on phantom12).
Usage (Thor, calib-py:3, PYTHONPATH = this folder + the repo's interface/):
    python 15_dump_corners.py <session.mcap> <out.npz> [--still-px 1.0] [--per-hold 3] [--rig PATH]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

TOPICS = ("/cam1/image", "/cam2/image")      # kalibr cam0, cam1 on phantom12 and mk13; main() reads the rig file


def pair_frames(path, topics=TOPICS, max_dt_ns=2_000_000):
    """[(t_left, t_right, off_left, len_left, off_right, len_right)] for left frames with a right frame within
    max_dt_ns; payload offsets so images are decoded only when used"""
    import mcap_reader
    idx = {topics[0]: [], topics[1]: []}

    def on_msg(ch, log_t, pub_t, data, off):
        if ch.topic in idx:
            idx[ch.topic].append((pub_t, off, len(data)))

    if mcap_reader.read_session(str(path), on_message=on_msg, offsets=True).truncated:
        print(f"{path}: truncated (recorder killed hard); frames up to the cut were read", file=sys.stderr)
    left = sorted(idx[topics[0]])
    right = sorted(idx[topics[1]])
    tr = np.array([r[0] for r in right], np.int64)
    out = []
    for t0, o0, n0 in left:
        if not len(tr):
            break
        k = int(np.clip(np.searchsorted(tr, t0), 1, len(tr) - 1)) if len(tr) > 1 else 0
        if len(tr) > 1 and abs(tr[k - 1] - t0) < abs(tr[k] - t0):
            k -= 1
        if abs(int(tr[k]) - t0) <= max_dt_ns:
            out.append((t0, int(tr[k]), o0, n0, right[k][1], right[k][2]))
    return out


def _gray(fh, off, n):
    """the frame as the export route reads it (cv2.imread(png, IMREAD_GRAYSCALE)): the record app writes
    3-channel PNGs, and cvtColor(BGR2GRAY) rounds differently on ~half the pixels (P1, 19 Oct), which moved
    30 of 17296 corners (0.17 %) by 0.3-5.4 px"""
    import cv2
    fh.seek(off)
    return cv2.imdecode(np.frombuffer(fh.read(n), np.uint8), cv2.IMREAD_GRAYSCALE)


def dump(path, board, out, still_px=1.0, per_hold=3, min_tags=6, topics=TOPICS):
    """detect the board in the left image of every pair; a hold = consecutive pairs whose left board centre
    stays within still_px of the hold's first; keep up to per_hold pairs from the middle of each hold of
    >= 3 pairs, with both images' corners. -> number of pairs kept. A hold here is a still stretch, not a
    neck hold: a 3 s neck hold can split in two when the head settles, or a tag at the image edge comes and
    goes, by more than still_px (P1: 87 still stretches in 49 neck holds), so cluster by time (img_t0) or by
    the neck's holds for error bars."""
    pairs = pair_frames(path, topics)
    keep_rows, holds, cur = [], [], []
    with open(path, "rb") as fh:
        cents = []
        for p in pairs:
            det = board.detect(_gray(fh, p[2], p[3]))
            ok = len({t for t, _ in det}) >= min_tags
            cents.append(np.mean(np.array(list(det.values())), axis=0) if ok else None)
        for i, c in enumerate(cents):
            if c is not None and cur and cents[cur[0]] is not None and np.hypot(*(c - cents[cur[0]])) <= still_px:
                cur.append(i)
                continue
            if len(cur) >= 3:
                holds.append(cur)
            cur = [i] if c is not None else []
        if len(cur) >= 3:
            holds.append(cur)
        img_hold, t0s, t1s, rows = [], [], [], []
        for h_i, h in enumerate(holds):
            mid = len(h) // 2
            for i in h[max(0, mid - per_hold // 2): max(0, mid - per_hold // 2) + per_hold]:
                p = pairs[i]
                dets = [board.detect(_gray(fh, p[2], p[3])), board.detect(_gray(fh, p[4], p[5]))]
                if min(len({t for t, _ in d}) for d in dets) < min_tags:
                    continue
                j = len(img_hold)
                img_hold.append(h_i)
                t0s.append(p[0])
                t1s.append(p[1])
                for cam, d in enumerate(dets):
                    for (tag, k), (u, v) in d.items():
                        rows.append((j, cam, tag, k, u, v))
    r = np.array(rows, float).reshape(-1, 6)
    np.savez_compressed(out, img_hold=np.array(img_hold, np.int64), img_t0=np.array(t0s, np.int64),
                        img_t1=np.array(t1s, np.int64), det_img=r[:, 0].astype(np.int64),
                        det_cam=r[:, 1].astype(np.int64), det_tag=r[:, 2].astype(np.int64),
                        det_k=r[:, 3].astype(np.int64), det_u=r[:, 4], det_v=r[:, 5])
    return len(img_hold)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("mcap", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--still-px", type=float, default=1.0)
    ap.add_argument("--per-hold", type=int, default=3)
    import rigcfg
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    rig = rigcfg.load(a.rig)
    # camera 0 = kalibr cam0. The record app names a camera's channels "/cam" + its --cameras index, which is the
    # DMA queue number (okvis2x_app_record.cpp), so the channel is /cam<dma_queue>/image; record_dir is an export
    # directory's name and need not match it
    topics = tuple(f"/cam{int(c['dma_queue'])}/image" for c in rig.kalibr)
    n = dump(a.mcap, rig.board, a.out, a.still_px, a.per_hold, topics=topics)
    print(f"{a.mcap}: {n} still stereo pairs -> {a.out}")


if __name__ == "__main__":
    main()
