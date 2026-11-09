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
MOVING-BOARD MODE (--max-blur, 4 Nov 2026)
A handheld board is rarely still: phantom09's four takes (4 Nov, Margaret's terminus session) held 45 still pairs
in 3,700. With hardware-synced cameras (left/right stamps within 0.016 ms there) a moving board is caught at the
same instant in both images, so the only cost of motion is blur within the exposure. --max-blur keeps a pair when
the board's speed to BOTH neighbouring frames (median displacement of the common corners / frame gap) times
--exposure-ms is <= max_blur px, then spaces the kept pairs >= --gap-s apart; each kept pair is its own hold. It
refuses a recording whose left/right stamps differ by more than --max-sync-ms (median): without a hardware sync a
moving board is seen at two places. Same npz layout; detection runs on --procs processes.
phantom09: 704 pairs at <= 0.5 px blur (2 ms exposure) gave a KB4T fit that validated at -0.19 % depth error on
the held-out far half, against +0.81 % for the kalibr equidistant fit of the same frames.

Usage (Thor, calib-py:3, PYTHONPATH = this folder + the repo's interface/):
    python 15_dump_corners.py <session.mcap> <out.npz> [--still-px 1.0] [--per-hold 3] [--rig PATH]
    python 15_dump_corners.py <session.mcap> <out.npz> --max-blur 0.5 --exposure-ms 2 [--gap-s 0.25] [--procs 8]
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


def steady_pairs(t_ns, dets, max_blur_px, exposure_ms, max_gap_ms=150.0):
    """indices i whose board moved slowly enough to both neighbours: speed (median displacement of the corners
    common to both frames, px, over the frame gap) x exposure <= max_blur_px. dets[i] is {(tag, k): (u, v)} or None
    (no board); a gap > max_gap_ms (dropped frames) leaves the speed unknown, so the pair is not kept"""
    def speed(a, b, dt_ms):
        k = set(a) & set(b)
        if len(k) < 8 or not 0 < dt_ms <= max_gap_ms:
            return np.inf
        return float(np.median([np.hypot(a[x][0] - b[x][0], a[x][1] - b[x][1]) for x in k])) / dt_ms
    out = []
    for i in range(1, len(dets) - 1):
        if dets[i] is None or dets[i - 1] is None or dets[i + 1] is None:
            continue
        v = max(speed(dets[i], dets[i - 1], (t_ns[i] - t_ns[i - 1]) / 1e6),
                speed(dets[i], dets[i + 1], (t_ns[i + 1] - t_ns[i]) / 1e6))
        if v * exposure_ms <= max_blur_px:
            out.append(i)
    return out


def spaced(idx, t_ns, gap_s):
    """the subset of idx (ascending) whose stamps are >= gap_s apart, greedily from the first"""
    out, last = [], None
    for i in idx:
        if last is None or t_ns[i] - last >= gap_s * 1e9:
            out.append(i)
            last = t_ns[i]
    return out


_W = {}   # the worker processes' view (fork): path, pairs, board, min_tags


def _det(job):
    i, both = job
    p = _W["pairs"][i]
    with open(_W["path"], "rb") as fh:
        d = [_W["board"].detect(_gray(fh, p[2], p[3]))]
        if both:
            d.append(_W["board"].detect(_gray(fh, p[4], p[5])))
    if not both:
        return d[0] if len({t for t, _ in d[0]}) >= _W["min_tags"] else None
    return d


def dump_moving(path, board, out, max_blur_px, exposure_ms, gap_s=0.25, min_tags=6, topics=TOPICS, procs=8,
                max_sync_ms=0.5):
    """moving-board mode (module docstring): -> (pairs kept, pairs read, pairs with the board, steady pairs)"""
    import multiprocessing as mp
    pairs = pair_frames(path, topics)
    if not pairs:
        return 0, 0, 0, 0
    sync = float(np.median([abs(p[1] - p[0]) for p in pairs])) / 1e6
    if sync > max_sync_ms:
        sys.exit(f"ERROR: {path}: left/right stamps differ by {sync:.3f} ms (median) > --max-sync-ms {max_sync_ms}: "
                 "the moving-board mode needs hardware-synced cameras; record still holds and use --still-px")
    _W.update(path=str(path), pairs=pairs, board=board, min_tags=min_tags)
    t = np.array([p[0] for p in pairs], np.int64)
    with mp.get_context("fork").Pool(procs) as pool:
        left = pool.map(_det, [(i, False) for i in range(len(pairs))], chunksize=8)
        steady = steady_pairs(t, left, max_blur_px, exposure_ms)
        sel = spaced(steady, t, gap_s)
        both = pool.map(_det, [(i, True) for i in sel], chunksize=4)
    t0s, t1s, rows = [], [], []
    for i, dets in zip(sel, both):
        if min(len({x for x, _ in d}) for d in dets) < min_tags:
            continue
        j = len(t0s)
        t0s.append(pairs[i][0])
        t1s.append(pairs[i][1])
        for cam, d in enumerate(dets):
            for (tag, k), (u, v) in d.items():
                rows.append((j, cam, tag, k, u, v))
    r = np.array(rows, float).reshape(-1, 6)
    np.savez_compressed(out, img_hold=np.arange(len(t0s), dtype=np.int64), img_t0=np.array(t0s, np.int64),
                        img_t1=np.array(t1s, np.int64), det_img=r[:, 0].astype(np.int64),
                        det_cam=r[:, 1].astype(np.int64), det_tag=r[:, 2].astype(np.int64),
                        det_k=r[:, 3].astype(np.int64), det_u=r[:, 4], det_v=r[:, 5])
    return len(t0s), len(pairs), sum(d is not None for d in left), len(steady)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("mcap", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--still-px", type=float, default=1.0)
    ap.add_argument("--per-hold", type=int, default=3)
    ap.add_argument("--max-blur", type=float, help="moving-board mode: keep pairs with <= this many px of blur")
    ap.add_argument("--exposure-ms", type=float, help="moving-board mode: the cameras' exposure time (required)")
    ap.add_argument("--gap-s", type=float, default=0.25, help="moving-board mode: min time between kept pairs")
    ap.add_argument("--max-sync-ms", type=float, default=0.5, help="moving-board mode: max median left/right offset")
    ap.add_argument("--procs", type=int, default=8, help="moving-board mode: detection processes")
    import rigcfg
    rigcfg.add_arg(ap)
    a = ap.parse_args(argv)
    rig = rigcfg.load(a.rig)
    # camera 0 = kalibr cam0. The record app names a camera's channels "/cam" + its --cameras index, which is the
    # DMA queue number (okvis2x_app_record.cpp), so the channel is /cam<dma_queue>/image; record_dir is an export
    # directory's name and need not match it
    topics = tuple(f"/cam{int(c['dma_queue'])}/image" for c in rig.kalibr)
    if a.max_blur is not None:
        if a.exposure_ms is None:
            sys.exit("ERROR: --max-blur needs --exposure-ms (blur = board speed x exposure)")
        n, total, seen, steady = dump_moving(a.mcap, rig.board, a.out, a.max_blur, a.exposure_ms, a.gap_s,
                                             topics=topics, procs=a.procs, max_sync_ms=a.max_sync_ms)
        print(f"{a.mcap}: {total} pairs, {seen} with the board, {steady} at <= {a.max_blur} px blur, "
              f"{n} kept >= {a.gap_s} s apart -> {a.out}")
        return
    n = dump(a.mcap, rig.board, a.out, a.still_px, a.per_hold, topics=topics)
    print(f"{a.mcap}: {n} still stereo pairs -> {a.out}")


if __name__ == "__main__":
    main()
