"""The collection protocol, measured on stored raw neck-grid takes (design plan 13.1): which of the 49 grid poses see
the board at each default distance and what they add, how many still frames per pose a fit needs, how far the board
can stand, and how many poses each coverage target costs.

Raw detection: every frame of every pose file (step 08's pose_NNN.npz: cmd / neck deg, tl / tr ns, L / R frames) is
detected in both cameras (rigcfg board.detect) and kept as PoseFrames. Step 15b's still-pair selection is reproduced
from those detections (still_frames, per_hold_slice, corner_dict: the same rule, the same output), so per_hold
variants and noise need one detection pass. corner_dict(detect_take(take), per_hold=3) equals 15b's file of the take.

Definitions:
  detected pose   >= 1 still pair (15b: >= MIN_TAGS tags in the left frame, left corners within STILL_PX of the pose's
                  first such frame, >= MIN_TAGS tags in the right frame)
  corner noise    pooled per-axis standard deviation (px) of each corner's position across the still frames of a pose
                  (corners seen in >= 2 of them); remove_shift: after subtracting each frame's median displacement
                  (the head still settling moves the whole board)
  tag size        mean side length (px) of the whole tags of a frame
  band            the nominal default distance nearest the measured one (edges at the midpoints of NOMINAL)
  swept poses     a target is met after the poses swept up to the hold that met it for the last time, counting every
                  pose of the earlier takes (detected or not) in capture order

Run as a script (python -m calibtool.protocol {detect, variant, analyse} ...; see main) to produce the numbers of
results/2026-11-calibtool-M1-protocol.md."""
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibtool import paths

MIN_TAGS = 6
STILL_PX = 1.0

# ---------------------------------------------------------------- the collection protocol's defaults (for M2)
# Measured in M1 C2 (results/2026-11-calibtool-M1-protocol.md) and decided with the user on 9 Nov 2026 (M1 C3).
# stereo pairs kept per still pose (15b --per-hold): 2 is the smallest with no measurable loss (results section 2:
# per_hold 1/2/3/5 on p10 6 Nov: 9 of 9 entries, median -0.05 %, rows 0.280 px); the stored sessions have 3
# (coverage.STORED_PER_HOLD). Captured from CAPTURE_FRAMES frames: 15b's still check compares later frames with the
# first detected one, so a single frame is never checked, and 15b keeps the middle PER_HOLD
PER_HOLD = 2
CAPTURE_FRAMES = 3
# the far placement (results section 3): the detector loses tags past ~1.6 m (8-14 of 35 at 1.90-1.95 m, none at
# ~2.5 m); ask for 1.8 m measured, accept 1.7-2.0 m (the FIT far band starts at 1.7 m), refuse beyond 2.1 m
FAR_ASK_M = 1.8
FAR_ACCEPT_M = (1.7, 2.0)
FAR_REFUSE_BEYOND_M = 2.1
# the neck hold ends once the frames are captured (M2 implements it): today's step 08 waits out a fixed 2.0 s hold,
# pose period POSE_PERIOD_S measured in all 24 raw takes; ending the hold at capture gives about 2.07 s at 3 frames
# (computed from the script's waits and the frame spacing, not measured)
HOLD_ENDS_AT_CAPTURE = True
POSE_PERIOD_S = 3.13
# pose lists, as indices of the 49-pose grid (pitch-major: planner.grid(), grid_index): all 49 up to 1.6 m; the far
# placement drops 8 and the tilted one 5 (results section 1, protocol.propose: detected in < half the takes and no
# edge-strip corner), which loses no edge-strip corner
POSES_UPRIGHT = tuple(range(49))
POSES_FAR = tuple(i for i in range(49) if i not in (0, 1, 2, 4, 7, 27, 33, 34))
POSES_TILTED = tuple(i for i in range(49) if i not in (0, 1, 2, 5, 6))


def poses_for(distance_m, tilted=False) -> tuple:
    """the pose list of a placement: the tilted one's, the far one's (from FAR_ACCEPT_M's lower end), else all 49"""
    if tilted:
        return POSES_TILTED
    return POSES_FAR if distance_m >= FAR_ACCEPT_M[0] else POSES_UPRIGHT
SEC_PER_POSE = 3.1                       # measured: neckgrid logs "(3.1 s)" per pose, ramp 1.0 + hold 2.0 s
NOMINAL = (0.5, 0.75, 1.1, 1.6, 2.0, 3.0)
GRID_PITCH = tuple(float(v) for v in np.arange(-24, 25, 8))          # = planner.GRID_PITCH
GRID_YAW = tuple(float(v) for v in np.linspace(-34, 34, 7))          # = planner.GRID_YAW


# ---------------------------------------------------------------- grid, bands, bookkeeping


def grid_index(pitch_deg, yaw_deg) -> int:
    """the nearest pose of the 49-pose grid (pitch-major, planner.grid() order)"""
    ip = int(np.argmin(np.abs(np.asarray(GRID_PITCH) - float(pitch_deg))))
    iy = int(np.argmin(np.abs(np.asarray(GRID_YAW) - float(yaw_deg))))
    return ip * len(GRID_YAW) + iy


_POSE_RE = re.compile(r"pose (\d+)/(\d+) cmd ([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)")


def pose_cmds_from_log(text):
    """a neck-grid log (08 / neckgrid*.py stdout) -> ({pose index from 0: (pitch, yaw) deg commanded}, poses planned)"""
    cmds, total = {}, 0
    for m in _POSE_RE.finditer(text):
        cmds[int(m.group(1)) - 1] = (float(m.group(3)), float(m.group(4)))
        total = int(m.group(2))
    return cmds, total


def band_of(z):
    """per distance (m): the nearest nominal distance of NOMINAL (nan stays nan)"""
    z = np.atleast_1d(np.asarray(z, float))
    nom = np.asarray(NOMINAL)
    edges = (nom[1:] + nom[:-1]) / 2
    out = nom[np.digitize(np.nan_to_num(z, nan=0.0), edges)].astype(float)
    out[np.isnan(z)] = np.nan
    return out


def swept(order, n_poses, take, pose) -> int:
    """poses swept up to and including `pose` of `take` (0-based), every pose of the takes before it in `order`"""
    return int(sum(n_poses[t] for t in order[:order.index(take)]) + pose + 1)


def met_index(flags):
    """the index of the hold after which a target was met for the last time (it stays met to the end); None if unmet
    at the end. flags[i]: met after hold i"""
    flags = list(flags)
    if not flags or not flags[-1]:
        return None
    i = len(flags) - 1
    while i > 0 and flags[i - 1]:
        i -= 1
    return i


def met_at(flags, swept_counts):
    """the swept count at which a target was met for the last time; None if unmet at the end"""
    i = met_index(flags)
    return None if i is None else swept_counts[i]


CENTRE_GRID = tuple(i * 7 + j for i in (2, 3, 4) for j in (2, 3, 4))    # pitch -8..8 x yaw -11..11: board in view


def centre_tags(rows) -> float:
    """median left-camera tag count over a take's central poses (the board is in view there at any distance >= 0.6
    m): how much of the board the detector finds, apart from the field of view"""
    t = [r["tags_l"] for r in rows if r.get("grid") in CENTRE_GRID and "tags_l" in r]
    return float(np.median(t)) if t else float("nan")


def session_minutes(poses_per_placement, sec_per_pose=SEC_PER_POSE, placement_s=60.0) -> float:
    """the session's grid time + one placement time per placement, minutes"""
    return (sum(poses_per_placement) * sec_per_pose + len(poses_per_placement) * placement_s) / 60.0


# ---------------------------------------------------------------- raw detection and 15b's selection


@dataclass
class PoseFrames:
    """one pose file: commanded / measured neck (pitch, yaw) deg, left / right stamps, per frame (left, right)
    detections {(tag, k): (u, v)}"""
    cmd: np.ndarray
    neck: np.ndarray
    tl: np.ndarray
    tr: np.ndarray
    dets: list


_W = {}     # fork-pool workers' view: the board


def _detect_file(f):
    board = _W["board"]
    z = np.load(f)
    dets = [(dict(board.detect(z["L"][i])), dict(board.detect(z["R"][i]))) for i in range(len(z["tl"]))]
    return PoseFrames(np.asarray(z["cmd"], float), np.asarray(z["neck"], float), np.asarray(z["tl"], np.int64),
                      np.asarray(z["tr"], np.int64), dets)


def detect_take(take_dir, board, procs=1) -> list:
    """every frame of every pose file of a take, both cameras -> [PoseFrames] in pose-file order (fork pool)"""
    files = sorted(glob.glob(f"{take_dir}/pose_*.npz"))
    _W["board"] = board
    if procs > 1 and len(files) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(procs, len(files))) as pool:
            return pool.map(_detect_file, files)
    return [_detect_file(f) for f in files]


def _ntags(d) -> int:
    return len({t for t, _ in d})


def still_frames(pf: PoseFrames, still_px=STILL_PX, min_tags=MIN_TAGS) -> list:
    """step 15b's kept frames of a pose (before the per_hold slice)"""
    kept, d0 = [], None
    for i, (dl, dr) in enumerate(pf.dets):
        if _ntags(dl) < min_tags:
            continue
        if d0 is None:
            d0 = dl
        common = set(d0) & set(dl)
        if not common or np.median([np.hypot(*(np.subtract(dl[k], d0[k]))) for k in common]) > still_px:
            continue
        if _ntags(dr) < min_tags:
            continue
        kept.append(i)
    return kept


def pose_detected(pf: PoseFrames, still_px=STILL_PX, min_tags=MIN_TAGS) -> bool:
    return bool(still_frames(pf, still_px, min_tags))


def per_hold_slice(kept, per_hold) -> list:
    """15b's per_hold pairs from the middle of the kept frames"""
    s = max(0, len(kept) // 2 - per_hold // 2)
    return list(kept[s:s + per_hold])


def corner_dict(frames, per_hold=3, still_px=STILL_PX, min_tags=MIN_TAGS) -> dict:
    """step 15b's corner file of a take from its detections (the same keys, order and types)"""
    img_hold, t0s, t1s, rows, yaws, pitches, ays, aps, prev = [], [], [], [], [], [], [], [], None
    for h, pf in enumerate(frames):
        cmd = pf.cmd
        ap_, ay_ = (0, 0) if prev is None else (int(np.sign(cmd[0] - prev[0])), int(np.sign(cmd[1] - prev[1])))
        prev = cmd
        pit, yaw = np.radians(pf.neck)
        for i in per_hold_slice(still_frames(pf, still_px, min_tags), per_hold):
            j = len(img_hold)
            img_hold.append(h)
            t0s.append(int(pf.tl[i]))
            t1s.append(int(pf.tr[i]))
            yaws.append(yaw)
            pitches.append(pit)
            ays.append(ay_)
            aps.append(ap_)
            for cam, det in enumerate(pf.dets[i]):
                for (tag, k), (u, v) in det.items():
                    rows.append((j, cam, tag, k, u, v))
    r = np.array(rows, float).reshape(-1, 6)
    return dict(img_hold=np.array(img_hold, np.int64), img_t0=np.array(t0s, np.int64), img_t1=np.array(t1s, np.int64),
                det_img=r[:, 0].astype(np.int64), det_cam=r[:, 1].astype(np.int64), det_tag=r[:, 2].astype(np.int64),
                det_k=r[:, 3].astype(np.int64), det_u=r[:, 4], det_v=r[:, 5], img_yaw=np.array(yaws),
                img_pitch=np.array(pitches), img_ay=np.array(ays, np.int64), img_ap=np.array(aps, np.int64))


def save_frames(frames, path):
    """PoseFrames list -> one npz (flat tables)"""
    fp, ft, fr, rows = [], [], [], []
    for p, pf in enumerate(frames):
        for i, dd in enumerate(pf.dets):
            j = len(fp)
            fp.append(p)
            ft.append(int(pf.tl[i]))
            fr.append(int(pf.tr[i]))
            for cam, det in enumerate(dd):
                for (tag, k), (u, v) in det.items():
                    rows.append((j, cam, tag, k, u, v))
    r = np.array(rows, float).reshape(-1, 6)
    np.savez_compressed(path, p_cmd=np.array([pf.cmd for pf in frames], float).reshape(-1, 2),
                        p_neck=np.array([pf.neck for pf in frames], float).reshape(-1, 2),
                        f_pose=np.array(fp, np.int64), f_tl=np.array(ft, np.int64), f_tr=np.array(fr, np.int64),
                        d_frame=r[:, 0].astype(np.int64), d_cam=r[:, 1].astype(np.int64),
                        d_tag=r[:, 2].astype(np.int64), d_k=r[:, 3].astype(np.int64), d_u=r[:, 4], d_v=r[:, 5])


def load_frames(path) -> list:
    z = np.load(path)
    nf = len(z["f_pose"])
    dets = [({}, {}) for _ in range(nf)]
    for j, c, t, k, u, v in zip(z["d_frame"], z["d_cam"], z["d_tag"], z["d_k"], z["d_u"], z["d_v"]):
        dets[int(j)][int(c)][(int(t), int(k))] = (float(u), float(v))
    out = []
    for p in range(len(z["p_cmd"])):
        idx = np.flatnonzero(z["f_pose"] == p)
        out.append(PoseFrames(z["p_cmd"][p], z["p_neck"][p], z["f_tl"][idx], z["f_tr"][idx], [dets[i] for i in idx]))
    return out


# ---------------------------------------------------------------- noise, size, timing, bearing


UNSTABLE_PX = 0.5            # a corner whose per-corner sd exceeds this is "unstable" (jumps between frames)


@dataclass
class Noise:
    median_sd: float            # median over corners of the per-corner sd (px; u and v pooled)
    p90_sd: float
    unstable: float             # share of corners with sd > UNSTABLE_PX
    pooled: float               # pooled sd over every corner (px): what a few jumping corners inflate
    n: int                      # corners seen in >= 2 frames


def corner_noise(dets, remove_shift=False) -> Noise:
    """repeatability of corner positions across frames (list of {(tag, k): (u, v)}), corners seen in >= 2 frames:
    per corner the sd of its position (u and v pooled), summarised; nan with none. remove_shift: each frame's median
    displacement subtracted first (the whole board moving while the head settles)"""
    cnt = {}
    for d in dets:
        for k in d:
            cnt[k] = cnt.get(k, 0) + 1
    keys = [k for k, n in cnt.items() if n >= 2]
    if not keys:
        return Noise(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    ki = {k: i for i, k in enumerate(keys)}
    P = np.full((len(dets), len(keys), 2), np.nan)
    for f, d in enumerate(dets):
        for k, uv in d.items():
            if k in ki:
                P[f, ki[k]] = uv
    shift = np.zeros((len(dets), 1, 2))
    if remove_shift:
        for _ in range(2):
            res = P - shift - np.nanmean(P - shift, axis=0)
            shift = shift + np.nan_to_num(np.nanmedian(res, axis=1, keepdims=True))
    res = P - shift - np.nanmean(P - shift, axis=0)
    dof = np.isfinite(P[..., 0]).sum(0) - 1
    sd = np.sqrt(np.nansum(res ** 2, axis=(0, 2)) / (2 * dof))
    return Noise(float(np.median(sd)), float(np.percentile(sd, 90)), float(np.mean(sd > UNSTABLE_PX)),
                 float(np.sqrt(np.nansum(res ** 2) / (2 * dof.sum()))), len(keys))


def tag_px(det) -> float:
    """mean side length (px) of the whole tags (all 4 corners) of one detection; nan without one"""
    tags = {}
    for (t, k), uv in det.items():
        tags.setdefault(t, {})[k] = uv
    sides = []
    for c in tags.values():
        if len(c) == 4:
            q = np.array([c[k] for k in range(4)], float)
            sides.append(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1).mean())
    return float(np.mean(sides)) if sides else float("nan")


def timing(tl_per_pose) -> dict:
    """from the left stamps (ns) of each pose: frame spacing (ms, median), capture window per pose (first to last
    frame, ms, median) and pose period (first frame to the next pose's first frame, s, median)"""
    within = np.concatenate([np.diff(np.asarray(t, np.int64)) for t in tl_per_pose if len(t) > 1])
    win = [int(t[-1] - t[0]) for t in tl_per_pose if len(t) > 1]
    firsts = np.array([int(t[0]) for t in tl_per_pose if len(t)], np.int64)
    return {"frame_ms": float(np.median(within) / 1e6), "window_ms": float(np.median(win) / 1e6),
            "period_s": float(np.median(np.diff(firsts)) / 1e9) if len(firsts) > 1 else float("nan")}


def bearing_deg(neck_yaw_rad, u_mean, focal_px, width) -> float:
    """the board's bearing from straight ahead (deg, + toward the robot's left), the median over poses of the neck
    yaw minus the board centre's angle right of the optical axis (equidistant lens, r = f theta; pitch ignored)"""
    y = np.degrees(np.asarray(neck_yaw_rad, float))
    a = np.degrees((np.asarray(u_mean, float) - width / 2.0) / focal_px)
    return float(np.median(y - a))


# ---------------------------------------------------------------- pose lists


def propose(detected, min_rate=0.5, adds_edge=None) -> list:
    """grid indices to sweep: detected in >= min_rate of the takes, or detected at least once and putting corners in
    an image edge strip (adds_edge, per grid pose); detected: (takes, 49) bool"""
    d = np.asarray(detected, bool)
    e = np.zeros(d.shape[1], bool) if adds_edge is None else np.asarray(adds_edge, bool)
    return [i for i in range(d.shape[1]) if d[:, i].mean() >= min_rate or (e[i] and d[:, i].any())]


def grid_text(values, title="pitch \\ yaw") -> str:
    """49 cells (planner.grid() order) as a markdown table: rows pitch -24..24, columns yaw -34..34"""
    head = "| " + title + " | " + " | ".join(f"{y:+.0f}" for y in GRID_YAW) + " |"
    sep = "|---" * (len(GRID_YAW) + 1) + "|"
    rows = [head, sep]
    for i, p in enumerate(GRID_PITCH):
        rows.append(f"| {p:+.0f} | " + " | ".join(str(values[i * 7 + j]) for j in range(7)) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------- per-pose table of a raw take


def pose_rows(frames, rig, take, cmds=None):
    """one dict per pose of a raw take: grid index, detected, tags (max over frames, left / right), distance (m, rough
    z of the middle still pair; of the frame with most left tags when none is still), tag size (px, camera 0),
    corner noise per camera (raw and shift removed), and the FIT regions / TEST cell of the middle still pair"""
    import lenscells
    from calibtool import coverage as cv
    d16 = paths.import_script("16_split_distance")
    W, H = rig.resolution
    out = []
    for p, pf in enumerate(frames):
        cmd = cmds[p] if cmds and p in cmds else tuple(pf.cmd)
        kept = still_frames(pf)
        best = int(np.argmax([_ntags(dl) for dl, _ in pf.dets])) if pf.dets else 0
        i = per_hold_slice(kept, 1)[0] if kept else best
        dl = pf.dets[i][0] if pf.dets else {}
        z = float("nan")
        if len(dl) >= 8:
            n0 = np.zeros(len(dl), np.int64)
            one = {"img_t0": np.zeros(1), "det_img": n0, "det_cam": n0,
                   "det_tag": np.array([t for t, _ in dl]), "det_k": np.array([k for _, k in dl]),
                   "det_u": np.array([u for u, _ in dl.values()]), "det_v": np.array([v for _, v in dl.values()])}
            z = float(d16.rough_z(one, rig.board, rig.nominal_focal_px, rig.resolution)[0])
        row = {"take": take, "pose": p, "grid": grid_index(*cmd), "cmd": tuple(float(c) for c in cmd),
               "neck": tuple(float(c) for c in pf.neck), "detected": bool(kept), "n_still": len(kept),
               "n_frames": len(pf.dets),
               "tags_l": max((_ntags(a) for a, _ in pf.dets), default=0),
               "tags_r": max((_ntags(b) for _, b in pf.dets), default=0), "z": z,
               "tag_px": tag_px(dl)}
        for cam in (0, 1):
            nz = corner_noise([pf.dets[k][cam] for k in kept], remove_shift=True)
            row[f"noise{cam}"], row[f"p90_{cam}"], row[f"unstable{cam}"] = nz.median_sd, nz.p90_sd, nz.unstable
            row[f"pooled{cam}"], row[f"noise_n{cam}"] = nz.pooled, nz.n
        if kept:
            for cam in (0, 1):
                d = pf.dets[i][cam]
                uv = np.array(list(d.values()), float).reshape(-1, 2)
                t = cv.corner_tally(uv[:, 0], uv[:, 1], (W, H))
                row[f"edges{cam}"] = dict(t.edges)
                row[f"boxes{cam}"] = dict(t.boxes)
                row[f"rim{cam}"] = t.rim
                row[f"n{cam}"] = t.n
            uv0 = np.array(list(pf.dets[i][0].values()), float)
            row["cell"] = str(lenscells.cell_of(uv0[:, 0].mean() - W / 2, uv0[:, 1].mean() - H / 2)[0])
            row["u_mean"] = float(uv0[:, 0].mean())
        out.append(row)
    return out


def take_kind(name) -> str:
    """a take's board placement by its name: "tilted" (leaning back, facing up), "floor" (on the floor), "upright" """
    n = str(name).lower()
    return "tilted" if "tilt" in n else "floor" if "floor" in n else "upright"


_ABBR = {"top-left": "tl", "top": "t", "top-right": "tr", "left": "l", "centre": "c", "right": "r",
         "bottom-left": "bl", "bottom": "b", "bottom-right": "br"}


def cell_abbrev(cell) -> str:
    return _ABBR.get(str(cell), "?")


def edge_letters(row) -> str:
    """the image edge strips (L, R, T, B) a pose put corners in, either camera"""
    out = ""
    for e, letter in (("left", "L"), ("right", "R"), ("top", "T"), ("bottom", "B")):
        if any(row.get(f"edges{c}", {}).get(e, 0) > 0 for c in (0, 1)):
            out += letter
    return out


def rows_from_corners(z, rig, take, cmds, n_poses):
    """pose_rows' rows (no noise, no tag counts of undetected poses) from a stored 15b corner file: a pose is detected
    when it has a pair; its distance is the median rough z of its pairs; tallies and cell from its middle pair"""
    import lenscells
    from calibtool import coverage as cv
    d16 = paths.import_script("16_split_distance")
    W, H = rig.resolution
    zz = d16.rough_z(z, rig.board, rig.nominal_focal_px, rig.resolution)
    hold = np.asarray(z["img_hold"])
    out = []
    for p in range(n_poses):
        idx = np.flatnonzero(hold == p)
        if p in cmds:
            cmd = cmds[p]
        elif len(idx):
            cmd = (np.degrees(z["img_pitch"][idx[0]]), np.degrees(z["img_yaw"][idx[0]]))
        else:
            cmd = (np.nan, np.nan)
        row = {"take": take, "pose": p, "grid": grid_index(*cmd) if np.isfinite(cmd).all() else -1,
               "cmd": tuple(float(c) for c in cmd), "detected": bool(len(idx)), "n_still": int(len(idx)),
               "z": float(np.nanmedian(zz[idx])) if len(idx) and np.isfinite(zz[idx]).any() else float("nan")}
        if len(idx):
            i = per_hold_slice(list(idx), 1)[0]
            d = {}
            for cam in (0, 1):
                m = (z["det_img"] == i) & (z["det_cam"] == cam)
                d[cam] = {(int(t), int(k)): (float(u), float(v)) for t, k, u, v in
                          zip(z["det_tag"][m], z["det_k"][m], z["det_u"][m], z["det_v"][m])}
                t = cv.corner_tally(z["det_u"][m], z["det_v"][m], (W, H))
                row[f"edges{cam}"], row[f"boxes{cam}"], row[f"rim{cam}"], row[f"n{cam}"] = (dict(t.edges),
                                                                                         dict(t.boxes), t.rim, t.n)
            row["tags_l"], row["tags_r"], row["tag_px"] = _ntags(d[0]), _ntags(d[1]), tag_px(d[0])
            m0 = (z["det_img"] == i) & (z["det_cam"] == 0)
            u, v = float(np.mean(z["det_u"][m0])), float(np.mean(z["det_v"][m0]))
            row["cell"] = str(lenscells.cell_of(u - W / 2, v - H / 2)[0])
            row["u_mean"] = u
            row["neck"] = (float(np.degrees(z["img_pitch"][i])), float(np.degrees(z["img_yaw"][i])))
        out.append(row)
    return out


def far_table(rows, edges=tuple(np.round(np.arange(0.25, 3.51, 0.25), 2))):
    """poses bucketed by distance [edges[i], edges[i+1]): poses, detected, median left tags, tag size (px), and over
    the detected poses the medians of camera 0's corner noise (median per-corner sd, px), unstable-corner share and
    pooled sd"""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = [r for r in rows if np.isfinite(r.get("z", np.nan)) and lo <= r["z"] < hi]
        det = [r for r in b if r["detected"]]
        med = (lambda xs: float(np.nanmedian(xs)) if len(xs) and np.isfinite(xs).any() else float("nan"))
        out.append({"lo": float(lo), "hi": float(hi), "n": len(b), "detected": len(det),
                    "tags_med": med(np.array([r.get("tags_l", np.nan) for r in b], float)),
                    "tag_px_med": med(np.array([r.get("tag_px", np.nan) for r in b], float)),
                    "noise_med": med(np.array([r.get("noise0", np.nan) for r in det], float)),
                    "unstable_med": med(np.array([r.get("unstable0", np.nan) for r in det], float)),
                    "pooled_med": med(np.array([r.get("pooled0", np.nan) for r in det], float))})
    return out


# ---------------------------------------------------------------- poses per coverage target


def target_keys(targets):
    """every target of coverage.Coverage.deficits as (kind, cam, region) keys (entries added from the data)"""
    from calibtool import coverage as cv
    keys = []
    for c in targets.fit_cams:
        keys += [("edge", c, r) for r in cv.EDGES] + [("box", c, r) for r in cv.BOXES]
        keys += [("third", c, r) for r in cv.CORNER_THIRDS] + [("rim", c, "rim")]
    keys += [("pairs", None, "pairs"), ("far", None, cv.FAR_FIT)]
    keys += [("side", None, s) for s in cv.EDGE_CELLS] + [("judged", None, "judged")]
    keys += [("far_band", None, b) for b in cv.FAR_BANDS]
    return keys


def target_costs(session, rig, z, assignment, n_poses, keep=None, targets=None, pp=None):
    """replay the session's holds in capture order through coverage's tallies (FIT on FIT holds, TEST per entry) and
    report, per target, the poses swept until it was met for the last time (None: unmet), the take that met it, and
    how many holds added to it (FIT targets: FIT holds with corners there; TEST targets: the TEST holds at the end;
    judged: entries judged at the end). n_poses: {take: poses swept in the take}; keep: {take: [pose indices
    swept]} (a reduced protocol: holds of other poses dropped, swept counts over the kept poses only; None: every
    pose). pp: the principal point of the TEST
    cells (default: the image centre, the nominal one the capture-time assignment uses)"""
    from calibtool import coverage as cv
    targets = targets or cv.Targets()
    W, H = rig.resolution
    pp = pp or (W / 2.0, H / 2.0)
    holds = list(session.holds())
    if keep is not None:
        holds = [h for h in holds if h.pose in set(keep.get(h.take, []))]
    order = []
    for h in holds:
        if h.take not in order:
            order.append(h.take)
    npos = {t: (len(keep[t]) if keep is not None else n_poses[t]) for t in order}
    pos = {t: ({p: i for i, p in enumerate(keep[t])} if keep is not None else None) for t in order}
    cells = cv.pair_cells(session, pp)
    import lenscells
    bins = lenscells.dist_bin(z)
    zz = np.asarray(z, float)
    fit = cv.fit_tally(session, np.zeros(session.n_pairs, bool), size=(W, H), z=z)
    entries, far_bands = {}, {b: 0 for b in cv.FAR_BANDS}
    keys = target_keys(targets)
    flags, swept_counts, adds, entry_keys = {k: [] for k in keys}, [], {k: 0 for k in keys}, set()
    states = []
    for h in holds:
        m = np.zeros(session.n_pairs, bool)
        m[h.pairs] = True
        if assignment.get(h.hold) == "FIT":
            add = cv.fit_tally(session, m, size=(W, H), z=z)
            fit = fit + add
            for k in keys:
                kind, c, r = k
                if kind in ("edge", "box", "third", "rim"):
                    t = add.cams[c]
                    v = {"edge": lambda: t.edges[r], "box": lambda: t.boxes[r],
                         "third": lambda: t.thirds[cv.CORNER_THIRDS[r]], "rim": lambda: t.rim}[kind]()
                    adds[k] += bool(v)
                elif kind == "pairs":
                    adds[k] += 1
                elif kind == "far":
                    adds[k] += bool(add.far_pairs)
        elif assignment.get(h.hold) == "TEST":
            for i, (lo, hi) in enumerate(lenscells.DIST_BINS):
                sel = m & (bins == i)
                for c in set(cells[sel]) - {None}:
                    entries[(c, (lo, hi))] = entries.get((c, (lo, hi)), 0) + 1
                    entry_keys.add((c, (lo, hi)))
            for b in cv.FAR_BANDS:
                if (m & (zz >= b[0]) & (zz < b[1])).any():
                    far_bands[b] += 1
        cov = cv.Coverage(fit, cv.TestTally(dict(entries), dict(far_bands)))
        short = {(d.kind, d.cam, d.region if not isinstance(d.region, list) else tuple(d.region))
                 for d in cov.deficits(targets)}
        states.append((short, dict(entries)))
        for k in keys:
            flags[k].append(k not in short)
        p = h.pose if keep is None else pos[h.take][h.pose]
        swept_counts.append(swept(order, npos, h.take, p))
    out = {}

    def res(f, n_add):
        i = met_index(f)
        return {"met_at": None if i is None else swept_counts[i], "met_in": None if i is None else holds[i].take,
                "holds_adding": n_add}
    for k in keys:
        kind, c, r = k
        if kind == "side":
            n_add = max([n for (cc, _), n in entries.items() if cc == r], default=0)
        elif kind == "judged":
            n_add = sum(n >= targets.test_holds for n in entries.values())
        elif kind == "far_band":
            n_add = far_bands[r]
        else:
            n_add = adds.get(k, 0)
        out[k] = res(flags[k], n_add)
    for e in sorted(entry_keys, key=str):
        out[("entry", None, e)] = res([st[1].get(e, 0) >= targets.test_holds for st in states], entries.get(e, 0))
    return {"targets": out, "swept_total": sum(npos.values()), "order": order, "n_poses": npos,
            "holds": len(holds), "fit": fit, "entries": entries, "far_bands": far_bands}


# ---------------------------------------------------------------- driver


def _rig(path):
    from calibtool import pipeline
    return pipeline.load_rig(path)


def cmd_detect(a):
    """detect every frame of each take -> <out>/<take>.frames.npz"""
    rig = _rig(a.rig)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in a.takes:
        f = out / f"{Path(t).name}.frames.npz"
        if f.is_file() and not a.force:
            print(f"{t}: {f} exists")
            continue
        frames = detect_take(t, rig.board, a.procs)
        save_frames(frames, f)
        print(f"{t}: {len(frames)} poses, {sum(pose_detected(p) for p in frames)} detected -> {f}")


def cmd_variant(a):
    """one per_hold variant: step 16 (holdout_every 2, max_fit_z 3) on the per-take corner files, step 45 (max_z 3),
    step 66 + summary on its own TEST and on a common TEST file -> <out>/variant.json"""
    from calibtool import pipeline
    from calibtool.session import Session
    rig = _rig(a.rig)
    out = Path(a.out)
    s = Session.from_files([(t, str(Path(a.corners) / f"{t}.npz")) for t in a.takes], name=out.name)
    sp = pipeline.split16(s, out, rig, max_fit_z=3.0, holdout_every=2)
    fit = pipeline.fit45(sp.fit_npz, rig, out, max_z=3.0)
    res = {"pairs": s.n_pairs, "fit_pairs": sp.fit_pairs, "test_pairs": sp.test_pairs, "fit": fit.as_dict()}
    tests = [("own", sp.test_npz)] + ([("common", Path(a.common_test))] if a.common_test else [])
    for name, tn in tests:
        v = pipeline.validate66(fit.json_path, tn, out / f"validate-{name}", rig)
        sm = pipeline.summary(fit.json_path, tn, rig)
        res[name] = {"validate": v.as_dict(), "summary": sm.as_dict(), "test_npz": str(tn)}
    (out / "variant.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("pairs", "fit_pairs", "test_pairs")}), fit.verdict)
    for name, _ in tests:
        r = res[name]
        print(name, r["validate"]["text"], "| median", round(r["summary"]["median_pct"], 3), "IQR",
              [round(x, 3) for x in r["summary"]["iqr_pct"]], "rows", round(r["summary"]["rows_px"], 3))


def _cmds_for(log_dir, take):
    for f in (Path(log_dir) / f"{take}.out", Path(log_dir) / take / "neckgrid.log"):
        if f.is_file():
            c, n = pose_cmds_from_log(f.read_text(errors="replace"))
            if c:
                return c, n
    return {}, 49


def robot_rows(robot, rig, det_dir, corner_dir, log_dir):
    """{take: rows} of one robot: from its raw detections where they exist, else from its stored corner files"""
    takes = {}
    for f in sorted(Path(det_dir).glob("*.frames.npz")):
        t = f.name[: -len(".frames.npz")]
        cmds, _ = _cmds_for(log_dir, t)
        takes[t] = ("raw", pose_rows(load_frames(f), rig, t, cmds))
    if corner_dir:
        for f in sorted(Path(corner_dir).glob("*.npz")):
            t = f.stem
            if t in takes:
                continue
            cmds, n = _cmds_for(log_dir, t)
            takes[t] = ("corners", rows_from_corners(dict(np.load(f)), rig, t, cmds, n))
    for t, (_, rows) in takes.items():
        for r in rows:
            r["robot"] = robot
    return takes


def take_summary(rows, rig):
    det = [r for r in rows if r["detected"]]
    z = np.array([r["z"] for r in det], float)
    zmed = float(np.nanmedian(z)) if len(z) and np.isfinite(z).any() else float("nan")
    b = bearing_deg([np.radians(r["neck"][1]) for r in det], [r["u_mean"] for r in det], rig.nominal_focal_px,
                    rig.resolution[0]) if det else float("nan")
    med = (lambda k: float(np.nanmedian([r.get(k, np.nan) for r in det])) if det else float("nan"))
    nf = [r.get("n_frames", np.nan) for r in det]
    still_all = float(np.mean([r["n_still"] == n for r, n in zip(det, nf)])) if det and np.isfinite(nf).all() \
        else float("nan")
    return {"poses": len(rows), "detected": len(det), "still_all": still_all, "centre_tags": centre_tags(rows),
            "z": zmed, "band": float(band_of(zmed)[0]), "bearing": b,
            "tag_px": med("tag_px"), "noise0": med("noise0"), "unstable0": med("unstable0"),
            "noise1": med("noise1"), "unstable1": med("unstable1")}


def _fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def cmd_analyse(a):
    """items 1, 3, 4 of the protocol report: markdown to stdout, everything to --json"""
    from calibtool import coverage as cv, sessions
    from calibtool.session import Session
    robots, allrows, summ, rigs = {}, [], {}, {}
    for spec in a.robot:
        name, rigp, det_dir, corner_dir, log_dir, sess = spec.split(":")
        rig = _rig(rigp)
        rigs[name] = (rig, sess, log_dir)
        robots[name] = robot_rows(name, rig, det_dir, corner_dir, log_dir)
        for t, (src, rows) in robots[name].items():
            summ[(name, t)] = dict(take_summary(rows, rig), source=src, kind=take_kind(t))
            allrows += rows
    P = print
    P("## Takes\n")
    P("| robot | take | source | kind | poses | detected | all frames still % | left tags at centre poses (of 35) | "
      "distance m (median) | band | bearing deg | tag px | noise L px / unstable % | noise R px / unstable % |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (r, t), s in summ.items():
        P(f"| {r} | {t} | {s['source']} | {s['kind']} | {s['poses']} | {s['detected']} | "
          f"{_fmt(100 * s['still_all'], 0)} | {_fmt(s['centre_tags'], 0)} | {_fmt(s['z'])} | "
          f"{_fmt(s['band'])} | {_fmt(s['bearing'], 1)} | {_fmt(s['tag_px'], 1)} | {_fmt(s['noise0'], 3)} / "
          f"{_fmt(100 * s['unstable0'], 1)} | {_fmt(s['noise1'], 3)} / {_fmt(100 * s['unstable1'], 1)} |")
    # ---- item 1: default placements per band
    default = {k: s for k, s in summ.items() if s["kind"] == "upright" and np.isfinite(s["bearing"])
               and abs(s["bearing"]) <= a.max_bearing and s["poses"] == 49}
    proposals, out_json = {}, {"takes": {f"{r}/{t}": s for (r, t), s in summ.items()}}
    groups = [(b, [k for k, s in default.items() if s["band"] == b]) for b in NOMINAL]
    tilt = [k for k, s in summ.items() if s["kind"] == "tilted" and s["poses"] == 49]
    groups.append(("tilted", tilt))
    P(f"\n## Pose usefulness per default distance (upright, |bearing| <= {a.max_bearing:g} deg, full 49-pose takes; "
      "tilted takes apart)\n")
    for b, keys in groups:
        if not keys:
            P(f"### {b}: no take\n")
            continue
        rows_by = [{r["grid"]: r for r in robots[rk][t][1]} for rk, t in keys]
        det = np.array([[g in rb and rb[g]["detected"] for g in range(49)] for rb in rows_by], bool)
        cells = []
        for g in range(49):
            d = [rb[g] for rb in rows_by if g in rb and rb[g]["detected"]]
            cs = sorted({cell_abbrev(r["cell"]) for r in d})
            el = "".join(sorted(set("".join(edge_letters(r) for r in d)), key="LRTB".index))
            cells.append(f"{len(d)}/{len(rows_by)} {'/'.join(cs)} {el}".strip())
        adds_edge = np.array([any(g in rb and rb[g]["detected"] and edge_letters(rb[g]) for rb in rows_by)
                              for g in range(49)])
        keep = propose(det, a.min_rate, adds_edge)
        proposals[str(b)] = keep
        tot = {k: 0 for k in ("edge", "box", "rim", "n")}
        kept = dict(tot)
        for rb in rows_by:
            for g, r in rb.items():
                if not r["detected"]:
                    continue
                v = {"edge": sum(sum(r[f"edges{c}"].values()) for c in (0, 1)),
                     "box": sum(sum(r[f"boxes{c}"].values()) for c in (0, 1)),
                     "rim": r["rim0"] + r["rim1"], "n": 1}
                for k in tot:
                    tot[k] += v[k]
                    kept[k] += v[k] if g in keep else 0
        cells_all = {(rr["cell"]) for rb in rows_by for rr in rb.values() if rr["detected"]}
        cells_kept = {(rr["cell"]) for rb in rows_by for g, rr in rb.items() if rr["detected"] and g in keep}
        names = ", ".join(f"{rk}/{t} ({_fmt(summ[(rk, t)]['z'])} m)" for rk, t in keys)
        P(f"### {b}{' m' if b != 'tilted' else ''}: {names}\n")
        P(grid_text(cells))
        P(f"\nCell = takes detecting the pose / takes; step-66 cells of its camera-0 board (nominal principal point); "
          f"edge strips (80 px) it puts corners in, either camera.\n")
        P(f"Proposed list (detected in >= {a.min_rate:.0%} of the takes, or at least once when it fills an edge "
          f"strip): {len(keep)} poses, dropped grid {[g for g in range(49) if g not in keep]}; keeps "
          f"{kept['n']}/{tot['n']} detected pose-takes, edge-strip corners {kept['edge']}/{tot['edge']}, corner-box "
          f"corners {kept['box']}/{tot['box']}, rim corners {kept['rim']}/{tot['rim']}, cells {len(cells_kept)}/"
          f"{len(cells_all)}; time {len(keep)} x {SEC_PER_POSE} s = {len(keep) * SEC_PER_POSE:.0f} s (all 49: "
          f"{49 * SEC_PER_POSE:.0f} s).\n")
    out_json["proposals"] = proposals
    # ---- item 3: far limit (raw takes only: noise needs the frames)
    raw = [r for (rk, t), s in summ.items() if s["source"] == "raw" for r in robots[rk][t][1]]
    P("\n## Detection and corner noise vs distance (raw takes, every pose; distance = rough z of the pose's middle "
      "still pair, or of its frame with most left tags)\n")
    P("| distance m | poses | detected | detected % | median left tags | tag px (cam0) | noise L px | "
      "unstable L % | pooled sd L px |")
    P("|---|---|---|---|---|---|---|---|---|")
    ft = far_table(raw)
    for b in ft:
        if b["n"]:
            P(f"| {b['lo']:.2f}-{b['hi']:.2f} | {b['n']} | {b['detected']} | {100 * b['detected'] / b['n']:.0f} | "
              f"{_fmt(b['tags_med'], 0)} | {_fmt(b['tag_px_med'], 1)} | {_fmt(b['noise_med'], 3)} | "
              f"{_fmt(100 * b['unstable_med'], 1)} | {_fmt(b['pooled_med'], 3)} |")
    out_json["far"] = ft
    # ---- capture timing
    P("\n## Capture timing (raw takes)\n")
    tim = []
    for (rk, t), s in summ.items():
        if s["source"] != "raw":
            continue
        fr = load_frames(Path([x for x in a.robot if x.startswith(rk + ":")][0].split(":")[2]) / f"{t}.frames.npz")
        tm = timing([pf.tl for pf in fr])
        tim.append(tm)
        P(f"- {rk}/{t}: frame {tm['frame_ms']:.2f} ms, 5-frame window {tm['window_ms']:.1f} ms, pose period "
          f"{tm['period_s']:.2f} s")
    out_json["timing"] = tim
    # ---- item 4: poses per target on the stored sessions
    for name, (rig, sess, log_dir) in rigs.items():
        if not sess:
            continue
        st = sessions.get(sess)
        s = Session.load(st)
        z = cv.session_z(s, rig)
        asg = cv.split16_assignment(s, z)
        n_poses = {t: _cmds_for(log_dir, t)[1] for t in s.take_order}
        extra = set(a.extra or [])
        dflt = [t for (rk, t), v in summ.items() if rk == name and (rk, t) in default and f"{rk}/{t}" not in extra]
        tilts = [t for (rk, t) in tilt if rk == name and f"{rk}/{t}" not in extra]
        scen = {"as recorded": None}
        cmds = {t: _cmds_for(log_dir, t)[0] for t in s.take_order}
        prop_keep = {}
        for t in s.take_order:
            b = str(summ[(name, t)]["band"]) if (name, t) in default else ("tilted" if t in tilts else None)
            if b in proposals:
                prop_keep[t] = [p for p in range(n_poses[t]) if p in cmds[t] and grid_index(*cmds[t][p]) in
                                proposals[b]]
            else:
                prop_keep[t] = list(range(n_poses[t]))
        scen["proposed pose lists (defaults + tilted reduced, extras whole)"] = prop_keep
        scen["defaults + tilted only, all poses"] = {t: list(range(n_poses[t])) for t in s.take_order
                                                     if t in dflt or t in tilts}
        scen["defaults + tilted only, proposed lists"] = {t: prop_keep[t] for t in s.take_order
                                                          if t in dflt or t in tilts}
        res = {k: target_costs(s, rig, z, asg, n_poses, keep=v) for k, v in scen.items()}
        P(f"\n## Poses per coverage target: {sess} (split16 holdout_every 2; TEST cells on the nominal principal "
          "point; swept poses in capture order, every pose of earlier takes counted)\n")
        P("Scenarios: " + "; ".join(f"**{k}**: {r['holds']} holds, {r['swept_total']} poses swept over "
                                    f"{len(r['order'])} takes ({', '.join(r['order'])})" for k, r in res.items()))
        P("")
        P("| target | " + " | ".join(f"{k}: met at swept pose, in take (holds)" for k in res) + " |")
        P("|---" * (len(res) + 1) + "|")
        keys = []
        for r in res.values():
            for k in r["targets"]:
                if k not in keys:
                    keys.append(k)
        for k in keys:
            kind, c, reg = k
            label = f"{kind} {'' if c is None else 'cam' + str(c) + ' '}{reg}"
            cells_ = []
            for r in res.values():
                v = r["targets"].get(k)
                cells_.append("-" if v is None else "unmet" + f" ({v['holds_adding']})" if v["met_at"] is None else
                              f"{v['met_at']} {v['met_in']} ({v['holds_adding']})")
            P(f"| {label} | " + " | ".join(cells_) + " |")
        P("\nFinal FIT tallies (corners; thirds = image-corner thirds TL/TR/BL/BR) and TEST entries with >= "
          f"{cv.Targets().test_holds} TEST holds:\n")
        for sk, r in res.items():
            f = r["fit"]
            parts = []
            for c in (0, 1):
                t = f.cams[c]
                th = "/".join(str(int(t.thirds[i, j])) for i, j in cv.CORNER_THIRDS.values())
                parts.append(f"cam{c}: thirds {th}, rim {t.rim}, edges L/R/T/B "
                             + "/".join(str(t.edges[e]) for e in cv.EDGES) + ", boxes TL/TR/BL/BR "
                             + "/".join(str(t.boxes[b]) for b in cv.BOXES))
            judged = sorted(f"{c} {lo:g}-{hi:g} ({n})" for (c, (lo, hi)), n in r["entries"].items()
                            if n >= cv.Targets().test_holds)
            P(f"- **{sk}**: FIT pairs {f.n_pairs}, far pairs {f.far_pairs}; " + "; ".join(parts)
              + f"; judged {len(judged)}: {', '.join(judged)}; far-band TEST holds "
              + ", ".join(f"{lo:g}-{hi:g}: {n}" for (lo, hi), n in r["far_bands"].items()))
        out_json[f"targets/{sess}"] = {sk: {str(k): v for k, v in r["targets"].items()} for sk, r in res.items()}
        out_json[f"scenarios/{sess}"] = {sk: {"holds": r["holds"], "swept": r["swept_total"], "order": r["order"]}
                                         for sk, r in res.items()}
    if a.json:
        Path(a.json).write_text(json.dumps(out_json, indent=1, default=str))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="calibtool collection-protocol measurements")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--rig", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--procs", type=int, default=1)
    d.add_argument("--force", action="store_true")
    d.add_argument("takes", nargs="+")
    v = sub.add_parser("variant")
    v.add_argument("--rig", required=True)
    v.add_argument("--corners", required=True, help="folder of per-take corner files <take>.npz")
    v.add_argument("--out", required=True)
    v.add_argument("--common-test", default=None)
    v.add_argument("takes", nargs="+")
    n = sub.add_parser("analyse")
    n.add_argument("--robot", action="append", required=True,
                   help="name:rig:frames dir:corner dir:raw-take/log dir:stored session (empty fields allowed)")
    n.add_argument("--min-rate", type=float, default=0.5)
    n.add_argument("--max-bearing", type=float, default=25.0)
    n.add_argument("--json", default=None)
    n.add_argument("--extra", nargs="*", default=[],
                   help="robot/take placements made as extras (left out of the defaults scenario)")
    a = ap.parse_args(argv)
    {"detect": cmd_detect, "variant": cmd_variant, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
