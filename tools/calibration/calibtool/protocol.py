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


def met_at(flags, swept_counts):
    """the swept count at which a target was met for the last time (it stays met to the end); None if unmet at the
    end. flags[i]: met after hold i; swept_counts[i]: poses swept by then"""
    flags = list(flags)
    if not flags or not flags[-1]:
        return None
    i = len(flags) - 1
    while i > 0 and flags[i - 1]:
        i -= 1
    return swept_counts[i]


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


def corner_noise(dets, remove_shift=False):
    """pooled per-axis std (px) of corner positions across frames (list of {(tag, k): (u, v)}) for corners seen in
    >= 2 frames -> (std, corners used); nan with none"""
    cnt = {}
    for d in dets:
        for k in d:
            cnt[k] = cnt.get(k, 0) + 1
    keys = [k for k, n in cnt.items() if n >= 2]
    if not keys:
        return float("nan"), 0
    ki = {k: i for i, k in enumerate(keys)}
    P = np.full((len(dets), len(keys), 2), np.nan)
    for f, d in enumerate(dets):
        for k, uv in d.items():
            if k in ki:
                P[f, ki[k]] = uv
    shift = np.zeros((len(dets), 1, 2))
    for _ in range(2 if remove_shift else 1):
        mean = np.nanmean(P - shift, axis=0)
        res = P - shift - mean
        if remove_shift:
            shift = shift + np.nan_to_num(np.nanmedian(res, axis=1, keepdims=True))
    res = P - shift - np.nanmean(P - shift, axis=0)
    dof = (np.isfinite(P[..., 0]).sum(0) - 1).sum()
    return float(np.sqrt(np.nansum(res ** 2) / (2 * dof))), len(keys)


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


def propose(detected, min_rate=0.5) -> list:
    """grid indices detected in >= min_rate of the takes; detected: (takes, 49) bool"""
    d = np.asarray(detected, bool)
    return [i for i in range(d.shape[1]) if d[:, i].mean() >= min_rate]


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
            one = {"img_t0": np.zeros(1), "det_img": np.zeros(len(dl), np.int64), "det_cam": np.zeros(len(dl), np.int64),
                   "det_tag": np.array([t for t, _ in dl]), "det_k": np.array([k for _, k in dl]),
                   "det_u": np.array([u for u, _ in dl.values()]), "det_v": np.array([v for _, v in dl.values()])}
            z = float(d16.rough_z(one, rig.board, rig.nominal_focal_px, rig.resolution)[0])
        row = {"take": take, "pose": p, "grid": grid_index(*cmd), "cmd": tuple(float(c) for c in cmd),
               "neck": tuple(float(c) for c in pf.neck), "detected": bool(kept), "n_still": len(kept),
               "tags_l": max((_ntags(a) for a, _ in pf.dets), default=0),
               "tags_r": max((_ntags(b) for _, b in pf.dets), default=0), "z": z,
               "tag_px": tag_px(dl)}
        for cam in (0, 1):
            ds = [pf.dets[k][cam] for k in kept]
            row[f"noise{cam}"], row[f"noise_n{cam}"] = corner_noise(ds) if len(ds) >= 2 else (float("nan"), 0)
            row[f"noise{cam}_ns"] = corner_noise(ds, remove_shift=True)[0] if len(ds) >= 3 else float("nan")
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
    report, per target, the poses swept until it was met for the last time (None: unmet), and how many holds added to
    it. n_poses: {take: poses swept in the take}; keep: {take: [pose indices swept]} (a reduced protocol: holds of
    other poses dropped, swept counts over the kept poses only; None: every pose). pp: the principal point of the TEST
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
    for k in keys:
        out[k] = {"met_at": met_at(flags[k], swept_counts), "holds_adding": adds.get(k, 0)}
    for e in sorted(entry_keys, key=str):
        f = [st[1].get(e, 0) >= targets.test_holds for st in states]
        out[("entry", None, e)] = {"met_at": met_at(f, swept_counts),
                                   "holds_adding": states[-1][1].get(e, 0) if states else 0}
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
    a = ap.parse_args(argv)
    {"detect": cmd_detect, "variant": cmd_variant}[a.cmd](a)


if __name__ == "__main__":
    main()
