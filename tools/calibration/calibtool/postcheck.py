"""The checks after a deploy (design plan 6.5, order fixed by the user; 13.2 row-health reference).

1. Depth check, always (depth_check): the board at one fresh distance; sync checked first (left/right stamps within
   SYNC_MS, else not checked); then the new and the old calibration scored on THE SAME frames by the same method
   (decide.score: lensscore's board-scale depth error + the rectified row error, via score_holds), compared on the
   pairs both scored. "worse" = new's median |depth error| above old's by more than DEPTH_WORSE_PCT, or its median
   |row error| above old's by more than ROWS_WORSE_PX.
2. OKVIS check, opt-out (okvis_check): the log names the new config (the entrypoint's `(<sha12>)`) and shows the
   reader's `KB4T pinhole camera` line; depth latency within LATENCY_FACTOR x the preflight baseline; the row offset
   on OKVIS's rectified pair, new vs old BY THE SAME METHOD (rowfeat: the rectified-pair method read a steady +1.78 px
   on phantom10's good config, so there is no absolute threshold, and two different methods are never compared);
   floor tilt only after VIO has settled (its gravity wandered 0.3-2.6 deg for minutes after restarts), reported.
decide(): automatic rollback ONLY on a crash loop or when the new calibration is worse than the old by the same method
(the depth check, or the rectified-pair rows); everything else is reported for the operator.

row_reference: what a future calibration-health monitor needs (13.2): the method, the median row offset, its pattern
(left/centre/right thirds, top/bottom edges), the frame conditions and the deployed config's sha, stored next to the
deployed config and in the session report.

The thresholds are named placeholders until fixed with the user (design Q2)."""
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from calibtool import decide as dc
from calibtool.deploy import crash_loop  # noqa: F401  (re-exported: the post-check's crash-loop rule)

SYNC_MS = 0.5
DEPTH_WORSE_PCT = 0.10          # median |depth error|, percentage points
ROWS_WORSE_PX = 0.10            # median |row error| on the board corners, px
RECT_WORSE_PX = 0.30            # |median row offset| on OKVIS's rectified pair, px
LATENCY_FACTOR = 1.5
FLOOR_TILT_WARN_DEG = 1.0
ROWFEAT_METHOD = "rowfeat-orb"
ROWFEAT_MAX_DY = 8.0
ROWFEAT_MIN_MATCHES = 30
REGION_MIN = 10


@dataclass
class DepthCheck:
    status: str                 # "ok" | "worse" | "not_checked"
    new: dict
    old: dict
    reasons: list

    def as_dict(self):
        return asdict(self)


def _side(sc: dc.Scores, common):
    m = np.isin(sc.src, common)
    cm = np.isin(sc.corner_src, common)
    e = np.asarray(sc.err_pct)[m]
    dy = np.abs(np.asarray(sc.corner_dy)[cm])
    return {"depth": float(np.median(np.abs(e))) if len(e) else None, "median_pct": float(np.median(e)) if len(e)
            else None, "rows": float(np.median(dy)) if len(dy) else None, "pairs": int(m.sum())}


def depth_check(sync_ms, new: dc.Scores, old: dc.Scores) -> DepthCheck:
    if sync_ms is None or sync_ms > SYNC_MS:
        return DepthCheck("not_checked", {}, {}, [f"sync: left/right stamps {sync_ms} ms apart (limit {SYNC_MS} ms): "
                                                  "the depth check needs synced pairs; fix the sync and measure again"])
    common = np.intersect1d(new.src, old.src)
    if not len(common):
        return DepthCheck("not_checked", {}, {}, ["no pair scored by both calibrations"])
    a, b = _side(new, common), _side(old, common)
    reasons, worse = [], False
    dd, dr = a["depth"] - b["depth"], a["rows"] - b["rows"]
    line = (f"median |depth error| new {a['depth']:.3f} % vs old {b['depth']:.3f} % ({dd:+.3f}); rows new "
            f"{a['rows']:.3f} px vs old {b['rows']:.3f} px ({dr:+.3f}); {a['pairs']} pairs, the same frames")
    if dd > DEPTH_WORSE_PCT:
        worse = True
        reasons.append(f"new worse on depth by the same method (> {DEPTH_WORSE_PCT} %): " + line)
    if dr > ROWS_WORSE_PX:
        worse = True
        reasons.append(f"new worse on rows by the same method (> {ROWS_WORSE_PX} px): " + line)
    if not worse:
        reasons.append(line)
    return DepthCheck("worse" if worse else "ok", a, b, reasons)


@dataclass
class RowHealth:
    method: str
    n: int
    median_dy: float            # right - left row of matched features, px
    thirds: dict                # {"left", "centre", "right"}: median dy per image third (None: too few)
    edges: dict                 # {"top", "bottom"}: median dy in the top / bottom fifth

    def as_dict(self):
        return asdict(self)


def rowfeat(img0, img1, max_dy=ROWFEAT_MAX_DY, min_matches=ROWFEAT_MIN_MATCHES):
    """the row offset of a rectified pair from ORB matches (right row - left row, positive disparity only); None with
    fewer than min_matches matches"""
    import cv2
    a, b = np.asarray(img0, np.uint8), np.asarray(img1, np.uint8)
    orb = cv2.ORB_create(nfeatures=3000)
    k0, d0 = orb.detectAndCompute(a, None)
    k1, d1 = orb.detectAndCompute(b, None)
    if d0 is None or d1 is None:
        return None
    ms = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
    rows = []
    for m in ms:
        u0, v0 = k0[m.queryIdx].pt
        u1, v1 = k1[m.trainIdx].pt
        if abs(v1 - v0) <= max_dy and u0 - u1 >= -2.0:
            rows.append((u0, v0, v1 - v0))
    if len(rows) < min_matches:
        return None
    r = np.asarray(rows)
    h, w = a.shape[:2]

    def med(mask):
        return float(np.median(r[mask, 2])) if mask.sum() >= REGION_MIN else None
    u, v = r[:, 0], r[:, 1]
    thirds = {"left": med(u < w / 3), "centre": med((u >= w / 3) & (u < 2 * w / 3)), "right": med(u >= 2 * w / 3)}
    edges = {"top": med(v < h / 5), "bottom": med(v > 4 * h / 5)}
    return RowHealth(ROWFEAT_METHOD, len(r), float(np.median(r[:, 2])), thirds, edges)


@dataclass
class OkvisCheck:
    kb4t_line: bool
    config_sha: bool
    latency_s: object
    latency_ok: object          # None: not measured
    rect_new: object
    rect_old: object
    rect_worse: object          # None: not compared
    floor_tilt_deg: object
    reasons: list = field(default_factory=list)

    def as_dict(self):
        d = asdict(self)
        return d


def okvis_check(log_lines, sha12, latency_s=None, baseline_latency_s=None, rect_new=None, rect_old=None,
                floor_tilt_deg=None) -> OkvisCheck:
    txt = "\n".join(log_lines or [])
    kb4t = "KB4T pinhole camera" in txt
    sha = bool(sha12) and f"({sha12})" in txt
    reasons = []
    if not sha:
        reasons.append(f"the OKVIS log does not name the deployed config ({sha12}): is it running the new file?")
    if not kb4t:
        reasons.append("no 'KB4T pinhole camera' line in the OKVIS log")
    lat_ok = None
    if latency_s is not None and baseline_latency_s:
        lat_ok = latency_s <= LATENCY_FACTOR * baseline_latency_s
        if not lat_ok:
            reasons.append(f"depth latency {latency_s:.2f} s vs {baseline_latency_s:.2f} s before (> "
                           f"{LATENCY_FACTOR:g}x)")
    worse = None
    if rect_new is not None and rect_old is not None:
        if rect_new.method != rect_old.method:
            reasons.append(f"rectified-pair rows not compared: different method ({rect_new.method} vs "
                           f"{rect_old.method})")
        else:
            worse = abs(rect_new.median_dy) - abs(rect_old.median_dy) > RECT_WORSE_PX
            reasons.append(f"rectified-pair rows ({rect_new.method}) new {rect_new.median_dy:+.2f} px vs old "
                           f"{rect_old.median_dy:+.2f} px" + (f": new worse by > {RECT_WORSE_PX} px" if worse else ""))
    if floor_tilt_deg is not None and abs(floor_tilt_deg) > FLOOR_TILT_WARN_DEG:
        reasons.append(f"floor tilt {floor_tilt_deg:.1f} deg after settle (> {FLOOR_TILT_WARN_DEG} deg; VIO's gravity "
                       "can wander for minutes after a restart: measure again later before blaming the calibration)")
    return OkvisCheck(kb4t, sha, latency_s, lat_ok, rect_new, rect_old, worse, floor_tilt_deg, reasons)


@dataclass
class PostDecision:
    rollback: bool
    reasons: list

    def as_dict(self):
        return asdict(self)


def decide(depth: DepthCheck = None, okvis: OkvisCheck = None, crash=False) -> PostDecision:
    """roll back only on a crash loop or new worse than old by the same method; the rest is reported"""
    if crash:
        return PostDecision(True, ["OKVIS is crash-looping on the new calibration: rolling back"])
    rb, reasons = [], []
    if depth is not None:
        (rb if depth.status == "worse" else reasons).extend(depth.reasons)
    if okvis is not None:
        if okvis.rect_worse:
            rb.append(next(r for r in okvis.reasons if r.startswith("rectified-pair rows")))
            reasons += [r for r in okvis.reasons if not r.startswith("rectified-pair rows")]
        else:
            reasons += okvis.reasons
    if rb:
        return PostDecision(True, rb + reasons)
    return PostDecision(False, reasons or ["post-check passed"])


def row_reference(row: RowHealth, conditions: dict, config_sha12, config_name=None, clock=time.time) -> dict:
    """the row-health reference stored at every deploy (design 13.2)"""
    return {"method": row.method, "median_dy_px": row.median_dy, "matches": row.n, "thirds": dict(row.thirds),
            "edges": dict(row.edges), "conditions": dict(conditions), "config_sha12": config_sha12,
            "config": config_name, "taken_at": float(clock())}


def merge_holds(holds) -> dict:
    """CaptureHolds -> one step-15b corner dict (pairs renumbered in order)"""
    out, off = {}, 0
    for h in holds:
        d = h.data
        n = len(d["img_t0"])
        for k, v in d.items():
            v = np.asarray(v)
            if k == "det_img":
                v = v + off
            out.setdefault(k, []).append(v)
        off += n
    return {k: np.concatenate(v) for k, v in out.items()}


def score_holds(calib_json, holds, rig, work_dir, placement="postcheck", scorer=None):
    """the post-check frames scored by one calibration (decide.score: the depth and row method of 6.1)"""
    d = merge_holds(holds)
    p = Path(work_dir) / "postcheck_holds.npz"
    np.savez(p, **d)
    pl = np.array([placement] * len(d["img_t0"]))
    return (scorer or dc.score)(calib_json, p, rig, pl)


def calib_json_for(okvis_yaml, size, work_dir, name):
    """an OKVIS config as the step-66 JSON decide.score reads, at the frames' size (livecfg.load(at_size))"""
    from calibtool import livecfg
    return livecfg.load(okvis_yaml, at_size=tuple(size)).write_json(Path(work_dir) / f"{name}.json")
