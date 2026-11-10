"""Live AprilGrid detection (design plan sections 3, 10.1 and decision D9).

find_refine: the ArUco detector FINDS the tags on a contrast-stretched copy of the frame (a dark board is still found;
camera-side contrast is never touched), and every corner is then REFINED on the raw frame by the board's own X-junction
refinement (rigcfg.AprilGrid.refine), so the stretch never shifts a corner. use_stretch=False is the board's own
detect().

DetectPool: a process pool sized to the robot's free CPU (pool_size: cpus - load - a reserve, capped; phantom10 runs
load ~9 on 9 cpus, so 1 worker there) that detects both frames of a pair per job; resize() rebuilds it when the load
changes. board_obs: the board's centroid, depth and 3-D centre (PnP with the nominal equidistant camera, as step 06),
tag count and whether it touches the image edge, for placement guidance; board_miss: why a frame shows no board
(too dark, too far, partly in view, few tags, nothing), so the guidance can say it. hold_dict: detected pairs -> the step-15b
corner dict a replay.Collection / assign.CaptureHold takes."""
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from calibtool import paths

MIN_TAGS = 6
BORDER_PX = 8.0
DARK = 40.0                     # mean grey (0-255) of the image's central region below this: too dark to see the board
SMALL_TAG_PX = 14.0             # tags found but their median edge below this: the board is too far to read
POOL_RESERVE = 2.0              # cpus left to the robot's own load
POOL_CAP = 4


def stretch(gray, lo=1.0, hi=99.0) -> np.ndarray:
    """a copy with the lo / hi percentiles mapped to 0 / 255 (a flat image comes back unchanged)"""
    g = np.asarray(gray)
    a, b = np.percentile(g, [lo, hi])
    if b - a < 1.0:
        return g.copy()
    return np.clip((g.astype(np.float32) - a) * (255.0 / (b - a)), 0, 255).astype(np.uint8)


def _detector(board):
    import cv2
    det = getattr(board, "_calibtool_det", None)
    if det is None:
        p = cv2.aruco.DetectorParameters()
        p.markerBorderBits = board.border_bits
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)
        board._calibtool_det = det
    return det


def quads(board, img) -> list:
    """[(tag id, quad 4x2)] of the board's tags ArUco finds in img (the same detector settings as board.detect)"""
    corners, ids, _ = _detector(board).detectMarkers(img)
    if ids is None:
        return []
    return [(int(t), q) for q, t in zip(corners, ids.ravel()) if 0 <= int(t) < board.n_tags]


def find_refine(board, gray, use_stretch=True) -> dict:
    """{(tag, k): (u, v)}: tags found on the stretched copy, corners refined on the raw image"""
    if not use_stretch:
        return board.detect(gray)
    out = {}
    for tid, q in quads(board, stretch(gray)):
        for k, uv in board.refine(gray, q).items():
            out[(tid, k)] = uv
    return out


def usable_cpus() -> int:
    """the cpus this process may run on (os.cpu_count() counts every core; phantom10 lets a process use 9)"""
    try:
        import os
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        import os
        return os.cpu_count() or 1


def pool_size(ncpu, load1, reserve=POOL_RESERVE, cap=POOL_CAP) -> int:
    """detection workers for the free CPU: floor(cpus - load - reserve), at least 1, at most cap"""
    return int(max(1, min(cap, math.floor(ncpu - load1 - reserve))))


_W = {}


def _init(spec):
    import rigcfg
    _W["board"] = rigcfg.AprilGrid(*spec)


def _pair_job(yl, yr, use_stretch):
    b = _W["board"]
    return find_refine(b, yl, use_stretch), find_refine(b, yr, use_stretch)


def guide_corners(board, gray, use_stretch=True) -> dict:
    """{(tag, k): (u, v)} for placement guidance only: tags found on the (stretched) left image, ArUco's quad corners
    without sub-pixel refinement, no right image. 9 Nov trial: full-size detection of
    both images (~2 per second on phantom10) made the spoken guidance lag behind the board."""
    g = np.asarray(gray)
    img = stretch(g) if use_stretch else g          # full size: at half size only 14 of 35 tags were found at 1.1 m
    out = {}
    for tid, q in quads(board, img):
        q = np.asarray(q, float).reshape(4, 2)
        for k in range(4):
            out[(tid, k)] = (float(q[k, 0]), float(q[k, 1]))
    return out


def brightness(gray) -> float:
    """mean grey of the image's central region (the middle half in each direction, where the board is placed),
    every 4th pixel"""
    g = np.asarray(gray)
    h, w = g.shape[:2]
    return float(g[h // 4:h - h // 4:4, w // 4:w - w // 4:4].mean())


def _guide_job(yl, use_stretch):
    return guide_corners(_W["board"], yl, use_stretch), {"brightness": brightness(yl)}


class DetectPool:
    """detection workers (forkserver: the daemon has threads running, so no plain fork; the board travels as its
    constructor arguments, a cv2 detector does not pickle); submit_pair(left Y, right Y) -> Future of (left corners,
    right corners)"""

    def __init__(self, board, workers, use_stretch=True):
        self.board, self.use_stretch = board, use_stretch
        self.workers = 0
        self._ex = None
        self.resize(workers)

    def resize(self, workers):
        workers = max(1, int(workers))
        if workers == self.workers and self._ex is not None:
            return
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=False)
        import multiprocessing as mp
        b = self.board
        spec = (b.cols, b.rows, b.tag_size, b.spacing, b.border_bits)
        self._ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("forkserver"), initializer=_init,
                                       initargs=(spec,))
        self.workers = workers

    def submit_pair(self, yl, yr):
        return self._ex.submit(_pair_job, yl, yr, self.use_stretch)

    def submit_guide(self, yl):
        """placement guidance: left image only, no refinement -> Future of (left corners, {"brightness": central mean
        grey of the left image}) (board_miss uses it when no board is seen)"""
        return self._ex.submit(_guide_job, yl, self.use_stretch)

    def close(self):
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=True)
            self._ex = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@dataclass
class BoardObs:
    centroid: np.ndarray        # (2,) mean pixel of the corners
    z: float                    # depth (m) of the seen corners' centre, camera frame
    n_tags: int
    n_corners: int
    border: bool                # the board is cut by the frame (cut is not empty)
    cut: tuple                  # image sides that cut it ("left", "right", "top", "bottom"): tags are missing and the
                                # seen corners come within 1.5 tag widths (+ BORDER_PX) of that side
    lo: np.ndarray              # (2,) min u, v
    hi: np.ndarray              # (2,) max u, v
    p: object = None            # (3,) the seen corners' centre in camera coordinates (m; z = .z), None when unknown


def board_obs(det, board, focal=None, pp=None, intr=None, size=(1280, 800), min_tags=MIN_TAGS,
              border_px=BORDER_PX):
    """the board as one frame sees it, or None (fewer than min_tags tags, or PnP failed). The camera: intr (KB4
    [fx fy cx cy k1..k4]) or the nominal equidistant one (focal, principal point pp, default the image centre)"""
    import cv2
    import lensmodel
    if len({t for t, _ in det}) < min_tags:
        return None
    W, H = size
    if intr is None:
        pp = pp or ((W - 1) / 2.0, (H - 1) / 2.0)
        intr = [focal, focal, pp[0], pp[1], 0.0, 0.0, 0.0, 0.0]
    keys = list(det)
    uv = np.array([det[k] for k in keys], float)
    obj = np.array([board.obj_points(t)[k] for t, k in keys], float)
    ray = lensmodel.unproject("KB4", intr, uv)
    front = ray[:, 2] > 0.1
    if front.sum() < 12:
        return None
    und = np.ascontiguousarray(ray[front, :2] / ray[front, 2:3]).reshape(-1, 1, 2)
    ok, rv, tv = cv2.solvePnP(np.ascontiguousarray(obj[front]).reshape(-1, 1, 3), und, np.eye(3), None)
    if not ok:
        return None
    p = cv2.Rodrigues(rv)[0] @ obj.mean(0) + tv.ravel()
    z = float(p[2])
    lo, hi = uv.min(0), uv.max(0)
    n_tags = len({t for t, _ in det})
    margin = border_px + (1.5 * tag_px(det) if n_tags < board.n_tags else 0.0)
    gaps = {"left": lo[0], "top": lo[1], "right": W - 1 - hi[0], "bottom": H - 1 - hi[1]}
    cut = tuple(side for side, g in gaps.items() if g < margin)
    return BoardObs(uv.mean(0), z, n_tags, len(keys), bool(cut), cut, lo, hi, p)


@dataclass
class Miss:
    """why one frame shows no usable board (board_obs gave None), for the spoken guidance"""
    reason: str                 # dark | far | partial | few | none
    n_tags: int
    tag_px: float               # median tag edge (px) of the tags found, 0 when none
    brightness: object          # mean grey of the image's central region, None when unknown
    cut: tuple                  # image sides the tags found sit at, the nearest first ("left", "right", "top", ...)

    def as_dict(self):
        return {"reason": self.reason, "tags": self.n_tags, "tag_px": round(self.tag_px, 1),
                "brightness": None if self.brightness is None else round(float(self.brightness), 1),
                "cut": list(self.cut)}


def board_miss(det, size=(1280, 800), brightness=None, border_px=BORDER_PX) -> Miss:
    """why det (the corners found in a frame board_obs could not use) shows no board: too dark (central mean grey
    below DARK), too far (tags found, median edge below SMALL_TAG_PX), partial (the tags found sit within 1.5 tag
    edges + border_px of an image side), few (some tags, none of those), none (no tags in a bright image)"""
    n = len({t for t, _ in det})
    tpx = tag_px(det)
    cut = ()
    if det:
        uv = np.array(list(det.values()), float)
        W, H = size
        lo, hi = uv.min(0), uv.max(0)
        gaps = {"left": lo[0], "top": lo[1], "right": W - 1 - hi[0], "bottom": H - 1 - hi[1]}
        margin = border_px + 1.5 * tpx
        cut = tuple(sorted((k for k, g in gaps.items() if g < margin), key=lambda k: gaps[k]))
    if brightness is not None and brightness < DARK:
        reason = "dark"
    elif n and 0 < tpx < SMALL_TAG_PX:
        reason = "far"
    elif n and cut:
        reason = "partial"
    elif n:
        reason = "few"
    else:
        reason = "none"
    return Miss(reason, n, tpx, None if brightness is None else float(brightness), cut)


def tag_px(det) -> float:
    """median side length (px) of the tags whose 4 corners were found (0 when none)"""
    by = {}
    for (t, k), uv in det.items():
        by.setdefault(t, {})[k] = np.asarray(uv, float)
    sides = [np.mean([np.linalg.norm(c[(k + 1) % 4] - c[k]) for k in range(4)]) for c in by.values() if len(c) == 4]
    return float(np.median(sides)) if sides else 0.0


def hold_dict(pairs, hold, neck_rad=(math.nan, math.nan), approach=(0, 0)) -> dict:
    """step 15b's corner dict of one hold: pairs [(t_left, t_right, left corners, right corners)], hold = the global
    hold id, neck_rad = measured (pitch, yaw) rad, approach = (img_ap, img_ay)"""
    rows = []
    for j, (_, _, dl, dr) in enumerate(pairs):
        for cam, det in enumerate((dl, dr)):
            for (tag, k), (u, v) in det.items():
                rows.append((j, cam, tag, k, u, v))
    r = np.array(rows, float).reshape(-1, 6)
    n = len(pairs)
    return {"img_hold": np.full(n, int(hold), np.int64),
            "img_t0": np.array([p[0] for p in pairs], np.int64), "img_t1": np.array([p[1] for p in pairs], np.int64),
            "det_img": r[:, 0].astype(np.int64), "det_cam": r[:, 1].astype(np.int64),
            "det_tag": r[:, 2].astype(np.int64), "det_k": r[:, 3].astype(np.int64), "det_u": r[:, 4],
            "det_v": r[:, 5], "img_yaw": np.full(n, float(neck_rad[1])), "img_pitch": np.full(n, float(neck_rad[0])),
            "img_ay": np.full(n, int(approach[1]), np.int64), "img_ap": np.full(n, int(approach[0]), np.int64)}


paths  # noqa: B018  (puts the scripts folder on sys.path for lensmodel)
