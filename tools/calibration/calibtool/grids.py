"""The collection controller (design plan sections 4 and 5): one placement's neck grid -> captured pairs -> detection ->
CaptureHolds -> replay.Collection, and the session's walk over placements. Pure orchestration: the neck grid, the
capture, the detector, the placement model and the collection are passed in (fakes in tests; on the robot
neck.run_grid bound to a Neck, Capture.grab, DetectPool.submit_pair, the placement model, replay.Collection).

Per pose (GridRun): the neck grid calls on_pose after the neck has settled; on_pose grabs CAPTURE_FRAMES fresh pairs
and submits them for detection, then returns, which ends the hold (protocol.HOLD_ENDS_AT_CAPTURE). The pose's
detections are collected while the neck moves to the next pose (one pose in flight), so detection costs no pose time.
A pose becomes a hold with step 15b's rule (protocol.still_frames + per_hold_slice: PER_HOLD pairs from the middle of
the still frames with >= MIN_TAGS tags in both cameras): detect.hold_dict -> assign.CaptureHold -> Collection.add, its
FIT/TEST label kept in the pose's record. A pose with no such pair adds no hold.

Early stop (design 4): judged only on poses where the placement model predicts the board visible (predict(pitch, yaw)
-> Visible; without a model nothing is judged). After EarlyStopRule.min_judged judged poses, the grid stops when the
board was missed on more than miss_frac of them; darkness (the predicted board region's mean brightness below
dark_level on most misses: D9's levers, exposure first, gain second, then a retry) is told apart from absence (bright
enough, still not found: the board is not where the model put it). A bumped board (BumpWatch: the seen board's centre
more than px from the prediction on `run` found poses in a row, after agreeing first) pauses the grid. Either way the
placement goes to the redo list with its whole pose list. A grid held by the operator (heartbeat, key) goes there with
the poses it did not reach; an abort ends the session. With stop_when_rule_met the grid also ends as soon as the
collection's stop rule is met (Collection.stop).

End of placement: Collection.end_placement -> the snapshot and its gap sentences, spoken while the operator moves the
board. Collector walks the placements: the defaults, then the planner's extras until the stop rule is met, then the
redo list once; a placement the operator cannot make (key C) is recorded as unreachable."""
import math
from dataclasses import dataclass, field

import numpy as np

from calibtool import assign, detect, protocol
from calibtool.protocol import CAPTURE_FRAMES, MIN_TAGS, PER_HOLD, STILL_PX

grid_index = protocol.grid_index
EARLY_MIN_JUDGED = 6
EARLY_MISS_FRAC = 0.5
DARK_LEVEL = 40.0               # mean grey level (0-255) of the predicted board region below which a miss is "dark"
BUMP_PX = 60.0
BUMP_RUN = 3


@dataclass
class Visible:
    """the placement model's prediction for one pose: the board detectable, its pixel box (u0, v0, u1, v1) and centre
    in the left image"""
    visible: bool
    bbox: object = None
    centroid: object = None


@dataclass
class PoseRecord:
    index: int                  # pose order within the placement
    grid: int                   # index in the 49-pose grid
    command: tuple
    measured: tuple
    n_pairs: int
    tags_l: int                 # most tags seen in a left frame
    tags_r: int
    found: bool                 # a hold came out of the pose
    kept: int                   # pairs kept in the hold
    hold: object = None
    label: object = None
    predicted: object = None    # True / False / None (no model)
    brightness: object = None   # mean grey level of the predicted board region (left, first pair)
    dark: object = None
    centroid: object = None     # seen board centre (left, first frame with the board)
    pred_centroid: object = None
    sync_ms: object = None      # median |right - left| stamp

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        for k in ("command", "measured", "centroid", "pred_centroid"):
            if d[k] is not None:
                d[k] = [float(x) for x in d[k]]
        return d


@dataclass
class EarlyStop:
    reason: str                 # "dark" | "absent" | "bumped"
    sentence: str
    poses_judged: int
    misses: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RedoItem:
    target: object              # the placement target (name, distance_m, tilted, ...)
    reason: str
    poses: list

    @property
    def name(self):
        return getattr(self.target, "name", str(self.target))

    def as_dict(self) -> dict:
        return {"placement": self.name, "reason": self.reason, "poses": [list(p) for p in self.poses]}


@dataclass
class GridResult:
    placement: str
    records: list
    holds: list
    stopped: object             # None, "abort", "hold", "error: ...", "early_stop", "stop_rule"
    early_stop: object
    snapshot: object            # replay.Snapshot (None: no hold in this placement)
    sentences: list
    redo: object                # RedoItem or None

    def as_dict(self) -> dict:
        snap = self.snapshot.as_dict() if self.snapshot is not None and hasattr(self.snapshot, "as_dict") else None
        return {"placement": self.placement, "poses": len(self.records), "holds": list(self.holds),
                "stopped": self.stopped, "early_stop": self.early_stop.as_dict() if self.early_stop else None,
                "sentences": list(self.sentences), "redo": self.redo.as_dict() if self.redo else None,
                "snapshot": snap, "last": self.records[-1].as_dict() if self.records else None}


class EarlyStopRule:
    def __init__(self, min_judged=EARLY_MIN_JUDGED, miss_frac=EARLY_MISS_FRAC, dark_level=DARK_LEVEL):
        self.min_judged, self.miss_frac, self.dark_level = min_judged, miss_frac, dark_level
        self.judged, self.misses = 0, []

    def update(self, rec):
        if rec.predicted is not True:
            return None
        self.judged += 1
        if not rec.found:
            self.misses.append(rec)
        m, j = len(self.misses), self.judged
        if j < self.min_judged or m <= self.miss_frac * j:
            return None
        dark = [r for r in self.misses if r.dark]
        if len(dark) > m / 2:
            lv = np.median([r.brightness for r in dark])
            return EarlyStop("dark", f"The board is too dark to find: missed on {m} of {j} poses where it should be "
                             f"in view, and the image there is dark (brightness {lv:.0f} of 255). Raising exposure "
                             "first, then gain, and trying this placement again.", j, m)
        return EarlyStop("absent", f"I can't find the board: missed on {m} of {j} poses where it should be in view, "
                         "though the image there is bright enough. Is it still standing where it was? This placement "
                         "goes on the redo list.", j, m)


class BumpWatch:
    def __init__(self, px=BUMP_PX, run=BUMP_RUN):
        self.px, self.run = px, run
        self.agreed, self.streak = False, 0

    def update(self, rec):
        if not rec.found or rec.centroid is None or rec.pred_centroid is None:
            return None
        err = float(np.hypot(*np.subtract(rec.centroid, rec.pred_centroid)))
        if err <= self.px:
            self.agreed, self.streak = True, 0
            return None
        self.streak += 1
        if self.agreed and self.streak >= self.run:
            return EarlyStop("bumped", f"The board moved: on the last {self.streak} poses it sits {err:.0f} px from "
                             "where it stood at the start. Paused. Please stand it back and check it is steady; this "
                             "placement goes on the redo list.", self.streak, 0)
        return None


def _result(x):
    return x.result() if hasattr(x, "result") else x


def _ntags(d) -> int:
    return len({t for t, _ in d})


class GridRun:
    """one placement. neck_grid(poses, on_pose, should_stop) -> neck.GridOutcome (neck.run_grid bound to a Neck);
    grab(n) -> pairs (.left.y, .right.y, .t0, .t1, .dt_ns); detect(yl, yr) -> (left, right) corners or a Future of
    them; collection: replay.Collection; placement: its name (the take); poses [(pitch, yaw)] deg; predict(pitch, yaw)
    -> Visible or None; hold_ids: an iterator of global hold ids; should_stop() -> None | "abort" | "hold";
    on_event(kind, **kw)"""

    def __init__(self, neck_grid, grab, detect, collection, placement, poses, predict=None, hold_ids=None,
                 per_hold=PER_HOLD, frames=CAPTURE_FRAMES, should_stop=None, on_event=None, stop_when_rule_met=False,
                 rule=None, bump=None, target=None, still_px=STILL_PX, min_tags=MIN_TAGS):
        self.neck_grid, self.grab, self.detect, self.col = neck_grid, grab, detect, collection
        self.placement, self.poses = str(placement), [tuple(map(float, p)) for p in poses]
        self.predict, self.per_hold, self.frames = predict, int(per_hold), int(frames)
        self.hold_ids = hold_ids if hold_ids is not None else iter(range(10**9))
        self.ext_stop = should_stop or (lambda: None)
        self.on_event = on_event or (lambda kind, **kw: None)
        self.stop_when_rule_met = stop_when_rule_met
        self.rule, self.bump = rule or EarlyStopRule(), bump or BumpWatch()
        self.target = target if target is not None else placement
        self.still_px, self.min_tags = still_px, min_tags
        self.records, self.holds = [], []
        self.early = None
        self._pending = None
        self._prev_cmd = None
        self._stop_rule = False

    # ------------------------------------------------------------ per pose

    def _stop(self):
        why = self.ext_stop()
        if why:
            return why
        if self.early is not None:
            return "early_stop"
        if self._stop_rule:
            return "stop_rule"
        return None

    def _on_pose(self, done):
        pairs = self.grab(self.frames)
        futs = [self.detect(p.left.y, p.right.y) for p in pairs]
        self._flush()
        self._pending = (len(self.records) + len(self._queued()), done, pairs, futs)

    def _queued(self):
        return [] if self._pending is None else [self._pending]

    def _flush(self):
        if self._pending is None:
            return
        idx, done, pairs, futs = self._pending
        self._pending = None
        self._process(idx, done, pairs, [_result(f) for f in futs])

    def _process(self, idx, done, pairs, dets):
        cmd, meas = tuple(done.command), tuple(done.measured)
        prev = self._prev_cmd
        self._prev_cmd = cmd
        ap, ay = (0, 0) if prev is None else (int(np.sign(cmd[0] - prev[0])), int(np.sign(cmd[1] - prev[1])))
        tl = np.array([p.t0 for p in pairs], np.int64)
        tr = np.array([p.t1 for p in pairs], np.int64)
        pf = protocol.PoseFrames(np.asarray(cmd, float), np.asarray(meas, float), tl, tr, dets)
        sel = protocol.per_hold_slice(protocol.still_frames(pf, self.still_px, self.min_tags), self.per_hold)
        rec = PoseRecord(idx, grid_index(*cmd), cmd, meas, len(pairs), max([_ntags(d[0]) for d in dets] or [0]),
                         max([_ntags(d[1]) for d in dets] or [0]), bool(sel), len(sel))
        dts = [abs(p.dt_ns) for p in pairs]
        rec.sync_ms = float(np.median(dts)) / 1e6 if dts else None
        if sel:
            hid = int(next(self.hold_ids))
            data = detect.hold_dict([(int(tl[i]), int(tr[i]), dets[i][0], dets[i][1]) for i in sel], hid,
                                    neck_rad=(math.radians(meas[0]), math.radians(meas[1])), approach=(ap, ay))
            rec.label = self.col.add(assign.CaptureHold(hid, self.placement, int(tl[sel[0]]), data, rec.grid))
            rec.hold = hid
            self.holds.append(hid)
        seen = next((d[0] for d in dets if _ntags(d[0]) >= self.min_tags), None)
        if seen:
            rec.centroid = tuple(np.mean(list(seen.values()), axis=0))
        if self.predict is not None:
            v = self.predict(*meas)
            if v is not None:
                rec.predicted = bool(v.visible)
                rec.pred_centroid = None if v.centroid is None else tuple(map(float, v.centroid))
                if v.visible and v.bbox is not None and pairs:
                    rec.brightness = _region_mean(pairs[0].left.y, v.bbox)
                    rec.dark = rec.brightness is not None and rec.brightness < self.rule.dark_level
        self.records.append(rec)
        self.on_event("pose", placement=self.placement, record=rec.as_dict())
        for watch in (self.rule, self.bump):
            if self.early is None:
                es = watch.update(rec)
                if es is not None:
                    self.early = es
                    self.on_event("grid_early_stop", placement=self.placement, **es.as_dict())
        if self.stop_when_rule_met and getattr(self.col, "stop", None) is not None:
            self._stop_rule = True

    # ------------------------------------------------------------ the placement

    def run(self) -> GridResult:
        out = self.neck_grid(self.poses, self._on_pose, self._stop)
        self._flush()
        stopped = out.stopped
        if stopped is None and self.early is not None:
            stopped = "early_stop"
        if stopped is None and self._stop_rule and len(out.done) < len(self.poses):
            stopped = "stop_rule"
        snap = self.col.end_placement(self.placement) if self.holds else None
        sentences = list(getattr(snap, "sentences", []) or [])
        redo = None
        if self.early is not None:
            redo = RedoItem(self.target, self.early.reason, list(self.poses))
        elif stopped == "hold" or (isinstance(stopped, str) and stopped.startswith("error")):
            redo = RedoItem(self.target, "hold" if stopped == "hold" else "error", self.poses[len(out.done):])
        res = GridResult(self.placement, self.records, self.holds, stopped, self.early, snap, sentences, redo)
        self.on_event("grid_end", placement=self.placement, stopped=stopped, holds=len(self.holds),
                      sentences=sentences)
        return res


def _region_mean(y, bbox):
    y = np.asarray(y)
    if y.ndim != 2:
        return None
    h, w = y.shape
    u0, v0, u1, v1 = (int(round(v)) for v in bbox)
    u0, u1 = max(0, min(w, u0)), max(0, min(w, u1))
    v0, v1 = max(0, min(h, v0)), max(0, min(h, v1))
    if u1 <= u0 or v1 <= v0:
        return None
    return float(y[v0:v1, u0:u1].mean())


# ---------------------------------------------------------------- the session's placements


@dataclass
class CollectorResult:
    results: list = field(default_factory=list)       # GridResult per grid run
    redo: list = field(default_factory=list)          # RedoItems still open
    stop_met: bool = False
    unreachable: list = field(default_factory=list)   # targets the operator could not make (key C)
    stopped: object = None                            # None, "abort", "hold"
    pending: list = field(default_factory=list)       # targets not started (after a hold/abort)

    def as_dict(self) -> dict:
        nm = lambda t: getattr(t, "name", str(t))  # noqa: E731
        return {"placements": [r.placement for r in self.results], "redo": [r.as_dict() for r in self.redo],
                "stop_met": self.stop_met, "unreachable": [nm(t) for t in self.unreachable],
                "stopped": self.stopped, "pending": [nm(t) for t in self.pending]}


class Collector:
    """make_grid(target, poses=None) -> a GridRun-like (.run() -> GridResult) for that placement (poses: a redo's
    subset); await_placement(target) -> True when the operator placed the board (guidance + countdown done), False
    for "can't put it there"; plan_next(collection) -> the next extra target or None (the planner); should_stop() ->
    None | "abort" | "hold" between placements"""

    def __init__(self, collection, make_grid, await_placement, plan_next=None, max_extras=6, should_stop=None):
        self.col, self.make_grid, self.await_placement = collection, make_grid, await_placement
        self.plan_next, self.max_extras = plan_next, int(max_extras)
        self.should_stop = should_stop or (lambda: None)

    def _met(self):
        return getattr(self.col, "stop", None) is not None

    def _one(self, res: CollectorResult, target, poses=None) -> bool:
        """one placement; False: the session stops here (abort / hold)"""
        why = self.should_stop()
        if why:
            res.stopped = why
            return False
        if not self.await_placement(target):
            res.unreachable.append(target)
            return True
        g = self.make_grid(target) if poses is None else self.make_grid(target, poses=poses)
        r = g.run()
        res.results.append(r)
        if r.redo is not None:
            res.redo.append(r.redo)
        if r.stopped in ("abort", "hold"):
            res.stopped = r.stopped
            return False
        return True

    def run(self, placements) -> CollectorResult:
        res = CollectorResult()
        queue = list(placements)
        while queue and not self._met():
            t = queue.pop(0)
            if not self._one(res, t):
                res.pending = queue
                return res
        n = 0
        while self.plan_next is not None and not self._met() and n < self.max_extras:
            t = self.plan_next(self.col)
            if t is None:
                break
            n += 1
            if not self._one(res, t):
                return res
        retry, res.redo = list(res.redo), []
        for item in retry:
            if self._met():
                res.redo.append(item)
                continue
            if not self._one(res, item.target, poses=item.poses):
                return res
        res.stop_met = self._met()
        return res
