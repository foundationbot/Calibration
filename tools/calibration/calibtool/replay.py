"""Replay a stored calibration session as if it were live (M1 C1; design plan section 4).

The session's holds go, in capture order (first left stamp), through the FIT/TEST assignment and the coverage tallies;
after each placement (take) a Snapshot is taken: tallies, what is short (coverage deficits), the gap sentences the
operator would hear and one progress line. After the last placement the planner suggests the next placements. Then the
split is written, fitted (step 45), validated (step 66: the new fit and the live config, a RADTAN live config through
livecfg's runner), both are scored on the same TEST pairs (decide.score) and decided (decide.verdict, with the default
required edges and with all four required). run() writes out_dir/report.json and out_dir/report.md.

The live tool (M2) reuses the collection part as it is: Collection.add(hold) per hold as it arrives (a
assign.CaptureHold: in-memory corner dict, nothing read from or written to disk), Collection.end_placement() after
each grid, plan() for the next placement. feed() is the replay's loop over those calls.

Counting, per pair (step 16's rough distance z, rough_z with the nominal focal and principal point):
  all     every corner captured so far, per camera (coverage.corner_tally: thirds, rim, edge strips, corner boxes),
          and the far pairs (coverage.FAR_FIT, z in [1.7, 3.0) m)
  fit     the same over FIT pairs: pairs of a FIT hold with z <= max_fit_z (a far pair is TEST whatever its hold, as in
          step 16 and assign.Assigner.pair_labels)
  test    TEST holds per step-66 entry (lenscells cell of the pair's camera-0 mean corner vs the principal point x
          DIST_BINS on z) and per far band, a hold counted once per entry: coverage.test_tally's definition
The deficits and sentences are computed on the corner regions and far pairs of ALL corners captured (the kit scripts'
and the handoff's numbers: p21 left edge 176 vs 1,099 counts every corner of the 8 takes; and what the planner's
simulated placements add, every pair of a grid), with the FIT pair count for the pairs target (what step 45 gets).
REGION_TALLY = "fit" counts the regions over FIT corners instead.

A snapshot counts with the labels the assignment's finish() (assign rule 3: held-back holds released to TEST) would
leave if the session ended after that placement: the split it would write, which is also what the stop rule judges.
Collection.labels() / tallies() without arguments are the capture-time labels.

The stop rule (coverage.Targets, Coverage.stop_met: no blocking deficit) is checked after every hold with the labels
finish() would leave if the session ended there (the split it would write); the first hold that meets it is
Collection.stop (a StopPoint). run(stop_at_rule=True) cuts the session there and fits only that part. Targets default
to Targets.for_per_hold(per_hold), per_hold = the stored sessions' 3 pairs per pose."""
import json
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from calibtool import assign, coverage as cv, decide, livecfg, paths, pipeline, placement as pl, planner as pn, sessions
from calibtool.session import Session

ASSIGNMENTS = ("capture", "split16")
REGION_TALLY = "all"            # "all" | "fit": which corners the region deficits count (see the module text)
MD_SENTENCES = 6                # gap sentences per placement in report.md (all of them are in report.json)
# the report's two verdicts: the default required edges (left, right, bottom; C3) and all four
VERDICT_KEYS = {"required_edges=left,right,bottom": cv.REQUIRED_EDGES, "required_edges=all": frozenset(cv.EDGES)}
# the errors that make the live config "not scorable" (the verdict says so, with the error); any other error while
# loading, scoring or validating it is a bug and stops the run
NOT_SCORABLE = (livecfg.NotScorable, pipeline.PipelineError)


# ---------------------------------------------------------------- labelers: the FIT/TEST assignment


class FixedLabeler:
    """labels known before the session (step 16's split as a hold assignment): the replay reference. The same calls
    as assign.Assigner: observe(hold) -> label, finish(); labels() of the holds observed"""

    def __init__(self, labels):
        self._all = {int(h): str(v) for h, v in labels.items()}
        self._seen = {}

    def observe(self, hold) -> str:
        if hold.hold in self._seen:
            raise ValueError(f"hold {hold.hold} observed twice")
        self._seen[hold.hold] = self._all[int(hold.hold)]
        return self._seen[hold.hold]

    def finish(self):
        pass

    def labels(self) -> dict:
        return dict(self._seen)


def capture_labeler(rig, pp=None, **kw) -> assign.Assigner:
    """the capture-time assignment (assign.Assigner) at the rig's nominal focal and principal point (image centre)"""
    rig = pipeline.load_rig(rig)
    pp = pp or (rig.resolution[0] / 2, rig.resolution[1] / 2)
    return assign.Assigner(rig, pp, rig.nominal_focal_px, **kw)


def labels_of(labeler) -> dict:
    """{hold id: FIT | TEST}: the labeler's current labels (an Assigner's are final after finish())"""
    if isinstance(labeler, assign.Assigner):
        return {h: i.label for h, i in labeler.holds.items()}
    return labeler.labels()


# ---------------------------------------------------------------- the per-placement step


@dataclass
class _HoldCov:
    """what one hold adds to the tallies"""
    hold: int
    placement: str
    n_pairs: int
    n_near: int
    board: bool                 # some pair shows >= placement.MIN_TAGS camera-0 tags
    all: cv.FitTally            # all its pairs
    near: cv.FitTally           # its pairs within max_fit_z (FIT when the hold is FIT)
    entries: set                # {(cell, (lo, hi))} its pairs fall in
    far_bands: set              # {(lo, hi)} its pairs fall in


def _empty_tally(size) -> cv.FitTally:
    z = cv.corner_tally(np.zeros(0), np.zeros(0), size)
    return cv.FitTally({0: z, 1: z}, 0, 0)


@dataclass
class StopPoint:
    """the first hold after which the stop rule was met"""
    index: int                  # holds fed by then (1, 2, ...)
    hold: int
    placement: str
    placement_index: int        # 1, 2, ...: the placement's place in capture order
    pose: int                   # the pose index within the placement's grid (-1: not known)
    neck_pitch_deg: float       # measured neck angles of the hold (mean over its pairs; nan without them)
    neck_yaw_deg: float

    def as_dict(self) -> dict:
        return {"index": self.index, "hold": self.hold, "placement": self.placement,
                "placement_index": self.placement_index, "pose": self.pose, "neck_pitch_deg": self.neck_pitch_deg,
                "neck_yaw_deg": self.neck_yaw_deg}


@dataclass
class Snapshot:
    index: int                  # 1, 2, ...: placements so far
    placement: str
    placement_holds: int
    placement_board_holds: int  # holds of this placement where the board was seen (>= MIN_TAGS camera-0 tags)
    placement_pairs: int
    holds: int                  # the session so far
    pairs: int
    fit_pairs: int
    test_pairs: int
    tallies: dict               # {"all": FitTally, "fit": FitTally, "test": TestTally}
    ready: list                 # step-66 entries with >= Targets.test_holds TEST holds
    deficits: list              # coverage.Deficit, most important first
    sentences: list             # coverage.sentences(deficits)
    progress: str               # one line
    stop_met: bool = False      # the stop rule has been met by the end of this placement (Collection.stop)

    def as_dict(self) -> dict:
        return {"index": self.index, "placement": self.placement, "placement_holds": self.placement_holds,
                "placement_board_holds": self.placement_board_holds, "placement_pairs": self.placement_pairs,
                "holds": self.holds, "pairs": self.pairs, "fit_pairs": self.fit_pairs, "test_pairs": self.test_pairs,
                "progress": self.progress, "stop_met": self.stop_met, "sentences": list(self.sentences),
                "deficits": [deficit_dict(d) for d in self.deficits], "ready": [region_str(e) for e in self.ready],
                "tallies": tallies_dict(self.tallies)}


class Collection:
    """the coverage of a session as its holds arrive: add() per hold, end_placement() after each placement. No file is
    read or written. labeler: assign.Assigner (capture) or FixedLabeler; pp / focal: the nominal principal point and
    focal length (rough distance and cells), by default the Assigner's or the rig's"""

    def __init__(self, rig, labeler, pp=None, focal=None, targets=None, max_fit_z=3.0, region_tally=REGION_TALLY):
        import lenscells  # noqa: F401  (scripts folder on sys.path through calibtool.paths)
        self.rig = pipeline.load_rig(rig)
        self.size = tuple(int(v) for v in self.rig.resolution)
        self.labeler = labeler
        lp = getattr(labeler, "pp", None)
        self.pp = tuple(float(v) for v in (pp or lp or (self.size[0] / 2, self.size[1] / 2)))
        self.focal = float(focal or getattr(labeler, "focal", None) or self.rig.nominal_focal_px)
        self.targets = targets or cv.Targets()
        self.max_fit_z = float(max_fit_z)
        if region_tally not in ("all", "fit"):
            raise ValueError(f"region_tally={region_tally!r}: 'all' or 'fit'")
        self.region_tally = region_tally
        self._d16 = paths.import_script("16_split_distance")
        self._holds = {}            # hold id -> _HoldCov, in add order
        self._open = []             # hold ids added since the last snapshot
        self._places = []           # placements in capture order
        self.snapshots = []
        self.stop = None            # StopPoint: the first hold after which the stop rule was met

    @classmethod
    def for_session(cls, session, rig, assignment="capture", **kw):
        """a Collection for replaying session: assignment "capture" (assign.Assigner) or "split16" (step 16's split,
        known beforehand: coverage.split16_assignment)"""
        rig = pipeline.load_rig(rig)
        if assignment == "capture":
            lab = capture_labeler(rig)
        elif assignment == "split16":
            lab = FixedLabeler(cv.split16_assignment(session, cv.session_z(session, rig), kw.get("max_fit_z", 3.0)))
        else:
            raise ValueError(f"assignment={assignment!r}: one of {ASSIGNMENTS}")
        return cls(rig, lab, **kw)

    # ------------------------------------------------------------ holds in

    def _measure(self, hold) -> _HoldCov:
        import lenscells
        d = hold.data
        n = len(d["img_t0"])
        z = self._d16.rough_z(d, self.rig.board, self.focal, (2 * self.pp[0], 2 * self.pp[1]))
        near = np.nan_to_num(z, nan=np.inf) <= self.max_fit_z
        img, cam = np.asarray(d["det_img"], np.int64), np.asarray(d["det_cam"], np.int64)
        u, v = np.asarray(d["det_u"], float), np.asarray(d["det_v"], float)

        def tally(pair_mask):
            sel = pair_mask[img]
            cams = {c: cv.corner_tally(u[sel & (cam == c)], v[sel & (cam == c)], self.size) for c in (0, 1)}
            zz = z[pair_mask]
            return cv.FitTally(cams, int(((zz >= cv.FAR_FIT[0]) & (zz < cv.FAR_FIT[1])).sum()), int(pair_mask.sum()))

        m0 = cam == 0
        cnt = np.bincount(img[m0], minlength=n)[:n]
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.bincount(img[m0], weights=u[m0], minlength=n)[:n] / cnt
            mv = np.bincount(img[m0], weights=v[m0], minlength=n)[:n] / cnt
        cells = lenscells.cell_of(mu - self.pp[0], mv - self.pp[1])
        bins = lenscells.dist_bin(z)
        entries = {(str(c), tuple(lenscells.DIST_BINS[b])) for c, b, k in zip(cells, bins, cnt) if k and b >= 0}
        bands = {(lo, hi) for lo, hi in cv.FAR_BANDS if ((z >= lo) & (z < hi)).any()}
        tags = [len(np.unique(np.asarray(d["det_tag"])[m0 & (img == j)])) for j in range(n)]
        return _HoldCov(int(hold.hold), str(hold.placement), n, int(near.sum()),
                        bool(tags) and max(tags) >= pl.MIN_TAGS, tally(np.ones(n, bool)), tally(near), entries, bands)

    def add(self, hold) -> str:
        """one hold (assign.CaptureHold) as it is captured: its FIT/TEST label at capture"""
        if hold.hold in self._holds:
            raise ValueError(f"hold {hold.hold} added twice")
        hc = self._measure(hold)
        label = self.labeler.observe(hold)
        self._holds[hc.hold] = hc
        self._open.append(hc.hold)
        if not self._places or self._places[-1] != hc.placement:
            self._places.append(hc.placement)
        if self.stop is None and self.stop_met():
            d = hold.data
            with np.errstate(invalid="ignore"):
                pitch = float(np.degrees(np.nanmean(d["img_pitch"]))) if "img_pitch" in d else math.nan
                yaw = float(np.degrees(np.nanmean(d["img_yaw"]))) if "img_yaw" in d else math.nan
            self.stop = StopPoint(len(self._holds), hc.hold, hc.placement, len(self._places),
                                  int(getattr(hold, "pose", -1)), pitch, yaw)
        return label

    # ------------------------------------------------------------ state

    def labels(self) -> dict:
        return labels_of(self.labeler)

    def stop_labels(self) -> dict:
        """the labels finish() would leave if the session ended now (an Assigner's held-back holds released; a fixed
        labeler's own): what the stop rule counts"""
        f = getattr(self.labeler, "labels_if_finished", None)
        return f() if f else self.labels()

    def stop_met(self) -> bool:
        """the stop rule (coverage.Coverage.stop_met) on the labels the split would get now"""
        return self.coverage(self.tallies(self.stop_labels())).stop_met(self.targets)

    def tallies(self, labels=None) -> dict:
        """{"all": FitTally of every pair, "fit": FitTally of the FIT pairs, "test": TestTally}; labels: {hold: FIT |
        TEST} (default: the current ones)"""
        lab = self.labels() if labels is None else labels
        out_all, out_fit = _empty_tally(self.size), _empty_tally(self.size)
        entries, bands = {}, {b: 0 for b in cv.FAR_BANDS}
        for h, hc in self._holds.items():
            out_all = out_all + hc.all
            if lab.get(h) == assign.FIT:
                out_fit = out_fit + hc.near
            elif lab.get(h) == assign.TEST:
                for e in hc.entries:
                    entries[e] = entries.get(e, 0) + 1
                for b in hc.far_bands:
                    bands[b] += 1
        return {"all": out_all, "fit": out_fit, "test": cv.TestTally(entries, bands)}

    def coverage(self, tallies=None) -> cv.Coverage:
        """the Coverage the deficits are judged on: regions and far pairs from region_tally, pairs = FIT pairs"""
        t = tallies or self.tallies()
        reg = t[self.region_tally]
        return cv.Coverage(fit=cv.FitTally(reg.cams, reg.far_pairs, t["fit"].n_pairs), test=t["test"])

    def counts(self, labels=None) -> dict:
        lab = self.labels() if labels is None else labels
        fit = sum(hc.n_near for h, hc in self._holds.items() if lab.get(h) == assign.FIT)
        pairs = sum(hc.n_pairs for hc in self._holds.values())
        return {"holds": len(self._holds), "pairs": pairs, "fit_pairs": fit, "test_pairs": pairs - fit}

    def ready(self, tallies=None) -> list:
        t = tallies or self.tallies()
        return sorted(e for e, n in t["test"].entries.items() if n >= self.targets.test_holds)

    # ------------------------------------------------------------ the placement step

    def end_placement(self, placement=None) -> Snapshot:
        """close the placement: the holds added since the previous snapshot. -> its Snapshot (also in .snapshots)"""
        if not self._open:
            raise ValueError("no hold added since the last snapshot")
        mine = [self._holds[h] for h in self._open]
        name = placement or mine[-1].placement
        lab = self.stop_labels()
        t = self.tallies(lab)
        defs = self.coverage(t).deficits(self.targets)
        c = self.counts(lab)
        ready = self.ready(t)
        snap = Snapshot(len(self.snapshots) + 1, name, len(mine), sum(h.board for h in mine),
                        sum(h.n_pairs for h in mine), c["holds"], c["pairs"], c["fit_pairs"], c["test_pairs"], t,
                        ready, defs, cv.sentences(defs), "", self.stop is not None)
        snap.progress = progress_line(snap, self.targets)
        self.snapshots.append(snap)
        self._open = []
        return snap

    def finish(self) -> dict:
        """close the session (the assignment's finish(): held-back holds may move to TEST) -> the final state"""
        before = self.labels()
        self.labeler.finish()
        lab = self.labels()
        t = self.tallies()
        c = self.counts()
        return {**c, "released_to_test": sum(before.get(h) != lab[h] for h in lab),
                "ready": [region_str(e) for e in self.ready(t)], "tallies": tallies_dict(t)}


def feed(col: Collection, holds, log=None) -> list:
    """the replay loop: holds (assign.CaptureHold, capture order) into col, a snapshot whenever the placement changes
    and after the last hold. log: called with each progress line"""
    snaps, cur = [], None
    for h in holds:
        if cur is not None and h.placement != cur:
            snaps.append(col.end_placement(cur))
            if log:
                log(snaps[-1].progress)
        col.add(h)
        cur = h.placement
    if cur is not None:
        snaps.append(col.end_placement(cur))
        if log:
            log(snaps[-1].progress)
    return snaps


def plan(deficits, head: pl.Head, board, pp=None, k=3, chain_=None, pairs_per_pose=None) -> list:
    """the planner's next k placements (planner.Suggestion) for these deficits"""
    kw = {} if pairs_per_pose is None else {"pairs_per_pose": pairs_per_pose}
    return pn.Planner(head, chain_ or pl.chain(), board, pp=pp, **kw).suggest(deficits, k=k)


# ---------------------------------------------------------------- words and JSON


def region_str(r) -> str:
    """("centre", (0.4, 0.6)) -> "centre 0.4-0.6"; (1.7, 3.0) -> "1.7-3"; "left" -> "left" """
    if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], tuple):
        return f"{r[0]} {r[1][0]:g}-{r[1][1]:g}"
    if isinstance(r, tuple) and len(r) == 2 and all(isinstance(x, (int, float)) for x in r):
        return f"{r[0]:g}-{r[1]:g}"
    return str(r)


def deficit_dict(d: cv.Deficit) -> dict:
    return {"kind": d.kind, "cam": d.cam, "region": region_str(d.region), "have": d.have, "want": d.want,
            "ref_region": None if d.ref_region is None else region_str(d.ref_region), "ref_have": d.ref_have,
            "weight": d.weight, "blocks": d.blocks, "sentence": cv.sentence(d)}


def _corner_dict(t: cv.CornerTally) -> dict:
    return {"n": t.n, "thirds": np.asarray(t.thirds).tolist(), "rim": t.rim, "edges": dict(t.edges),
            "boxes": dict(t.boxes)}


def tallies_dict(t: dict) -> dict:
    out = {k: {"n_pairs": t[k].n_pairs, "far_pairs": t[k].far_pairs,
               "cams": {str(c): _corner_dict(x) for c, x in t[k].cams.items()}} for k in ("all", "fit")}
    out["test"] = {"entries": {region_str(e): n for e, n in sorted(t["test"].entries.items())},
                   "far_bands": {region_str(b): n for b, n in t["test"].far_bands.items()}}
    return out


def suggestion_dict(s: pn.Suggestion) -> dict:
    c = s.candidate
    return {"say": s.say(), "distance_m": c.distance_m, "bearing_deg": c.bearing_deg, "tilted": c.tilted,
            "describe": c.describe(), "score": s.score, "target": deficit_dict(s.target),
            "fills": [{**deficit_dict(f), "gain": g} for f, g in zip(s.fills, s.gains)],
            "poses": [list(p) for p in s.poses]}


def progress_line(s: Snapshot, targets: cv.Targets) -> str:
    e = s.tallies["all"].cams[0].edges
    ready = ", ".join(region_str(x) for x in s.ready) or "none"
    return (f"Placement {s.index} ({s.placement}): {s.placement_holds} holds, board seen in "
            f"{s.placement_board_holds}, {s.placement_pairs:,} pairs. Session: {s.pairs:,} pairs (FIT "
            f"{s.fit_pairs:,}, TEST {s.test_pairs:,}); held-out entries with >= {targets.test_holds} holds: "
            f"{len(s.ready)} ({ready}); left camera edge strips: left {e['left']:,}, right {e['right']:,}, top "
            f"{e['top']:,}, bottom {e['bottom']:,} corners.")


def _js(o):
    """JSON-ready: tuples to lists, numpy to Python, non-finite floats to None, Paths to str"""
    if isinstance(o, dict):
        return {str(k): _js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_js(v) for v in (sorted(o, key=str) if isinstance(o, (set, frozenset)) else o)]
    if isinstance(o, np.ndarray):
        return _js(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


# ---------------------------------------------------------------- the decision


def decision(new, live, step66_new, gates: decide.Gates, live_error=None) -> dict:
    """decide.verdict with each VERDICT_KEYS edge set, the paired statistics (new - live) and each side's own
    statistics. new / live: decide.Scores (live None: not scorable, live_error saying why); step66_new: step 66 of the
    new fit"""
    out = {"verdicts": {}, "paired": None, "stats": {"new": decide.stats(new) if new is not None else None,
                                                     "live": decide.stats(live) if live is not None else None}}
    for key, edges in VERDICT_KEYS.items():
        out["verdicts"][key] = decide.verdict(new, live, step66_new, replace(gates, required_edges=edges),
                                              live_error=live_error).as_dict()
    if new is not None and live is not None:
        out["paired"] = decide.paired(new, live, gates).as_dict()
    return _js(out)


def _num(x, fmt="{:.3f}"):
    return "-" if x is None else fmt.format(x)


def decision_md(dec: dict) -> list:
    """report.md lines of a decision() result"""
    out = ["## Decision", ""]
    for key in VERDICT_KEYS:
        v = dec["verdicts"][key]
        out.append(f"**{v['decision']}** with {key}")
        out += [f"- {r}" for r in v["reasons"]]
        out.append("")
    st = dec.get("stats") or {}
    for side in ("new", "live"):
        s = st.get(side)
        if s:
            out.append(f"- {side}: median depth error {s['median_pct']:+.2f} %, median |depth error| "
                       f"{s['depth']:.3f} %, spread (IQR) {s['spread']:.3f} %, rows {s['rows']:.3f} px, "
                       f"{s['pairs']} pairs")
    out.append("")
    pr = dec.get("paired")
    if pr is None:
        out += ["No paired statistics: the live config was not scored.", ""]
        return out
    out += ["## Paired statistics (new - live on the same TEST pairs)", "",
            "How to read it:",
            "- depth: median |depth error| (%), the median of the absolute errors; spread: interquartile range of the "
            "signed depth error (%); rows: median |row error| after rectification (px); band lo-hi: median |depth "
            "error| (%) of the pairs whose board is lo-hi m away",
            "- new - live below zero: the new fit is better; the 95 % interval comes from a bootstrap that resamples "
            "whole placements",
            "- judged: enough pairs and placements for an interval (else the reason)", "",
            "| statistic | new | live | new - live | 95 % interval | pairs | placements | judged |",
            "|---|---|---|---|---|---|---|---|"]
    for s in pr["stats"]:
        ci = f"{s['ci'][0]:+.3f} .. {s['ci'][1]:+.3f}" if s.get("ci") else "-"
        out.append(f"| {s['name']} | {_num(s['new'])} | {_num(s['live'])} | {_num(s['diff'], '{:+.3f}')} | {ci} | "
                   f"{s['n_pairs']} | {s['n_placements']} | {'yes' if s['judged'] else s['why']} |")
    out.append("")
    return out


class _Now:
    """an executor that runs each call at once (jobs=1: one process at a time)"""

    def submit(self, fn, *a, **kw):
        f = Future()
        try:
            f.set_result(fn(*a, **kw))
        except Exception as e:      # noqa: BLE001  (re-raised by .result())
            f.set_exception(e)
        return f

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def evaluate(session, rig_path, split: pipeline.Split, live_config, out_dir, gates: decide.Gates, jobs=2,
             max_fit_z=3.0, solver=pipeline.DEFAULT_SOLVER) -> dict:
    """step 45 on the split, step 66 (new and live), decide.score of both, decision(). jobs: processes at a time
    (2: the step-66 runs in a second process while this one scores); solver: pipeline.SOLVERS, for every step"""
    pipeline.check_solver(solver)
    out_dir = Path(out_dir)
    rig = pipeline.load_rig(rig_path)
    tm = {}
    t = time.time()
    fit = pipeline.fit45(split.fit_npz, rig_path, out_dir / "fit", max_z=max_fit_z, solver=solver)
    tm["fit45_s"] = time.time() - t
    live, live_err = None, None
    try:
        live = livecfg.load(live_config, at_size=rig.resolution)
    except NOT_SCORABLE as e:   # a config the package cannot score; anything else is a bug and stops the run
        live_err = f"live config not loaded: {type(e).__name__}: {e}"
    res = {"fit": fit.as_dict(), "validate_new": None, "validate_live": None,
           "live": {"config": str(live_config), "model": live.model if live else None,
                    "size": list(live.size) if live else None,
                    "scaled_from": list(live.scaled_from) if live and live.scaled_from else None,
                    "scorable": False, "error": live_err}}
    t = time.time()
    if not fit.converged:
        v_new = SimpleNamespace(verdict="NO RESULT", text=f"step 45 did not converge: {fit.verdict}", entries={})
        dec = decision(None, None, v_new, gates)
        res["validate_new"] = {"verdict": "NO RESULT", "text": v_new.text}
    else:
        place = decide.placements(split.test_npz, session)
        new_sc = live_sc = None
        with (_Now() if jobs <= 1 else ThreadPoolExecutor(max(1, jobs - 1))) as ex:
            f_new = ex.submit(pipeline.validate66, fit.json_path, split.test_npz, out_dir / "validate-new", rig_path,
                              fit_report=fit.json_path, solver=solver)
            f_live = ex.submit(livecfg.validate66, live, split.test_npz, out_dir / "validate-live", rig_path,
                               solver=solver) if live else None
            new_sc = decide.score(fit.json_path, split.test_npz, rig_path, place, solver=solver)
            if live:
                try:
                    js = live.write_json(out_dir / f"live-{live.model.lower()}.json")
                    live_sc = decide.score(js, split.test_npz, rig_path, place, solver=solver)
                except NOT_SCORABLE as e:
                    live_err = f"live config not scored: {type(e).__name__}: {e}"
            v_new = f_new.result()
            v_live = None
            if f_live is not None:
                try:
                    v_live = f_live.result()
                except NOT_SCORABLE as e:
                    live_err = live_err or f"step 66 on the live config: {type(e).__name__}: {e}"
        res["validate_new"] = v_new.as_dict()
        res["validate_live"] = v_live.as_dict() if v_live else None
        res["live"].update(scorable=live_sc is not None, error=live_err)
        dec = decision(new_sc, live_sc, v_new, gates, live_error=live_err)
    tm["validate_score_s"] = time.time() - t
    res.update(dec)
    res["timings"] = tm
    return _js(res)


# ---------------------------------------------------------------- the whole replay


def load_session(stored, exclude_takes=()) -> Session:
    """the stored session's corner files in its take order, without exclude_takes (ValueError for a take it lacks)"""
    unknown = [t for t in exclude_takes if t not in stored.take_order]
    if unknown:
        raise ValueError(f"exclude_takes {unknown} not in session {stored.name!r}'s takes {list(stored.take_order)}")
    files = dict(zip(stored.take_order, stored.corner_files))
    keep = [t for t in stored.take_order if t not in exclude_takes]
    name = stored.name + (f" without {', '.join(exclude_takes)}" if exclude_takes else "")
    return Session.from_files([(t, files[t]) for t in keep], name=name)


def run(stored, assignment="capture", out_dir=None, gates=None, exclude_takes=(), *, fit=True, targets=None, k=3,
        jobs=2, head=None, log=None, stop_at_rule=False, per_hold=cv.STORED_PER_HOLD,
        solver=pipeline.DEFAULT_SOLVER) -> dict:
    """replay a stored session (sessions.StoredSession or its name; anything with name, robot, rig_path, take_order,
    corner_files, live_config, live_model) -> the report (dict, as written to out_dir/report.json; report.md beside
    it). assignment: "capture" (assign.Assigner) | "split16" (step 16's split, the recorded runs'). fit=False stops
    after the planner (no split written, no fit). head: placement.Head for the planner (default: from the live
    config). jobs: processes at a time in the fit/validate part. per_hold: pairs per pose of the session's corner
    files (targets default to coverage.Targets.for_per_hold(per_hold); the planner simulates that many pairs a pose).
    stop_at_rule: cut the session at the stop rule's point (Collection.stop; the whole session when never met).
    solver: pipeline.SOLVERS ("fast", calibtool's default; "scipy": the recorded runs' least squares)"""
    pipeline.check_solver(solver)
    t_all = time.time()
    if isinstance(stored, str):
        stored = sessions.get(stored)
    if assignment not in ASSIGNMENTS:
        raise ValueError(f"assignment={assignment!r}: one of {ASSIGNMENTS}")
    exclude_takes = tuple(exclude_takes)
    gates = gates or decide.Gates()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tm = {}

    t = time.time()
    session = load_session(stored, exclude_takes)
    rig = pipeline.load_rig(stored.rig_path)
    targets = targets or cv.Targets.for_per_hold(per_hold)
    col = Collection.for_session(session, rig, assignment, targets=targets)
    holds = list(assign.capture_holds(session))
    snaps = feed(col, holds, log=None if stop_at_rule else log)
    stop = {**(col.stop.as_dict() if col.stop else {"index": None}), "holds_total": len(holds), "stopped": False}
    if stop_at_rule and col.stop is not None:
        session = session.subset([h.hold for h in holds[:col.stop.index]], name=f"{session.name} (stop rule)")
        col = Collection.for_session(session, rig, assignment, targets=targets)
        snaps = feed(col, assign.capture_holds(session), log=log)
        stop["stopped"] = True
    tm["collect_s"] = time.time() - t

    t = time.time()
    head = head or pl.head_from_okvis(stored.live_config, rig.resolution)
    sug = plan(snaps[-1].deficits, head, rig.board, pp=col.pp, k=k, pairs_per_pose=per_hold)
    tm["plan_s"] = time.time() - t
    final = col.finish()

    rep = {"session": stored.name, "robot": getattr(stored, "robot", ""), "assignment": assignment,
           "exclude_takes": list(exclude_takes), "take_order": [x for x in stored.take_order if x not in exclude_takes],
           "capture_order": [s.placement for s in snaps], "placements": [s.as_dict() for s in snaps],
           "final": final, "suggestions": [suggestion_dict(s) for s in sug], "split": None, "fit": None,
           "validate_new": None, "validate_live": None,
           "live": {"config": str(stored.live_config), "model": getattr(stored, "live_model", None), "scorable": None,
                    "error": None},
           "stats": None, "paired": None, "verdicts": None, "timings": tm, "stop": stop, "per_hold": per_hold,
           "targets": asdict(targets), "solver": solver}
    if fit:
        t = time.time()
        sd = out_dir / "split"
        split = col.labeler.write_split(session, sd) if assignment == "capture" else \
            pipeline.split16(session, sd, rig, max_fit_z=col.max_fit_z, holdout_every=2)
        tm["split_s"] = time.time() - t
        rep["split"] = {"fit_pairs": split.fit_pairs, "test_pairs": split.test_pairs,
                        "near_test_pairs": split.near_test_pairs, "far_test_pairs": split.far_test_pairs,
                        "fit_npz": str(split.fit_npz), "test_npz": str(split.test_npz)}
        ev = evaluate(session, stored.rig_path, split, stored.live_config, out_dir, gates, jobs=jobs,
                      max_fit_z=col.max_fit_z, solver=solver)
        tm.update(ev.pop("timings"))
        rep.update(ev)
    tm["total_s"] = time.time() - t_all
    rep = _js(rep)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    (out_dir / "report.md").write_text(render_md(rep))
    return rep


def render_md(rep: dict) -> str:
    """report.md: the collection placement by placement (progress line, gap sentences), the planner's next
    placements, fit and validation, the decision with its reasons and the paired statistics"""
    L = [f"# Calibration replay: {rep['session']} ({rep['robot']})", "",
         f"Assignment: {rep['assignment']}. Takes: {', '.join(rep['take_order'])}"
         + (f" (left out: {', '.join(rep['exclude_takes'])})" if rep["exclude_takes"] else "")
         + f". Capture order: {', '.join(rep['capture_order'])}.", "",
         "## Collection, placement by placement", "",
         "Each placement: one progress line, then what is still short, most important first (left camera = OKVIS "
         "camera 0; directions name the operator's side first, the robot's in brackets).", ""]
    for p in rep["placements"]:
        L.append(f"- {p['progress']}")
        for s in p["sentences"][:MD_SENTENCES]:
            L.append(f"  - {s}")
        if len(p["sentences"]) > MD_SENTENCES:
            L.append(f"  - (+{len(p['sentences']) - MD_SENTENCES} more in report.json)")
    st = rep.get("stop") or {}
    if st.get("index"):
        L += ["", f"Stop rule met after hold {st['index']} of {st['holds_total']}: placement {st['placement_index']} "
                  f"({st['placement']}), pose {st['pose']} (neck pitch {st['neck_pitch_deg']:+.1f}, yaw "
                  f"{st['neck_yaw_deg']:+.1f} deg)" + ("; the session was cut there." if st["stopped"] else ".")]
    else:
        L += ["", f"Stop rule not met in the {st.get('holds_total', 0)} holds."]
    f = rep["final"]
    L += ["", f"At the end (assignment closed): {f['pairs']:,} pairs, FIT {f['fit_pairs']:,}, TEST "
              f"{f['test_pairs']:,}; {f['released_to_test']} held-back holds released to TEST; held-out entries "
              f"ready: {len(f['ready'])} ({', '.join(f['ready']) or 'none'}).", "",
          "## Next placements (planner, after the last placement)", ""]
    if rep["suggestions"]:
        L += [f"{i}. {s['say']}" for i, s in enumerate(rep["suggestions"], 1)]
    else:
        L.append("None: no candidate placement reduces a deficit.")
    L += ["", "## Fit and validation", "", f"Least squares: {rep.get('solver', 'fast')} solver.", ""]
    if rep["fit"] is None:
        L += ["Fit not run (collection-only replay).", ""]
    else:
        sp = rep["split"]
        L.append(f"- split: FIT {sp['fit_pairs']} pairs, TEST {sp['test_pairs']} pairs ({sp['far_test_pairs']} "
                 f"beyond the fit's distance)")
        L.append(f"- step 45: {rep['fit']['verdict']}")
        vn = rep["validate_new"] or {}
        L.append(f"- step 66, new fit: {vn.get('text') or vn.get('verdict')}")
        lv = rep["live"]
        vl = rep["validate_live"]
        if vl:
            L.append(f"- step 66, live config ({lv['model']}, {Path(lv['config']).name}): {vl.get('text')}")
        if lv.get("error"):
            L.append(f"- live config: {lv['error']}")
        L.append("")
        L += decision_md({"verdicts": rep["verdicts"], "paired": rep["paired"], "stats": rep["stats"]})
    tm = rep["timings"]
    L += ["## Timings", "", ", ".join(f"{k[:-2]} {v:.0f} s" for k, v in tm.items()), ""]
    return "\n".join(L)
