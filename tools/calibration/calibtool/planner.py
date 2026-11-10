"""The placement planner: which board placement to ask for next, and which neck poses of it to sweep.

Candidates: the board standing at distance {0.5, 0.75, 1.1, 1.6, 2.0} m (along the floor from camera 0) x bearing
{-40, -20, 0, 20, 40} deg (+: toward the robot's LEFT = the operator's right), plus the tilted-low placement (on the
chair seat leaning back against the backrest, facing up toward the head) at 0.75 m on each bearing; every board faces
the head. Heights: an upright board's centre UPRIGHT_DZ below camera 0, the tilted one's TILT_DZ below (measured on
the stored takes' first poses, 9 Nov 2026: upright p21 -0.08..-0.50 m, p10 -0.11..-0.50 m, median about -0.35 m;
tilted p21 dtiltleft -0.85 m, p10 n6_tiltL/R -1.19/-1.16 m).

Each candidate is simulated over the 49-pose neck grid (pitch -24..24 deg step 8 x yaw -34..34 deg in 7 steps) with
the placement model (placement.BoardModel): per pose, both cameras' predicted corners (x pairs_per_pose stereo pairs),
the far pairs, and the step-66 entry / far band of the camera-0 board. suggest() gives one suggestion per top deficit
(C3): each pick serves the most important deficit no earlier pick served, with the candidate predicted to fill most of
it; the summed weighted deficit reduction breaks ties (each deficit's gain capped at what it is short, divided by its
target, times its weight, coverage.Targets: edge strips 3, corner boxes 2, the rest mostly 1); the deficits are
updated with each pick's predicted gains."""
from dataclasses import dataclass, field, replace

import numpy as np

from calibtool import coverage as cv, placement as pl
from calibtool.protocol import PER_HOLD

DISTANCES = (0.5, 0.75, 1.1, 1.6, 2.0)
BEARINGS = (-40, -20, 0, 20, 40)
TILT_DISTANCE = 0.75
UPRIGHT_DZ = -0.35
TILT_DZ = -1.0
GRID_PITCH = tuple(float(v) for v in np.arange(-24, 25, 8))
GRID_YAW = tuple(float(v) for v in np.linspace(-34, 34, 7))
PAIRS_PER_POSE = PER_HOLD   # stereo pairs kept per still pose: the protocol's 2 (Planner(pairs_per_pose=3) for the
                            # stored sessions' 3)
TEST_FRAC = 0.5             # share of a placement's holds that go to TEST (holdout_every = 2)


def grid() -> list:
    """the 49 neck poses, (pitch, yaw) in degrees, pitch-major"""
    return [(p, y) for p in GRID_PITCH for y in GRID_YAW]


@dataclass(frozen=True)
class Candidate:
    distance_m: float           # along the floor from camera 0
    bearing_deg: int            # + toward the robot's left (the operator's right)
    tilted: bool

    def describe(self) -> str:
        if self.bearing_deg == 0:
            where = "straight ahead"
        else:
            side = cv.SIDE["left" if self.bearing_deg > 0 else "right"]
            where = f"{abs(self.bearing_deg)} degrees {side}"
        if self.tilted:
            return (f"Board low and tilted: on the chair seat leaning back against the backrest, facing up toward the "
                    f"head, {self.distance_m:g} m out along the floor, {where}.")
        return f"Board standing {self.distance_m:g} m out along the floor, {where}, facing the head."


def candidates() -> list:
    out = [Candidate(d, b, False) for d in DISTANCES for b in BEARINGS]
    return out + [Candidate(TILT_DISTANCE, b, True) for b in BEARINGS]


def _scaled(t: cv.CornerTally, n: int) -> cv.CornerTally:
    return cv.CornerTally(t.n * n, t.thirds * n, t.rim * n, {k: v * n for k, v in t.edges.items()},
                          {k: v * n for k, v in t.boxes.items()})


def _empty_fit() -> cv.FitTally:
    z = cv.corner_tally(np.zeros(0), np.zeros(0), (1, 1))
    return cv.FitTally({0: z, 1: z}, 0, 0)


@dataclass
class PoseSim:
    pose: tuple                 # (pitch, yaw) deg
    detected: bool
    fit: cv.FitTally
    entries: dict               # {(cell, (lo, hi)): holds}
    far_bands: dict             # {(lo, hi): holds}


@dataclass
class Sim:
    candidate: Candidate
    poses: list

    @property
    def n_detected(self) -> int:
        return sum(p.detected for p in self.poses)

    @property
    def fit(self) -> cv.FitTally:
        out = _empty_fit()
        for p in self.poses:
            out = out + p.fit
        return out

    @property
    def entries(self) -> dict:
        return _sum_dicts(p.entries for p in self.poses)

    @property
    def far_bands(self) -> dict:
        return _sum_dicts(p.far_bands for p in self.poses)


def _sum_dicts(ds) -> dict:
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def gain(fit: cv.FitTally, entries: dict, far_bands: dict, d: cv.Deficit, test_holds=10) -> float:
    """what a (simulated) contribution adds to deficit d, in d's units"""
    if d.kind in ("edge", "strip", "box", "third", "rim"):
        t = fit.cams[d.cam]
        if d.kind in ("edge", "strip"):
            return t.edges[d.region]
        if d.kind == "box":
            return t.boxes[d.region]
        if d.kind == "third":
            i, j = cv.CORNER_THIRDS[d.region]
            return t.thirds[i, j]
        return t.rim
    if d.kind == "pairs":
        return fit.n_pairs
    if d.kind == "far":
        return fit.far_pairs
    if d.kind == "entry":
        return entries.get(d.region, 0) * TEST_FRAC
    if d.kind == "side":
        return max([n for (c, _), n in entries.items() if d.region in cv.edges_of_cell(c)], default=0) * TEST_FRAC
    if d.kind == "judged":
        return sum(n * TEST_FRAC >= test_holds for n in entries.values())
    if d.kind == "far_band":
        return far_bands.get(d.region, 0) * TEST_FRAC
    raise ValueError(f"unknown deficit kind {d.kind!r}")


def _value(g, d: cv.Deficit) -> float:
    """weighted share of d's target that gain g fills (capped at what d is short)"""
    return d.weight * min(g, d.short) / max(d.want, 1e-9)


@dataclass
class Suggestion:
    candidate: Candidate
    score: float
    target: cv.Deficit          # the deficit it fills most (weighted)
    fills: list                 # deficits it adds to, most first (fills[0] is target)
    gains: list                 # predicted gain for each, in the deficit's units
    poses: list = field(default_factory=list)     # (pitch, yaw) deg predicted to fill the target

    def say(self) -> str:
        t = self.target
        what = cv.sentence(t).split(":")[0]
        return f"{self.candidate.describe()} For: {what} (about {int(round(self.gains[0])):,} more predicted)."


class Planner:
    """candidates simulated over the grid for one head (placement.Head), neck chain and board"""

    def __init__(self, head: pl.Head, chain_, board, pp=None, upright_dz=UPRIGHT_DZ, tilt_dz=TILT_DZ,
                 cands=None, pairs_per_pose=None):
        self.head, self.chain, self.board = head, chain_, board
        self.pairs_per_pose = int(pairs_per_pose or PAIRS_PER_POSE)
        K = head.cams[0].K
        self.pp = (float(K[0, 2]), float(K[1, 2])) if pp is None else tuple(pp)
        self.upright_dz, self.tilt_dz = upright_dz, tilt_dz
        self.candidates = list(cands) if cands is not None else candidates()
        self._sims = {}

    def model(self, c: Candidate) -> pl.BoardModel:
        T = pl.facing_pose(self.chain, self.head, self.board, c.distance_m, c.bearing_deg,
                           self.tilt_dz if c.tilted else self.upright_dz)
        return pl.BoardModel(T, self.board, self.head, self.chain)

    def simulate(self, c: Candidate) -> Sim:
        if c in self._sims:
            return self._sims[c]
        import lenscells
        m = self.model(c)
        poses = []
        for p, y in grid():
            qp, qy = np.radians(p), np.radians(y)
            p0 = m.predict(qp, qy, 0)
            if not p0.detected:
                poses.append(PoseSim((p, y), False, _empty_fit(), {}, {}))
                continue
            size = self.head.cams[0].size
            cams = {0: _scaled(cv.corner_tally(p0.uv[:, 0], p0.uv[:, 1], size), self.pairs_per_pose)}
            if len(self.head.cams) > 1:
                p1 = m.predict(qp, qy, 1)
                t1 = cv.corner_tally(p1.uv[:, 0], p1.uv[:, 1], self.head.cams[1].size) if p1.detected else \
                    cv.corner_tally(np.zeros(0), np.zeros(0), size)
                cams[1] = _scaled(t1, self.pairs_per_pose)
            else:
                cams[1] = cv.corner_tally(np.zeros(0), np.zeros(0), size)
            far = self.pairs_per_pose if cv.FAR_FIT[0] <= p0.z < cv.FAR_FIT[1] else 0
            fit = cv.FitTally(cams, far, self.pairs_per_pose)
            entries, bands = {}, {}
            b = int(lenscells.dist_bin(p0.z)[0])
            if b >= 0:
                cell = str(lenscells.cell_of(p0.centroid[0] - self.pp[0], p0.centroid[1] - self.pp[1])[0])
                entries[(cell, lenscells.DIST_BINS[b])] = 1
            for lo, hi in cv.FAR_BANDS:
                if lo <= p0.z < hi:
                    bands[(lo, hi)] = 1
            poses.append(PoseSim((p, y), True, fit, entries, bands))
        self._sims[c] = Sim(c, poses)
        return self._sims[c]

    def _gains(self, c: Candidate, deficits) -> list:
        s = self.simulate(c)
        fit, ent, far = s.fit, s.entries, s.far_bands
        return [gain(fit, ent, far, d) for d in deficits]

    def suggest(self, deficits, k=3) -> list:
        """up to k placements, one per top deficit (C3): pick i serves the first deficit, in the given order
        (Coverage.deficits: most important first), that is still short after the earlier picks' predicted gains and
        that no earlier pick served; once every short deficit has been served, a new round starts. Its placement is
        the unused candidate predicted to add most to that deficit (not capped at what it is short: of two boards
        that both cover it, the one that fills more), ties broken by the summed weighted value over all deficits
        (Suggestion.score). A deficit no candidate adds to is skipped.
        So a deficit an earlier pick already addressed does not dominate the next pick, and a bottom deficit gets the
        tilted board when that is its best fill"""
        left = [replace(d) for d in deficits if d.short > 0]
        out, used, served, skip = [], set(), set(), set()
        while len(out) < k:
            short = [i for i, d in enumerate(left) if d.short > 0 and i not in skip]
            if not short:
                break
            todo = [i for i in short if i not in served] or short
            if len(todo) == len(short):
                served = set()
            ti = todo[0]
            best = None
            for c in self.candidates:
                if c in used:
                    continue
                g = self._gains(c, left)
                key = (g[ti], sum(_value(gi, d) for gi, d in zip(g, left)))
                if key[0] > 0 and (best is None or key > best[0]):
                    best = (key, c, g)
            if best is None:
                skip.add(ti)
                continue
            (_, score), c, g = best
            order = [ti] + sorted([i for i in range(len(left)) if g[i] > 0 and i != ti],
                                  key=lambda i: -_value(g[i], left[i]))
            fills = [replace(left[i]) for i in order]
            out.append(Suggestion(c, score, fills[0], fills, [g[i] for i in order], self.pose_subset(c, fills[0])))
            used.add(c)
            served.add(ti)
            for i, d in enumerate(left):
                d.have = d.have + g[i]
        return out

    def pose_subset(self, c: Candidate, deficit: cv.Deficit) -> list:
        """the grid poses (pitch, yaw) deg of candidate c predicted to add to deficit"""
        return [p.pose for p in self.simulate(c).poses
                if p.detected and gain(p.fit, p.entries, p.far_bands, deficit) > 0]
