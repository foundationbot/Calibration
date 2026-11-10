"""FIT/TEST assigned at capture time: each hold is marked FIT or TEST as it is captured, stratified per step-66 entry,
so the tool knows the TEST side while the session runs (design plan section 5; step 16 is the replay reference only).

What a hold is measured by, at capture time (the fitted lens is not known yet):
  distance  per pair, step 16's rough distance (16_split_distance.rough_z: PnP of the camera-0 corners with a plain
            equidistant lens at the nominal focal length and principal point); the hold's distance = the median of
            its near pairs
  cell      lenscells.cell_of of the mean camera-0 corner of its near pairs, relative to the NOMINAL principal point
            (step 66 uses the fitted one: a hold near a cell border can change cell after the fit)
  entry     (cell, (lo, hi)): the step-66 distance bins (lenscells.DIST_BINS, [lo, hi)) and, outside them, the bands
            NEAR_BAND and MID_BANDS (stratified the same way, judged per band by the summary row, not by step 66)

The rule, per hold in capture order:
  1. a pair beyond max_fit_z, or with no distance (too few corners), is TEST, as in step 16; a hold whose pairs are
     all such is TEST and feeds no entry
  2. otherwise the hold joins its entry: k = holds of the entry so far (this one included), t = its TEST holds
     before this one, N = holdout_every
     a. hold back: while fewer than min_placements placements have fed the entry and t >= ceil(target_test / 2),
        FIT (the entry's TEST side waits for another placement: holds of one placement are correlated with its own
        FIT holds)
     b. share: TEST when t < k / N (the entry's TEST count is ceil(k / N): an odd entry gives TEST the extra hold)
     c. new placement: TEST when the hold's placement has no TEST hold in the entry yet and t <= k / N (so a second
        placement's first hold is TEST, one hold ahead of the share at most)
     d. FIT otherwise
  3. finish() (write_split calls it), the end of the session: an entry still short of its share because it held
     holds back (2a) gets that many of its held-back holds released to TEST, spread evenly over them in capture
     order (release_at_end=True; False keeps them FIT). Holds observe() returned FIT can thus end up TEST; the
     final labels are self.holds[h].label, the capture-time ones observe()'s return values.
So every entry ends with |TEST - holds / N| <= 1, and an entry fed by >= 2 placements tests at least two of them (for
N = 2; for larger N when the second placement brings enough holds).
"""
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from calibtool import paths, pipeline

NEAR_BAND = (0.0, 0.4)
MID_BANDS = ((1.3, 1.7), (1.7, math.inf))
FIT, TEST = "FIT", "TEST"


@dataclass(frozen=True)
class CaptureHold:
    """one hold as the capture delivers it: its global hold id, placement (the take: one board placement), first left
    stamp, and its pairs as a step-15b corner dict (pairs numbered from 0, img_hold = the global hold id)"""
    hold: int
    placement: str
    t0: int
    data: dict


def capture_holds(session):
    """the session's holds in capture order (Session.holds), each with its pairs and corners"""
    for h in session.holds():
        m = np.zeros(session.n_pairs, bool)
        m[h.pairs] = True
        yield CaptureHold(hold=h.hold, placement=h.take, t0=h.t0, data=session.to_npz_dict(m))


@dataclass
class HoldInfo:
    hold: int
    placement: str
    t0: int
    pair_t0: np.ndarray         # left stamps of its pairs
    pair_z: np.ndarray          # rough distance per pair (nan: none)
    near: np.ndarray            # per pair: within max_fit_z
    u: float                    # mean camera-0 corner of its near pairs (nan without one)
    v: float
    z: float                    # median rough distance of its near pairs (nan without one)
    entry: tuple                # (cell, (lo, hi)) or None (no near pair)
    label: str                  # FIT | TEST: the final label (after finish())
    captured: str = ""          # FIT | TEST: what observe() returned
    held_back: bool = False     # FIT by rule 2a
    released: bool = False      # held back, then released to TEST by finish()


@dataclass
class EntryCount:
    holds: int = 0
    fit: int = 0
    test: int = 0
    placements: int = 0         # placements that fed the entry
    test_placements: int = 0    # placements with a TEST hold in it
    held_back: int = 0          # holds rule 2a kept FIT at capture
    released: int = 0           # of those, released to TEST by finish()
    placement_names: list = field(default_factory=list)


def entry_of(cell, z):
    """(cell, (lo, hi)) for a hold at rough distance z"""
    import lenscells
    for lo, hi in (NEAR_BAND,) + tuple(lenscells.DIST_BINS) + MID_BANDS:
        if lo <= z < hi:
            return (str(cell), (lo, hi))
    return (str(cell), MID_BANDS[-1])


class Assigner:
    def __init__(self, rig, pp_nominal, focal_nominal, max_fit_z=3.0, target_test=10, min_placements=2,
                 holdout_every=2, release_at_end=True):
        self.rig = pipeline.load_rig(rig)
        self.pp = (float(pp_nominal[0]), float(pp_nominal[1]))
        self.focal = float(focal_nominal)
        self.max_fit_z = float(max_fit_z)
        self.target_test = int(target_test)
        self.min_placements = int(min_placements)
        self.holdout_every = int(holdout_every)
        self.release_at_end = bool(release_at_end)
        self.cap = math.ceil(self.target_test / 2)
        self.finished = False
        if self.holdout_every < 1:
            raise ValueError("holdout_every must be >= 1")
        self.holds = {}             # hold id -> HoldInfo, in observation order
        self._entries = {}          # entry -> [hold ids] in observation order
        self._d16 = paths.import_script("16_split_distance")

    # ------------------------------------------------------------ measuring a hold

    def _measure(self, h):
        import lenscells
        d = h.data
        # rough_z puts the principal point at resolution / 2: pass twice the nominal principal point
        z = self._d16.rough_z(d, self.rig.board, self.focal, (2 * self.pp[0], 2 * self.pp[1]))
        near = np.nan_to_num(z, nan=np.inf) <= self.max_fit_z
        img, cam = np.asarray(d["det_img"]), np.asarray(d["det_cam"])
        m = (cam == 0) & np.isin(img, np.flatnonzero(near))
        if near.any() and m.any():
            u, v = float(np.mean(d["det_u"][m])), float(np.mean(d["det_v"][m]))
            zh = float(np.median(z[near]))
            cell = lenscells.cell_of(u - self.pp[0], v - self.pp[1])[0]
            entry = entry_of(cell, zh)
        else:
            u = v = zh = float("nan")
            entry = None
        return z, near, u, v, zh, entry

    # ------------------------------------------------------------ the rule

    def observe(self, hold) -> str:
        """FIT or TEST for one hold (CaptureHold), called once per hold in capture order"""
        if self.finished:
            raise RuntimeError("observe() after finish(): the session is closed")
        if hold.hold in self.holds:
            raise ValueError(f"hold {hold.hold} observed twice")
        z, near, u, v, zh, entry = self._measure(hold)
        held = False
        if entry is None:
            label = TEST
        else:
            ids = self._entries.setdefault(entry, [])
            rows = [self.holds[i] for i in ids]
            k, N = len(ids) + 1, self.holdout_every
            t = sum(r.label == TEST for r in rows)
            placements = {r.placement for r in rows} | {hold.placement}
            tested = {r.placement for r in rows if r.label == TEST}
            if len(placements) < self.min_placements and t >= self.cap:
                label, held = FIT, True                                     # 2a
            elif t < k / N:
                label = TEST                                                # 2b
            elif hold.placement not in tested and t <= k / N:
                label = TEST                                                # 2c
            else:
                label = FIT                                                 # 2d
            ids.append(hold.hold)
        self.holds[hold.hold] = HoldInfo(hold.hold, hold.placement, int(hold.t0),
                                         np.asarray(hold.data["img_t0"], np.int64), z, near, u, v, zh, entry, label,
                                         captured=label, held_back=held)
        return label

    def finish(self):
        """close the session: release held-back holds of entries short of their share (rule 3). Idempotent"""
        if self.finished:
            return
        self.finished = True
        if not self.release_at_end:
            return
        for ids in self._entries.values():
            rows = [self.holds[i] for i in ids]
            need = math.ceil(len(rows) / self.holdout_every) - sum(r.label == TEST for r in rows)
            hb = [r for r in rows if r.held_back]
            need = min(need, len(hb))
            for i in range(need):
                r = hb[int((i + 0.5) * len(hb) / need)]
                r.label, r.released = TEST, True

    def entry_counts(self) -> dict:
        """{entry: EntryCount} with the current labels (final after finish())"""
        out = {}
        for e, ids in self._entries.items():
            rows = [self.holds[i] for i in ids]
            names = list(dict.fromkeys(r.placement for r in rows))
            test = sum(r.label == TEST for r in rows)
            out[e] = EntryCount(len(rows), len(rows) - test, test, len(names),
                                len({r.placement for r in rows if r.label == TEST}),
                                sum(r.held_back for r in rows), sum(r.released for r in rows), names)
        return out

    # ------------------------------------------------------------ results

    def recell(self, pp) -> dict:
        """{hold: (capture-time cell, cell against pp)} for the holds whose cell differs against principal point pp
        (after the fit: the fitted one, which step 66 uses)"""
        import lenscells
        out = {}
        for h in self.holds.values():
            if h.entry is None:
                continue
            c = str(lenscells.cell_of(h.u - pp[0], h.v - pp[1])[0])
            if c != h.entry[0]:
                out[h.hold] = (h.entry[0], c)
        return out

    def pair_labels(self):
        """{left stamp: FIT | TEST} per observed pair: a far or unmeasured pair is TEST whatever its hold"""
        out = {}
        for hi in self.holds.values():
            for t, n in zip(hi.pair_t0, hi.near):
                out[int(t)] = hi.label if n else TEST
        return out

    def write_split(self, session, out_dir) -> pipeline.Split:
        """out_dir/fit.npz and out_dir/test.npz in step 16's layout (global hold ids, img_z = the rough distance)"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        lab = self.pair_labels()
        t0 = session.pairs["t0"]
        missing = [int(t) for t in t0 if int(t) not in lab]
        if missing:
            raise ValueError(f"{len(missing)} pairs of session {session.name!r} were not observed (first left stamp "
                             f"{missing[0]}): feed every hold before writing the split")
        self.finish()
        lab = self.pair_labels()
        zof = {int(t): float(z) for hi in self.holds.values() for t, z in zip(hi.pair_t0, hi.pair_z)}
        is_test = np.array([lab[int(t)] == TEST for t in t0])
        zz = np.array([zof[int(t)] for t in t0])
        files = []
        for name, mask in (("fit", ~is_test), ("test", is_test)):
            d = session.to_npz_dict(mask)
            out = {k: d[k] for k in ("img_hold", "img_t0", "img_t1")}
            out["img_z"] = zz[mask]
            out.update({k: d[k] for k in ("det_img", "det_cam", "det_tag", "det_k", "det_u", "det_v")})
            f = out_dir / f"{name}.npz"
            np.savez_compressed(f, **out)
            files.append(f)
        far = is_test & ~(np.nan_to_num(zz, nan=np.inf) <= self.max_fit_z)
        return pipeline.Split(files[0], files[1], int((~is_test).sum()), int(is_test.sum()),
                              int((is_test & ~far).sum()), int(far.sum()), [])
