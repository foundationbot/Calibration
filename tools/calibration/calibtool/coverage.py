"""Coverage of a calibration session: what the FIT side and the TEST side hold, what is still short (deficits), and the
gap sentences the operator hears.

FIT tally, per camera (0 = left = OKVIS cam0, 1 = right), over the corners of the FIT pairs, with the definitions of
the stored kit scripts (p21/kit/cov21.py, edge21.py; p10/kit/cov1280.py; targets of COLLECT-lens-data.md):
  thirds        3 x 3 image thirds, row = min(int(v / H * 3), 2), col = min(int(u / W * 3), 2)
  rim           corners with r / half-diagonal > 0.8, r from the image centre (W/2, H/2)
  edge strips   corners within EDGE_PX = 80 px of each image edge (u < 80, u > W - 80, v < 80, v > H - 80)
  corner boxes  corners in the EDGE_BOX_PX = 160 px square at each image corner
  far pairs     pairs whose rough distance (step 16's rough_z) is in FAR_FIT = [1.7, 3.0) m
TEST tally: per step-66 entry (lenscells cell of the pair's camera-0 mean corner vs the principal point x DIST_BINS on
the rough distance) the number of TEST holds, and per far band (1.3-1.7, 1.7-2.3 m) the TEST holds.

Directions (the rule that was misread on real robots): the head camera looks forward and image x grows toward the
ROBOT's right, so an image-left deficit is filled by a board toward the robot's LEFT, which is the OPERATOR's right
(the operator faces the robot). Every sentence names the operator's side first, the robot's in brackets:
"to YOUR right (the robot's left)"; never "robot's right" alone. Image top is filled by a higher board, image bottom
by a lower one (the tilted placement: leaning back on the chair, facing up toward the head)."""
from dataclasses import dataclass, field

import numpy as np

from calibtool.protocol import PER_HOLD

EDGE_PX = 80
EDGE_BOX_PX = 160
RIM_FRAC = 0.8
FAR_FIT = (1.7, 3.0)
FAR_BANDS = ((1.3, 1.7), (1.7, 2.3))
EDGES = ("left", "right", "top", "bottom")
BOXES = ("top-left", "top-right", "bottom-left", "bottom-right")
CORNER_THIRDS = {"top-left": (0, 0), "top-right": (0, 2), "bottom-left": (2, 0), "bottom-right": (2, 2)}
MIRROR = {"left": "right", "right": "left", "top": "bottom", "bottom": "top", "top-left": "top-right",
          "top-right": "top-left", "bottom-left": "bottom-right", "bottom-right": "bottom-left"}
CAMERA = {0: "Left camera", 1: "Right camera"}
EDGE_CELLS = ("left", "right", "top", "bottom")       # step-66 cells on each image side
# the image edges that need a judged entry (M1 C3, 9 Nov 2026, decided with the user): left, right and bottom block a
# deploy (decide.Gates.required_edges) and the stop rule (Targets.required_sides); the top is asked for during
# collection (a deficit and its sentence) but blocks neither. p21 5 Nov judged no top entry and deployed well
REQUIRED_EDGES = frozenset({"left", "right", "bottom"})


def edges_of_cell(cell) -> set:
    """the image edges a step-66 cell is on: a side cell its edge, a corner cell (top-left, ...) both of its edges,
    centre none. The one rule for "an entry on each image edge": coverage's side deficit, the planner's side gain and
    decide's verdict (decide.edges_judged) all use it"""
    return {e for e in EDGE_CELLS if e in str(cell).split("-")}


# the operator faces the robot: the robot's left is the operator's right
SIDE = {"left": "to YOUR right (the robot's left)", "right": "to YOUR left (the robot's right)"}
TILTED = "the tilted placement (leaning back on the chair, facing up toward the head)"


@dataclass
class CornerTally:
    n: int
    thirds: np.ndarray          # 3 x 3 int, [row (top, middle, bottom), col (left, centre, right)]
    rim: int
    edges: dict                 # {"left" | "right" | "top" | "bottom": corners}
    boxes: dict                 # {"top-left" | ...: corners}

    def __add__(self, o):
        return CornerTally(self.n + o.n, self.thirds + o.thirds, self.rim + o.rim,
                           {k: self.edges[k] + o.edges[k] for k in EDGES},
                           {k: self.boxes[k] + o.boxes[k] for k in BOXES})


def corner_tally(u, v, size) -> CornerTally:
    """the tallies of corners at pixels (u, v) in an image of size (W, H)"""
    W, H = size
    u, v = np.asarray(u, float).ravel(), np.asarray(v, float).ravel()
    g = np.zeros((3, 3), int)
    if u.size:
        np.add.at(g, (np.minimum((v / H * 3).astype(int), 2), np.minimum((u / W * 3).astype(int), 2)), 1)
    rim = int((np.hypot(u - W / 2, v - H / 2) / np.hypot(W / 2, H / 2) > RIM_FRAC).sum())
    E, B = EDGE_PX, EDGE_BOX_PX
    edges = {"left": int((u < E).sum()), "right": int((u > W - E).sum()), "top": int((v < E).sum()),
             "bottom": int((v > H - E).sum())}
    boxes = {"top-left": int(((u < B) & (v < B)).sum()), "top-right": int(((u > W - B) & (v < B)).sum()),
             "bottom-left": int(((u < B) & (v > H - B)).sum()),
             "bottom-right": int(((u > W - B) & (v > H - B)).sum())}
    return CornerTally(int(u.size), g, rim, edges, boxes)


@dataclass
class FitTally:
    cams: dict                  # {0: CornerTally, 1: CornerTally}
    far_pairs: int
    n_pairs: int

    def __add__(self, o):
        return FitTally({c: self.cams[c] + o.cams[c] for c in self.cams}, self.far_pairs + o.far_pairs,
                        self.n_pairs + o.n_pairs)


@dataclass
class TestTally:
    entries: dict               # {(cell, (lo, hi)): TEST holds}, only entries with data
    far_bands: dict             # {(lo, hi): TEST holds} for every FAR_BANDS band


def session_z(session, rig):
    """per pair of the session: step 16's rough camera-0 distance (m; nan where PnP fails)"""
    from calibtool import paths
    d16 = paths.import_script("16_split_distance")
    return d16.rough_z(session.to_npz_dict(), rig.board, rig.nominal_focal_px, rig.resolution)


def fit_tally(session, mask=None, rig=None, z=None, size=None) -> FitTally:
    """the FIT tally of the pairs in mask (bool per pair; None: all). z: per-pair distance (default: session_z with
    rig); size: (W, H) (default: rig.resolution)"""
    if size is None:
        size = rig.resolution
    if z is None:
        z = session_z(session, rig)
    keep = np.ones(session.n_pairs, bool) if mask is None else np.asarray(mask, bool)
    C = session.corners
    sel = keep[C["pair"]]
    cams = {c: corner_tally(C["u"][sel & (C["cam"] == c)], C["v"][sel & (C["cam"] == c)], size) for c in (0, 1)}
    zz = np.asarray(z, float)[keep]
    far = int(((zz >= FAR_FIT[0]) & (zz < FAR_FIT[1])).sum())
    return FitTally(cams, far, int(keep.sum()))


def pair_cells(session, pp):
    """per pair: lenscells cell of its camera-0 mean corner relative to the principal point pp (None: no corner)"""
    import lenscells  # noqa: F401  (scripts dir is on sys.path through calibtool.paths)
    C = session.corners
    m = C["cam"] == 0
    n = np.bincount(C["pair"][m], minlength=session.n_pairs)
    su = np.bincount(C["pair"][m], weights=C["u"][m], minlength=session.n_pairs)
    sv = np.bincount(C["pair"][m], weights=C["v"][m], minlength=session.n_pairs)
    with np.errstate(invalid="ignore", divide="ignore"):
        u, v = su / n, sv / n
    cells = lenscells.cell_of(u - pp[0], v - pp[1])
    cells[n == 0] = None
    return cells


def test_tally(session, assignment, pp, z) -> TestTally:
    """TEST holds per step-66 entry and per far band. assignment: {global hold id: "FIT" | "TEST"} (holds missing from
    it are not counted); pp: the principal point the cells are taken against; z: per-pair distance (m)"""
    from calibtool import paths  # noqa: F401
    import lenscells
    hold = session.pairs["hold"]
    test = np.array([assignment.get(int(h)) == "TEST" for h in hold], bool)
    cells = pair_cells(session, pp)
    z = np.asarray(z, float)
    bins = lenscells.dist_bin(z)
    entries = {}
    for i, (lo, hi) in enumerate(lenscells.DIST_BINS):
        for c in lenscells.CELLS:
            m = test & (bins == i) & (cells == c)
            if m.any():
                entries[(c, (lo, hi))] = int(len(np.unique(hold[m])))
    far = {}
    for lo, hi in FAR_BANDS:
        m = test & (z >= lo) & (z < hi)
        far[(lo, hi)] = int(len(np.unique(hold[m])))
    return TestTally(entries, far)


def split16_assignment(session, z, max_fit_z=3.0, holdout_every=2) -> dict:
    """step 16's split as a hold assignment {global hold id: "FIT" | "TEST"}: per take, in time order, every
    holdout_every-th hold with near pairs (rough distance <= max_fit_z) is TEST, and so is a hold with no near pair.
    Step 16 splits at the pair level: a hold mixing FIT near pairs and far pairs is FIT here (its far pairs are TEST
    in 16); the stored sessions have none (no pair beyond 2.3 m)"""
    from calibtool import paths
    d16 = paths.import_script("16_split_distance")
    P = session.pairs
    near = np.nan_to_num(np.asarray(z, float), nan=np.inf) <= max_fit_z
    out = {}
    for t in session.take_order:
        m = P["take"] == t
        held = d16.near_holdout(P["hold"][m], P["t0"][m], near[m], holdout_every)
        for h in np.unique(P["hold"][m]):
            k = P["hold"][m] == h
            out[int(h)] = "TEST" if (held[k].any() or not near[m][k].any()) else "FIT"
    return out


# ---------------------------------------------------------------- targets and deficits

# Targets re-based in M1 C3 (9 Nov 2026, decided with the user). The 4 Nov handheld-route targets (COLLECT-lens-data.md:
# thirds >= ~5,000, rim >= ~1,500, >= 50 far pairs, >= 6 entries at 10 TEST holds) were met by neither stored neck-grid
# session, yet both gave the best results so far (held-out median -0.04 / +0.04 %, rows 0.28 / 0.21 px). So each target
# is now what BOTH sessions reached, the lower of the two, measured the way the stop rule counts (replay.Collection at
# the end of the session: capture-time assignment, labels as finish() leaves them; regions and far pairs over all
# corners captured, pairs = FIT pairs). Values at the stored sessions' 3 pairs per pose:
STORED_PER_HOLD = 3         # the stored sessions' corner files: 15b per_hold 3
REACHED_FIT_PAIRS = 476     # FIT pairs: p21 476, p10 557
REACHED_THIRD = 3665        # corners per image-corner third, each camera: p21 min 3,665 (cam0 top-left), p10 min 5,814
REACHED_RIM = 486           # rim corners, each camera: p21 912 / 972, p10 486 (cam0) / 1,034
REACHED_FAR_PAIRS = 79      # pairs at 1.7-3.0 m: p21 79, p10 127
REACHED_STRIP_BOTTOM = 3004     # corners within EDGE_PX of the image bottom, each camera: p21 3,648 / 4,428,
                                # p10 3,004 (cam0) / 3,465
# hold counts (a hold is a pose: the same at any per_hold)
REACHED_JUDGED = 6          # entries with >= 10 TEST holds: p21 6, p10 7
REACHED_SIDE_HOLDS = 9      # most TEST holds in one cell of a required side, weakest side: p21 left 9 (p10 left 11,
                            # bottom 12); step 66 needs 8
REACHED_FAR_BAND = 21       # TEST holds per far band: p21 21 / 21, p10 51 / 25


def per_hold_scaled(n, per_hold) -> int:
    """a pair or corner count reached at STORED_PER_HOLD pairs per pose, at per_hold (rounded down): per_hold 2 keeps
    2/3 of the pairs and corners (protocol results section 2: 783 vs 1,159 pairs on p10 6 Nov)"""
    return int(n * per_hold) // STORED_PER_HOLD


@dataclass(frozen=True)
class Targets:
    """what a session should reach before it stops (design section 5, re-based in M1 C3: the REACHED_* constants),
    by default for PER_HOLD (2) pairs per pose; Targets.for_per_hold(3) for the stored sessions.

    The stop rule (Coverage.stop_met): no blocking deficit. Blocking: pairs, the corner thirds, the rim, the far pairs,
    the bottom edge strip (strip_min, per camera), the judged-entry count, each far band, and each required side
    (required_sides: left, right, bottom) with side_holds TEST holds in one of its cells. Asked for but not blocking:
    the comparative edge/box balance (a strip or box with fewer than ratio x its mirror image's count; p21 5 Nov: left
    176 vs right 1,099; neither passing session met it, p10's right edge 143 vs 408), each single entry short of
    test_holds, and the top side."""
    pairs_min: int = per_hold_scaled(REACHED_FIT_PAIRS, PER_HOLD)
    third_min: int = per_hold_scaled(REACHED_THIRD, PER_HOLD)          # each image-corner third, each camera
    rim_min: int = per_hold_scaled(REACHED_RIM, PER_HOLD)
    far_pairs_min: int = per_hold_scaled(REACHED_FAR_PAIRS, PER_HOLD)
    # absolute edge-strip targets {edge: corners}, each camera: the bottom (the tilted placement's job)
    strip_min: dict = field(default_factory=lambda: {"bottom": per_hold_scaled(REACHED_STRIP_BOTTOM, PER_HOLD)})
    edge_ratio: float = 0.5
    box_ratio: float = 0.5
    edge_min: int = 0
    box_min: int = 0
    fit_cams: tuple = (0, 1)
    test_holds: int = 10                # TEST holds that judge an entry (8 required + margin)
    judged_min: int = REACHED_JUDGED
    side_holds: int = REACHED_SIDE_HOLDS
    required_sides: frozenset = REQUIRED_EDGES
    far_band_min: int = REACHED_FAR_BAND
    # the planner's weights per deficit kind (default 1): edge strips weigh most, corner boxes next (the handoff's
    # order: p21's left edge first, its bottom-left box second), a side of the image short of TEST holds 2
    weights: dict = field(default_factory=lambda: {"edge": 3.0, "strip": 3.0, "box": 2.0, "side": 2.0})

    @classmethod
    def for_per_hold(cls, per_hold, **kw):
        """the targets for a session with per_hold pairs per pose (pair and corner counts scaled; kw override)"""
        base = dict(pairs_min=per_hold_scaled(REACHED_FIT_PAIRS, per_hold),
                    third_min=per_hold_scaled(REACHED_THIRD, per_hold), rim_min=per_hold_scaled(REACHED_RIM, per_hold),
                    far_pairs_min=per_hold_scaled(REACHED_FAR_PAIRS, per_hold),
                    strip_min={"bottom": per_hold_scaled(REACHED_STRIP_BOTTOM, per_hold)})
        base.update(kw)
        return cls(**base)


@dataclass
class Deficit:
    kind: str           # edge | strip | box | third | rim | pairs | far | entry | side | judged | far_band
    cam: object         # 0 / 1 for camera tallies, None otherwise
    region: object      # edge / box / third name; entry: (cell, (lo, hi)); far_band: (lo, hi); side: edge name
    have: int
    want: float
    ref_region: object = None
    ref_have: object = None
    weight: float = 1.0
    blocks: bool = True     # holds up the stop rule (False: asked for only)

    @property
    def short(self) -> float:
        return max(0.0, self.want - self.have)


@dataclass
class Coverage:
    fit: FitTally
    test: object = None         # TestTally or None

    def deficits(self, targets: Targets = Targets()) -> list:
        """what is short, most important first: the comparative edge strips (most lopsided first), the absolute edge
        strips, the corner boxes, then the rest: what blocks the stop rule (least met first), then what is only asked
        for (least met first)"""
        w = targets.weights
        out = []
        for c in targets.fit_cams:
            t = self.fit.cams[c]
            for kind, names, counts, ratio, amin in (("edge", EDGES, t.edges, targets.edge_ratio, targets.edge_min),
                                                     ("box", BOXES, t.boxes, targets.box_ratio, targets.box_min)):
                for r in names:
                    ref = counts[MIRROR[r]]
                    want = max(ratio * ref, amin)
                    if counts[r] < want:
                        out.append(Deficit(kind, c, r, counts[r], want, MIRROR[r], ref, w.get(kind, 1.0),
                                           blocks=False))
            for r, want in targets.strip_min.items():
                if t.edges[r] < want:
                    out.append(Deficit("strip", c, r, t.edges[r], want, weight=w.get("strip", 1.0)))
            for r, (i, j) in CORNER_THIRDS.items():
                if t.thirds[i, j] < targets.third_min:
                    out.append(Deficit("third", c, r, int(t.thirds[i, j]), targets.third_min, weight=w.get("third", 1)))
            if t.rim < targets.rim_min:
                out.append(Deficit("rim", c, "rim", t.rim, targets.rim_min, weight=w.get("rim", 1.0)))
        if self.fit.n_pairs < targets.pairs_min:
            out.append(Deficit("pairs", None, "pairs", self.fit.n_pairs, targets.pairs_min, weight=w.get("pairs", 1)))
        if self.fit.far_pairs < targets.far_pairs_min:
            out.append(Deficit("far", None, FAR_FIT, self.fit.far_pairs, targets.far_pairs_min,
                               weight=w.get("far", 1.0)))
        if self.test is not None:
            ent = self.test.entries
            judged = [k for k, n in ent.items() if n >= targets.test_holds]
            for k, n in ent.items():
                if n < targets.test_holds:
                    out.append(Deficit("entry", None, k, n, targets.test_holds, weight=w.get("entry", 1.0),
                                       blocks=False))
            for side in EDGE_CELLS:
                have = max([n for (c, _), n in ent.items() if side in edges_of_cell(c)], default=0)
                if have < targets.side_holds:
                    out.append(Deficit("side", None, side, have, targets.side_holds, weight=w.get("side", 1.0),
                                       blocks=side in targets.required_sides))
            if len(judged) < targets.judged_min:
                out.append(Deficit("judged", None, "judged", len(judged), targets.judged_min,
                                   weight=w.get("judged", 1.0)))
            for band, n in self.test.far_bands.items():
                if n < targets.far_band_min:
                    out.append(Deficit("far_band", None, band, n, targets.far_band_min, weight=w.get("far_band", 1)))
        edges = sorted([d for d in out if d.kind == "edge"], key=lambda d: d.have / max(d.ref_have, 1))
        strips = sorted([d for d in out if d.kind == "strip"], key=lambda d: d.have / max(d.want, 1e-9))
        boxes = sorted([d for d in out if d.kind == "box"], key=lambda d: d.have / max(d.ref_have, 1))
        rest = sorted([d for d in out if d.kind not in ("edge", "strip", "box")],
                      key=lambda d: (not d.blocks, d.have / max(d.want, 1e-9)))
        return edges + strips + boxes + rest

    def blocking(self, targets: Targets = Targets()) -> list:
        """the deficits that hold up the stop rule (Targets' docstring); needs the TEST tally"""
        if self.test is None:
            raise ValueError("the stop rule needs the TEST tally (Coverage.test)")
        return [d for d in self.deficits(targets) if d.blocks]

    def stop_met(self, targets: Targets = Targets()) -> bool:
        """the stop rule: no blocking deficit"""
        return not self.blocking(targets)


# ---------------------------------------------------------------- sentences


def ratio_words(have, ref) -> str:
    """176 vs 1,099 -> "6x fewer"; 100 vs 160 -> "1.6x fewer"; 0 -> "none there" """
    if have <= 0:
        return "none there"
    r = ref / have
    return f"{round(r)}x fewer" if r >= 2 else f"{r:.1f}x fewer"


def where(region) -> str:
    """the board's move that fills an image region ("left", "bottom-left", "top", ...), the operator's side first:
    "lower and to YOUR right (the robot's left)"; "straight ahead" for the centre"""
    parts = str(region).split("-")
    vert = next((p for p in parts if p in ("top", "bottom")), None)
    horiz = next((p for p in parts if p in ("left", "right")), None)
    words = ([("higher" if vert == "top" else "lower")] if vert else []) + ([SIDE[horiz]] if horiz else [])
    return " and ".join(words) if words else "straight ahead"


def direction(region) -> str:
    """the spoken instruction for an image region"""
    s = f"Put the board {where(region)}."
    if "bottom" in str(region):
        s += f" If it already stands as low as it goes, use {TILTED}."
    return s


def sentence(d: Deficit) -> str:
    cam = CAMERA.get(d.cam, "")
    if d.kind == "edge":
        return (f"{cam}, {d.region} edge: {d.have:,} corners vs {d.ref_have:,} on the {d.ref_region} edge "
                f"({ratio_words(d.have, d.ref_have)}). {direction(d.region)}")
    if d.kind == "strip":
        return (f"{cam}, {d.region} edge strip: {d.have:,} corners of {int(d.want):,} wanted ({int(d.short):,} short). "
                f"{direction(d.region)}")
    if d.kind == "box":
        return (f"{cam}, {d.region} corner: {d.have:,} corners vs {d.ref_have:,} in the {d.ref_region} corner "
                f"({ratio_words(d.have, d.ref_have)}). {direction(d.region)}")
    if d.kind == "third":
        return (f"{cam}, {d.region} third: {d.have:,} corners of {int(d.want):,} wanted ({int(d.short):,} short). "
                f"{direction(d.region)}")
    if d.kind == "rim":
        return (f"{cam}, image rim: {d.have:,} corners of {int(d.want):,} wanted ({int(d.short):,} short). "
                "Off-centre placements fill it: the board higher or lower, and toward either side.")
    if d.kind == "pairs":
        return f"Usable pairs: {d.have:,} of {int(d.want):,} wanted ({int(d.short):,} short). Any placement adds some."
    if d.kind == "far":
        lo, hi = d.region
        return (f"Far boards ({lo:.1f}-{hi:.1f} m): {d.have:,} pairs of {int(d.want):,} wanted ({int(d.short):,} "
                f"short). Put the board {lo:.1f}-{hi:.1f} m from the head.")
    if d.kind == "entry":
        cell, (lo, hi) = d.region
        return (f"Held-out check, {cell} cell at {lo:g}-{hi:g} m: {d.have} of {int(d.want)} held-out holds. "
                f"Put the board {where(cell)}, {lo:g}-{hi:g} m from the head.")
    if d.kind == "side":
        return (f"Held-out check, image {d.region} side: best {d.have} of {int(d.want)} held-out holds in a "
                f"{d.region} cell. {direction(d.region)}" + ("" if d.blocks else " (Asked for, not required.)"))
    if d.kind == "judged":
        return (f"Held-out check: {d.have} of {int(d.want)} entries judged. Placements at 0.4-1.3 m with the board "
                "off-centre add entries.")
    if d.kind == "far_band":
        lo, hi = d.region
        return (f"Held-out check, far band {lo:g}-{hi:g} m: {d.have} of {int(d.want)} held-out holds. "
                f"Put the board {lo:g}-{hi:g} m from the head.")
    raise ValueError(f"unknown deficit kind {d.kind!r}")


def sentences(deficits) -> list:
    return [sentence(d) for d in deficits]
