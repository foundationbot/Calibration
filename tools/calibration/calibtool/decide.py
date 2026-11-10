"""Deploy the new lens calibration or keep the live one (design plan 6.1-6.2).

Statistics, one definition each, on the TEST pairs both calibrations scored:
  depth error   per pair, lensscore's board scale - 1, in % (+ = reads too far)
  depth         median |depth error|: the median of the absolute values
  spread        interquartile range of the signed depth error (p75 - p25, %)
  rows          median |dy| over the corners seen by both cameras, after rectification with the calibration's own
                stereo pose, at the rig's nominal focal (pipeline.summary's row error, px)
  band lo-hi    median |depth error| of the pairs whose board is at z in [lo, hi) m, on one table: 0.4-0.6, 0.6-0.9,
                0.9-1.3 (step 66's bins), 1.3-1.7, 1.7-2.3; z = the NEW calibration's board depth, so both sides
                judge the same pairs
Paired: new - live on the same pairs (matched by their index in the TEST file); a 95 % interval by a seeded bootstrap
that resamples placements (takes), whole, with replacement: holds of one placement share the board's position and are
not independent. A statistic is judged only with enough pairs and placements (Gates).

Verdict (verdict()):
  NOT_ENOUGH_DATA           step 66 on the new fit gave no result; or fewer than Gates.min_judged judged (validated)
                            entries; or, with require_all_edges, an image edge (left/right/top/bottom; a corner cell
                            counts for both its edges) without a judged entry; or too few placements to compare
  KEEP_LIVE                 step 66 FAILs the new fit; or the paired interval does not show new better on depth and
                            on rows; or a judged band where the interval shows new worse
                            ; or no live scores (the live config could not be scored, the error in the reasons)
                            without Gates.allow_deploy_without_live
  LIVE_NOT_SCORABLE_DEPLOY  no live scores and Gates.allow_deploy_without_live=True: step 66 PASS alone decides
  DEPLOY                    otherwise
Every threshold is a field of Gates; the defaults are PLACEHOLDERS, to be fixed from the stored sessions (M1 C3)."""
from dataclasses import asdict, dataclass, field

import numpy as np

BANDS = ((0.4, 0.6), (0.6, 0.9), (0.9, 1.3), (1.3, 1.7), (1.7, 2.3))     # [lo, hi) m
EDGES = ("left", "right", "top", "bottom")
DECISIONS = ("DEPLOY", "KEEP_LIVE", "NOT_ENOUGH_DATA", "LIVE_NOT_SCORABLE_DEPLOY")


@dataclass(frozen=True)
class Gates:
    """the decision's thresholds. PLACEHOLDER defaults (M1 B3): fixed from the stored sessions in M1 C3"""
    min_judged: int = 6                 # step-66 entries validated on the new fit (design 5: >= 6)
    require_all_edges: bool = True      # each image edge has a judged entry (design 5)
    depth_margin_pct: float = 0.0       # new better on depth: interval's upper end < -margin
    rows_margin_px: float = 0.0         # new better on rows: interval's upper end < -margin
    band_worse_margin_pct: float = 0.0  # a band is worse when its interval's lower end > +margin
    band_min_pairs: int = 10            # a band is judged from this many pairs ...
    band_min_placements: int = 2        # ... from this many placements
    min_placements: int = 2             # the whole comparison needs this many placements
    boot: int = 2000                    # bootstrap resamples
    seed: int = 20261109                # bootstrap seed (fixed: the same data gives the same intervals)
    level_pct: float = 95.0             # interval
    min_valid_frac: float = 0.5         # a band's interval needs data in this share of the resamples
    # without live scores, deploy on step 66 alone (LIVE_NOT_SCORABLE_DEPLOY) only when this is set: "deploy only if
    # shown better than live" cannot be checked then, so it takes an explicit choice (else NOT_ENOUGH_DATA)
    allow_deploy_without_live: bool = False


# ---------------------------------------------------------------- scores of one calibration


@dataclass
class Scores:
    """one calibration on one TEST file: per scored pair its index in the file (src), hold, placement (take), depth
    error (%) and board depth z (m); per corner seen by both cameras its pair (src) and row error dy (px)"""
    src: np.ndarray
    hold: np.ndarray
    placement: np.ndarray
    err_pct: np.ndarray
    z: np.ndarray
    corner_src: np.ndarray
    corner_dy: np.ndarray
    summary: object = field(default=None, repr=False)     # pipeline.SummaryResult, when scored by score()

    @classmethod
    def from_records(cls, rec, placement, corner_src, corner_dy, summary=None):
        """lensscore.score's records + placement per TEST-file pair (array indexed by src)"""
        src = np.asarray(rec["src"], int)
        return cls(src, np.asarray(rec["hold"], int), np.asarray(placement)[src],
                   100 * (np.asarray(rec["scale"], float) - 1), np.asarray(rec["z"], float),
                   np.asarray(corner_src, int), np.asarray(corner_dy, float), summary)


def band_of(z):
    """band index per z, -1 outside the table"""
    z = np.atleast_1d(np.asarray(z, float))
    out = np.full(z.shape, -1, int)
    for i, (lo, hi) in enumerate(BANDS):
        out[(z >= lo) & (z < hi)] = i
    return out


def band_name(i):
    return f"band {BANDS[i][0]:g}-{BANDS[i][1]:g}"


def stats(sc: Scores):
    """the statistics of one calibration over all its pairs (no pairing): depth, spread, rows, median signed error,
    per band median |error| (None without pairs)"""
    e = np.asarray(sc.err_pct)
    b = band_of(sc.z)
    out = {"depth": float(np.median(np.abs(e))), "spread": float(np.subtract(*np.percentile(e, [75, 25]))),
           "rows": float(np.median(np.abs(sc.corner_dy))), "median_pct": float(np.median(e)), "pairs": int(len(e))}
    for i in range(len(BANDS)):
        m = b == i
        out[band_name(i)] = float(np.median(np.abs(e[m]))) if m.any() else None
    return out


def placements(test_npz, session):
    """the placement (take) of every pair of a TEST file, joined to the session by left stamp (img_t0)"""
    t0 = np.load(test_npz)["img_t0"]
    return session.pairs["take"][session.index_of_t0(t0)]


def row_errors(model, intr, C, test_npz, rig):
    """(pair index per corner, dy per corner): pipeline.summary's row error, per corner: the corners both cameras saw,
    rectified with the calibration's stereo pose, dy at the rig's nominal focal. The lens model must be known to
    lensmodel (livecfg.model_scope)"""
    import cv2
    import lensmodel
    from calibtool import pipeline
    rig = pipeline.load_rig(rig)
    T10 = np.linalg.inv(np.asarray(C, float))
    R0, R1, *_ = cv2.stereoRectify(np.eye(3), np.zeros(5), np.eye(3), np.zeros(5), tuple(rig.resolution),
                                   np.ascontiguousarray(T10[:3, :3]), np.ascontiguousarray(T10[:3, 3].reshape(3, 1)),
                                   flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    z = np.load(test_npz)
    srcs, dys = [], []
    for j in range(len(z["img_t0"])):
        m = z["det_img"] == j
        d = [{(t, k): (u, v) for t, k, u, v, c in zip(z["det_tag"][m], z["det_k"][m], z["det_u"][m], z["det_v"][m],
                                                       z["det_cam"][m]) if c == cam} for cam in (0, 1)]
        keys = sorted(set(d[0]) & set(d[1]))
        if keys:
            b0 = (R0 @ lensmodel.unproject(model, intr[0], [d[0][k] for k in keys]).T).T
            b1 = (R1 @ lensmodel.unproject(model, intr[1], [d[1][k] for k in keys]).T).T
            dys.append(rig.nominal_focal_px * (b0[:, 1] / b0[:, 2] - b1[:, 1] / b1[:, 2]))
            srcs.append(np.full(len(keys), j))
    return (np.concatenate(srcs) if srcs else np.zeros(0, int)), (np.concatenate(dys) if dys else np.zeros(0))


def score(calib_json, test_npz, rig, placement) -> Scores:
    """a calibration (any step-66 JSON: step 45's fit, livecfg's live config) scored on the TEST file: pipeline.summary
    (lensscore) and row_errors, the model registered while it runs (a RADTAN live config)"""
    import stereofit as sf
    from calibtool import livecfg, pipeline
    model, intr, C = sf.load_calibration(str(calib_json))
    with livecfg.model_scope(model):
        sm = pipeline.summary(calib_json, test_npz, rig)
        cs, dy = row_errors(model, intr, C, test_npz, rig)
    return Scores.from_records(sm.records, placement, cs, dy, sm)


# ---------------------------------------------------------------- paired comparison


@dataclass
class Stat:
    name: str
    new: float
    live: float
    diff: float                 # new - live (negative: new better)
    ci: tuple                   # 95 % interval of diff, None when not judged
    n_pairs: int
    n_placements: int
    judged: bool
    why: str = ""               # why not judged


@dataclass
class Paired:
    stats: list
    placements: list
    dropped_pairs: dict         # pairs scored by one side only: {"new": n, "live": n}
    gates: Gates

    def stat(self, name):
        return next(s for s in self.stats if s.name == name)

    def as_dict(self):
        return {"stats": [asdict(s) for s in self.stats], "placements": list(self.placements),
                "dropped_pairs": self.dropped_pairs}


def _absmed(x):
    return float(np.median(np.abs(x))) if len(x) else np.nan


def _iqr(x):
    return float(np.subtract(*np.percentile(x, [75, 25]))) if len(x) else np.nan


def paired(new: Scores, live: Scores, gates: Gates = Gates()) -> Paired:
    """new - live on the pairs both scored, with bootstrap intervals over placements"""
    common = np.intersect1d(new.src, live.src)
    ni = {s: i for i, s in enumerate(new.src)}
    li = {s: i for i, s in enumerate(live.src)}
    a = np.array([ni[s] for s in common], int)
    b = np.array([li[s] for s in common], int)
    en, el = new.err_pct[a], live.err_pct[b]
    pl, band = new.placement[a], band_of(new.z[a])
    cn = np.isin(new.corner_src, common)
    cl = np.isin(live.corner_src, common)
    dn, dl = np.abs(new.corner_dy[cn]), np.abs(live.corner_dy[cl])
    pl_of_src = dict(zip(common.tolist(), pl.tolist()))
    pcn = np.array([pl_of_src[s] for s in new.corner_src[cn]], dtype=object)
    pcl = np.array([pl_of_src[s] for s in live.corner_src[cl]], dtype=object)
    ids = sorted(set(pl.tolist()))
    K = len(ids)
    by_pl = [np.flatnonzero(pl == p) for p in ids]
    by_pl_cn = [np.flatnonzero(pcn == p) for p in ids]
    by_pl_cl = [np.flatnonzero(pcl == p) for p in ids]

    def compute(ip, icn, icl):
        """{name: (new, live)} on pair rows ip and corner rows icn / icl"""
        out = {"depth": (_absmed(en[ip]), _absmed(el[ip])), "spread": (_iqr(en[ip]), _iqr(el[ip])),
               "rows": (_absmed(dn[icn]), _absmed(dl[icl]))}
        bb = band[ip]
        for i in range(len(BANDS)):
            m = ip[bb == i]
            out[band_name(i)] = (_absmed(en[m]), _absmed(el[m]))
        return out

    full = compute(np.arange(len(common)), np.arange(len(dn)), np.arange(len(dl)))
    names = list(full)
    boot = {k: np.full(gates.boot, np.nan) for k in names}
    if K:
        rng = np.random.default_rng(gates.seed)
        for r in range(gates.boot):
            pick = rng.integers(0, K, K)
            res = compute(np.concatenate([by_pl[j] for j in pick]), np.concatenate([by_pl_cn[j] for j in pick]),
                          np.concatenate([by_pl_cl[j] for j in pick]))
            for k in names:
                boot[k][r] = res[k][0] - res[k][1]
    q = ((100 - gates.level_pct) / 2, 100 - (100 - gates.level_pct) / 2)
    out = []
    for k in names:
        if k.startswith("band "):
            i = [band_name(j) for j in range(len(BANDS))].index(k)
            m = band == i
            n, npl = int(m.sum()), len(set(pl[m].tolist()))
            need_n, need_pl = gates.band_min_pairs, gates.band_min_placements
        else:
            n, npl = len(common), K
            need_n, need_pl = 1, gates.min_placements
        why = ""
        if n < need_n:
            why = f"{n} pairs (< {need_n})"
        elif npl < need_pl:
            why = f"{npl} placement{'s' if npl != 1 else ''} (< {need_pl})"
        valid = boot[k][np.isfinite(boot[k])]
        if not why and len(valid) < gates.min_valid_frac * gates.boot:
            why = f"data in {len(valid)} of {gates.boot} resamples"
        ci = None if why else tuple(float(x) for x in np.percentile(valid, q))
        nv, lv = full[k]
        out.append(Stat(k, nv, lv, float(nv - lv) if n else float("nan"), ci, n, npl, not why, why))
    return Paired(out, ids, {"new": int(len(new.src) - len(common)), "live": int(len(live.src) - len(common))}, gates)


# ---------------------------------------------------------------- the verdict


def edges_judged(entries):
    """the image edges with a judged entry ({(cell, dist)}): a side cell counts for its edge, a corner cell
    (top-left, ...) for both of its edges, centre for none"""
    out = set()
    for cell, _ in entries:
        out |= {e for e in EDGES if e in str(cell).split("-")}
    return out


@dataclass
class Verdict:
    decision: str
    reasons: list
    paired: Paired = None
    judged: list = field(default_factory=list)          # step 66's validated entries on the new fit
    missing_edges: list = field(default_factory=list)
    gates: Gates = Gates()

    def as_dict(self):
        return {"decision": self.decision, "reasons": list(self.reasons),
                "paired": self.paired.as_dict() if self.paired else None,
                "judged": [f"{c} {lo:g}-{hi:g}" for c, (lo, hi) in self.judged],
                "missing_edges": list(self.missing_edges), "gates": asdict(self.gates)}


def _fmt(s: Stat, unit):
    ci = f"{s.ci[0]:+.3f} .. {s.ci[1]:+.3f}" if s.ci else "-"
    return f"{s.new:.3f} vs live {s.live:.3f} {unit}, difference {s.diff:+.3f} (95 % {ci})"


def verdict(new: Scores, live, step66_new, gates: Gates = Gates(), live_error=None) -> Verdict:
    """the decision (DECISIONS) with its reasons. new / live: Scores on the same TEST file (live None: the live
    config could not be scored, live_error saying why); step66_new: step 66 of the new fit (pipeline.ValidateResult:
    .verdict, .entries)"""
    v66 = getattr(step66_new, "verdict", None)
    judged = sorted(getattr(step66_new, "entries", {}) or {})
    edges = edges_judged(judged)
    missing = [e for e in EDGES if e not in edges]
    base = dict(judged=judged, missing_edges=missing, gates=gates)
    text = getattr(step66_new, "text", "")
    if v66 not in ("PASS", "FAIL"):
        return Verdict("NOT_ENOUGH_DATA", [f"step 66 gave no result on the new fit: {text or v66}"], **base)
    if v66 == "FAIL":
        return Verdict("KEEP_LIVE", [f"the new fit fails step 66: {text}"], **base)
    no_live = f"the live config could not be scored ({live_error or 'no reason given'})"
    short = []
    if len(judged) < gates.min_judged:
        short.append(f"{len(judged)} judged entries (step 66 validated), {gates.min_judged} needed")
    if gates.require_all_edges and missing:
        short.append("no judged entry on the image's " + ", ".join(missing) + " edge" + ("s" if len(missing) > 1
                                                                                         else ""))
    if short:
        return Verdict("NOT_ENOUGH_DATA", short + ([no_live] if live is None else []), **base)
    if live is None:
        if not gates.allow_deploy_without_live:
            return Verdict("NOT_ENOUGH_DATA",
                           [f"{no_live}: the new fit is not compared with the robot's current calibration; deploying "
                            f"on its step-66 PASS alone needs allow_deploy_without_live=True"], **base)
        return Verdict("LIVE_NOT_SCORABLE_DEPLOY",
                       [f"{no_live}: deploy rests on the new fit's step-66 PASS alone ({len(judged)} judged entries; "
                        f"allow_deploy_without_live=True)"], **base)
    pr = paired(new, live, gates)
    depth, rows = pr.stat("depth"), pr.stat("rows")
    if not (depth.judged and rows.judged):
        return Verdict("NOT_ENOUGH_DATA", [f"new vs live not comparable: {depth.why or rows.why}"], paired=pr, **base)
    reasons, keep = [], False
    if depth.ci[1] < -gates.depth_margin_pct:
        reasons.append("new better on median |depth error|: " + _fmt(depth, "%"))
    else:
        keep = True
        reasons.append("new not shown better on median |depth error|: " + _fmt(depth, "%"))
    if rows.ci[1] < -gates.rows_margin_px:
        reasons.append("new better on row error: " + _fmt(rows, "px"))
    else:
        keep = True
        reasons.append("new not shown better on row error: " + _fmt(rows, "px"))
    for s in pr.stats:
        if s.name.startswith("band ") and s.judged and s.ci[0] > gates.band_worse_margin_pct:
            keep = True
            reasons.append(f"new worse in {s.name} m: " + _fmt(s, "%"))
    return Verdict("KEEP_LIVE" if keep else "DEPLOY", reasons, paired=pr, **base)
