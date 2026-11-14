"""calibtool.decide: deploy the new lens or keep the live one (design plan 6.1-6.2).

Statistics (one definition each): depth error per TEST pair = lensscore board scale - 1 (%); median |error| = the
median of the absolute values; one band table [lo, hi) 0.4-0.6 / 0.6-0.9 / 0.9-1.3 / 1.3-1.7 / 1.7-2.3 m; row error =
median |dy| over the corners; paired new - live on the same pairs, 95 % interval by a seeded bootstrap over placements
(takes). Fast tests on synthetic scores; `replay` tests on the two stored sessions."""
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from dataclasses import replace  # noqa: E402

from calibtool import decide  # noqa: E402

BAND_Z = (0.5, 0.75, 1.1, 1.5, 2.0)          # one z inside each band
ALL_EDGES = {("top", (0.9, 1.3)): 0.1, ("left", (0.6, 0.9)): -0.1, ("right", (0.6, 0.9)): 0.2,
             ("bottom", (0.6, 0.9)): 0.0, ("centre", (0.4, 0.6)): 0.05, ("bottom-left", (0.9, 1.3)): -0.3}


def step66(entries=ALL_EDGES, verdict="PASS"):
    return SimpleNamespace(verdict=verdict, entries=dict(entries), text=f"{verdict}: synthetic")


def synthetic(err_scale=1.0, rows_scale=1.0, placements=6, holds=10, seed=7, band_shift=None, drop_src=()):
    """scores of one calibration on a synthetic TEST set: `placements` takes, each with holds at every band, 2 pairs a
    hold, 20 corners a pair. The underlying errors are the same for every call with the same seed (paired); err_scale
    / rows_scale multiply them; band_shift = (band index, add %) moves one band"""
    rng = np.random.default_rng(seed)
    src, hold, pl, err, z, csrc, cdy = [], [], [], [], [], [], []
    s = h = 0
    for p in range(placements):
        for _ in range(holds):
            b = h % len(BAND_Z)
            for _ in range(2):
                e = rng.normal(0.4, 0.5) * err_scale
                if band_shift and band_shift[0] == b:
                    e += band_shift[1]
                src.append(s), hold.append(h), pl.append(f"take{p}"), err.append(e), z.append(BAND_Z[b])
                csrc += [s] * 20
                cdy += list(np.abs(rng.normal(0, 0.5, 20)) * rows_scale)
                s += 1
            h += 1
    keep = ~np.isin(src, list(drop_src))
    ck = ~np.isin(csrc, list(drop_src))
    return decide.Scores(src=np.array(src)[keep], hold=np.array(hold)[keep], placement=np.array(pl)[keep],
                         err_pct=np.array(err)[keep], z=np.array(z)[keep], corner_src=np.array(csrc)[ck],
                         corner_dy=np.array(cdy)[ck])


# ---------------------------------------------------------------- statistics


def test_median_abs_error_is_median_of_absolute_values():
    sc = synthetic()
    st = decide.stats(sc)
    assert st["depth"] == pytest.approx(np.median(np.abs(sc.err_pct)))
    assert st["spread"] == pytest.approx(np.subtract(*np.percentile(sc.err_pct, [75, 25])))
    assert st["rows"] == pytest.approx(np.median(np.abs(sc.corner_dy)))


def test_band_table_is_lo_inclusive():
    assert decide.BANDS == ((0.4, 0.6), (0.6, 0.9), (0.9, 1.3), (1.3, 1.7), (1.7, 2.3))
    assert decide.band_of([0.4, 0.5999, 0.6, 1.3, 2.2999, 2.3, 0.39]).tolist() == [0, 0, 1, 3, 4, -1, -1]


def test_paired_identical_is_zero_with_zero_interval():
    a = synthetic()
    pr = decide.paired(a, a, decide.Gates())
    for s in pr.stats:
        if s.judged:
            assert s.diff == 0 and s.ci == (0.0, 0.0), s


def test_paired_is_seeded_and_clustered_by_placement():
    a, b = synthetic(0.5, 0.5), synthetic()
    p1, p2 = decide.paired(a, b, decide.Gates()), decide.paired(a, b, decide.Gates())
    assert [s.ci for s in p1.stats] == [s.ci for s in p2.stats]
    p3 = decide.paired(a, b, decide.Gates(seed=1))
    assert [s.ci for s in p3.stats] != [s.ci for s in p1.stats]
    depth = p1.stat("depth")
    assert depth.n_placements == 6 and depth.n_pairs == len(a.src)
    assert depth.ci[0] <= depth.diff <= depth.ci[1] < 0


def test_paired_matches_pairs_by_src():
    """a pair the live scoring lost (PnP failed) is left out of both sides"""
    a, b = synthetic(0.5), synthetic(drop_src=(0, 1, 2))
    pr = decide.paired(a, b, decide.Gates())
    assert pr.stat("depth").n_pairs == len(a.src) - 3 and pr.dropped_pairs == {"new": 3, "live": 0}


def test_one_placement_is_not_judged():
    a = synthetic(placements=1)
    pr = decide.paired(a, a, decide.Gates())
    assert not pr.stat("depth").judged and "placement" in pr.stat("depth").why


def _signed(err, dy, placements=4, z=0.75):
    """Scores with the given signed depth errors (%) and signed row errors (px), one corner per pair, pairs dealt round
    robin over `placements` takes"""
    err, dy = np.asarray(err, float), np.asarray(dy, float)
    n = len(err)
    return decide.Scores(src=np.arange(n), hold=np.arange(n), placement=np.array([f"t{i % placements}"
                                                                                  for i in range(n)]),
                         err_pct=err, z=np.full(n, z), corner_src=np.arange(n), corner_dy=dy)


def test_paired_uses_absolute_values_for_depth_and_rows():
    """median |x| is not |median x| on mixed-sign errors: paired() must take the absolute values first, for the depth
    error and for the row error dy (which is signed)"""
    err = [-3.0, -2.5, -2.0, 0.5, 1.0, 1.5, -2.2, 0.7]
    dy = [-0.9, -0.8, -0.7, 0.1, 0.2, 0.3, -0.75, 0.15]
    new = _signed(err, dy)
    live = _signed(np.zeros(8), np.zeros(8))
    assert np.median(np.abs(err)) != abs(np.median(err)) and np.median(np.abs(dy)) != abs(np.median(dy))
    pr = decide.paired(new, live, decide.Gates(boot=50))
    assert pr.stat("depth").new == pytest.approx(np.median(np.abs(err)))
    assert pr.stat("rows").new == pytest.approx(np.median(np.abs(dy)))
    assert pr.stat("band 0.6-0.9").new == pytest.approx(np.median(np.abs(err)))
    st = decide.stats(new)
    assert st["depth"] == pytest.approx(np.median(np.abs(err))) and st["rows"] == pytest.approx(np.median(np.abs(dy)))


@pytest.mark.parametrize("level", [95.0, 80.0])
def test_interval_holds_level_pct_of_the_resamples(level):
    """the interval is the central level_pct % of the bootstrap distribution (default 95 %): its ends are the
    distribution's percentiles, and it holds about level_pct % of the resamples (a little more: medians of few
    placements repeat, so resamples tie at the ends)"""
    assert decide.Gates().level_pct == 95.0
    pr = decide.paired(synthetic(0.5, 0.5), synthetic(), decide.Gates(level_pct=level))
    for name in ("depth", "rows"):
        s, b = pr.stat(name), pr.boot[name]
        inside = np.mean((b >= s.ci[0]) & (b <= s.ci[1]))
        assert level / 100 - 0.01 <= inside <= level / 100 + 0.05, (name, inside)
        assert s.ci[0] == pytest.approx(np.percentile(b, (100 - level) / 2))
        assert s.ci[1] == pytest.approx(np.percentile(b, 100 - (100 - level) / 2))


def test_a_band_from_too_few_placements_is_not_judged():
    """a band with enough pairs from one placement only is not judged (band_min_placements): its pairs share the
    board's position. Here that band is much worse for the new fit, yet the verdict stays DEPLOY"""
    full = synthetic(placements=6, holds=30)
    one = [s for s, p, z in zip(full.src, full.placement, full.z) if 0.9 <= z < 1.3 and p != "take0"]
    new = synthetic(0.3, 0.3, placements=6, holds=30, band_shift=(2, 3.0), drop_src=one)
    live = synthetic(placements=6, holds=30, drop_src=one)
    v = decide.verdict(new, live, step66())
    band = v.paired.stat("band 0.9-1.3")
    assert band.n_pairs >= decide.Gates().band_min_pairs and band.n_placements == 1
    assert not band.judged and "placement" in band.why
    assert v.decision == "DEPLOY", v.reasons


# ---------------------------------------------------------------- the verdict


def test_identical_keeps_live():
    a = synthetic()
    v = decide.verdict(a, a, step66())
    assert v.decision == "KEEP_LIVE"
    assert any("median |depth error|" in r for r in v.reasons)


def test_new_better_everywhere_deploys():
    v = decide.verdict(synthetic(0.3, 0.3), synthetic(), step66())
    assert v.decision == "DEPLOY", v.reasons
    assert all(s.diff < 0 for s in v.paired.stats if s.judged)


def test_new_worse_in_one_band_keeps_live():
    new = synthetic(0.3, 0.3, band_shift=(2, 3.0))
    v = decide.verdict(new, synthetic(), step66())
    assert v.decision == "KEEP_LIVE", v.reasons
    band = v.paired.stat("band 0.9-1.3")
    assert band.judged and band.ci[0] > 0
    assert any("0.9-1.3" in r and "worse" in r for r in v.reasons)


def test_new_not_better_on_rows_keeps_live():
    v = decide.verdict(synthetic(0.3, 1.0), synthetic(), step66())
    assert v.decision == "KEEP_LIVE" and any("row" in r for r in v.reasons)


def test_fewer_than_six_judged_entries_is_not_enough_data():
    few = dict(list(ALL_EDGES.items())[:5])
    v = decide.verdict(synthetic(0.3, 0.3), synthetic(), step66(few))
    assert v.decision == "NOT_ENOUGH_DATA" and any("5 judged" in r for r in v.reasons)


def test_required_edges_default_left_right_bottom():
    """C3 (9 Nov): a deploy needs judged entries on the left, right and bottom image edges; the top is asked for during
    collection but does not block, and the reasons say when it is not judged. required_edges=EDGES needs all four"""
    no_top = {k: x for k, x in ALL_EDGES.items() if k[0] != "top"}
    no_top[("right", (0.9, 1.3))] = 0.1
    new, live = synthetic(0.3, 0.3), synthetic()
    assert decide.Gates().required_edges == frozenset({"left", "right", "bottom"})
    v = decide.verdict(new, live, step66(no_top))
    assert v.decision == "DEPLOY", v.reasons
    assert v.missing_edges == ["top"]
    assert any("top" in r and "not judged" in r and "not required" in r for r in v.reasons), v.reasons
    strict = decide.verdict(new, live, step66(no_top), decide.Gates(required_edges=frozenset(decide.EDGES)))
    assert strict.decision == "NOT_ENOUGH_DATA" and strict.missing_edges == ["top"]
    assert any("top" in r for r in strict.reasons)
    assert decide.verdict(new, live, step66(), decide.Gates(required_edges=frozenset(decide.EDGES))).decision \
        == "DEPLOY"
    full = decide.verdict(new, live, step66())
    assert full.decision == "DEPLOY" and not any("not required" in r for r in full.reasons), full.reasons


@pytest.mark.parametrize("edge", ["left", "right", "bottom"])
def test_a_missing_required_edge_is_not_enough_data(edge):
    """an entry set with no judged cell on a required edge (a corner cell would count for it, so those go too)"""
    ent = {k: x for k, x in ALL_EDGES.items() if edge not in k[0]}
    ent.update({("centre", (0.6, 0.9)): 0.0, ("centre", (0.9, 1.3)): 0.0, ("top", (0.6, 0.9)): 0.0})
    assert len(ent) >= decide.Gates().min_judged
    v = decide.verdict(synthetic(0.3, 0.3), synthetic(), step66(ent))
    assert v.decision == "NOT_ENOUGH_DATA", v.reasons
    assert edge in v.missing_edges and any(edge in r for r in v.reasons)


def test_required_edges_are_checked():
    with pytest.raises(ValueError, match="sideways"):
        decide.Gates(required_edges=frozenset({"left", "sideways"}))


def test_band_min_placements_is_three():
    assert decide.Gates().band_min_placements == 3


def test_corner_cell_counts_for_both_its_edges():
    assert decide.edges_judged([("bottom-left", (0.9, 1.3))]) == {"bottom", "left"}
    assert decide.edges_judged([("centre", (0.4, 0.6))]) == set()


def test_step66_fail_keeps_live_and_no_result_is_not_enough_data():
    new, live = synthetic(0.3, 0.3), synthetic()
    v = decide.verdict(new, live, step66(verdict="FAIL"))
    assert v.decision == "KEEP_LIVE" and any("step 66" in r for r in v.reasons)
    v = decide.verdict(new, live, step66({}, verdict="NO RESULT"))
    assert v.decision == "NOT_ENOUGH_DATA"


def _near_miss_step66(median=0.574, limit=0.5):
    """step 66 FAILing one validated entry (13 Nov, phantom10: top 0.9-1.3 m +0.57 % vs 0.5 %)"""
    table = [{"cell": "top", "dist_m": [0.9, 1.3], "median_pct": median, "limit_pct": limit, "ok": False},
             {"cell": "left", "dist_m": [0.6, 0.9], "median_pct": -0.1, "limit_pct": 0.5, "ok": True}]
    return SimpleNamespace(verdict="FAIL", entries=dict(ALL_EDGES), text="FAIL: synthetic near miss",
                           failing={("top", (0.9, 1.3)): median}, report={"table": table})


def test_step66_failing_reads_the_failed_rows_of_the_report():
    assert decide.step66_failing(_near_miss_step66()) == [["top 0.9-1.3 m", 0.574, 0.5]]
    assert decide.step66_failing(step66()) == []
    assert decide.near_miss([["top 0.9-1.3 m", 0.574, 0.5]], 2.0)
    assert not decide.near_miss([["top 0.9-1.3 m", 1.2, 0.5]], 2.0)
    assert not decide.near_miss([["top 0.9-1.3 m", 0.574, 0.5]], 1.0) and not decide.near_miss([], 2.0)


def test_a_step66_near_miss_keeps_live_by_the_rule_and_deploys_only_with_the_exception_and_new_better():
    new, live = synthetic(0.3, 0.3), synthetic()
    v = decide.verdict(new, live, _near_miss_step66())
    assert v.decision == "KEEP_LIVE" and v.step66_failing == [["top 0.9-1.3 m", 0.574, 0.5]]
    assert v.as_dict()["step66_failing"] == [["top 0.9-1.3 m", 0.574, 0.5]]
    g = decide.Gates(step66_fail_factor=2.0)
    v = decide.verdict(new, live, _near_miss_step66(), g)
    assert v.decision == "DEPLOY", v.reasons
    assert v.reasons[0].startswith("one-off exception") and "top 0.9-1.3 m +0.57 % vs 0.5 %" in v.reasons[0]
    assert v.gates.step66_fail_factor == 2.0
    # the exception never skips the comparison: new not better than live keeps live
    assert decide.verdict(synthetic(), synthetic(), _near_miss_step66(), g).decision == "KEEP_LIVE"
    # nor accepts a big miss, nor a FAIL without a report to judge it from
    assert decide.verdict(new, live, _near_miss_step66(1.2), g).decision == "KEEP_LIVE"
    assert decide.verdict(new, live, step66(verdict="FAIL"), g).decision == "KEEP_LIVE"
    # nor deploys without live scores
    v = decide.verdict(new, None, _near_miss_step66(), replace(g, allow_deploy_without_live=True), live_error="x")
    assert v.decision == "KEEP_LIVE"


def test_the_exception_reuses_a_bootstrap_made_with_its_gates():
    new, live = synthetic(0.3, 0.3), synthetic()
    g = decide.Gates(step66_fail_factor=2.0)
    pr = decide.paired(new, live, g)
    assert decide.verdict(new, live, _near_miss_step66(), g, paired_=pr).decision == "DEPLOY"


def test_live_not_scorable_needs_the_explicit_gate():
    """no live scores: deploying without comparing with the robot's current calibration contradicts "deploy only if
    shown better", so it takes Gates.allow_deploy_without_live=True; by default NOT_ENOUGH_DATA, the error in the
    reasons"""
    v = decide.verdict(synthetic(0.3, 0.3), None, step66(), live_error="NotScorable: distortion_type ['omni']")
    assert v.decision == "NOT_ENOUGH_DATA" and v.paired is None
    assert any("omni" in r and "allow_deploy_without_live" in r for r in v.reasons), v.reasons
    g = decide.Gates(allow_deploy_without_live=True)
    v = decide.verdict(synthetic(0.3, 0.3), None, step66(), g, live_error="NotScorable: distortion_type ['omni']")
    assert v.decision == "LIVE_NOT_SCORABLE_DEPLOY" and v.paired is None
    assert any("omni" in r for r in v.reasons), v.reasons
    assert decide.verdict(synthetic(), None, step66(verdict="FAIL"), g).decision == "KEEP_LIVE"
    assert decide.Gates().allow_deploy_without_live is False


def test_depth_is_judged_on_the_intervals_upper_end():
    """depth interval straddling 0 (new not shown better on depth) while rows are clearly better: KEEP_LIVE. No band is
    judged here, so only the depth rule can keep live"""
    new, live = synthetic(rows_scale=0.3, seed=8), synthetic()
    v = decide.verdict(new, live, step66(), decide.Gates(band_min_pairs=10**6))
    d, r = v.paired.stat("depth"), v.paired.stat("rows")
    assert d.ci[0] < 0 < d.ci[1], d
    assert r.ci[1] < 0, r
    assert v.decision == "KEEP_LIVE", v.reasons
    assert any("not shown better on median |depth error|" in x for x in v.reasons)


def test_without_live_the_entry_and_edge_checks_still_apply():
    """live not scorable, deploy without live allowed: fewer than 6 judged entries, or an image edge with no judged
    entry, is still NOT_ENOUGH_DATA"""
    g = decide.Gates(allow_deploy_without_live=True, required_edges=frozenset(decide.EDGES))
    few = dict(list(ALL_EDGES.items())[:5])
    assert len(few) == 5 and decide.edges_judged(few) == set(decide.EDGES)
    v = decide.verdict(synthetic(0.3, 0.3), None, step66(few), g)
    assert v.decision == "NOT_ENOUGH_DATA" and any("5 judged" in r for r in v.reasons), v.reasons
    no_top = {k: x for k, x in ALL_EDGES.items() if k[0] != "top"}
    no_top[("right", (0.9, 1.3))] = 0.1
    v = decide.verdict(synthetic(0.3, 0.3), None, step66(no_top), g)
    assert v.decision == "NOT_ENOUGH_DATA" and v.missing_edges == ["top"], v.reasons
    assert decide.verdict(synthetic(0.3, 0.3), None, step66(), g).decision == "LIVE_NOT_SCORABLE_DEPLOY"


def test_pairs_live_scored_but_new_did_not_are_gated():
    """pairs the live config scored and the new fit could not (PnP failed on its lens) leave the comparison; more than
    Gates.max_new_unscored_frac of live's pairs that way keeps live, with the count in the reasons. Pairs only the new
    fit scored are not gated"""
    live = synthetic()
    n = len(live.src)
    assert decide.Gates().max_new_unscored_frac == 0.05
    few, many = range(3), range(int(0.1 * n))
    assert decide.verdict(synthetic(0.3, 0.3, drop_src=few), live, step66()).decision == "DEPLOY"
    v = decide.verdict(synthetic(0.3, 0.3, drop_src=many), live, step66())
    assert v.decision == "KEEP_LIVE", v.reasons
    assert any(f"{len(many)} of the {n} pairs" in r for r in v.reasons), v.reasons
    loose = decide.Gates(max_new_unscored_frac=0.2)
    assert decide.verdict(synthetic(0.3, 0.3, drop_src=many), live, step66(), loose).decision == "DEPLOY"
    assert decide.verdict(synthetic(0.3, 0.3), synthetic(drop_src=many), step66()).decision == "DEPLOY"


def test_thresholds_are_named_parameters():
    g = decide.Gates()
    for k in ("min_judged", "required_edges", "depth_margin_pct", "rows_margin_px", "band_worse_margin_pct",
              "band_min_pairs", "band_min_placements", "min_placements", "boot", "seed"):
        assert hasattr(g, k)
    # a margin larger than the improvement turns DEPLOY into KEEP_LIVE
    new, live = synthetic(0.9, 0.3), synthetic()
    assert decide.verdict(new, live, step66()).decision == "DEPLOY"
    assert decide.verdict(new, live, step66(), decide.Gates(depth_margin_pct=0.5)).decision == "KEEP_LIVE"


def test_verdict_as_dict_is_json():
    v = decide.verdict(synthetic(0.3, 0.3), synthetic(), step66())
    d = json.loads(json.dumps(v.as_dict()))
    assert d["decision"] == "DEPLOY" and d["paired"]["stats"][0]["name"] == "depth"


# ---------------------------------------------------------------- replay: the two stored sessions


@pytest.mark.replay
@pytest.mark.usefixtures("recorded_trim")  # recorded numbers: at most 4 trim rounds
@pytest.mark.parametrize("name", ["p21_1105", "p10_1106"])
def test_stored_sessions_new_beats_live(name, data_root, tmp_path):
    """the recorded fit vs the robot's live config on the recorded TEST split, scored with scipy (the recorded
    numbers are scipy's): new better on every paired statistic,
    DEPLOY with the default required edges (left, right, bottom); with all four required p21 is NOT_ENOUGH_DATA (no
    judged top entry)"""
    from calibtool import livecfg, pipeline, sessions
    from calibtool.session import Session
    st = sessions.get(name)
    f = st.recorded.files
    fit_json, test_npz = data_root / f["fit_json"], data_root / f["test_npz"]
    live = livecfg.load(st.live_config)
    assert live.model == st.live_model
    place = decide.placements(test_npz, Session.load(st))
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(pipeline.validate66, fit_json, test_npz, tmp_path / "validate-new", st.rig_path,
                        fit_report=fit_json, solver="scipy")
        new = decide.score(fit_json, test_npz, st.rig_path, place, solver="scipy")
        old = decide.score(live.write_json(tmp_path / "live.json"), test_npz, st.rig_path, place, solver="scipy")
        v66 = fut.result()
    assert v66.verdict == "PASS"
    loose = decide.verdict(new, old, v66, decide.Gates())
    strict = decide.verdict(new, old, v66, decide.Gates(required_edges=frozenset(decide.EDGES)))
    out = {"session": name, "summary_new": decide.stats(new), "summary_live": decide.stats(old),
           "loose": loose.as_dict(), "strict": strict.as_dict()}
    print(f"DECIDE {name}: " + json.dumps(out))
    for s in loose.paired.stats:
        print(f"  {s.name:14s} new {s.new:8.4f} live {s.live:8.4f} diff {s.diff:+8.4f} "
              f"CI {s.ci} pairs {s.n_pairs} placements {s.n_placements} judged {s.judged} {s.why}")
    print(f"  loose: {loose.decision} {loose.reasons}\n  strict: {strict.decision} {strict.reasons}")
    worse = [s.name for s in loose.paired.stats if s.judged and not s.diff < 0]
    assert not worse, f"new not better on {worse}"
    assert np.median(new.err_pct) == pytest.approx(st.recorded.summary_new.median_pct, abs=0.02)
    assert np.median(old.err_pct) == pytest.approx(st.recorded.summary_live.median_pct, abs=0.02)
    assert decide.stats(new)["rows"] == pytest.approx(st.recorded.summary_new.rows_px, abs=0.005)
    assert decide.stats(old)["rows"] == pytest.approx(st.recorded.summary_live.rows_px, abs=0.005)
    assert loose.decision == "DEPLOY", loose.reasons
    assert strict.decision == {"p21_1105": "NOT_ENOUGH_DATA", "p10_1106": "DEPLOY"}[name], strict.reasons
