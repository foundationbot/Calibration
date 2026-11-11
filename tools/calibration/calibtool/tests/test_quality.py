"""calibtool.quality: the per-section data status on the operator's screen (10 Nov 2026, phantom11: "quality status
so one can see quality among different sections of the camera")."""
import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from calibtool import coverage as cv, quality  # noqa: E402


def ct(thirds, edges=None):
    e = {"left": 10, "right": 20, "top": 30, "bottom": 40}
    e.update(edges or {})
    return cv.CornerTally(int(np.sum(thirds)), np.array(thirds), 5, e, {b: 1 for b in cv.BOXES})


def col(test_entries, far=None, deficits=()):
    tg = cv.Targets()
    fit = cv.FitTally({0: ct([[tg.third_min, 0, 0], [0, 50, 0], [0, 0, tg.third_min // 2]]),
                       1: ct([[0, 0, 0], [0, 0, 0], [0, 0, 0]])}, 0, 0)
    tallies = {"all": fit, "fit": fit, "test": cv.TestTally(dict(test_entries), dict(far or {}))}
    return SimpleNamespace(targets=tg, region_tally="all", tallies=lambda: tallies,
                           coverage=lambda t: SimpleNamespace(deficits=lambda targets: list(deficits)))


def test_sections_percent_of_target_and_heldout_grid_json_ready():
    c = col({("top-left", (0.6, 0.9)): 12, ("bottom", (0.4, 0.6)): 7}, {(1.7, 2.3): 10})
    q = quality.snapshot(c)
    json.dumps(q)
    assert q["cams"]["0"]["pct"][0][0] == 100 and q["cams"]["0"]["pct"][2][2] == 50
    assert q["cams"]["0"]["sections"][1][1] == 50 and q["cams"]["1"]["pct"][0][0] == 0
    h = q["heldout"]
    assert h["sections"][0][0] == [0, 12, 0] and h["sections"][2][1] == [7, 0, 0]
    assert h["judged"] == 1 and h["want"] == 10 and h["far"] == {"1.7-2.3": 10}
    assert q["stop_met"] is True and q["blocking"] == []


def test_only_blocking_deficits_are_listed():
    must = cv.Deficit("far_band", None, (1.7, 2.3), 10, 21)
    nice = cv.Deficit("edge", 0, "left", 38, 300, "right", 600, blocks=False)
    q = quality.snapshot(col({}, deficits=[nice, must]))
    assert len(q["blocking"]) == 1 and "1.7" in q["blocking"][0] and q["stop_met"] is False
