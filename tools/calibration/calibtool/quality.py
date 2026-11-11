"""The per-section data status the operator sees during a session (10 Nov 2026, phantom11: "show progress for each
take and quality status so one can see quality among different sections of the camera").

snapshot(collection) -> a JSON-ready dict for calibd's state["quality"]:
  cams       per camera ("0" left, "1" right): the image in 3 x 3 sections (rows top / middle / bottom, columns left /
             centre / right), the corners captured in each and the percent of the per-section target
             (Targets.third_min, the target of the image-corner thirds; the stop rule judges the four corners), the
             edge strips and the bottom strip against its target
  heldout    the held-out check (what decides a deploy): per lenscells section, the TEST holds in each distance band
             against Targets.test_holds; "judged" counts the (section, band) entries that reached it, against
             Targets.judged_min; the far bands against Targets.far_band_min
  blocking   the sentences of what still blocks the stop rule (most important first), at most 4
  stop_met   the collection has met the stop rule
Counted the way the stop rule counts them (Collection.tallies / coverage)."""
from calibtool import coverage as cv

ROWS = ("top", "middle", "bottom")
COLS = ("left", "centre", "right")
SECTIONS = (("top-left", "top", "top-right"), ("left", "centre", "right"),
            ("bottom-left", "bottom", "bottom-right"))           # lenscells.CELLS as a 3 x 3 grid
BANDS = ((0.4, 0.6), (0.6, 0.9), (0.9, 1.3))                    # lenscells.DIST_BINS


def _pct(have, want) -> int:
    return 100 if not want else int(min(999, round(100.0 * float(have) / float(want))))


def snapshot(col) -> dict:
    tg = col.targets
    t = col.tallies()
    reg = t[getattr(col, "region_tally", "all")]
    cams = {}
    for c in (0, 1):
        ct = reg.cams.get(c) if isinstance(reg.cams, dict) else None
        if ct is None:
            continue
        th = [[int(ct.thirds[i][j]) for j in range(3)] for i in range(3)]
        strip = (tg.strip_min or {}).get("bottom")
        cams[str(c)] = {"sections": th, "pct": [[_pct(v, tg.third_min) for v in row] for row in th],
                        "edges": {k: int(v) for k, v in ct.edges.items()},
                        "bottom_strip_pct": None if not strip else _pct(ct.edges.get("bottom", 0), strip)}
    test = t.get("test")
    ent = dict(getattr(test, "entries", None) or {})
    held = [[[int(ent.get((cell, b), 0)) for b in BANDS] for cell in row] for row in SECTIONS]
    judged = sum(1 for n in ent.values() if n >= tg.test_holds)
    far = {f"{lo:g}-{hi:g}": int(n) for (lo, hi), n in (getattr(test, "far_bands", None) or {}).items()}
    cov = col.coverage(t)
    blocking = [d for d in cov.deficits(tg) if getattr(d, "blocks", True)]
    sentences = []
    for d in blocking[:4]:
        try:
            sentences.append(cv.sentence(d))
        except Exception:  # noqa: BLE001  (a sentence is a courtesy)
            sentences.append(f"{d.kind} {d.region}: {d.have} of {d.want:g}")
    return {"cams": cams, "third_min": int(tg.third_min),
            "heldout": {"sections": held, "bands": [f"{lo:g}-{hi:g}" for lo, hi in BANDS],
                        "want": int(tg.test_holds), "judged": judged, "judged_min": int(tg.judged_min),
                        "far": far, "far_min": int(tg.far_band_min)},
            "blocking": sentences, "stop_met": not blocking}
