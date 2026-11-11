"""readb2: stability and accuracy of one bench config (b2.py outputs r_<tag>_<variant>.json in a work dir).

  python3 stab.py WORKDIR TAG [TAG...]  -> per tag: the per-capture deviation of each perturbation from "base"
                                           (wrist offset, mm), clean flips, and stage 2 (leave-one-arm-pose-out
                                           CV on wave 2 of both arms, as H-1) for M12+head and M12; JSON on stdout's
                                           last line
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from armcal import urdffit  # noqa: E402
from armcal.urdf import Urdf  # noqa: E402

DATA = Path.home() / "calibtool" / "data" / "p14arm"
VARIANTS = ("jit", "phase", "f2", "f2b", "start", "f3")


def stage2(recs, models=("M12+head", "M12"), only=None):
    """only: a set of burst names: those captures (clean or not), instead of the clean ones"""
    base = Urdf.load(DATA / "phantom_mk1.urdf")
    if only is not None:
        recs = [dict(r, clean=True) for r in recs if r["burst"] in only]
    R = urdffit.observations([r for r in recs if "1104a-right" in r["burst"]], "Right")
    L = urdffit.observations([r for r in recs if "1104b-left" in r["burst"]], "Left")
    R, dR = urdffit.select(R)
    L, dL = urdffit.select(L)
    out = dict(n_right=len(R), n_left=len(L), dropped=dR + dL)
    for m in models:
        spec = urdffit.MODELS[m]
        f = urdffit.fit(spec, R, L, base, m)
        folds = urdffit.lopo(spec, R, L, base, m)
        cv = urdffit.cv_summary(folds)
        out[m] = dict(x=np.round(f.x, 2).tolist(), se=np.round(f.se, 2).tolist(), cond=round(f.cond),
                      right=[round(cv["Right"][k], 2) for k in ("median", "p90", "max")],
                      left=[round(cv["Left"][k], 2) for k in ("median", "p90", "max")],
                      pooled=round(cv["pooled_median"], 2),
                      per_pose={r["pose"]: [r["n"], round(float(np.median(r["new"])), 1)] for r in folds})
    return out


def dev_stats(d):
    d = np.asarray(d)
    if not len(d):
        return None
    return dict(n=len(d), median=round(float(np.median(d)), 2), p90=round(float(np.percentile(d, 90)), 2),
                max=round(float(d.max()), 2), over1=int((d > 1).sum()), over3=int((d > 3).sum()))


FAST_CLEAN = set()


def main():
    work = Path(sys.argv[1])
    summary = {}
    fast = work / "r_fast_base.json"
    if fast.exists():                  # "common": the captures clean under both this config and FAST
        FAST_CLEAN.update(r["burst"] for r in json.loads(fast.read_text())["res"] if r["clean"])
    for tag in sys.argv[2:]:
        runs = {v: json.loads(p.read_text()) for v in ("base",) + VARIANTS
                if (p := work / f"r_{tag}_{v}.json").exists()}
        base = {r["burst"]: r for r in runs["base"]["res"]}
        names = sorted(base)
        clean = np.array([base[n]["clean"] for n in names])
        s = dict(fit_s_mean=round(runs["base"]["summary"]["fit_s_mean"], 2),
                 fit_s_max=round(runs["base"]["summary"]["fit_s_max"], 2), clean=int(clean.sum()),
                 steps_median=runs["base"]["summary"]["steps_median"], variants={})
        spread = np.zeros(len(names))
        flips_any = set()
        print(f"\n===== {tag}: {runs['base']['summary']['cfg']}")
        print(f"fit s/capture (8 parallel) mean {s['fit_s_mean']} max {s['fit_s_max']}, clean {clean.sum()}/48")
        per = {}
        for v, run in runs.items():
            if v == "base":
                continue
            rv = {r["burst"]: r for r in run["res"]}
            d = np.array([np.linalg.norm(np.subtract(rv[n]["off"], base[n]["off"])) for n in names])
            flips = [n[12:-4] for n in names if rv[n]["clean"] != base[n]["clean"]]
            flips_any |= set(flips)
            spread = np.maximum(spread, d)
            per[v] = d
            s["variants"][v] = dict(all=dev_stats(d), clean=dev_stats(d[clean]), flips=len(flips))
            print(f"  {v:6s} all {dev_stats(d)}  clean {dev_stats(d[clean])}  flips {len(flips)}")
        s["spread"] = dict(all=dev_stats(spread), clean=dev_stats(spread[clean]), flips=len(flips_any))
        print(f"  SPREAD all {s['spread']['all']}  clean {s['spread']['clean']}  flips {len(flips_any)}")
        s["per_capture"] = {n[12:-4]: dict(spread=round(float(spread[i]), 2), clean=bool(clean[i]),
                                           off=np.round(base[n]["off"], 1).tolist(), rot=round(base[n]["rot"], 1),
                                           **{v: round(float(per[v][i]), 1) for v in per})
                            for i, n in enumerate(names)}
        s["stage2"] = stage2(runs["base"]["res"])
        both = {n for n in names if base[n]["clean"] and n in FAST_CLEAN}
        s["stage2_common"] = stage2(runs["base"]["res"], only=both)
        for lab, st in (("own", s["stage2"]), ("common", s["stage2_common"])):
            for m in ("M12+head", "M12"):
                print(f"  CV {lab:6s} {m:9s} right {st[m]['right']} left {st[m]['left']} pooled {st[m]['pooled']}  "
                      f"(n {st['n_right']}/{st['n_left']}, dropped {st['dropped']})")
        print("  per held-out pose (own, M12+head): [captures, median mm]", s["stage2"]["M12+head"]["per_pose"])
        for v in ("f2", "f2b", "f3"):  # the burst-length question: stage 2 on 2-frame fusions
            if v in runs:
                st = s["stage2_" + v] = stage2(runs[v]["res"])
                for m in ("M12+head", "M12"):
                    print(f"  CV {v:6s} {m:9s} right {st[m]['right']} left {st[m]['left']} pooled {st[m]['pooled']}  "
                          f"(n {st['n_right']}/{st['n_left']}, dropped {st['dropped']})")
        summary[tag] = s
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
