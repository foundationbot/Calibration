"""readb2: per-capture stability like for like -- over the captures clean (FAST) or clean and settled (robust) in
both the base run and the perturbed one, the wrist offset's deviation (mm).

  python3 both.py WORKDIR"""
import json, sys
import numpy as np
W = sys.argv[1]
def load(t, v):
    try: return {r["burst"]: r for r in json.load(open(f"{W}/r_{t}_{v}.json"))["res"]}
    except FileNotFoundError: return None
def st(d):
    d = np.asarray(d); return "n %2d median %.2f p90 %.2f max %.2f >3mm %d" % (len(d), np.median(d), np.percentile(d, 90), d.max(), (d > 3).sum())
for tag, p1 in (("fast", None), ("K2", "K1")):
    print("==", tag)
    def clean(t, v):
        R = load(t, v)
        if p1 is None: return R, {n: r["clean"] for n, r in R.items()}
        A = load(p1, v)
        return R, {n: r["clean"] and np.linalg.norm(np.subtract(r["off"], A[n]["off"])) <= 3 for n, r in R.items()}
    B, cb = clean(tag, "base")
    worst = {}
    allv = {n: cb[n] for n in B}
    for v in ("jit", "phase", "f2", "f2b", "f3", "start"):
        if load(tag, v) is None: continue
        R, cv = clean(tag, v)
        both = [n for n in B if cb[n] and cv[n]]
        d = [np.linalg.norm(np.subtract(R[n]["off"], B[n]["off"])) for n in both]
        print(f"  {v:6s} clean in both: {st(d)}")
        for n in B:
            allv[n] = allv[n] and cv[n]
            if cb[n] and cv[n]:
                worst[n] = max(worst.get(n, 0), np.linalg.norm(np.subtract(R[n]["off"], B[n]["off"])))
    w = [worst[n] for n in B if allv[n]]
    print(f"  worst variant, captures clean in base and in every variant: {st(w)}")
