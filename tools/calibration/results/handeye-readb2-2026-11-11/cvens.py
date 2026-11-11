"""readb2: stage 2's leave-one-arm-pose-out CV under every perturbation of a config (the CV of one run hangs on one
or two captures; the spread over runs that should not differ shows how much).

  python3 cvens.py WORKDIR TAG[:PASS1TAG] ...   (with PASS1TAG: clean also needs pass 2 within 3 mm of pass 1)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stab  # noqa: E402

work = Path(sys.argv[1])
res = {}
for spec in sys.argv[2:]:
    tag, _, p1 = spec.partition(":")
    rows = {}
    for v in ("base",) + stab.VARIANTS:
        f = work / f"r_{tag}_{v}.json"
        if not f.exists() or (p1 and not (work / f"r_{p1}_{v}.json").exists()):
            continue
        recs = json.loads(f.read_text())["res"]
        if p1:
            A = {r["burst"]: r for r in json.loads((work / f"r_{p1}_{v}.json").read_text())["res"]}
            for r in recs:
                r["clean"] = bool(r["clean"] and np.linalg.norm(np.subtract(r["off"], A[r["burst"]]["off"])) <= 3.0)
        st = stab.stage2(recs)
        rows[v] = {m: st[m]["right"][:2] + st[m]["left"][:2] for m in ("M12+head", "M12")}
        rows[v]["n"] = [st["n_right"], st["n_left"]]
    print(f"\n== {spec}   (right median, p90 | left median, p90; n right/left)")
    for m in ("M12+head", "M12"):
        for v, r in rows.items():
            print(f"  {m:9s} {v:6s} {r[m][0]:5.2f} {r[m][1]:5.2f} | {r[m][2]:5.2f} {r[m][3]:5.2f}   n {r['n']}")
        same = [r[m] for v, r in rows.items() if v in ("base", "jit", "phase", "start")]
        two = [r[m] for v, r in rows.items() if v in ("f2", "f2b")]
        three = [r[m] for v, r in rows.items() if v == "f3"]
        for lab, g in (("5-frame runs, mean", same), ("2-frame runs, mean", two), ("3-frame run", three)):
            if g:
                a = np.mean(g, axis=0)
                print(f"  {m:9s} {lab}: {a[0]:5.2f} {a[1]:5.2f} | {a[2]:5.2f} {a[3]:5.2f}  (of {len(g)})")
    res[spec] = rows
print(json.dumps(res))
