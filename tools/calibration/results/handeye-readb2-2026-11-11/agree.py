"""readb2: the two passes as a check. A capture whose pass 2 (started at the wave's median offset) lands within
AGREE_MM of its pass 1 (started at the encoders) is "settled"; stage 2 and the stability numbers on the settled ones.

  python3 agree.py WORKDIR PASS1TAG PASS2TAG [AGREE_MM]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stab  # noqa: E402

work, t1, t2 = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
AGREE = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
out = {}
runs = {}
for v in ("base",) + stab.VARIANTS:
    p1, p2 = work / f"r_{t1}_{v}.json", work / f"r_{t2}_{v}.json"
    if not (p1.exists() and p2.exists()):
        continue
    A = {r["burst"]: r for r in json.loads(p1.read_text())["res"]}
    B = json.loads(p2.read_text())["res"]
    for r in B:
        r["agree_mm"] = float(np.linalg.norm(np.subtract(r["off"], A[r["burst"]]["off"])))
        r["clean0"] = r["clean"]
        r["clean"] = bool(r["clean"] and r["agree_mm"] <= AGREE)
    runs[v] = {r["burst"]: r for r in B}
base = runs["base"]
names = sorted(base)
settled = np.array([base[n]["clean"] for n in names])
print(f"{t1} -> {t2}, agree within {AGREE} mm: settled {settled.sum()}/48; not settled:",
      [n[12:-4] for n in names if not base[n]["clean"]])
spread = np.zeros(len(names))
for v, R in runs.items():
    if v == "base":
        continue
    d = np.array([np.linalg.norm(np.subtract(R[n]["off"], base[n]["off"])) for n in names])
    both = settled & np.array([R[n]["clean"] for n in names])
    spread = np.maximum(spread, np.where(settled, d, 0))
    print(f"  {v:6s} settled in base ({settled.sum()}): {stab.dev_stats(d[settled])}; settled in both ({both.sum()}): "
          f"{stab.dev_stats(d[both])}   flips {int((settled != np.array([R[n]['clean'] for n in names])).sum())}")
print(f"  SPREAD (worst variant per capture) over the captures settled in base: {stab.dev_stats(spread[settled])}")
for v in ("base", "f2", "f2b"):
    if v in runs:
        st = stab.stage2(list(runs[v].values()))
        for m in ("M12+head", "M12"):
            print(f"  CV {v:5s} {m:9s} right {st[m]['right']} left {st[m]['left']} (n {st['n_right']}/{st['n_left']}, "
                  f"dropped {st['dropped']})")
        print("     per pose:", st["M12+head"]["per_pose"])
