"""H0 burst length: read B on the stored phantom14 bursts fused from their first N depth frames (bench.py --depth numpy
--frames N), against all 5 frames: per-capture wrist-offset deviation, and what stage 2 (M12+head, leave one arm pose
out, wave 2 of both arms as in H-1) reaches with each.

  python3 frames.py OUT5.json OUT1.json OUT2.json ... > frames.log"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from armcal import urdffit  # noqa: E402
from armcal.urdf import Urdf  # noqa: E402

DATA = Path.home() / "calibtool" / "data" / "p14arm"


def stage2(recs):
    base = Urdf.load(DATA / "phantom_mk1.urdf")
    R = urdffit.observations([r for r in recs if "1104a-right" in r["burst"]], "Right")
    L = urdffit.observations([r for r in recs if "1104b-left" in r["burst"]], "Left")
    R, dR = urdffit.select(R)
    L, dL = urdffit.select(L)
    spec = urdffit.MODELS["M12+head"]
    f = urdffit.fit(spec, R, L, base, "M12+head")
    cv = urdffit.cv_summary(urdffit.lopo(spec, R, L, base, "M12+head"))
    return dict(n_right=len(R), n_left=len(L), dropped=dR + dL, x=np.round(f.x, 2).tolist(),
                se=np.round(f.se, 2).tolist(), cond=round(f.cond),
                cv_right=[round(cv["Right"][k], 1) for k in ("median", "p90", "max")],
                cv_left=[round(cv["Left"][k], 1) for k in ("median", "p90", "max")])


def main():
    runs = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
    ref = {r["burst"]: r for r in runs[0]["res"]}
    print("stage 2 M12+head per frame count (CV = held-out wrist error, mm: median / p90 / max)")
    for run in runs:
        recs = run["res"]
        n = run["summary"]["frames"] or 5
        d = np.array([np.linalg.norm(np.subtract(r["off"], ref[r["burst"]]["off"])) for r in recs])
        cl = np.array([ref[r["burst"]]["clean"] for r in recs])
        flips = [r["burst"][12:-4] for r in recs if r["clean"] != ref[r["burst"]]["clean"]]
        print(f"\n== {n} frame(s): vs 5 frames, all {len(d)}: median {np.median(d):.2f} max {d.max():.2f} mm; "
              f"clean-at-5 ({cl.sum()}): median {np.median(d[cl]):.2f} max {d[cl].max():.2f} mm; "
              f"clean flips {len(flips)} {flips}")
        print("   fuse s mean %.2f, fit s mean %.1f" % (np.mean([r["fuse_s"] for r in recs]),
                                                       np.mean([r["fit_s"] for r in recs])))
        print("   stage 2:", json.dumps(stage2(recs)))


if __name__ == "__main__":
    main()
