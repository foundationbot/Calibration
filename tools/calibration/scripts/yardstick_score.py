#!/usr/bin/env python3
"""Score a tag-free method's hand poses against the frozen plate yardstick (yardstick.py, yardstick_build.py).

candidate.json:
  {"method": "<name and commit>",
   "captures": {"<capture>": {"T_C0_hand": 4x4, "in_sample": true|false}, ...}}
T_C0_hand = RightGripperYawLink in Fit C's camera 0 (T_BS·T_SC[0]): convert a burst-camera pose first, as the
point-cloud agent's score_blind.py does (R4 = T_SC0^-1 · burst.T_SC, T_C0 = R4 · T_C). in_sample = the method fitted
on that capture; only held-out captures count. Captures are named as yardstick.json names them.

Every reference model is shown (B1 decides; M1 and M0 show whether a model choice could flip a verdict).

usage: yardstick_score.py yardstick_dir candidate.json [out.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yardstick as ys  # noqa: E402


def main(root, cand_path, out=None):
    ys.check_manifest(root)
    doc = json.loads((Path(root) / "yardstick.json").read_text())
    cand = json.loads(Path(cand_path).read_text())
    res = {"method": cand.get("method"), "reference": doc["reference"]}
    for model in ("B1", "M1", "M0"):
        ref = {n: c[model] for n, c in doc["captures"].items()}
        res[model] = ys.score(ref, cand["captures"])
    unknown = sorted(set(cand["captures"]) - set(doc["captures"]))
    res["unknown_captures"] = unknown
    b = res["B1"]
    print(f"{res['method']}: verdict {b['verdict']} vs B1 ({b['held_out_pass']}/{b['held_out']} held-out pass "
          f"{ys.GATE_MM} mm / {ys.GATE_DEG} deg); vs M1 {res['M1']['verdict']}, vs M0 {res['M0']['verdict']}")
    for n, c in sorted(b["captures"].items()):
        m1, m0 = res["M1"]["captures"][n], res["M0"]["captures"][n]
        print(f"  {n:24s} {'held-out ' if c['counted'] else 'in-sample'} B1 {c['wrist_mm']:6.2f} mm {c['rot_deg']:6.3f} deg"
              f" {'pass' if c['pass'] else 'FAIL'} | M1 {m1['wrist_mm']:6.2f} {m1['rot_deg']:6.3f} | M0 {m0['wrist_mm']:6.2f}"
              f" {m0['rot_deg']:6.3f}")
    if b["missing"]:
        print("  missing:", " ".join(b["missing"]))
    if unknown:
        print("  not in the yardstick:", " ".join(unknown))
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:4])
