"""The point-cloud agent's committed blind read-B json -> yardstick candidates in Fit C's camera 0 (Phase C3 baseline).

Read B's poses are in the burst's camera C (its T_SC); the agreed conversion (plan Phase C decision 5, the point-cloud
agent's score_blind.py): R4 = T_SC0^-1 · burst.T_SC, T_C0_hand = R4 · T_C_hand. Its held-out captures are blind
(in_sample false). One candidate file per pose kind: B, X-only and the plain chain (the last must match the
yardstick's M0 to ~0: a check of the conversion).

usage: readb_to_yardstick.py blind_b.json T_SC0-fitc.json bursts_dir out_prefix method_label
"""
import json
import sys
from pathlib import Path

import numpy as np

blind = json.loads(Path(sys.argv[1]).read_text())
T_SC0 = np.asarray(json.loads(Path(sys.argv[2]).read_text())["T_SC0"], float).reshape(4, 4)
bursts, prefix, label = Path(sys.argv[3]), sys.argv[4], sys.argv[5]
session = blind["session"]
for key, kind in (("B_T_C_hand", "B"), ("Xonly_T_C_hand", "X-only"), ("chain_T_C_hand", "chain")):
    caps = {}
    for n, p in blind["predictions"].items():
        T_SC = np.asarray(np.load(bursts / session / f"{n}.npz")["T_SC"], float).reshape(4, 4)
        R4 = np.linalg.inv(T_SC0) @ T_SC
        caps[n] = {"T_C0_hand": (R4 @ np.asarray(p[key], float)).tolist(), "in_sample": False}
    out = Path(f"{prefix}-{kind}.json")
    out.write_text(json.dumps({"method": f"{label} {kind}", "captures": caps}, indent=1))
    print(out)
