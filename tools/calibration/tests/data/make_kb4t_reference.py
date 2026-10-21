"""Writes kb4t_reference.json next to this file: Fit A's two cameras (20 Oct) projected by lensfit.project, the
analysis code of 18-20 Oct that fitted Fit A (outside the repo), so tests/test_lensmodel.py pins the package's KB4T
arithmetic to the code that produced the calibration. Not a test. Run once, from the repo root:
    uv run --with numpy --with scipy python tools/calibration/tests/data/make_kb4t_reference.py <lensfit folder>"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
import lensfit  # noqa: E402

INTR = [[564.8806398823662, 563.9275439964279, 642.8295109855806, 384.9454915650745, -0.00828172594207738,
         0.011863175838579886, -0.0005047335197465951, -0.001418336722152802, 0.0008523071658447929,
         -0.0011130944412204317],
        [567.205603114578, 566.1503426681027, 642.5170962119012, 383.0559544886883, -0.006095703857029558,
         0.00663703918848934, 0.0030852663661844874, -0.0023449544984341826, -0.00037091569127222347,
         -7.265917199549953e-05]]
rng = np.random.default_rng(20261020)
th, ph, r = np.radians(rng.uniform(0, 75, 60)), rng.uniform(-np.pi, np.pi, 60), rng.uniform(0.4, 2.0, 60)
P = np.c_[r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)]
out = {"source": "lensfit.project('KB4T', intr, points), analysis code of 18-20 Oct (tbs-2026-10-17/lens/lensfit.py)",
       "calibration": "Fit A, Thor /root/calib-proto/data/final-fit-nofar.json key KB4T@all@800",
       "intr": INTR, "points": P.tolist(),
       "pixels": [lensfit.project("KB4T", np.array(v), P).tolist() for v in INTR]}
Path(__file__).with_name("kb4t_reference.json").write_text(json.dumps(out, indent=1))
print(f"wrote {len(P)} points x 2 cameras")
