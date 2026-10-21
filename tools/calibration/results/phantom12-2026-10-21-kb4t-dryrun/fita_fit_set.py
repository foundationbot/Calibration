"""Fit A's fit set (final_fit.sets() with FF_NOFAR=1, half == 0) as one step-15 npz; img_hold = Fit A's clusters"""
import os
import sys

import numpy as np

os.environ["FF_NOFAR"] = "1"
sys.path.insert(0, "/d")
import final_fit as ff  # noqa: E402

d, sid, half, clus = ff.sets()
keep = np.flatnonzero(half == 0)
new = np.full(len(half), -1)
new[keep] = np.arange(len(keep))
m = np.isin(d["det_img"], keep)
out = {"img_hold": np.asarray(clus)[keep].astype(np.int64), "img_t0": np.asarray(d["img_t0"])[keep],
       "img_t1": np.asarray(d["img_t0"])[keep], "det_img": new[d["det_img"][m]].astype(np.int64)}
for k in ("det_cam", "det_tag", "det_k", "det_u", "det_v"):
    out[k] = np.asarray(d[k])[m]
np.savez_compressed(sys.argv[1], **out)
print(f"{len(keep)} pairs in {len(np.unique(out['img_hold']))} holds -> {sys.argv[1]}")
