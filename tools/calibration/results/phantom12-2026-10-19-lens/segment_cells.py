"""Where a lens-take segment's still pairs fall (plan of 19 Oct, revision 2, Task 5 Step 5): pairs per cell
(board-centre band x side, as bands590.py) from step 15's corner files, the board centre being the mean of
camera 0's corners. Usage (Thor, calib-py:3, bands590.py in /d): python segment_cells.py <corners.npz>..."""
import sys

import numpy as np

sys.path.insert(0, "/d")
import bands590 as bd  # noqa: E402

for path in sys.argv[1:]:
    d = np.load(path)
    m0 = d["det_cam"] == 0
    img, u, v = d["det_img"][m0], d["det_u"][m0], d["det_v"][m0]
    n = len(d["img_hold"])
    lab = bd.cells({"u": [u[img == j].mean() for j in range(n)], "v": [v[img == j].mean() for j in range(n)]})
    counts = [(c, int((lab == c).sum())) for c in bd.CELLS if (lab == c).any()]
    print(f"{path}: {n} still pairs in {len(np.unique(d['img_hold']))} still stretches | "
          + ", ".join(f"{c} {k}" for c, k in counts), flush=True)
