# Copy a step-10 tree leaving out images with any tag corner more than LIMIT px from the image
# centre: the 4-parameter lens models cannot fit this lens's outer ring (phantom12, 15 Oct 2026).
#   stereo tree (no imu0.csv): pairs are dropped together, judged on the left image, as for the
#     recommended stereo fit (calib_stereo3noedge).
#   imucam tree (imu0.csv present): step 10 keeps every frame of each camera, unpaired, so each
#     camera is filtered on its own corners.
# Hard links; imu0.csv and SOURCE.json are carried over; frames with no board are kept.
# usage: python filter_edges.py <tree> <out> <limit px>      (JOBS=n detector processes, default 2)
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, "/w/scripts")
import rigcfg  # noqa: E402

b = rigcfg.load("/w/config/rig-phantom12.yaml").board
src, dst, LIMIT = sys.argv[1], sys.argv[2], float(sys.argv[3])
JOBS = int(os.environ.get("JOBS", "2"))


def outside(path):
    det = b.detect(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    if not det:
        return False
    pts = np.array(list(det.values()), float)
    return float(np.max(np.hypot(pts[:, 0] - 640, pts[:, 1] - 400))) > LIMIT


def pngs(cam):
    return sorted(f for f in os.listdir(f"{src}/{cam}") if f.endswith(".png"))


L, R = pngs("cam0"), pngs("cam1")
imucam = os.path.exists(f"{src}/imu0.csv")
if not imucam and len(L) != len(R):
    sys.exit(f"stereo tree, but left/right counts differ: {len(L)} vs {len(R)}")
for c in ("cam0", "cam1"):
    os.makedirs(f"{dst}/{c}", exist_ok=True)
for extra in ("imu0.csv", "SOURCE.json"):
    if os.path.exists(f"{src}/{extra}") and not os.path.exists(f"{dst}/{extra}"):
        os.link(f"{src}/{extra}", f"{dst}/{extra}")

with Pool(JOBS) as pool:
    if imucam:
        for cam, names in (("cam0", L), ("cam1", R)):
            out = pool.map(outside, [f"{src}/{cam}/{f}" for f in names], chunksize=16)
            kept = 0
            for f, o in zip(names, out):
                if not o:
                    os.link(f"{src}/{cam}/{f}", f"{dst}/{cam}/{f}")
                    kept += 1
            print(f"FILTER-EDGES {cam}: kept {kept} dropped {len(names) - kept} (corners beyond {LIMIT:.0f} px)")
    else:
        out = pool.map(outside, [f"{src}/cam0/{f}" for f in L], chunksize=16)
        kept = 0
        for fl, fr, o in zip(L, R, out):
            if not o:
                os.link(f"{src}/cam0/{fl}", f"{dst}/cam0/{fl}")
                os.link(f"{src}/cam1/{fr}", f"{dst}/cam1/{fr}")
                kept += 1
        print(f"FILTER-EDGES pairs: kept {kept} dropped {len(L) - kept} (corners beyond {LIMIT:.0f} px)")
print("FILTER-EDGES-DONE")
