# Board sharpness against distance: variance of the Laplacian inside the detected board area, and
# the board distance from the median tag edge in pixels (z ~ f * 0.1 m / edge). Every 8th left frame.
import sys, os, numpy as np, cv2
sys.path.insert(0, "/w/scripts"); import rigcfg
b = rigcfg.load("/w/config/rig-phantom12.yaml").board
d = "/e/cam1/data"; fs = sorted(os.listdir(d))[::8]; rows = []
for f in fs:
    g = cv2.imread(f"{d}/{f}", cv2.IMREAD_GRAYSCALE); det = b.detect(g)
    tags = {t for t, _ in det}
    if len(tags) < 6: continue
    pts = np.array(list(det.values()), float); x0, y0 = pts.min(0).astype(int); x1, y1 = pts.max(0).astype(int)
    roi = g[y0:y1, x0:x1]; lap = cv2.Laplacian(roi, cv2.CV_64F).var()
    edges = [np.hypot(*np.subtract(det[(t, 0)], det[(t, 1)])) for t in tags if (t, 0) in det and (t, 1) in det]
    if edges: rows.append((562 * 0.1 / np.median(edges), lap))
rows.sort()
for lo, hi in ((0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.2), (1.2, 1.6), (1.6, 2.6)):
    s = [l for z, l in rows if lo <= z < hi]
    if s: print(f"  {lo:.1f}-{hi:.1f} m: n {len(s):3d}  sharpness median {np.median(s):7.0f}  (p10 {np.percentile(s,10):6.0f})")
print("SHARP-DONE", len(rows))
