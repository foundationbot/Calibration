# Build a filtered copy of tree/stereo3 (hard links) for the combination tests:
#   nonear: drop pairs whose board is closer than LIMIT metres (z ~ f * 0.1 m / median tag edge px)
#   noedge: drop pairs with any detected corner more than LIMIT px from the image centre (left image)
# Left and right files of a pair carry each camera's own stamp, so pairs are matched by sorted order
# (step 10 writes them as pairs; the counts are checked to be equal).
import sys, os, numpy as np, cv2
sys.path.insert(0, "/w/scripts"); import rigcfg
b = rigcfg.load("/w/config/rig-phantom12.yaml").board
mode, LIMIT = sys.argv[1], float(sys.argv[2])
src = "/w/tree/stereo3"; dst = f"/w/tree/stereo3{mode}"
L = sorted(f for f in os.listdir(f"{src}/cam0") if f.endswith(".png"))
R = sorted(f for f in os.listdir(f"{src}/cam1") if f.endswith(".png"))
if len(L) != len(R): sys.exit(f"left/right counts differ: {len(L)} vs {len(R)}")
for c in ("cam0", "cam1"): os.makedirs(f"{dst}/{c}", exist_ok=True)
kept = dropped = 0; worst_gap = 0
for fl, fr in zip(L, R):
    worst_gap = max(worst_gap, abs(int(fl[:-4]) - int(fr[:-4])))
    det = b.detect(cv2.imread(f"{src}/cam0/{fl}", cv2.IMREAD_GRAYSCALE))
    drop = False
    if det:
        pts = np.array(list(det.values()), float)
        if mode == "nonear":
            tags = {t for t, _ in det}
            edges = [np.hypot(*np.subtract(det[(t, 0)], det[(t, 1)])) for t in tags if (t, 0) in det and (t, 1) in det]
            drop = bool(edges) and 562 * 0.1 / float(np.median(edges)) < LIMIT
        else:
            drop = float(np.max(np.hypot(pts[:, 0] - 640, pts[:, 1] - 400))) > LIMIT
    if drop: dropped += 1; continue
    os.link(f"{src}/cam0/{fl}", f"{dst}/cam0/{fl}"); os.link(f"{src}/cam1/{fr}", f"{dst}/cam1/{fr}"); kept += 1
print(f"FILTER-DONE {mode} {LIMIT}: kept {kept} dropped {dropped}; worst left/right stamp gap {worst_gap / 1e6:.3f} ms")
