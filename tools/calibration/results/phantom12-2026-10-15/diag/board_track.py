# Per-frame board track for finding still holds: tags seen, median tag-corner motion
# against the previous analysed frame (same tag id + corner), median tag edge in px.
import sys, os, csv, numpy as np, cv2
sys.path.insert(0, "/w/scripts"); import rigcfg
rig = rigcfg.load("/w/config/rig-phantom12.yaml"); board = rig.board
d = sys.argv[1]; step = int(sys.argv[2]); out = sys.argv[3]
names = sorted(f for f in os.listdir(d) if f.endswith(".png"))[::step]
prev = {}
with open(out, "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["ts_ns", "tags", "motion_px", "edge_px"])
    for n in names:
        g = cv2.imread(os.path.join(d, n), cv2.IMREAD_GRAYSCALE)
        found = board.detect(g)
        tags = {tid for (tid, k) in found}
        common = [k for k in found if k in prev]
        mot = float(np.median([np.hypot(*(np.subtract(found[k], prev[k]))) for k in common])) if common else -1
        edges = []
        for tid in tags:
            if all((tid, k) in found for k in range(2)):
                edges.append(float(np.hypot(*np.subtract(found[(tid, 0)], found[(tid, 1)]))))
        w.writerow([n[:-4], len(tags), f"{mot:.3f}", f"{np.median(edges) if edges else -1:.1f}"])
        prev = found
print("TRACK-DONE", len(names))
