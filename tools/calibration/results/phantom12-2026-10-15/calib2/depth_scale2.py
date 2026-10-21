# Scale (triangulated / ruler pitch) versus board distance, on every Nth left/right pair of the
# given exports that the fit did not use (not in tree/stereo2/cam0). The right calibration is flat at 1.
import sys, os, csv, importlib, numpy as np, yaml, cv2
sys.path.insert(0, "/w/scripts"); import rigcfg
V = importlib.import_module("60_validate_stereo"); S = importlib.import_module("65_scale_check")
rig = rigcfg.load("/w/config/rig-phantom12.yaml"); board = rig.board; pitch = board.pitch
chain, step, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]; exports = sys.argv[4:]
used = {int(f[:-4]) for f in os.listdir("/w/tree/stereo2/cam0") if f.endswith(".png")}
CC = yaml.safe_load(open(f"/w/{chain}-camchain.yaml")); lens = rig.camchain_lens(CC, "cam0")
K0, D0 = V.KD(CC["cam0"]); K1, D1 = V.KD(CC["cam1"]); T10 = np.array(CC["cam1"]["T_cn_cnm1"])
rows = []
for E in exports:
    L = sorted(int(f[:-4]) for f in os.listdir(f"{E}/cam1/data")); Rt = np.array(sorted(int(f[:-4]) for f in os.listdir(f"{E}/cam2/data")))
    for t in L[::step]:
        if t in used: continue
        j = int(np.argmin(np.abs(Rt - t)))
        if abs(int(Rt[j]) - t) > 1_000_000: continue
        im0 = cv2.imread(f"{E}/cam1/data/{t}.png", cv2.IMREAD_GRAYSCALE); im1 = cv2.imread(f"{E}/cam2/data/{Rt[j]}.png", cv2.IMREAD_GRAYSCALE)
        keys, p0, p1 = V.match_tags(board.detect(im0), board.detect(im1))
        if len({k for k, _ in keys}) < 6: continue
        X = S.triangulate(lens, p0, p1, K0, D0, K1, D1, T10)
        r = S.pitch_ratios(keys, X, board, pitch)
        if r: rows.append([t, float(np.median(X[:, 2])), float(np.median(r)), len({k for k, _ in keys})])
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ts", "z", "ratio", "tags"]); w.writerows(rows)
print("DEPTH-SCALE2-DONE", len(rows))
