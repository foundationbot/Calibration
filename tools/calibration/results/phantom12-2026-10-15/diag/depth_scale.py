# Scale (triangulated / ruler pitch) versus board distance for several stereo camchains, on pairs
# with board speed < MAXV px/s that are NOT in the slow fit. The right calibration is flat at 1.
import sys, os, csv, importlib, numpy as np, yaml, cv2
sys.path.insert(0, "/w/scripts"); import rigcfg
V = importlib.import_module("60_validate_stereo"); S = importlib.import_module("65_scale_check")
rig = rigcfg.load("/w/config/rig-phantom12.yaml"); board = rig.board; pitch = board.pitch
MAXV = float(sys.argv[1]); chains = sys.argv[2:]
used = {int(f[:-4]) for f in os.listdir("/w/tree/slow/cam0") if f.endswith(".png")}
spd = {}; prev = None
for r in csv.DictReader(open("/w/board_track_cam1_all.csv")):
    t = int(r["ts_ns"])
    if prev and int(r["tags"]) >= 4 and float(r["motion_px"]) >= 0 and 0 < (t - prev) / 1e9 < 0.15:
        spd[t] = float(r["motion_px"]) / ((t - prev) / 1e9)
    prev = t
E = "/e"; Rn = sorted(os.listdir(E + "/cam2/data")); rt = np.array([int(n[:-4]) for n in Rn])
cams = []
for c in chains:
    CC = yaml.safe_load(open(f"/w/{c}-camchain.yaml")); lens = rig.camchain_lens(CC, "cam0")
    K0, D0 = V.KD(CC["cam0"]); K1, D1 = V.KD(CC["cam1"]); cams.append((c, lens, K0, D0, K1, D1, np.array(CC["cam1"]["T_cn_cnm1"])))
rows = []
for t, v in sorted(spd.items()):
    if v >= MAXV or t in used: continue
    j = int(np.argmin(np.abs(rt - t)))
    if abs(int(rt[j]) - t) > 1_000_000: continue
    im0 = cv2.imread(f"{E}/cam1/data/{t}.png", cv2.IMREAD_GRAYSCALE); im1 = cv2.imread(f"{E}/cam2/data/{Rn[j]}", cv2.IMREAD_GRAYSCALE)
    keys, p0, p1 = V.match_tags(board.detect(im0), board.detect(im1))
    if len({k for k, _ in keys}) < 6: continue
    row = [t, v]
    for c, lens, K0, D0, K1, D1, T10 in cams:
        X = S.triangulate(lens, p0, p1, K0, D0, K1, D1, T10)
        r = S.pitch_ratios(keys, X, board, pitch)
        row += [float(np.median(X[:, 2])), float(np.median(r)) if r else float("nan")]
    rows.append(row)
with open("/w/depth_scale.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ts", "speed"] + sum([[f"z_{c}", f"ratio_{c}"] for c in chains], [])); w.writerows(rows)
print("DEPTH-SCALE-DONE", len(rows))
