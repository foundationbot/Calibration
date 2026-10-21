# Does the rectified row error track the board vertical speed? If left and right expose at
# different instants, dy = -vy * dt (dt = right minus left exposure time). Per sampled pair:
# signed median dy, median |dy|, vertical speed vy of the corners in the left image (from the
# previous left frame), mean corner radius from the image centre, and tag count.
import sys, os, re, importlib, csv, numpy as np, yaml
sys.path.insert(0, "/w/scripts"); import rigcfg
V = importlib.import_module("60_validate_stereo")
rig = rigcfg.load("/w/config/rig-phantom12.yaml"); board = rig.board
CC = yaml.safe_load(open(os.environ.get("CC", "/w/calib_stereo-camchain.yaml")))  # CC=<camchain> overrides
lens = rig.camchain_lens(CC, "cam0"); rig.camchain_lens(CC, "cam1")
K0, D0 = V.KD(CC["cam0"]); K1, D1 = V.KD(CC["cam1"])
T10 = np.array(CC["cam1"]["T_cn_cnm1"]); R0, R1, P0, P1 = V.rectify(lens, K0, D0, K1, D1, T10, rig.resolution)
E = "/e"; L = sorted(os.listdir(E + "/cam1/data")); Rn = sorted(os.listdir(E + "/cam2/data"))
lt = np.array([int(n[:-4]) for n in L]); rt = np.array([int(n[:-4]) for n in Rn])
step = int(sys.argv[1]); out = sys.argv[2]
def det(cam, n): return board.detect(V._read_grey(f"{E}/{cam}/data/{n}", rig.resolution))
rows = []
for i in range(1, len(L), step):
    j = int(np.argmin(np.abs(rt - lt[i])))
    if abs(int(rt[j]) - int(lt[i])) > 1_000_000: continue
    d0, d1, dp = det("cam1", L[i]), det("cam2", Rn[j]), det("cam1", L[i - 1])
    keys = sorted(set(d0) & set(d1))
    if len({t for t, _ in keys}) < 4: continue
    p0 = np.array([d0[k] for k in keys], float).reshape(-1, 1, 2); p1 = np.array([d1[k] for k in keys], float).reshape(-1, 1, 2)
    u0 = lens.undistort(p0, K0, D0, R=R0, P=P0); u1 = lens.undistort(p1, K1, D1, R=R1, P=P1)
    dy = u0[:, 0, 1] - u1[:, 0, 1]
    kv = [k for k in keys if k in dp]; dt_prev = (lt[i] - lt[i - 1]) * 1e-9
    vy = float(np.median([(d0[k][1] - dp[k][1]) / dt_prev for k in kv])) if len(kv) >= 8 and dt_prev < 0.2 else float("nan")
    rad = float(np.mean(np.hypot(p0[:, 0, 0] - 640, p0[:, 0, 1] - 400)))
    rows.append([lt[i], len({t for t, _ in keys}), float(np.median(dy)), float(np.median(np.abs(dy))), vy, rad])
with open(out, "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["ts", "tags", "dy_med", "absdy_med", "vy_pxs", "radius_px"]); w.writerows(rows)
print("SYNC-DIAG-DONE", len(rows))
