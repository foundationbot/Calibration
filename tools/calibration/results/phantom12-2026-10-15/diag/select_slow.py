# Build a filtered export holding only pairs where the board moved < MAXSPEED px/s in the left image
# (left/right exposures are ~8.5 ms apart on phantom12, so slow frames carry < 0.2 px of offset error).
import csv, os, sys
SRC, DST, MAXSPEED = sys.argv[1], sys.argv[2], float(sys.argv[3])
R = list(csv.DictReader(open("/root/calib-work/board_track_cam1_all.csv")))
slow = set(); prev = None
for r in R:
    t = int(r["ts_ns"])
    if prev is not None and int(r["tags"]) >= 4 and float(r["motion_px"]) >= 0:
        dt = (t - prev) / 1e9
        if 0 < dt < 0.15 and float(r["motion_px"]) / dt < MAXSPEED: slow.add(t)
    prev = t
def rows(cam): 
    with open(f"{SRC}/{cam}/timestamps_full.csv") as f:
        lines = f.read().splitlines()
    return lines[0], [l for l in lines[1:] if l and not l.startswith("#")]
h1, l1 = rows("cam1"); h2, l2 = rows("cam2")
t2 = sorted(int(l.split(",")[0]) for l in l2)
import bisect
keep1, keep2 = set(), set()
for l in l1:
    t = int(l.split(",")[0])
    if t not in slow: continue
    i = bisect.bisect_left(t2, t); best = min((abs(t2[j] - t), t2[j]) for j in (i - 1, i) if 0 <= j < len(t2))
    if best[0] <= 1_000_000: keep1.add(t); keep2.add(best[1])
for cam, h, lines, keep in (("cam1", h1, l1, keep1), ("cam2", h2, l2, keep2)):
    os.makedirs(f"{DST}/{cam}/data", exist_ok=True)
    with open(f"{DST}/{cam}/timestamps_full.csv", "w") as f:
        f.write(h + "\n")
        for l in lines:
            t = int(l.split(",")[0])
            if t in keep:
                f.write(l + "\n"); fn = l.split(",")[-1]
                d = f"{DST}/{cam}/data/{fn}"
                if not os.path.exists(d): os.link(f"{SRC}/{cam}/data/{fn}", d)
print(f"SELECT-DONE slow frames {len(slow)}, kept pairs {len(keep1)}")
