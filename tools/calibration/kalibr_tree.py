#!/usr/bin/env python3
"""Lay out the folder kalibr_bagcreater reads, from one EuRoC take, with RAW stamps.

  python3 tools/calibration/kalibr_tree.py <euroc take> <tree out> --cams 3,0

<euroc take> is what tools/mcap_to_euroc.py writes (cam<N>/data/<ns>.png, imu0/data.csv). The tree is
  <out>/cam0/<ns>.png -> <take>/cam<first of --cams>/data/<ns>.png    (symlinks, absolute)
  <out>/cam1/<ns>.png -> <take>/cam<second>/data/<ns>.png
  <out>/imu0.csv                                                       (header dropped, sorted by stamp)

--cams is the DMA camera ids in OKVIS pair order, so kalibr's cam0/cam1 are OKVIS's
(phantom21: 2,0; P10: 3,0). The PNG names are the capture stamps exactly as recorded -- already
on the IMU's clock. The old prep_kalibr_tree.py renamed them by min(publish - capture), which
kalibr reported back as a -27 ms timeshift (docs/phantom21-calibration-runbook.md §2); this
script never renames.

The links are absolute, so mount the data at the SAME path inside the kalibr container. Prints
how many frames each camera has and how many cam0 stamps have a cam1 partner within 1 ms (an
OAK stereo pair is hardware-synced, so that should be nearly all of them).
"""
from __future__ import annotations

import argparse
import bisect
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("out")
    ap.add_argument("--cams", required=True, help="DMA camera ids, OKVIS pair order (P10: 3,0)")
    a = ap.parse_args(argv)
    take, out = os.path.abspath(a.take), os.path.abspath(a.out)
    cams = [int(x) for x in a.cams.split(",")]
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} is not empty; remove it first (a stale link would be bagged)")
    stamps = []
    for k, c in enumerate(cams):
        src = os.path.join(take, f"cam{c}", "data")
        if not os.path.isdir(src):
            sys.exit(f"{src} missing: the take has no DMA camera {c}")
        dst = os.path.join(out, f"cam{k}")
        os.makedirs(dst)
        names = sorted(f for f in os.listdir(src) if f.endswith(".png"))
        for f in names:
            os.symlink(os.path.join(src, f), os.path.join(dst, f))
        stamps.append([int(f[:-4]) for f in names])
        print(f"cam{k} <- DMA cam{c}: {len(names)} frames")
    imu_src = os.path.join(take, "imu0", "data.csv")
    with open(imu_src) as f:
        rows = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    rows.sort(key=lambda ln: int(ln.split(",", 1)[0]))
    with open(os.path.join(out, "imu0.csv"), "w") as f:
        f.writelines(rows)
    print(f"imu0.csv: {len(rows)} samples")
    if not rows:
        print("WARNING: no IMU samples -- fine for the stereo take, fatal for the camera-IMU one")
    if len(stamps) == 2 and stamps[1]:
        other, paired = stamps[1], 0
        for t in stamps[0]:
            i = bisect.bisect_left(other, t)
            if min(abs(other[j] - t) for j in (i - 1, i) if 0 <= j < len(other)) <= 1_000_000:
                paired += 1
        print(f"cam0 frames with a cam1 frame within 1 ms: {paired}/{len(stamps[0])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
