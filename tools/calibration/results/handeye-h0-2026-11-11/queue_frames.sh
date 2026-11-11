#!/bin/bash
# H0 burst length: numpy fusion of the first N frames, FAST matcher, all 48 captures; then the comparison
cd ~/calibtool/wt-h0/tools/calibration/results/handeye-h0-2026-11-11
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CALIBTOOL_DATA=~/calibtool/data
W=~/calibtool/h0-work
for n in 0 1 2 3 4; do
  echo "== frames $n $(date +%T)"
  nice -n 10 ~/calibtool/venv/bin/python bench.py --config fast --depth numpy --frames $n --out $W/f_$n.json -j 8 > $W/f_$n.log 2>&1
  python3 -c "import json;s=json.load(open('$W/f_$n.json'))['summary'];print({k:s[k] for k in ('max_off_mm','median_off_mm','max_rot_deg','clean_flips','fit_s_mean','fuse_s_mean','wall_s')})"
done
nice -n 10 ~/calibtool/venv/bin/python frames.py $W/f_0.json $W/f_1.json $W/f_2.json $W/f_3.json $W/f_4.json > $W/frames.log 2>&1
cat $W/frames.log
echo "queue done $(date +%T)"
