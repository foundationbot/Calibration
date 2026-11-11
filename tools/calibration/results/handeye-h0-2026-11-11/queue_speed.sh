#!/bin/bash
# H0 speed queue 1: sequential bench runs, 8 processes each, niced, one BLAS thread
cd ~/calibtool/wt-h0/tools/calibration/results/handeye-h0-2026-11-11
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CALIBTOOL_DATA=~/calibtool/data
W=~/calibtool/h0-work
for c in "$@"; do
  n=$(echo "$c" | tr ',=' '_-')
  echo "== $c $(date +%T)"
  nice -n 10 ~/calibtool/venv/bin/python bench.py --config "$c" --depth cpp --out $W/b_$n.json -j 8 > $W/b_$n.log 2>&1
  python3 -c "import json;s=json.load(open('$W/b_$n.json'))['summary'];print({k:s[k] for k in ('max_off_mm','median_off_mm','max_rot_deg','max_cov','clean_flips','fit_s_mean','fit_s_max','steps_median','wall_s')})"
done
echo "queue done $(date +%T)"
