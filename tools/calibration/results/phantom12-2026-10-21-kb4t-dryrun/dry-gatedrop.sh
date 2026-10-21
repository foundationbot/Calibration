#!/usr/bin/env bash
# productise plan Task 17 rule (c), completed after the results review: the gate on (0.6 px) with the recording-rejection
# limit lifted, so step 45 drops its holds and refits; then the depth map against Fit A by cell
set -u
D=/root/calib-work/kb4t-dryrun
SHA=$(cat /root/calib-tools-kb4t/COMMIT)
sed 's/^  lens_bad_hold_max_frac: 0.20 .*/  lens_bad_hold_max_frac: 1.0/' /root/calib-tools-kb4t/tools/calibration/config/rig-phantom12.yaml > $D/rig-gatedrop.yaml
sed -i 's#^target: target-aprilgrid.yaml#target: /t/tools/calibration/config/target-aprilgrid.yaml#' $D/rig-gatedrop.yaml
grep -n "lens_hold_rms_max_px\|lens_bad_hold_max_frac\|^target:" $D/rig-gatedrop.yaml
date +%T > $D/fit-gatedrop.start
docker run --rm --name kb4t-dry-gatedrop --cpus=2 --cpu-shares=2 --memory=8g \
  -v /root/calib-tools-kb4t:/t:ro -v $D:/o -e OMP_NUM_THREADS=2 -e CALIB_GIT_SHA=$SHA \
  -e PYTHONPATH=/t/tools/calibration/scripts:/t/interface calib-py:3 \
  python -B /t/tools/calibration/scripts/45_fit_lens.py /o/fita-fit-corners.npz /o/fit-gatedrop.json --rig /o/rig-gatedrop.yaml \
  > $D/fit-gatedrop.log 2>&1 < /dev/null
echo "fit exit $?" >> $D/fit-gatedrop.log
for lim in none 450; do
  if [ $lim = none ]; then extra=""; else extra="--max-radius-px 450"; fi
  docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -e OMP_NUM_THREADS=2 -v /root/calib-tools-kb4t:/t:ro -v $D:/o \
    -e PYTHONPATH=/t/tools/calibration/scripts calib-py:3 python -B /t/tools/calibration/scripts/depthmap.py \
    /o/fit-gatedrop.json /t/tools/calibration/results/phantom12-2026-10-20-lens/fit-a-kb4t.json:KB4T@all@800 \
    --rig /t/tools/calibration/config/rig-phantom12.yaml $extra
done > $D/depthmap-gatedrop.txt 2>&1
echo done >> $D/fit-gatedrop.log
