#!/usr/bin/env bash
# productise plan Task 17 Step 3: two step-45 fits from the nominal start on Fit A's fit set (gate on / off)
set -u
D=/root/calib-work/kb4t-dryrun
SHA=$(cat /root/calib-tools-kb4t/COMMIT)
sed 's/^  lens_hold_rms_max_px: 0.6 .*/  lens_hold_rms_max_px: 100/' /root/calib-tools-kb4t/tools/calibration/config/rig-phantom12.yaml > $D/rig-nogate.yaml
sed -i 's#^target: target-aprilgrid.yaml#target: /t/tools/calibration/config/target-aprilgrid.yaml#' $D/rig-nogate.yaml
grep -n "lens_hold_rms_max_px\|^target:" $D/rig-nogate.yaml
for v in gate:/t/tools/calibration/config/rig-phantom12.yaml nogate:/o/rig-nogate.yaml; do
  name=${v%%:*}; rig=${v#*:}
  date +%T > $D/fit-$name.start
  setsid nohup docker run --rm --name kb4t-dry-$name --cpus=2 --cpu-shares=2 --memory=8g \
    -v /root/calib-tools-kb4t:/t:ro -v $D:/o -e OMP_NUM_THREADS=2 -e CALIB_GIT_SHA=$SHA \
    -e PYTHONPATH=/t/tools/calibration/scripts:/t/interface calib-py:3 \
    python -B /t/tools/calibration/scripts/45_fit_lens.py /o/fita-fit-corners.npz /o/fit-$name.json --rig $rig \
    > $D/fit-$name.log 2>&1 < /dev/null &
done
sleep 3; docker ps --format "{{.Names}} {{.Status}}" | grep kb4t-dry
