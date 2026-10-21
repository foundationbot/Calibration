#!/usr/bin/env bash
# Step 81 on the 17 Oct powered-neck takes from their corner files (plan 2026-10-20-phantom12-handeye-fit-a.md).
#   bash run.sh repro     the old lens (the config's), radius 450: must reproduce 18 Oct (Decision 3)
#   bash run.sh fita      Fit A, radius 450 (primary) and 560 (secondary)
set -u
T=/root/calib-tools-he; D=/root/calib-proto/data; O=/root/calib-proto/handeye
CFG=/t/tools/calibration/results/phantom12-2026-10-15/imucam/phantom12_stereo.yaml
QC=/t/tools/calibration/results/phantom12-2026-10-17-tbs/reports
FIT=/t/tools/calibration/results/phantom12-2026-10-20-lens/fit-a-kb4t.json
s81() {   # name take drops extra-args...
  local name=$1 take=$2 drops=$3; shift 3
  docker run --rm --cpus=2 --cpu-shares=2 --memory=4g -v $T:/t:ro -v $D:/d:ro -v $O:/o \
    -e CALIB_RIG=/t/tools/calibration/config/rig-phantom12.yaml -w /t/tools/calibration/scripts calib-py:3 \
    python -B 81_tbs_handeye.py --corners /d/tbs-neck-$take-corners.npz --config $CFG \
      --urdf /t/config/foundation_rig/20250206_phantom_mk1.urdf --per-hold 5 --drop-holds "$drops" \
      --qc $QC/tbs-neck-$take-qc.json --out /o/$name-$take.json "$@"
  echo "exit $? ($name $take) $(date +%T)"
}
case "$1" in
  repro) s81 repro P1 4,33,48; s81 repro P2 49 --compare /o/repro-P1.json
         s81 cfglens P1 4,33,48 --lens-json /t/tools/calibration/results/phantom12-2026-10-20-handeye/cfg-as-fit.json \
           --lens-key cfg ;;
  fita)  for r in 450 560; do
           s81 fita$r P1 4,33,48 --lens-json $FIT --lens-key KB4T@all@800 --max-radius-px $r
           s81 fita$r P2 49 --lens-json $FIT --lens-key KB4T@all@800 --max-radius-px $r --compare /o/fita$r-P1.json
         done ;;
esac
echo done
