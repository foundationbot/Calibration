#!/usr/bin/env bash
# Export a camera-IMU take and run the quick check on it:  bash /root/qc-take.sh <take name>
# The export is skipped when it already finished (run_info.txt is written last).
set -euo pipefail
R=/data/okvis-recordings; W=/root/calib-work; T=/root/calib-tools; TAKE=${1:?usage: qc-take.sh <take>}
if [ ! -f "$R/$TAKE-euroc/run_info.txt" ]; then
  nice -n 15 ionice -c3 python3 "$T/tools/mcap_to_euroc.py" "$R/$TAKE/session.mcap" "$R/$TAKE-euroc" | tail -2
fi
echo "EXPORT-DONE $(date +%T)"
docker run --rm --cpus=2 --cpu-shares=2 --memory=4g -v "$W:/w" -v "$R/$TAKE-euroc:/e:ro" calib-py:1 \
  python /w/qc_imucam.py /e 3
echo "QC-DONE $(date +%T)"
