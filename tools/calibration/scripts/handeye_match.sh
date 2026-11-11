#!/bin/bash
# Read B (armcal.readb, the robust matcher by default) over one arm's wave, ON the robot, JOBS captures in parallel.
# usage: handeye_match.sh <wave_dir> <right|left> <urdf> [jobs=4] [frames=0]
#   <wave_dir>: handeye_wave.py's --out (the bursts NN-<pose>-<pitch>-<yaw>.npz); writes <wave_dir>/readb-<side>[-fN].json
#   <urdf>: the URDF the robot's consumers read now (phantom10: /etc/phantomos/okvis/mk1_1.urdf)
#   frames: fuse only the first N burst frames (0 = all) -- the short-burst check
# Low priority (nice 15, one BLAS thread per process) so OKVIS and the depth engine keep their CPU.
# Env: CODE (default /data/calib/handeye/code, holding armcal/ and scripts/), PY (default /root/p11-cal/venv/bin/python).
set -u
WAVE=$(realpath "${1:?wave dir}"); SIDE=${2:?right|left}; URDF=$(realpath "${3:?urdf}"); JOBS=${4:-4}; FRAMES=${5:-0}
[ -f "$URDF" ] || { echo "no URDF $URDF"; exit 2; }   # absolute: readb runs from $CODE
CODE=${CODE:-/data/calib/handeye/code}; PY=${PY:-/root/p11-cal/venv/bin/python}
SUF=$([ "$FRAMES" -gt 0 ] && echo "-f$FRAMES" || echo "")
OUT="$WAVE/readb-$SIDE$SUF.json"
mapfile -t B < <(ls "$WAVE"/[0-9][0-9]-*.npz 2>/dev/null)
[ ${#B[@]} -gt 0 ] || { echo "no bursts in $WAVE"; exit 2; }
t0=$(date +%s)
# one readb over the whole wave: the robust matcher's pass 2 starts from the WAVE's median offset, so the captures
# must not be split into separate processes (readb --jobs parallelises inside)
( cd "$CODE" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$CODE:$CODE/scripts" \
    nice -n 15 "$PY" -m armcal.readb "${B[@]}" --side "$SIDE" --urdf "$URDF" --frames "$FRAMES" --jobs "$JOBS" \
    --out "$OUT" ) || { echo "readb failed"; exit 3; }
"$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); r=d['res']; print(sys.argv[1]+':', len(r), 'captures,', sum(x['clean'] for x in r), 'clean')" "$OUT"
echo "readb wall time $(( $(date +%s) - t0 )) s for ${#B[@]} captures, $JOBS processes"
