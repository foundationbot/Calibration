#!/bin/bash
# Read B (armcal.readb) over one arm's wave, ON the robot, in parallel chunks, then one merged records file.
# usage: handeye_match.sh <wave_dir> <right|left> <urdf> [jobs=4] [frames=0]
#   <wave_dir>: handeye_wave.py's --out (the bursts NN-<pose>-<pitch>-<yaw>.npz); writes <wave_dir>/readb-<side>[-fN].json
#   <urdf>: the URDF the robot's consumers read now (phantom10: /etc/phantomos/okvis/mk1_1.urdf)
#   frames: fuse only the first N burst frames (0 = all) -- the short-burst check
# Low priority (nice 15, one BLAS thread per process) so OKVIS and the depth engine keep their CPU.
# Env: CODE (default /data/calib/handeye/code, holding armcal/ and scripts/), PY (default /root/p11-cal/venv/bin/python).
set -u
WAVE=${1:?wave dir}; SIDE=${2:?right|left}; URDF=${3:?urdf}; JOBS=${4:-4}; FRAMES=${5:-0}
CODE=${CODE:-/data/calib/handeye/code}; PY=${PY:-/root/p11-cal/venv/bin/python}
SUF=$([ "$FRAMES" -gt 0 ] && echo "-f$FRAMES" || echo "")
OUT="$WAVE/readb-$SIDE$SUF.json"
mapfile -t B < <(ls "$WAVE"/[0-9][0-9]-*.npz 2>/dev/null)
[ ${#B[@]} -gt 0 ] || { echo "no bursts in $WAVE"; exit 2; }
TMP=$(mktemp -d "$WAVE/.readb-XXXX")
t0=$(date +%s)
for ((k = 0; k < JOBS; k++)); do
  chunk=()
  for ((i = k; i < ${#B[@]}; i += JOBS)); do chunk+=("${B[$i]}"); done
  [ ${#chunk[@]} -gt 0 ] || continue
  ( cd "$CODE" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$CODE:$CODE/scripts" \
      nice -n 15 "$PY" -m armcal.readb "${chunk[@]}" --side "$SIDE" --urdf "$URDF" --frames "$FRAMES" \
      --out "$TMP/part$k.json" > "$TMP/part$k.log" 2>&1 ) &
done
wait
cat "$TMP"/part*.log
"$PY" - "$OUT" "$TMP" <<'EOF'
import json, sys
from pathlib import Path
out, tmp = sys.argv[1], Path(sys.argv[2])
parts = [json.loads(p.read_text()) for p in sorted(tmp.glob("part*.json"))]
if not parts:
    sys.exit("no readb output (see the logs above)")
d = dict(parts[0])
d["res"] = sorted([r for p in parts for r in p["res"]], key=lambda r: r["name"])
Path(out).write_text(json.dumps(d, indent=1))
n = len(d["res"]); c = sum(r["clean"] for r in d["res"])
print(f"{out}: {n} captures, {c} clean")
EOF
echo "readb wall time $(( $(date +%s) - t0 )) s for ${#B[@]} captures, $JOBS processes"
