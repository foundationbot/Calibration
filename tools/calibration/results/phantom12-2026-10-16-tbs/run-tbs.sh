#!/usr/bin/env bash
# phantom12 T_BS measurement on one neck take: export -> 75 quick check -> 81 hand-eye fit -> 82 gyro check.
# Usage (on Thor): bash /root/calib-work/run-tbs.sh <take> [<earlier take to compare with>]
# Takes live in /root/calib-work/takes; reports go to /root/calib-work/tbs. Env: TORSO_INDEX (default 9),
# DROP_HOLDS (comma-separated hold numbers; set, it replaces the list step 75 writes), CALIB_TOOLS (the
# package copy, default /root/calib-tools).
set -u
TAKE=${1:?usage: run-tbs.sh <take> [<earlier take>]}
PREV=${2:-}
for n in "$TAKE" ${PREV:+"$PREV"}; do      # a take is a folder name: nothing that leaves takes/ or tbs/
  case "$n" in */*|.*) echo "bad take name: $n"; exit 2;; esac
done
W=/root/calib-work; T=${CALIB_TOOLS:-/root/calib-tools}; TK=$W/takes
S=$TK/$TAKE/session.mcap
[ -f "$S" ] || { echo "no $S"; exit 2; }
mkdir -p "$W/tbs"
# the recorder keeps the current minute's segment open: let the one covering the take's end close
age=$(( $(date +%s) - $(stat -c %Y "$S") ))
if [ "$age" -lt 75 ]; then echo "waiting $((75 - age)) s for the recorder's segment to close"; sleep $((75 - age)); fi
# export again when there is none or the take was recorded again since
if [ ! -f "$TK/$TAKE-euroc/run_info.txt" ] || [ "$S" -nt "$TK/$TAKE-euroc/run_info.txt" ]; then
  nice -n 15 ionice -c3 python3 "$T/tools/mcap_to_euroc.py" "$S" "$TK/$TAKE-euroc" | tail -2
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "export failed"; exit 2; }
fi
PY="docker run --rm --cpus=2 --cpu-shares=2 --memory=8g -v $T:/t:ro -v $W:/w -v /root/recordings:/rec:ro -w /t/tools/calibration/scripts calib-py:2 python"
C="--session /w/takes/$TAKE/session.mcap --export /w/takes/$TAKE-euroc --config /w/phantom12_stereo.yaml --urdf /t/config/foundation_rig/20250206_phantom_mk1.urdf --rig /w/config/rig-phantom12.yaml"
verdict_of() {   # the verdict a step wrote, or "no report" (a crash leaves none: stale ones are removed first)
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("verdict", "no verdict"))' "$1" 2>/dev/null || echo "no report"
}
rm -f "$W/tbs/$TAKE-qc.json" "$W/tbs/$TAKE-handeye.json" "$W/tbs/$TAKE-gyro.json"
$PY 75_qc_necktake.py $C --dmarec /rec --dmarec-copy "/w/takes/$TAKE-dmarec" --torso-index "${TORSO_INDEX:-9}" \
  --qc-out "/w/tbs/$TAKE-qc.json"; QC=$?; echo "QC-EXIT $QC ($(verdict_of "$W/tbs/$TAKE-qc.json"))"
# the holds step 75 says to leave out (a moved torso, a /joints gap while the neck moved), unless DROP_HOLDS is set
if [ -n "${DROP_HOLDS:-}" ]; then
  FROM="DROP_HOLDS from the shell"
elif [ -f "$W/tbs/$TAKE-qc.json" ]; then
  FROM="step 75"
  DROP_HOLDS=$(python3 -c 'import json, sys; print(",".join(map(str, json.load(open(sys.argv[1]))["drop_holds"])))' "$W/tbs/$TAKE-qc.json")
else
  FROM="none: step 75 wrote no report"
fi
echo "step 81 leaves out holds: ${DROP_HOLDS:-none} ($FROM)"
[ "$QC" -ne 0 ] && echo "QC did not pass: the step 81/82 verdicts below are diagnostics, not the decision"
CMP=""
if [ -n "$PREV" ]; then
  if [ -f "$W/tbs/$PREV-handeye.json" ]; then CMP="--compare /w/tbs/$PREV-handeye.json"
  else echo "no report for $PREV: not compared"; fi
fi
$PY 81_tbs_handeye.py $C --drop-holds "${DROP_HOLDS:-}" $CMP --qc "/w/tbs/$TAKE-qc.json" --out "/w/tbs/$TAKE-handeye.json"
echo "HANDEYE-EXIT $? ($(verdict_of "$W/tbs/$TAKE-handeye.json"))"
$PY 82_tbs_gyro_axes.py $C --out "/w/tbs/$TAKE-gyro.json"
echo "GYRO-EXIT $? ($(verdict_of "$W/tbs/$TAKE-gyro.json"))"
echo "RUN-TBS-DONE $(date +%T)"
