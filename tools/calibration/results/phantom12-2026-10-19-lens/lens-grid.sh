#!/usr/bin/env bash
# Lens take (phantom12, plan of 19 Oct, revision 2): walks the neck through a grid of held poses with
# DMA.ethercat's dma_motor_motion (targets mode: the neck ramps to absolute targets, every other joint
# holds its measured pose; --no-return keeps holding the last pose between calls). The board stays on its
# stand; the neck's yaw carries it across the image out to the depth image's edge (~54 deg off-axis
# needs yaw ~+-55 deg), its pitch moves it up and down. Grid: NeckYaw over YAWS (default -60..60 deg in
# 15 deg steps) x NeckPitch over PITCHES (default -24 -12 0; +pitch = head down), snaked along the yaw
# rows; a 3 s move to the first hold, then 1.5 s ramps and 3 s holds; then back to yaw 0 at the middle
# pitch. MK1 neck limits: yaw +-90 deg, pitch -28.4..+27.5 deg; this script refuses yaw outside +-75 and
# pitch outside -26..25.
# Stops (exit 1, no return move: the neck holds its last pose) when a call prints no RESULT line (the
# tool's usage and start-up errors also exit 1, and nothing moved), exits above 1, or leaves a joint more
# than 5 deg from its target; exit 1 with a RESULT line is a hold that missed the tool's 1 deg tolerance
# (gravity sag, ~0.5-0.9 deg at Kp 30): expected, the grid goes on. Ctrl-C or a kill stops it (exit 130):
# dma_motor_motion catches SIGINT and exits normally, so without the trap bash would run the next hold. A RESULT
# line whose ss_err is not a number also stops it. The tool's own per-second rows are captured, not shown: during
# a move the console is quiet until its RESULT lines (stderr stays live).
# Usage (on the robot):   bash lens-grid.sh
#   DRY=1 bash lens-grid.sh                       prints the holds, moves nothing (DRY must be 0 or 1)
#   YAWS="-45 -30 -15 0 15 30 45" PITCHES="-20 -10 0" bash lens-grid.sh
#   MOTION=<stand-in> bash lens-grid.sh           a stand-in for /usr/bin/dma_motor_motion (testing)
set -u
case "${DRY:-0}" in 0|1) DRY=${DRY:-0} ;; *) echo "DRY=${DRY} not understood (0 or 1): refused"; exit 2 ;; esac
MOTION=${MOTION:-/usr/bin/dma_motor_motion}
read -r -a YAWS <<< "${YAWS:--60 -45 -30 -15 0 15 30 45 60}"
read -r -a PITCHES <<< "${PITCHES:--24 -12 0}"
RAMP=${RAMP:-1.5}
HOLD=${HOLD:-3}
for a in "${YAWS[@]}"; do (( a >= -75 && a <= 75 )) || { echo "yaw ${a} outside -75..75 deg: refused"; exit 2; }; done
for a in "${PITCHES[@]}"; do (( a >= -26 && a <= 25 )) || { echo "pitch ${a} outside -26..25 deg: refused"; exit 2; }; done
trap 'echo "interrupted $(date +%T.%N): grid stopped, the neck holds its last pose"; exit 130' INT TERM
move() {   # yaw pitch ramp hold -> 0 or 1 (hold done; 1 = a 1 deg tolerance miss), 3 nothing moved, 4 a joint > 5 deg off
  [ "${DRY}" = 1 ] && return 0
  local out rc res
  out=$("${MOTION}" "NeckYaw=$1,NeckPitch=$2" targets position --ramp "$3" --hold "$4" --no-return; echo "exit=$?")
  rc=${out##*exit=}
  res=$(printf '%s\n' "${out}" | grep '^RESULT')
  [ -n "${res}" ] && printf '%s\n' "${res}"
  if [ -z "${res}" ] || (( rc > 1 )); then echo "dma_motor_motion exit ${rc}, ${res:+with}${res:-no} RESULT: nothing reliable moved"; return 3; fi
  if printf '%s\n' "${res}" | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^ss_err=/) {if ($i !~ /^ss_err=-?[0-9]+(\.[0-9]+)?$/) bad = 1; v = substr($i, 8) + 0; if (v < 0) v = -v; if (v > 5) bad = 1}} END {exit !bad}'; then
    echo "a joint ended more than 5 deg from its target, or its error is not a number"; return 4
  fi
  return "${rc}"
}
stop() { echo "grid stopped $(date +%T.%N): $1; the neck holds its last pose"; exit 1; }
L=()
for i in "${!PITCHES[@]}"; do
  if (( i % 2 )); then order=$(seq $(( ${#YAWS[@]} - 1 )) -1 0); else order=$(seq 0 $(( ${#YAWS[@]} - 1 ))); fi
  for j in ${order}; do L+=("${YAWS[j]},${PITCHES[i]}"); done
done
MID=${PITCHES[$(( ${#PITCHES[@]} / 2 ))]}
echo "grid: ${#L[@]} holds, ramp ${RAMP} s, hold ${HOLD} s; start $(date +%T.%N)"
move "${L[0]%,*}" "${L[0]#*,}" 3 0.5
rc=$?; (( rc > 1 )) && stop "the move to the first hold failed (${rc})"
n=0
for yp in "${L[@]}"; do
  n=$((n+1))
  y=${yp%,*}; p=${yp#*,}
  echo "hold ${n} yaw=${y} pitch=${p} start $(date +%T.%N)"
  move "${y}" "${p}" "${RAMP}" "${HOLD}"
  rc=$?; (( rc > 1 )) && stop "hold ${n} failed (${rc})"
done
echo "back to yaw 0, pitch ${MID} $(date +%T.%N)"
move 0 "${MID}" 3 1
rc=$?; (( rc > 1 )) && stop "the return move failed (${rc})"
echo "grid done $(date +%T.%N)"
