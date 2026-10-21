#!/usr/bin/env bash
# T_BS powered-neck take: walks the neck through a 5 x 5 grid of held poses with DMA.ethercat's
# dma_motor_motion (targets mode: the neck ramps to absolute targets, every other joint holds its
# measured pose; with --no-return dma_main keeps holding the last pose between calls).
# Grid: NeckYaw -30..30 deg x NeckPitch -24..0 deg (+pitch = head down; the board is seen only with
# the head level or up), snaked along the yaw rows and back = 49 holds; 1.5 s ramps, 3 s holds;
# then back to level. Usage (on the robot): bash tbs-grid.sh
# WARNING (review of 19 Oct): the record of the 17 Oct take, kept as run. Ctrl-C does NOT stop it:
# dma_motor_motion catches SIGINT and exits normally, so bash runs the next hold; stop it with a kill from
# another terminal. Exit 1 is also the tool's usage and start-up error code, so a tool that moves
# nothing runs the whole grid silently. For a new take use results/phantom12-2026-10-19-lens/lens-grid.sh,
# which stops on both (check-lens-grid.sh tests it).
set -u
YAWS=(-30 -15 0 15 30)
PITCHES=(-24 -18 -12 -6 0)
RAMP=1.5
HOLD=3
L=()
for i in 0 1 2 3 4; do
  if (( i % 2 )); then idx=(4 3 2 1 0); else idx=(0 1 2 3 4); fi
  for j in "${idx[@]}"; do L+=("${YAWS[j]},${PITCHES[i]}"); done
done
for ((k=${#L[@]}-2; k>=0; k--)); do L+=("${L[k]}"); done
echo "grid: ${#L[@]} holds, ramp ${RAMP} s, hold ${HOLD} s; start $(date +%T.%N)"
n=0
for yp in "${L[@]}"; do
  n=$((n+1))
  y=${yp%,*}; p=${yp#*,}
  echo "hold ${n} yaw=${y} pitch=${p} start $(date +%T.%N)"
  /usr/bin/dma_motor_motion "NeckYaw=${y},NeckPitch=${p}" targets position \
    --ramp "${RAMP}" --hold "${HOLD}" --no-return | grep RESULT
  rc=${PIPESTATUS[0]}
  # exit 1 = a hold missed the tool's 1 deg tolerance (gravity sag): expected; anything else stops
  if (( rc > 1 )); then echo "dma_motor_motion exit ${rc}: stopping the grid"; break; fi
done
# single-joint sweeps for step 82's gyro check (the grid has almost no pitch-only motion): only the
# listed joint moves, the other neck joint holds; linear 1.5-2 s strokes = 750-1000 IMU samples each
if [ "${SWEEPS:-1}" = 1 ]; then
  go() { /usr/bin/dma_motor_motion "$1" targets position --ramp "$2" --hold "$3" --no-return | grep RESULT; }
  echo "sweeps start $(date +%T.%N)"
  go "NeckYaw=0,NeckPitch=-12" 1.5 1
  echo "nod start $(date +%T.%N)"
  for p in -24 0 -24 0 -24 0 -12; do go "NeckPitch=${p}" 1.5 0.5; done
  echo "turn start $(date +%T.%N)"
  for y in -30 30 -30 30 -30 30 0; do go "NeckYaw=${y}" 2 0.5; done
  echo "sweeps done $(date +%T.%N)"
fi
echo "back to level $(date +%T.%N)"
/usr/bin/dma_motor_motion "NeckYaw=0,NeckPitch=0" targets position --ramp 2 --hold 1 --no-return | grep RESULT
echo "grid done $(date +%T.%N)"
