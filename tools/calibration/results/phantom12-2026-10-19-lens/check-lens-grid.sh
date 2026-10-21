#!/usr/bin/env bash
# Checks of lens-grid.sh that need no robot: stand-ins for dma_motor_motion (MOTION=...) and DRY=1.
# Each case prints PASS or FAIL; the script exits 1 if any case fails. Usage: bash check-lens-grid.sh
set -u
G=$(cd "$(dirname "$0")" && pwd)/lens-grid.sh
W=$(mktemp -d)
trap 'rm -rf -- "${W}"' EXIT
R='RESULT targets joint=15 target_deg=0'
mk() { printf '#!/bin/bash\necho called >> %s/calls\n%s\n' "${W}" "$2" > "${W}/$1"; chmod +x "${W}/$1"; }
mk ok    "echo '${R} ss_err=0.6 rms=0.6 max=0.9 samples=1500'; echo '${R} ss_err=-0.8 rms=0.8 max=1 samples=1500'; exit 1"
mk usage "echo 'ERROR: targets signal supports position mode only' >&2; exit 1"
mk big   "echo '${R} ss_err=7.2 rms=7 max=9 samples=1500'; exit 1"
mk two   "echo '${R} ss_err=0.2 rms=0.2 max=0.3 samples=1500'; exit 2"
mk nan   "echo '${R} ss_err=-nan rms=-nan max=-nan samples=0'; exit 0"
mk slow  "trap \"echo '${R} ss_err=0.1 rms=0.1 max=0.1 samples=10'; exit 0\" INT; sleep 3 & wait \$!; echo '${R} ss_err=0.1 rms=0.1 max=0.1 samples=1500'; exit 0"
fails=0
check() {   # name expected-exit expected-calls expected-text -- command...
  local name=$1 want_rc=$2 want_calls=$3 want_text=$4; shift 5
  rm -f "${W}/calls"
  local out rc calls
  out=$("$@" 2>&1); rc=$?
  calls=$( [ -f "${W}/calls" ] && wc -l < "${W}/calls" || echo 0 )
  if [ "${rc}" = "${want_rc}" ] && [ "${calls}" = "${want_calls}" ] && grep -q -- "${want_text}" <<< "${out}"; then
    echo "PASS ${name}"
  else
    echo "FAIL ${name}: exit ${rc} (want ${want_rc}), calls ${calls} (want ${want_calls}), last: $(tail -n 1 <<< "${out}")"
    fails=$((fails+1))
  fi
}
check "a full grid with a well-behaved tool"   0 29 "grid done"                            -- env MOTION="${W}/ok" bash "${G}"
check "no RESULT (usage error) stops at once"  1 1  "the move to the first hold failed (3)" -- env MOTION="${W}/usage" bash "${G}"
check "a joint 7.2 deg off stops at once"      1 1  "the move to the first hold failed (4)" -- env MOTION="${W}/big" bash "${G}"
check "exit 2 stops at once"                   1 1  "the move to the first hold failed (3)" -- env MOTION="${W}/two" bash "${G}"
check "a non-numeric ss_err stops at once"     1 1  "the move to the first hold failed (4)" -- env MOTION="${W}/nan" bash "${G}"
check "DRY=1 never calls the tool"             0 0  "grid done"                            -- env DRY=1 MOTION="${W}/ok" bash "${G}"
check "DRY=true is refused"                    2 0  "not understood"                       -- env DRY=true MOTION="${W}/ok" bash "${G}"
check "yaw +-75 accepted"                      0 0  "grid done"                            -- env DRY=1 YAWS="-75 75" bash "${G}"
check "yaw -76 refused"                        2 0  "yaw -76 outside"                      -- env DRY=1 YAWS="-76" bash "${G}"
check "yaw 76 refused"                         2 0  "yaw 76 outside"                       -- env DRY=1 YAWS="76" bash "${G}"
check "pitch -26 and 25 accepted"              0 0  "grid done"                            -- env DRY=1 PITCHES="-26 25" bash "${G}"
check "pitch 26 refused"                       2 0  "pitch 26 outside"                     -- env DRY=1 PITCHES="26" bash "${G}"
check "pitch -27 refused"                      2 0  "pitch -27 outside"                    -- env DRY=1 PITCHES="-27" bash "${G}"
holds=$(DRY=1 bash "${G}" | grep -c '^hold ')
order=$(DRY=1 bash "${G}" | grep '^hold \|^back' | sed -n '1p;9p;10p;18p;19p;27p;28p' | cut -d' ' -f1-4 | tr '\n' '|')
want='hold 1 yaw=-60 pitch=-24|hold 9 yaw=60 pitch=-24|hold 10 yaw=60 pitch=-12|hold 18 yaw=-60 pitch=-12|hold 19 yaw=-60 pitch=0|hold 27 yaw=60 pitch=0|back to yaw 0,|'
if [ "${holds}" = 27 ] && [ "${order}" = "${want}" ]; then echo "PASS 27 holds in snake order"; else echo "FAIL snake order: ${holds} holds, ${order}"; fails=$((fails+1)); fi
# Ctrl-C: SIGINT to the grid's process group while hold 1 runs (as a terminal does); the stand-in catches it
# and exits normally, like dma_motor_motion. The grid must stop (exit 130) after 2 calls.
rm -f "${W}/calls"
python3 - "${G}" "${W}/slow" <<'PY'
import os, signal, subprocess, sys, time
p = subprocess.Popen(["bash", sys.argv[1]], env={**os.environ, "MOTION": sys.argv[2]}, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
time.sleep(5)
os.killpg(p.pid, signal.SIGINT)
try:
    p.wait(timeout=20)
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
sys.exit(0 if p.returncode == 130 else 1)
PY
rc=$?; calls=$(wc -l < "${W}/calls")
if [ "${rc}" = 0 ] && [ "${calls}" = 2 ]; then echo "PASS Ctrl-C stops the grid (exit 130, 2 calls)"; else echo "FAIL Ctrl-C: python exit ${rc}, calls ${calls}"; fails=$((fails+1)); fi
echo "${fails} failed"
exit $(( fails > 0 ))
