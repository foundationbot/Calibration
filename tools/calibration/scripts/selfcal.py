"""The one-command arm-wave self-calibration session (plan docs/superpowers/plans/2026-10-25-tagfree-selfcal.md,
Phase D, on the stereo-depth-eval branch): the runner phantom12 executes and the checks before it may start.

The runner is the 24 Oct sessions' shell (left-1023L.sh, right-long-1024.sh), generated from a route whose every move
was cleared in MuJoCo against the body at the robot's pose (left_clear.py / right_clear.py -> moves.json):
- every move lists all seven arm joints and the neck (dma_motor_motion holds an unlisted joint at its MEASURED pose,
  so a sagging arm sinks with each partial command), kp 100, --no-return;
- motors 29/29 are read before every move; a motor drop, a failed move or a failed burst aborts the session;
- a burst (tools/burst_pointcloud/grab.py, depth + joints) follows each move marked for one;
- the held-out captures are written into the runner before any data exists.
No recording: the tag-free fit reads the bursts only.
"""
from __future__ import annotations

import re

JOINTS = ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "ElbowPitch", "WristYaw", "WristRoll", "GripperYaw")
MIN_CLEAR_M = 0.025             # the plan's path clearance against the body
LONG_DEG, RAMP_S, RAMP_LONG_S = 40.0, 6, 14
DISK_MARGIN_GB = 5.0            # above kubelet's eviction line (24 Oct 23:20: OKVIS evicted on disk pressure)
POSE_TOL_DEG = 2.0              # go: the robot within this of the pose the route was cleared at
LOCK = "/root/burst-grab/selfcal.lock"


def _tag(name: str) -> str:
    """A shell-safe word: 'rev 1023d/11' -> 'rev_1023d_11' (review 25 Oct: a space shifted mv's arguments)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def capture_name(k: int, move: dict) -> str:
    return f"{k:02d}-{_tag(move['name'])}"


def runner(moves, neck: dict, session: str, arm: str = "Left", held=()) -> str:
    """The shell for phantom12 (run as root in /root/burst-grab). moves: [{name, deg (7, JOINTS order), burst,
    max_travel_deg, path_min_clear_m}] in order; neck {NeckYaw, NeckPitch} deg, held for the whole session."""
    for m in moves:
        if m["path_min_clear_m"] < MIN_CLEAR_M:
            raise ValueError(f"{m['name']}: path clearance {m['path_min_clear_m']} m < {MIN_CLEAR_M} m")
    caps = [capture_name(k, m) for k, m in enumerate(moves, 1) if m["burst"]]
    if set(held) - set(caps):
        raise ValueError(f"held-out not in the route: {sorted(set(held) - set(caps))}")
    log = f"/root/burst-grab/{session}.log"
    sh = ["#!/bin/bash",
          f"set -u; LOG={log}; cd /root/burst-grab",
          f"# {arm}-arm self-calibration session {session} (selfcal.py, generated). Aborts on motors < 29/29, a failed"
          " move or a failed burst.",
          f"# captures: {' '.join(caps)}",
          f"# held-out: {' '.join(held)}",
          f"# stop: pkill -f {session}.sh  (the move in flight finishes first, up to ~16 s; then the arm holds that pose)",
          f'exec 9>{LOCK}; flock -n 9 || {{ echo "another self-calibration session holds {LOCK}"; exit 5; }}',
          'log(){ echo "$(date +%T.%N | cut -c1-12) $*" | tee -a $LOG; }',
          'chk(){ s=$(timeout 10 dma_cmd_client state 2>&1); echo "$s" | grep -q "Motors enabled:    29/29" || '
          '{ log "ABORT motors: $(echo $s | tr -s " ")"; exit 2; }; }',
          'mv(){ chk; log "MOVE $1 ramp $2: $3"; out=$(dma_motor_motion "$3" targets position --ramp $2 --hold 2 '
          '--no-return --kp 100 --tolerance 30 2>&1); rc=$?; echo "$out" | grep -E "RESULT|FAIL|PASS|error" | tee -a $LOG; '
          '[ $rc -eq 0 ] || { log "ABORT rc $rc at $1"; exit 1; }; }',
          'grab(){ sleep 1; log "BURST $1"; PYTHONPATH=pydeps nice python3 repo/tools/burst_pointcloud/grab.py --once 5 '
          f'--joints all --out /root/burst-grab/{session}-$1.npz 2>&1 | tail -2 | tee -a $LOG; '
          '[ ${PIPESTATUS[0]} -eq 0 ] || { log "ABORT burst $1"; exit 3; }; }',
          'log "START"; PYTHONPATH=pydeps python3 armsnap.py >> $LOG']
    nk = f"NeckPitch={neck['NeckPitch']:.2f},NeckYaw={neck['NeckYaw']:.2f}"
    for k, m in enumerate(moves, 1):
        tgt = ",".join(f"{arm}{j}={v:.2f}" for j, v in zip(JOINTS, m["deg"]))
        ramp = RAMP_LONG_S if m["max_travel_deg"] > LONG_DEG else RAMP_S
        sh.append(f'mv {capture_name(k, m)} {ramp} "{tgt},{nk}"')
        if m["burst"]:
            sh.append(f"grab {capture_name(k, m)}")
    sh.append('PYTHONPATH=pydeps python3 armsnap.py >> $LOG; log "DONE"')
    return "\n".join(sh) + "\n"


def precheck(snap: dict, state: str, okvis_log: str, evict_free_gb: float, okvis_ready: bool, running: str = "",
             stale=()) -> list:
    """What stops the session from starting (empty: go). snap = armsnap.py's JSON, state = `dma_cmd_client state`,
    okvis_log = the OKVIS pod's recent log, evict_free_gb = kubelet's eviction line on Thor's root disk,
    okvis_ready = the okvis2x container is running and ready (an empty log alone says nothing), running = any
    dma_motor_motion or session runner already running, stale = this session's files already on the robot."""
    fails = []
    if running.strip():
        fails.append(f"running: another runner or dma_motor_motion is running: {running.strip()[:200]}")
    if stale:
        fails.append(f"stale: this session's files exist on the robot (use a new session name): {list(stale)}")
    if not okvis_ready:
        fails.append("okvis: the okvis2x container is not running/ready (no depth for the bursts)")
    if "Motors enabled:    29/29" not in state:
        fails.append("motors: not 29/29 enabled")
    if snap.get("desired", {}).get("advancing", True):
        fails.append("walker: /desired is advancing (or unknown)")
    if snap.get("disk_free_gb", 0.0) < evict_free_gb + DISK_MARGIN_GB:
        fails.append(f"disk: {snap.get('disk_free_gb')} GB free, need {evict_free_gb + DISK_MARGIN_GB}")
    if "[cam-stall]" in okvis_log:
        fails.append("cam-stall: OKVIS reports a stalled camera (RUNBOOK §3: dma-video producer 0 -> 1)")
    return fails


def pose_changes(planned: dict, now: dict, arm: str, tol_deg: float = POSE_TOL_DEG) -> list:
    """[(joint, deg)] that moved more than tol_deg since the route's clearance was computed (armsnap q_deg): the
    arm's seven joints, the spine and the neck (the body the arm is cleared against)."""
    names = [arm + j for j in JOINTS] + ["SpineYaw", "NeckPitch", "NeckYaw"]
    a, b = planned["q_deg"], now["q_deg"]
    return [(n, round(b[n] - a[n], 2)) for n in names if n in a and n in b and abs(b[n] - a[n]) > tol_deg]
