#!/usr/bin/env python3
"""The one-command arm-wave session, driven from the laptop (plan docs/superpowers/plans/2026-10-25-tagfree-selfcal.md,
Phase D, stereo-depth-eval branch). Three commands, each safe to repeat:

  check SESSION         read-only on the robot: motors, walker, disk vs kubelet's eviction line, [cam-stall]
  plan  SESSION ARM     ARM = Left | Right. The route: a snapshot of the robot's pose (read-only), MuJoCo path clearance of the arm's
                        route against the body at that pose (Moonstone, arm-venv), the runner written locally;
                        nothing is sent to the robot
  go    SESSION         the operator's yes first: check again, copy the runner to the robot, run it, copy each burst
                        to Moonstone as it lands; prints the read-B command for the bursts (the point-cloud agent's
                        blind_b.py with --dir/--side, once it has them)

Work dir: ./selfcal-SESSION/ (snap.json, moves.json, runner.sh, run.log). Robot: ssh phantom12 (root).
Moonstone: ssh Moonstone, MOONSTONE_DATA (the user's /data folder) holds tagfree/ (this repo's tools/calibration),
arm-pr-816/ (left-plan.json, left_clear.py), arm-task2/mjcf/ and arm-venv/.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selfcal  # noqa: E402

ROBOT, MOON = "phantom12", "Moonstone"
DATA = os.environ.get("MOONSTONE_DATA", "")
EVICT_GB = float(os.environ.get("EVICT_FREE_GB", "151"))      # kubelet's line on phantom12, 24 Oct (the stereo agent)
GRAB = "/root/burst-grab"


def sh(host, cmd, check=True) -> str:
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", host, cmd], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        if check:
            raise
        return ""
    if check and r.returncode:
        raise SystemExit(f"{host}: {cmd!r} -> rc {r.returncode}\n{r.stderr[-2000:]}")
    return r.stdout


def snapshot() -> dict:
    return json.loads(sh(ROBOT, f"cd {GRAB} && PYTHONPATH=pydeps python3 armsnap.py").strip().splitlines()[-1])


def busy() -> str:
    """A session runner holding the lock, or a dma_motor_motion running (by lock and exact name: a `pgrep -f` here
    would match this ssh command line itself)."""
    return sh(ROBOT, f"flock -n {selfcal.LOCK} true || echo 'a session holds {selfcal.LOCK}'; "
                     "pgrep -a -x dma_motor_motio", check=False)


def check(session) -> list:
    snap = snapshot()
    state = sh(ROBOT, "timeout 10 dma_cmd_client state", check=False)
    stale = sh(ROBOT, f"cd {GRAB} && ls {session}.log {session}.sh 2>/dev/null", check=False).split()
    log = sh(ROBOT, "k0s kubectl -n positronic logs ds/okvis2x --since=120s 2>&1 | tail -2000", check=False)
    pods = sh(ROBOT, "k0s kubectl -n positronic get pods -o name 2>/dev/null | grep okvis2x", check=False).split()
    ready = bool(pods) and sh(ROBOT, f"k0s kubectl -n positronic get {pods[0]} -o jsonpath="
                                     "'{.status.containerStatuses[?(@.name==\"okvis2x\")].ready}'", check=False).strip() == "true"
    fails = selfcal.precheck(snap, state, log, EVICT_GB, okvis_ready=ready, running=busy(), stale=stale)
    print("\n".join(fails) if fails else f"{session}: pre-checks OK (disk {snap['disk_free_gb']:.0f} GB free)")
    return fails


# The right arm's blind session (plan, D6 reference fixed 25 Oct ~00:20 before data): rl1024's route at neck 28, one
# burst at each of its 13 distinct poses (no burst at a revisit: the A3 review's leakage), held-out as rl1024's.
RIGHT_BURSTS = ("1023d/0", "1023d/1", "1023d/4", "1023d/5", "1023d/6", "1023d/7", "1023d/8", "1023d/9", "1023d/11",
                "1023d/12", "1023c/12", "1023c/13", "1023c/16")
RIGHT_HELD = ("1023d/1", "1023d/4", "1023c/13")
ROUTE = {"Left": ("left_clear.py", "left-plan.json"), "Right": ("right_clear.py", "right-long-28.json")}


def plan(session, arm, neck=None):
    if not DATA:
        raise SystemExit("set MOONSTONE_DATA to the /data folder on Moonstone")
    wd = Path(f"selfcal-{session}"); wd.mkdir(exist_ok=True)
    snap = snapshot(); (wd / "snap.json").write_text(json.dumps(snap)); (wd / "arm").write_text(arm)
    rd = f"{DATA}/selfcal-{session}"
    sh(MOON, f"mkdir -p {rd}")
    subprocess.run(["scp", "-q", str(wd / "snap.json"), f"{MOON}:{rd}/snap.json"], check=True)
    clear, route_file = ROUTE[arm]
    if neck is not None:                                     # T2 (head angles): the right route at another neck pitch
        assert arm == "Right", "only the right route exists at other neck angles"
        route_file = f"right-long-{int(neck)}.json"
    out = sh(MOON, f"cd {DATA}/arm-pr-816 && nice taskset -c 0-7 {DATA}/arm-venv/bin/python {clear} "
                   f"{DATA}/tagfree {rd}/snap.json {route_file} {DATA}/arm-task2/mjcf/phantom_mk1.xml {rd}/moves.json")
    print(out[-3000:])
    subprocess.run(["scp", "-q", f"{MOON}:{rd}/moves.json", str(wd / "moves.json")], check=True)
    mv = json.loads((wd / "moves.json").read_text())
    if arm == "Right":
        done = set()
        for m in mv["moves"]:
            m["burst"] = m["name"] in RIGHT_BURSTS and m["name"] not in done
            done |= {m["name"]} if m["burst"] else set()
        assert done == set(RIGHT_BURSTS), set(RIGHT_BURSTS) - done
        held_src = set(RIGHT_HELD)
    else:
        route = json.loads(sh(MOON, f"cat {DATA}/arm-pr-816/{route_file}"))["route"]
        held_src = {r["src"] for r in route if r.get("heldout")}
    held = [selfcal.capture_name(k, m) for k, m in enumerate(mv["moves"], 1) if m["burst"] and m["name"] in held_src]
    neck = {k: round(v, 2) for k, v in mv["neck_deg"].items()}
    (wd / "moves.json").write_text(json.dumps(mv, indent=1))
    (wd / "runner.sh").write_text(selfcal.runner(mv["moves"], neck, session, arm, held=held))
    print(f"{wd}/runner.sh: {sum(m['burst'] for m in mv['moves'])} bursts, held-out {held}; "
          f"worst path clearance {min(m['path_min_clear_m'] for m in mv['moves']) * 100:.1f} cm")


def go(session):
    wd = Path(f"selfcal-{session}")
    if check(session):
        raise SystemExit("pre-checks failed: not starting")
    moved = selfcal.pose_changes(json.loads((wd / "snap.json").read_text()), snapshot(), (wd / "arm").read_text().strip())
    if moved:
        raise SystemExit(f"the robot moved since the route was cleared: {moved}; run plan again")
    print(f"stop at any time: ssh {ROBOT} 'pkill -f {session}.sh'  (the move in flight still finishes, up to ~16 s; "
          "the arm then holds that pose)")
    subprocess.run(["scp", "-q", str(wd / "runner.sh"), f"{ROBOT}:{GRAB}/{session}.sh"], check=True)
    # 25 Oct 19:20: a `nohup ... &` over ssh kept the connection open, the call timed out after 300 s and this monitor
    # died while the runner went on alone. Launch fully detached (setsid) and never wait on that ssh.
    subprocess.Popen(["ssh", "-n", "-o", "ConnectTimeout=10", ROBOT,
                      f"cd {GRAB} && setsid nohup bash {session}.sh > /dev/null 2>&1 < /dev/null &"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    seen, dest = set(), f"burst-pc/arm/{session}"
    sh(MOON, f"mkdir -p {dest}")
    t0 = time.time()
    while True:
        log = sh(ROBOT, f"cat {GRAB}/{session}.log 2>/dev/null", check=False)
        (wd / "run.log").write_text(log)
        lines = log.splitlines()
        for i, l in enumerate(lines):
            if " BURST " in l and i + 1 < len(lines):                          # the grab has printed: file is complete
                n = l.split(" BURST ")[1].strip()
                if n not in seen:
                    subprocess.run(["scp", "-q", "-3", f"{ROBOT}:{GRAB}/{session}-{n}.npz", f"{MOON}:{dest}/"], check=True)
                    seen.add(n); print(f"{time.time() - t0:6.0f} s  burst {n} -> Moonstone")
        if any(" ABORT " in l for l in lines):
            raise SystemExit("ABORTED: " + next(l for l in lines if " ABORT " in l))
        if any(l.endswith(" DONE") for l in lines):
            break
        free = sh(ROBOT, f"flock -n {selfcal.LOCK} true && echo FREE || echo HELD", check=False).strip()
        if time.time() - t0 > 60 and free == "FREE":                       # an ssh failure reads "", not FREE
            again = sh(ROBOT, f"cat {GRAB}/{session}.log 2>/dev/null", check=False)   # 25 Oct 22:09: DONE can land
            if any(l.endswith(" DONE") for l in again.splitlines()):              # between the log read and the lock
                continue                                                          # check; read once more
            raise SystemExit("the runner is gone without DONE or ABORT: check the robot (" + (lines[-1] if lines else "no log") + ")")
        time.sleep(5)
    print(f"route done in {time.time() - t0:.0f} s, {len(seen)} bursts on Moonstone ~/{dest}")
    arm = (wd / "arm").read_text().strip()
    held = next(l for l in (wd / "runner.sh").read_text().splitlines() if l.startswith("# held-out:")).split()[2:]
    print("read B (Moonstone, a frozen copy of the point-cloud agent's ~/burst-pc/plan6g; gate on 'converged', not "
          "neck_ok, under --trans-only):\n"
          f"  cd ~/burst-pc/plan6d && OMP_NUM_THREADS=1 PYTHONPATH=build taskset -c 0-7 nice venv/bin/python -u "
          f"repo/tools/burst_pointcloud/experiments/arm816/blind_b.py --session {session} --dir $HOME/{dest} "
          f"--side {arm} --held {','.join(session + '-' + h for h in held)} --trans-only --jobs 8 --min-coverage 0.70 "
          f"--out ../out/{session} --yardstick ../out/{session}/yardstick.json")


if __name__ == "__main__":
    cmd, session = sys.argv[1], sys.argv[2]
    {"check": lambda: check(session), "plan": lambda: plan(session, sys.argv[3], *(sys.argv[4:5])), "go": lambda: go(session)}[cmd]()
