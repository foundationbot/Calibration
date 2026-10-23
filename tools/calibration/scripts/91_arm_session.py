#!/usr/bin/env python3
"""Step 91 — the arm session's sweep (docs/superpowers/plans/2026-10-22-phantom12-arm-session.md, Task 3; rev 3's
T2-5). Runs on the laptop; every robot command goes over ssh to the robot.

For each stop of step 90's poses.json: send the arm to the stop's command (dma_motor_motion, every right-arm joint
listed, --no-return, one kp), read where each joint stopped from the RESULT lines (reached = target - ss_err), and
at a measured stop (fit, held out, repeat) judge the reached pose. A pose more than CORRECT_DEG off the desired one
in any joint, or failing the view or clearance check at the reached joints, gets one correction (target += desired -
reached); if it still fails it is skipped. A kept pose gets a depth burst with every joint (the point-cloud
agent's grab.py --once). Any joint more than ABORT_DEG from where it was sent, a missing RESULT line or a failed
remote call stops the sweep at once: the arm holds where it is (--no-return). Every move is logged with wall-clock
stamps (JSON lines) so the analysis can find each hold in the recording.

--dry-run prints and logs the commands and sends nothing. The recording (record app, both cameras, /joints) and
the pre-checks are main()'s; the sweep logic is Session's, testable against a fake robot.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from armchain import RIGHT_ARM  # noqa: E402

CORRECT_DEG, ABORT_DEG = 4.0, 15.0
START_TOL_DEG, NECK_TOL_DEG = 3.0, 1.0
DESIRED_QUIET_S = 60.0
MOTORS_ON_NM = 1.0              # a raised arm held at kp 100 needs ~10 Nm at the shoulder and elbow
LEG_MAX_NM = 10.0               # the robot hangs: its legs carry nothing (22 Oct: <= 5.4 Nm)
REC_GB_PER_MIN = 3.4            # two-camera PNG, measured 22 Oct (775 MB / 15 s)
JIDX = {n: 7 + i for i, n in enumerate(RIGHT_ARM)}       # /config indices (checked against the snapshot's order)
MEASURED = ("fit", "heldout", "repeat_same", "repeat_opp")
GRAB = ("cd /root/burst-grab && PYTHONPATH=/root/burst-grab/pydeps nice -n 10 python3 "
        "repo/tools/burst_pointcloud/grab.py --once {n} --out {out} --joints all")


class Abort(RuntimeError):
    pass


def parse_result(text: str, jidx=JIDX) -> np.ndarray:
    """dma_motor_motion's RESULT lines -> where each right-arm joint stopped (deg), in RIGHT_ARM order."""
    got = {}
    for m in re.finditer(r"RESULT targets joint=(\d+) target_deg=(\S+) ss_err=(\S+)", text):
        got[int(m.group(1))] = float(m.group(2)) - float(m.group(3))
    missing = [n for n in RIGHT_ARM if jidx[n] not in got]
    if missing:
        raise Abort(f"no RESULT line for {', '.join(missing)}")
    return np.array([got[jidx[n]] for n in RIGHT_ARM])


def motion_command(target_deg, ramp_s: float, hold_s: float, kp: float, tol_deg: float = 30.0) -> str:
    spec = ",".join(f"{n}={v:.2f}" for n, v in zip(RIGHT_ARM, target_deg))
    return (f'dma_motor_motion "{spec}" targets position --ramp {ramp_s:.1f} --hold {hold_s:.1f} --no-return '
            f"--kp {kp:.0f} --tolerance {tol_deg:.0f}")


class Ssh:
    def __init__(self, host: str = "phantom12"):
        self.host = host

    def __call__(self, cmd: str, timeout_s: float):
        try:
            p = subprocess.run(["ssh", "-o", "BatchMode=yes", self.host, cmd], capture_output=True, text=True,
                               timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout_s:.0f} s"
        return p.returncode, p.stdout + p.stderr


class DryRun:
    """Sends nothing: a motion 'reaches' its target exactly; everything else answers 'ok'."""
    dry = True

    def __call__(self, cmd: str, timeout_s: float):
        print("DRY", cmd)
        if cmd.startswith("dma_motor_motion"):
            t = [float(kv.split("=")[1]) for kv in cmd.split('"')[1].split(",")]
            return 0, "\n".join(f"RESULT targets joint={JIDX[n]} target_deg={v:.3f} ss_err=0.000 rms=0 max=0 "
                                f"samples=0" for n, v in zip(RIGHT_ARM, t))
        return 0, "ok"


class Session:
    def __init__(self, plan: dict, remote, check, log_path, tag: str, burst_n: int = 10, jidx=JIDX):
        self.plan, self.remote, self.check = plan, remote, check
        self.log_path, self.tag, self.burst_n, self.jidx = Path(log_path), tag, burst_n, jidx
        self.kp, self.hold = float(plan["kp"]), float(plan["hold_s"])
        self.dry = bool(getattr(remote, "dry", False))

    def _log(self, **row) -> None:
        row["t_ns"] = time.time_ns()
        if self.dry:
            row["dry_run"] = True
        with self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _call(self, cmd: str, timeout_s: float) -> str:
        rc, out = self.remote(cmd, timeout_s)
        if rc != 0:
            raise Abort(f"rc {rc} from {cmd.split()[0]}: {out.strip()[-300:]}")
        return out

    def _move(self, i: int, target, ramp_s: float, desired) -> np.ndarray:
        cmd = motion_command(target, ramp_s, self.hold, self.kp)
        t0 = time.time_ns()
        out = self._call(cmd, ramp_s + self.hold + 30.0)
        reached = parse_result(out, self.jidx)
        self._log(event="move", stop=i, command=cmd, t0_ns=t0, target_deg=list(map(float, target)),
                  reached_deg=reached.tolist(), result="\n".join(ln for ln in out.splitlines() if "RESULT" in ln))
        off = np.abs(reached - np.asarray(desired, float))
        if off.max() > ABORT_DEG:
            bad = ", ".join(f"{RIGHT_ARM[j]} {off[j]:.1f} deg" for j in np.flatnonzero(off > ABORT_DEG))
            raise Abort(f"stop {i}: the arm stopped far from where it was sent ({bad}); holding here")
        return reached

    def _judge(self, desired, reached) -> tuple:
        why = []
        off = np.abs(reached - np.asarray(desired, float))
        if off.max() > CORRECT_DEG:
            why.append("off the desired pose: " + ", ".join(f"{RIGHT_ARM[j]} {off[j]:.1f} deg"
                                                            for j in np.flatnonzero(off > CORRECT_DEG)))
        ok, more = self.check(reached)
        return (not why and ok), why + list(more)

    def sweep(self) -> None:
        for i, s in enumerate(self.plan["stops"]):
            desired, target = np.array(s["q_deg"], float), np.array(s["command_deg"], float)
            reached = self._move(i, target, float(s["ramp_s"]), desired)
            row = {"event": "stop", "stop": i, "kind": s["kind"], "pose": s.get("pose"), "corrections": 0,
                   "status": "ok", "why": [], "reached_first_deg": reached.tolist()}
            if s["kind"] in MEASURED:
                ok, why = self._judge(desired, reached)
                if not ok:
                    target = target + (desired - reached)
                    reached = self._move(i, target, 6.0, desired)
                    row["corrections"] = 1
                    ok, why = self._judge(desired, reached)
                    if not ok:
                        row["status"] = "skipped"
                row["why"] = why
                if row["status"] == "ok":
                    out = f"/root/burst-grab/arm-{self.tag}-{i:02d}-p{s.get('pose')}.npz"
                    rc, msg = self.remote(GRAB.format(n=self.burst_n, out=out), 60.0)
                    row["burst"] = out if rc == 0 else None
                    if rc != 0:
                        row["why"] = row["why"] + [f"burst failed (rc {rc}): {msg.strip()[-200:]}"]
            row["reached_deg"] = reached.tolist()
            self._log(**row)


def prechecks(snap: dict, plan: dict, need_gb: float) -> list:
    """Reasons not to start (empty: go). snap: armsnap.py's JSON."""
    why = []
    d = snap["desired"]
    if d["advancing"] or d["age_s"] < DESIRED_QUIET_S:
        why.append(f"/desired written {d['age_s']:.0f} s ago{' and still advancing' if d['advancing'] else ''}: "
                   "another writer?")
    q = np.array([snap["q_deg"][n] for n in RIGHT_ARM])
    off = np.abs(q - np.array(plan["start_deg"]))
    if off.max() > START_TOL_DEG:
        why.append("arm not at the plan's start pose: " + ", ".join(f"{RIGHT_ARM[j]} {off[j]:.1f} deg"
                                                                     for j in np.flatnonzero(off > START_TOL_DEG)))
    for n, v in plan["neck_deg"].items():
        if abs(snap["q_deg"][n] - v) > NECK_TOL_DEG:
            why.append(f"{n} at {snap['q_deg'][n]:.1f}, the plan's {v:.1f}")
    tq = snap["joint_torques"]
    if max(abs(tq["RightShoulderPitch"]), abs(tq["RightElbowPitch"])) < MOTORS_ON_NM:
        why.append("motors off? the raised arm holds no torque at the shoulder or elbow")
    for n, v in tq.items():
        if re.search("Hip|Knee|Ankle", n) and abs(v) > LEG_MAX_NM:
            why.append(f"{n} carries {v:.1f} Nm: is the robot standing?")
    if snap["disk_free_gb"] < need_gb:
        why.append(f"disk: {snap['disk_free_gb']:.0f} GB free, the recording needs {need_gb:.0f}")
    return why


REC_POD = """apiVersion: v1
kind: Pod
metadata: {{ name: {name}, namespace: positronic, labels: {{ app: stereo-eval }} }}
spec:
  restartPolicy: Never
  nodeSelector: {{ kubernetes.io/hostname: phantom12 }}
  tolerations: [ {{ operator: Exists, effect: NoSchedule }} ]
  hostIPC: true
  imagePullSecrets: [ {{ name: dockerhub-creds }} ]
  terminationGracePeriodSeconds: 30
  containers:
    - name: record
      image: {image}
      imagePullPolicy: IfNotPresent
      args: ["record", "/output/{take}", "--cameras", "1,2", "--duration", "{duration}", "--imu-index", "2"]
      env: [ {{ name: OKVIS_DMA_IMU_SHM, value: /raw_imu_actuals }}, {{ name: OKVIS_SKIP_JETSON_CLOCKS, value: "1" }} ]
      volumeMounts: [ {{ name: shm, mountPath: /dev/shm }}, {{ name: output, mountPath: /output }} ]
      resources: {{ requests: {{ cpu: "500m", memory: "2Gi" }}, limits: {{ cpu: "4", memory: "4Gi" }} }}
      securityContext: {{ capabilities: {{ add: ["IPC_LOCK"] }} }}
  volumes:
    - {{ name: shm, hostPath: {{ path: /dev/shm, type: Directory }} }}
    - {{ name: output, hostPath: {{ path: /root/stereo-eval/takes, type: DirectoryOrCreate }} }}
"""


def make_check(plan: dict, urdf: Path, mjcf: Path):
    """The reached-pose check: the plate in view in both cameras (T2-3) and the arm clear of the body (T2-4), at the
    reached joints, with the plan's cameras and plate prior."""
    import armchain
    import armplan as ap
    import armsafety
    import rigcfg
    arm = armchain.ArmChain(urdf)
    g = plan["grid"]
    grid = rigcfg.AprilGrid(g["cols"], g["rows"], g["tag_size"], g["tag_spacing"])
    P, _ = ap.plate_points(grid)
    cams = [ap.Camera(np.array(c["T_TC"]), c["model"], np.array(c["intr"]), tuple(c["size"])) for c in plan["cameras"]]
    X0 = np.array(plan["X0"])
    body = armsafety.Body(mjcf)
    body.set_baseline({n: np.radians(v) for n, v in plan["baseline_deg"].items()})

    def check(q_deg):
        q = np.radians(np.asarray(q_deg, float))[None]
        why = []
        v = ap.view(cams, ap.plate_in_torso(arm, X0, q), P, grid)
        if not ap.visible(v)[0]:
            why.append("plate not in view: range {} m, facing {} deg, tag {} px, margin {} px".format(
                *(np.round(v[k][0], 2).tolist() for k in ("range_m", "facing_deg", "tag_px", "margin_px"))))
        c = float(body.clearance(q)[0])
        if c < 0.03:
            why.append(f"clearance {c * 100:.1f} cm")
        return not why, why
    return check


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--poses", required=True, help="step 90's poses.json")
    p.add_argument("--urdf", required=True)
    p.add_argument("--mjcf", required=True)
    p.add_argument("--tag", required=True, help="names the take and the bursts, e.g. 1023a")
    p.add_argument("--image", default="foundationbot/okvis2x:thor-5b592e01daadd3251ae857d1772745cc4f3258bb")
    p.add_argument("--snap-script", default=str(Path(__file__).resolve().parent / "armsnap.py"))
    p.add_argument("--log", default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    plan = json.loads(Path(a.poses).read_text())
    log = Path(a.log or f"arm-session-{a.tag}.jsonl")
    ssh = Ssh()
    duration = int(plan["motion_s"] * 1.5 + 60)
    need_gb = REC_GB_PER_MIN * duration / 60.0 * 1.5
    if not a.dry_run:
        subprocess.run(["scp", "-q", a.snap_script, "phantom12:/root/burst-grab/armsnap.py"], check=True)
        rc, out = ssh("cd /root/burst-grab && PYTHONPATH=pydeps nice python3 armsnap.py", 60)
        if rc:
            print(out)
            return 3
        snap = json.loads(out.strip().splitlines()[-1])
        if snap["names"][7:14] != list(RIGHT_ARM):
            print("the /config joint order is not the expected one:", snap["names"][7:14])
            return 3
        why = prechecks(snap, plan, need_gb)
        if why:
            print("NOT STARTED:\n  " + "\n  ".join(why))
            return 2
        take = f"se-arm-sweep-{a.tag}"
        pod = REC_POD.format(name=f"se-rec-{a.tag}", image=a.image, take=take, duration=duration)
        rc, out = ssh(f"cat > /root/stereo-eval/{take}.yaml <<'EOF'\n{pod}EOF\n"
                      f"k0s kubectl apply -f /root/stereo-eval/{take}.yaml && sleep 8 && "
                      f"k0s kubectl -n positronic get pod se-rec-{a.tag} -o jsonpath='{{.status.phase}}'", 60)
        if rc or "Running" not in out:
            print("recording did not start:", out)
            return 3
        time.sleep(4)
    s = Session(plan, DryRun() if a.dry_run else ssh, make_check(plan, Path(a.urdf), Path(a.mjcf)), log, a.tag)
    clock = None
    if not a.dry_run:                       # the robot's clock minus the laptop's, for finding holds in the take
        t0 = time.time_ns()
        rc, out = ssh("date +%s%N", 30)
        t1 = time.time_ns()
        clock = {"robot_minus_laptop_ns": int(out.strip()) - (t0 + t1) // 2, "round_trip_ns": t1 - t0} if rc == 0 \
            else None
    s._log(event="start", plan=a.poses, tag=a.tag, duration_s=duration, clock=clock)
    try:
        s.sweep()
    except Abort as e:
        s._log(event="abort", why=str(e))
        print("ABORTED:", e)
        return 1
    except KeyboardInterrupt:
        s._log(event="abort", why="interrupted by the operator")
        print("INTERRUPTED: the arm holds where it is")
        return 1
    s._log(event="done")
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    stops = [r for r in rows if r["event"] == "stop" and r["kind"] in MEASURED]
    print(f"done: {sum(r['status'] == 'ok' for r in stops)} of {len(stops)} measured stops kept, "
          f"{sum(r['corrections'] for r in stops)} corrections; log {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
