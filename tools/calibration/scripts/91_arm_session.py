#!/usr/bin/env python3
"""Step 91 — the arm session's sweep (docs/superpowers/plans/2026-10-22-phantom12-arm-session.md, Task 3; rev 3's
T2-5; the second code review's S1-S6). Runs on the laptop; every robot command goes over ssh to the robot.

For each stop of step 90's poses.json (refused unless its gate passed):
- the path from where the arm actually stands to the stop's desired pose is re-checked (step 90's Checks.path);
- the arm is sent to the stop's command (dma_motor_motion, every right-arm joint listed, --no-return, one kp), which
  must be finite and inside the joint ranges;
- where each joint stopped is read from the RESULT lines (reached = target - ss_err), which must echo the target;
- any joint further from the desired pose than its ABORT_DEG (the sag band step 90 checked) stops the sweep: the arm
  then holds its last target (--no-return);
- at a measured stop (fit, held out, repeat): a reached pose more than CORRECT_DEG off, or failing the reached-pose
  check (plate detectable, clearance, workspace), gets one correction (target += desired - reached) if every joint's
  correction is within CORRECTION_CAP_DEG and the new target stays CMD_IN_DEG inside the ranges, else it is skipped;
  a kept pose gets a depth burst with every joint (the point-cloud agent's grab.py --once).
Every move is logged (JSON lines, a real run's log is never appended to; a dry run writes *.dry.jsonl).

--dry-run sends nothing and lands each move exactly on the stop's desired pose. The recording (record app, both
cameras, /joints) and the pre-checks are main()'s; the sweep logic is Session's, testable against a fake robot.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from armchain import RIGHT_ARM  # noqa: E402

CORRECT_DEG = 4.0
ABORT_DEG = np.array([8.0, 5.0, 5.0, 8.0, 5.0, 5.0, 5.0])            # = step 90's sag band (shoulder pitch, elbow)
CORRECTION_CAP_DEG = np.array([8.0, 4.0, 4.0, 8.0, 4.0, 4.0, 4.0])
CMD_IN_DEG = 5.0                                                     # = step 90's CMD_IN_DEG
TARGET_ECHO_DEG = 0.01
START_TOL_DEG, NECK_TOL_DEG, BODY_TOL_DEG, GRAVITY_TOL_DEG = 3.0, 1.0, 3.0, 2.0
ACTUALS_MAX_AGE_S = 1.0
DESIRED_QUIET_S = 60.0
MOTORS_ON_NM = 1.0              # a raised arm held at kp 100 needs ~10 Nm at the shoulder and elbow
LEG_MAX_NM = 10.0               # the robot hangs: its legs carry nothing (22 Oct: <= 5.4 Nm)
REC_GB_PER_MIN = 3.4            # two-camera PNG, measured 22 Oct (775 MB / 15 s)
PER_STOP_EXTRA_S = 10.0         # a burst (grab.py start + ssh) and maybe a correction, on top of step 90's motion_s
JIDX = {n: 7 + i for i, n in enumerate(RIGHT_ARM)}       # /config indices (checked against the snapshot's order)
MEASURED = ("fit", "heldout", "repeat_same", "repeat_opp")
PELVIS_IMU_R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # MJCF site pelvis_imu: +90 deg z
GRAB = ("cd /root/burst-grab && PYTHONPATH=/root/burst-grab/pydeps nice -n 10 python3 "
        "repo/tools/burst_pointcloud/grab.py --once {n} --out {out} --joints all")


class Abort(RuntimeError):
    pass


def parse_result(text: str, jidx=JIDX, sent=None) -> np.ndarray:
    """dma_motor_motion's RESULT lines -> where each right-arm joint stopped (deg), in RIGHT_ARM order. Every value
    must be finite and, given `sent`, every target_deg must echo what was sent (which also re-checks the joint-index
    mapping on every move)."""
    got, tgt = {}, {}
    for m in re.finditer(r"RESULT targets joint=(\d+) target_deg=(\S+) ss_err=(\S+)", text):
        tgt[int(m.group(1))] = float(m.group(2))
        got[int(m.group(1))] = float(m.group(2)) - float(m.group(3))
    missing = [n for n in RIGHT_ARM if jidx[n] not in got]
    if missing:
        raise Abort(f"no RESULT line for {', '.join(missing)}")
    r = np.array([got[jidx[n]] for n in RIGHT_ARM])
    if not np.isfinite(r).all():
        raise Abort("a RESULT value is not finite: " + ", ".join(n for n, v in zip(RIGHT_ARM, r) if not np.isfinite(v)))
    if sent is not None:
        t = np.array([tgt[jidx[n]] for n in RIGHT_ARM])
        off = np.abs(t - np.asarray(sent, float))
        if off.max() > TARGET_ECHO_DEG:
            raise Abort("RESULT target_deg does not echo the target sent: " + ", ".join(
                f"{RIGHT_ARM[j]} {t[j]:.3f} vs {sent[j]:.3f}" for j in np.flatnonzero(off > TARGET_ECHO_DEG)))
    return r


def motion_command(target_deg, ramp_s: float, hold_s: float, kp: float, tol_deg: float = 30.0, limits=None) -> str:
    """The dma_motor_motion line; refused (Abort) for a non-finite target or one outside `limits` ([lower, upper]
    deg) minus nothing: callers keep CMD_IN_DEG themselves."""
    t = np.asarray(target_deg, float)
    if not np.isfinite(t).all() or not np.isfinite([ramp_s, hold_s, kp]).all():
        raise Abort(f"refusing a non-finite command {t.tolist()}")
    if limits is not None:
        lo, hi = np.asarray(limits[0], float), np.asarray(limits[1], float)
        if (t < lo).any() or (t > hi).any():
            raise Abort(f"refusing a command outside the joint ranges: {t.round(2).tolist()}")
    spec = ",".join(f"{n}={v:.2f}" for n, v in zip(RIGHT_ARM, t))
    return (f'dma_motor_motion "{spec}" targets position --ramp {ramp_s:.1f} --hold {hold_s:.1f} --no-return '
            f"--kp {kp:.0f} --tolerance {tol_deg:.0f}")


class Ssh:
    def __init__(self, host: str = "phantom12"):
        self.host = host

    def __call__(self, cmd: str, timeout_s: float):
        try:
            p = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", self.host, cmd], capture_output=True, text=True,
                               timeout=timeout_s, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout_s:.0f} s (the command may still be running on the robot)"
        return p.returncode, p.stdout + p.stderr


class DryRun:
    """Sends nothing: a motion lands on `landing` (the stop's desired pose, set by Session) or on its target."""
    dry = True
    landing = None

    def __call__(self, cmd: str, timeout_s: float):
        print("DRY", cmd)
        if cmd.startswith("dma_motor_motion"):
            t = [float(kv.split("=")[1]) for kv in cmd.split('"')[1].split(",")]
            r = t if self.landing is None else list(self.landing)
            return 0, "\n".join(f"RESULT targets joint={JIDX[n]} target_deg={v:.3f} ss_err={v - x:.3f} rms=0 max=0 "
                                 f"samples=0" for n, v, x in zip(RIGHT_ARM, t, r))
        return 0, "ok"


class Session:
    def __init__(self, plan: dict, remote, check, log_path, tag: str, burst_n: int = 10, jidx=JIDX,
                 path_check=None):
        self.plan, self.remote, self.check = plan, remote, check
        self.log_path, self.tag, self.burst_n, self.jidx = Path(log_path), tag, burst_n, jidx
        self.kp, self.hold = float(plan["kp"]), float(plan["hold_s"])
        self.limits = np.array(plan["limits_deg"], float)
        if path_check is None:
            raise ValueError("a path check is required (step 90's Checks.path, or a test's fake)")
        self.path_check = path_check
        self.dry = bool(getattr(remote, "dry", False))
        self.at = np.array(plan["start_deg"], float)          # where the arm stands now (deg)

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
        why = self.path_check(self.at, desired)
        if why:
            raise Abort(f"stop {i}: the path from where the arm stands to the stop fails: {'; '.join(why)}")
        cmd = motion_command(target, ramp_s, self.hold, self.kp, limits=self.limits)
        if self.dry:
            self.remote.landing = desired
        t0 = time.time_ns()
        out = self._call(cmd, ramp_s + self.hold + 30.0)
        reached = parse_result(out, self.jidx, sent=np.round(np.asarray(target, float), 2))
        self.at = reached
        self._log(event="move", stop=i, command=cmd, t0_ns=t0, target_deg=list(map(float, target)),
                  reached_deg=reached.tolist(), result="\n".join(ln for ln in out.splitlines() if "RESULT" in ln))
        off = np.abs(reached - np.asarray(desired, float))
        if (off > ABORT_DEG).any():
            bad = ", ".join(f"{RIGHT_ARM[j]} {off[j]:.1f} deg" for j in np.flatnonzero(off > ABORT_DEG))
            raise Abort(f"stop {i}: the arm stopped outside the band step 90 checked ({bad}); holding here")
        return reached

    def _judge(self, desired, reached) -> tuple:
        why = []
        off = np.abs(reached - np.asarray(desired, float))
        if off.max() > CORRECT_DEG:
            why.append("off the desired pose: " + ", ".join(f"{RIGHT_ARM[j]} {off[j]:.1f} deg"
                                                            for j in np.flatnonzero(off > CORRECT_DEG)))
        ok, more = self.check(reached)
        return (not why and ok), why + list(more)

    def _correction(self, target, desired, reached):
        """(new target, None) or (None, why not)."""
        corr = np.asarray(desired, float) - reached
        if (np.abs(corr) > CORRECTION_CAP_DEG).any():
            return None, "correction over its cap: " + ", ".join(
                f"{RIGHT_ARM[j]} {corr[j]:+.1f} deg" for j in np.flatnonzero(np.abs(corr) > CORRECTION_CAP_DEG))
        new = np.asarray(target, float) + corr
        lo, hi = self.limits[0] + CMD_IN_DEG, self.limits[1] - CMD_IN_DEG
        if (new < lo).any() or (new > hi).any():
            return None, "correction would command within 5 deg of a joint range: " + ", ".join(
                RIGHT_ARM[j] for j in np.flatnonzero((new < lo) | (new > hi)))
        return new, None

    def sweep(self) -> None:
        for i, s in enumerate(self.plan["stops"]):
            desired, target = np.array(s["q_deg"], float), np.array(s["command_deg"], float)
            reached = self._move(i, target, float(s["ramp_s"]), desired)
            row = {"event": "stop", "stop": i, "kind": s["kind"], "pose": s.get("pose"), "corrections": 0,
                   "status": "ok", "why": [], "reached_first_deg": reached.tolist()}
            if s["kind"] in MEASURED:
                ok, why = self._judge(desired, reached)
                if not ok:
                    new, no = self._correction(target, desired, reached)
                    if new is None:
                        row["status"], why = "skipped", why + [no]
                    else:
                        reached = self._move(i, new, 6.0, desired)
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


def gravity_of(snap) -> np.ndarray:
    a = snap["imu"]["Pelvis"]
    f = PELVIS_IMU_R @ np.array([a["accel_x"], a["accel_y"], a["accel_z"]])
    return -f / np.linalg.norm(f)


def prechecks(snap: dict, plan: dict, need_gb: float) -> list:
    """Reasons not to start (empty: go). snap: armsnap.py's JSON."""
    why = []
    d = snap["desired"]
    if d["advancing"] or d["age_s"] < DESIRED_QUIET_S:
        why.append(f"/desired written {d['age_s']:.0f} s ago{' and still advancing' if d['advancing'] else ''}: "
                   "another writer?")
    if snap.get("actuals_age_s", np.inf) > ACTUALS_MAX_AGE_S:
        why.append(f"/actuals is {snap.get('actuals_age_s', float('inf')):.1f} s old")
    q = np.array([snap["q_deg"][n] for n in RIGHT_ARM])
    off = np.abs(q - np.array(plan["start_deg"]))
    if off.max() > START_TOL_DEG:
        why.append("arm not at the plan's start pose: " + ", ".join(f"{RIGHT_ARM[j]} {off[j]:.1f} deg"
                                                                     for j in np.flatnonzero(off > START_TOL_DEG)))
    for n, v in plan["neck_deg"].items():
        if abs(snap["q_deg"][n] - v) > NECK_TOL_DEG:
            why.append(f"{n} at {snap['q_deg'][n]:.1f}, the plan's {v:.1f}")
    for n, v in plan["baseline_deg"].items():                  # the body the clearance model was built on
        if n in RIGHT_ARM or n in plan["neck_deg"] or n not in snap["q_deg"]:
            continue
        if abs(snap["q_deg"][n] - v) > BODY_TOL_DEG:
            why.append(f"{n} at {snap['q_deg'][n]:.1f}, the plan's {v:.1f} (the clearance model assumes it)")
    g = gravity_of(snap)
    ang = np.degrees(np.arccos(np.clip(g @ np.asarray(plan["gravity_in_pelvis"], float), -1, 1)))
    if ang > GRAVITY_TOL_DEG:
        why.append(f"gravity {ang:.1f} deg off the plan's: the robot moved")
    tq = snap["joint_torques"]
    if max(abs(tq["RightShoulderPitch"]), abs(tq["RightElbowPitch"])) < MOTORS_ON_NM:
        why.append("motors off? the raised arm holds no torque at the shoulder or elbow")
    for n, v in tq.items():
        if re.search("Hip|Knee|Ankle", n) and abs(v) > LEG_MAX_NM:
            why.append(f"{n} carries {v:.1f} Nm: is the robot standing?")
    if snap["disk_free_gb"] < need_gb:
        why.append(f"disk: {snap['disk_free_gb']:.0f} GB free, the recording needs {need_gb:.0f}")
    return why


def log_path(given, tag: str, dry: bool) -> Path:
    """The session log: a dry run writes *.dry.jsonl; a real run never appends to an existing log (review 2, W4)."""
    p = Path(given) if given else Path(f"arm-session-{tag}{'.dry' if dry else ''}.jsonl")
    if not dry and p.exists():
        raise SystemExit(f"{p} exists: a real run starts a new log (use another --tag or --log)")
    return p


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


REACHED_VIEW = dict(margin_px=15.0, range_m=(0.30, 0.85), max_facing_deg=60.0, min_tag_px=30.0)   # review 2, W8


def make_check(plan: dict, urdf: Path, mjcf: Path):
    """The reached-pose check: the plate still detectable in both cameras (REACHED_VIEW: looser than T2-3's planning
    margins, which the selection sits on), the arm clear of the body (T2-4: contact geoms and mesh surfaces) and
    inside the workspace (T2-4b), at the reached joints, with the plan's cameras and plate prior."""
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
    body = armsafety.Body(mjcf, surface_step=0.008)
    body.set_baseline({n: np.radians(v) for n, v in plan["baseline_deg"].items()})
    up = np.array(plan["up_in_torso"])

    def check(q_deg):
        q = np.radians(np.asarray(q_deg, float))[None]
        why = []
        v = ap.view(cams, ap.plate_in_torso(arm, X0, q), P, grid)
        if not ap.visible(v, **REACHED_VIEW)[0]:
            why.append("plate not in view: range {} m, facing {} deg, tag {} px, margin {} px".format(
                *(np.round(v[k][0], 2).tolist() for k in ("range_m", "facing_deg", "tag_px", "margin_px"))))
        c = float(body.clearance(q)[0])
        if c < 0.03:
            why.append(f"clearance {c * 100:.1f} cm")
        if not ap.workspace_ok(arm, q, up)[0]:
            why.append("outside the workspace (T2-4b)")
        return not why, why
    return check


def make_path_check(plan: dict, urdf: Path, mjcf: Path):
    """step 90's Checks.path, in degrees, on the plan's body model: re-run before every move from where the arm
    actually stands (review 2, S2)."""
    import armchain
    import armsafety
    s90 = importlib.import_module("90_arm_poses")
    body = armsafety.Body(mjcf, surface_step=0.008)
    body.set_baseline({n: np.radians(v) for n, v in plan["baseline_deg"].items()})
    chk = s90.Checks(armchain.ArmChain(urdf), body, plan["up_in_torso"], plan["gravity_in_pelvis"], kp=plan["kp"])
    return lambda a, b: chk.path(np.radians(a), np.radians(b))


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
    if not plan.get("gate", {}).get("pass") or not plan.get("stops"):
        print(f"{a.poses}: its gate did not pass (or it has no stops); step 91 runs only a list that passed step 90")
        return 3
    for k, f in (("urdf", a.urdf), ("mjcf", a.mjcf)):
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()[:12] != plan["inputs"][k]:
            print(f"--{k} {f} is not the file step 90 planned with ({plan['inputs'][k]})")
            return 3
    log = log_path(a.log, a.tag, a.dry_run)
    ssh = Ssh()
    duration = int(plan["motion_s"] * 1.5 + PER_STOP_EXTRA_S * len(plan["stops"]) + 120)
    need_gb = REC_GB_PER_MIN * duration / 60.0 * 1.5
    take = f"se-arm-sweep-{a.tag}"
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
        pod = REC_POD.format(name=f"se-rec-{a.tag}", image=a.image, take=take, duration=duration)
        rc, out = ssh(f"cat > /root/stereo-eval/{take}.yaml <<'EOF'\n{pod}EOF\n"
                      f"k0s kubectl apply -f /root/stereo-eval/{take}.yaml && sleep 8 && "
                      f"k0s kubectl -n positronic get pod se-rec-{a.tag} -o jsonpath='{{.status.phase}}'", 60)
        if rc or "Running" not in out:
            print("recording did not start:", out)
            return 3
        size = "stat -c %s /root/stereo-eval/takes/" + take + "/session.mcap"
        rc1, s1 = ssh(size, 30)
        time.sleep(4)
        rc2, s2 = ssh(size, 30)
        if rc1 or rc2 or int(s2.strip() or 0) <= int(s1.strip() or 0):
            print(f"the recording is not growing ({s1.strip()} -> {s2.strip()} bytes): not starting")
            return 3
    s = Session(plan, DryRun() if a.dry_run else ssh, make_check(plan, Path(a.urdf), Path(a.mjcf)), log, a.tag,
                path_check=make_path_check(plan, Path(a.urdf), Path(a.mjcf)))
    clock = None
    if not a.dry_run:                       # the robot's clock minus the laptop's (holds are found from /joints)
        t0 = time.time_ns()
        rc, out = ssh("date +%s%N", 30)
        t1 = time.time_ns()
        clock = {"robot_minus_laptop_ns": int(out.strip()) - (t0 + t1) // 2, "round_trip_ns": t1 - t0} if rc == 0 \
            else None
    s._log(event="start", plan=a.poses, tag=a.tag, duration_s=duration, clock=clock)
    try:
        s.sweep()
    except BaseException as e:              # noqa: BLE001 -- every way out is logged; the arm keeps its last target
        why = "interrupted by the operator" if isinstance(e, KeyboardInterrupt) else f"{type(e).__name__}: {e}"
        s._log(event="abort", why=why)
        print("STOPPED:", why)
        print("The arm holds its last target (--no-return). A command in flight may still finish its ramp on the robot.")
        return 1
    if not a.dry_run:
        rc, out = ssh(f"k0s kubectl -n positronic get pod se-rec-{a.tag} -o jsonpath='{{.status.phase}}'", 30)
        if "Running" not in out:
            s._log(event="warning", why=f"the recording pod is {out.strip()!r} at the end: some holds may be unrecorded")
    s._log(event="done")
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    stops = [r for r in rows if r["event"] == "stop" and r["kind"] in MEASURED]
    print(f"done: {sum(r['status'] == 'ok' for r in stops)} of {len(stops)} measured stops kept, "
          f"{sum(r['corrections'] for r in stops)} corrections; log {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
