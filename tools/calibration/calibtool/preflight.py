"""Preflight (design section 4 step 0; nothing moves).

run(probes, profile) -> Result(checks, snapshot, baseline). probes is a dict of callables, one per fact (RobotProbes
gathers them on the robot; tests pass fakes); a probe that raises becomes a failed check with the error, never a crash.
Every Check has a status (ok / warn / fail / info), a plain detail and a gate:
  collect  needed to read the cameras and collect (M2: all of these ok)
  motion   needed before the motors are enabled (M5); reported now, they do not block collection
snapshot: what a restore needs (producer YAML, head_camera.json, robot.env, OKVIS state, master version, motor tool
md5). baseline: what the health watch compares against (fps, load, boot id, head-camera USB serials and ports, the
producer's generation / args)."""
import json
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibtool import robots

SYNC_MS = 0.5
CLOCK_MIN_S = 1.577e9                   # 2020-01-01
SUPERSPEED = 5000
MOVIDIUS = "03e7"                       # the head cameras' USB vendor (Luxonis / Intel Movidius)
POLICY_PATTERNS = ("ghost-wbc", "omni-wbc", "locomotion", "loco-", "walker", "policy", "sonic", "foundation-vla")
CAMERA_USER_PATTERNS = ("okvis", "recorder", "vla", "mediamtx", "rtsp", "viewer", "tag-detector")


@dataclass
class Check:
    name: str
    status: str                 # ok | warn | fail | info
    detail: str
    gate: str = "collect"       # collect | motion

    def as_dict(self):
        return {"name": self.name, "status": self.status, "detail": self.detail, "gate": self.gate}


@dataclass
class Result:
    checks: list
    snapshot: dict
    baseline: dict


def green(checks, gate="collect") -> bool:
    """no failed check of the gate"""
    return not any(c.status == "fail" and c.gate == gate for c in checks)


# ---------------------------------------------------------------- pure helpers

def disparity_sign(yl, yr, max_d=None, min_ncc=0.3, min_matches=30, min_frac=0.75):
    """(+1, d) when the right image is the left one shifted left by d px (a correct left/right pair), (-1, d) when
    shifted right (swapped queues), (0, d) when it cannot be told (flat or ambiguous image, |d| < 1).

    Feature matches (SIFT, ratio test) give per-match dx = x_left - x_right; d = median dx, decided only when at least
    `min_frac` of the matches agree in sign. Raw (unrectified) head pairs carry a vertical offset (~15 px on
    phantom10) that a horizontal-only correlation cannot absorb: on 9 Nov it called a correct pair with a board at
    0.76 m swapped. Without enough matches it falls back to that correlation (fine on synthetic, texture-only pairs)."""
    try:
        import cv2
        a8 = np.clip(np.asarray(yl, np.float32), 0, 255).astype(np.uint8)
        b8 = np.clip(np.asarray(yr, np.float32), 0, 255).astype(np.uint8)
        sift = cv2.SIFT_create(3000)
        k1, d1 = sift.detectAndCompute(a8, None)
        k2, d2 = sift.detectAndCompute(b8, None)
        if d1 is not None and d2 is not None and len(k1) >= 2 and len(k2) >= 2:
            good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.7 * n.distance]
            if len(good) >= min_matches:
                dx = np.array([k1[m.queryIdx].pt[0] - k2[m.trainIdx].pt[0] for m in good])
                med, pos = float(np.median(dx)), float(np.mean(dx > 0))
                if abs(med) >= 1 and pos >= min_frac:
                    return 1, med
                if abs(med) >= 1 and pos <= 1 - min_frac:
                    return -1, med
                return 0, med
    except ImportError:
        pass
    return _disparity_sign_ncc(yl, yr, max_d, min_ncc)


def _disparity_sign_ncc(yl, yr, max_d=None, min_ncc=0.3):
    """horizontal-only whole-image correlation (the original check; fallback for feature-poor images)"""
    L = np.asarray(yl, np.float32)
    R = np.asarray(yr, np.float32)
    W = L.shape[1]
    m = int(max_d if max_d is not None else min(32, W // 8))
    b = R[:, m:W - m]
    b = b - b.mean()
    nb = np.sqrt((b * b).sum())
    best = (-2.0, 0)
    for d in range(-m, m + 1):
        a = L[:, m + d:W - m + d]
        a = a - a.mean()
        den = np.sqrt((a * a).sum()) * nb
        if den <= 0:
            continue
        ncc = float((a * b).sum() / den)
        if ncc > best[0]:
            best = (ncc, d)
    ncc, d = best
    if ncc < min_ncc or abs(d) < 1:
        return 0, d
    return (1 if d > 0 else -1), d


def head_serials(head_camera_json, queues) -> list:
    """USB serials (mxid) of the boards carrying the given camera queues"""
    j = json.loads(head_camera_json)
    boards = {str(c.get("board")) for c in (j.get("cameras") or {}).values() if c.get("queue_id") in queues}
    return sorted({str(j["boards"][b]["mxid"]) for b in boards if b in (j.get("boards") or {})})


def usb_speeds(sysfs="/sys/bus/usb/devices") -> dict:
    """{serial: (Mbit/s, port)} of the Movidius (camera board) USB devices"""
    out = {}
    for d in sorted(Path(sysfs).iterdir()):
        try:
            if (d / "idVendor").read_text().strip() != MOVIDIUS:
                continue
            out[(d / "serial").read_text().strip()] = (int(float((d / "speed").read_text().strip())), d.name)
        except (OSError, ValueError):
            continue
    return out


def policy_pods(names) -> list:
    return [n for n in names if any(p in n.split("/")[-1] for p in POLICY_PATTERNS)]


def camera_users(names) -> list:
    return [n for n in names if any(p in n.split("/")[-1] for p in CAMERA_USER_PATTERNS)]


# ---------------------------------------------------------------- the checks

def _probe(probes, name):
    try:
        return probes[name](), None
    except Exception as e:  # noqa: BLE001  (a broken probe is a failed check)
        return None, f"{type(e).__name__}: {e}"


def run(probes, prof) -> Result:
    checks, snap, base = [], {}, {}

    def add(name, status, detail, gate="collect"):
        checks.append(Check(name, status, detail, gate))

    def get(name, check_name=None, gate="collect"):
        v, err = _probe(probes, name)
        if err:
            add(check_name or name, "fail", f"could not check: {err}", gate)
        return v, err

    v, err = get("identity")
    if not err:
        host, mid = v
        try:
            idn = robots.identify(host, mid)
            if idn.robot.bot != prof.bot:
                add("identity", "fail", f"this is {idn.robot.bot} ({host}), not {prof.bot}")
            else:
                add("identity", "ok", f"{idn.robot.bot} = {host}, machine-id {str(mid)[:12]} "
                    f"({'verified' if idn.machine_id_verified else 'by host name only'})")
        except robots.IdentityError as e:
            add("identity", "fail", f"identity by machine-id refused: {e}")
    v, err = get("lock")
    if not err:
        add("lock", "ok" if v[0] else "fail", f"one calibration job per robot: {v[1]}" if v[0] else
            f"another calibd holds the robot lock: {v[1]}")
    v, err = get("clock")
    if not err:
        add("clock", "ok" if v >= CLOCK_MIN_S else "fail", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v))
            if v >= CLOCK_MIN_S else
            f"the clock reads {time.strftime('%Y-%m-%d', time.gmtime(v))} (1970-like): not set")
    boot, _ = _probe(probes, "boot_id")
    base["boot_id"] = boot
    v, err = get("k0s")
    if not err:
        add("k0s", "ok" if v[0] else "fail", v[1] if v[0] else f"k0s kubectl refused: {v[1]}")
    v, err = get("autonomy")
    if not err:
        if v is None:
            add("autonomy", "ok", "nav-cli is not installed on this robot: nothing to arm")
        elif re.search(r"\barmed\b", v, re.I) and not re.search(r"\b(dis|un|not\s+)armed\b", v, re.I):
            add("autonomy", "fail", f"autonomy is armed ({v.strip()[:80]}): disarm it first")
        else:
            add("autonomy", "ok", v.strip()[:80])
    pods, err = get("pods", "policy_pods", "motion")
    if not err:
        pol = policy_pods(pods)
        add("policy_pods", "fail" if pol else "ok", ("running: " + ", ".join(pol) + " (must be off before the motors "
            "are enabled; the robot is reserved for the session)") if pol else "no locomotion/policy pod running",
            "motion")
        users = camera_users(pods)
        add("camera_users", "info", ("also using the cameras: " + ", ".join(users)) if users else
            "nothing else reads the cameras")
    v, err = get("producer_spec", "producer")
    if not err:
        from calibtool import producer as pr
        spec = pr.Spec.from_yaml(v)
        snap["producer_yaml"] = v
        base["producer"] = {"generation": spec.generation, "args": spec.args}
        w, h, fps = spec.size()
        add("producer", "ok", f"{w}x{h} @ {fps} fps, image {spec.image}, generation {spec.generation} (saved)")
        base["producer_size"] = (w, h, fps)
    hj, err = get("head_camera_json")
    serials = []
    if not err:
        snap["head_camera_json"] = hj
        try:
            serials = head_serials(hj, (prof.left, prof.right))
            add("head_camera_json", "ok", f"saved; head cameras on board {', '.join(serials) or '?'}")
        except (ValueError, KeyError) as e:
            add("head_camera_json", "fail", f"unreadable: {e}")
    base["head_serials"] = serials
    env, _ = _probe(probes, "robot_env")
    snap["robot_env"] = env
    v, err = get("okvis")
    if not err:
        snap["okvis"] = v
        add("okvis", "info", ("OKVIS running: " if v.get("running") else "OKVIS not running: ") + str(v.get("detail")))
    v, err = get("usb", "usb_link")
    ports = []
    if not err:
        bad = []
        for s in serials:
            sp = v.get(s)
            if sp is None:
                bad.append(f"{s} not on the USB bus")
            else:
                ports.append(sp[1])
                if sp[0] < SUPERSPEED:
                    bad.append(f"{s} at {sp[0]} Mbit/s (below SuperSpeed: a cable/hub/port fault)")
        if not serials:
            bad.append("no head camera board known (head_camera.json)")
        add("usb_link", "fail" if bad else "ok", "; ".join(bad) if bad else ", ".join(
            f"{s} {v[s][0]} Mbit/s on {v[s][1]}" for s in serials))
    base["head_ports"] = ports
    f, err = get("frames")
    if not err:
        want_w, want_h = (base.get("producer_size") or (None, None, None))[:2]
        size = tuple(f["size"]) if f.get("size") else None
        prob = []
        if f["pairs"] < f["wanted"]:
            prob.append(f"{f['pairs']} of {f['wanted']} pairs within the timeout")
        if f.get("none_reads"):
            prob.append(f"{f['none_reads']} frame reads came back empty (pool out of step)")
        if size and want_w and size != (want_w, want_h):
            prob.append(f"frames are {size[0]}x{size[1]}, the producer says {want_w}x{want_h}")
        add("frames", "fail" if prob else "ok", "; ".join(prob) if prob else
            f"{f['pairs']} pairs from queues {prof.left}/{prof.right}, {size[0]}x{size[1]}, fps " +
            ", ".join(f"q{q} {x:.1f}" for q, x in sorted((f.get("fps") or {}).items())))
        fps_vals = [x for x in (f.get("fps") or {}).values() if x]
        base["fps"] = float(np.median(fps_vals)) if fps_vals else None
        sm = f.get("sync_median_ms")
        if sm is None:
            add("sync", "fail", "no pair to measure the left/right stamp offset")
        else:
            add("sync", "ok" if sm <= SYNC_MS else "fail", f"left/right capture stamps {sm:.3f} ms apart (limit "
                f"{SYNC_MS} ms)")
        if f.get("left") is not None and f.get("right") is not None:
            yl, yr = f["left"][::2, ::2], f["right"][::2, ::2]
            s, d = disparity_sign(yl, yr)
            if s > 0:
                add("left_right", "ok", f"queue {prof.left} is the left camera (scene shifted {abs(d) * 2:.0f} px left "
                    f"in queue {prof.right})")
            elif s < 0:
                add("left_right", "fail", f"queues swapped: queue {prof.left} sees the scene {abs(d) * 2:.0f} px left "
                    f"of queue {prof.right}; fix the queue mapping")
            else:
                add("left_right", "warn", "could not tell left from right on this scene (too flat or too far); "
                    "the board check at the first placement decides")
    v, err = get("imu")
    if not err:
        add("imu", "ok" if v[1] != v[0] else "fail", f"IMU ring advancing ({v[1] - v[0]} samples)" if v[1] != v[0]
            else "the IMU ring is not advancing: check the head power switch first")
    v, err = get("disk")
    if not err:
        add("disk", "ok" if v[0] else "fail", f"{v[1]:.1f} GB free under {prof.root}")
    v, err = get("shm_count", "shm")
    if not err:
        snap["shm_count"] = v
        add("shm", "info", f"{v} segments in /dev/shm")
    v, err = get("load")
    if not err:
        base["load"] = float(v[0])
        snap["cpus"] = int(v[1])
    m, err = get("master", gate="motion")
    if not err:
        snap["master"] = m.as_dict()
        add("master", "ok" if m.active and not m.error else "fail", f"dma-ethercat {m.version or '?'} active, "
            f"uptime {m.uptime_s:.0f} s" if m.active and not m.error else f"master not ready: {m.error}", "motion")
        if m.motor_state is not None:
            add("motors_off", "ok" if not m.motors_on else "warn", "motors are off (the tool enables them at M5)"
                if not m.motors_on else f"motors are ON already (motor_state {m.motor_state})", "motion")
        add("estop", "ok" if m.estop is False else "warn", "e-stop not engaged" if m.estop is False else
            f"e-stop state: {m.estop}", "motion")
    v, err = get("desired_still", gate="motion")
    if not err:
        add("desired_still", "ok" if v else "fail", "/dev/shm/desired still for 1 s: nothing commanding" if v else
            "/dev/shm/desired is advancing: something is commanding the motors", "motion")
    v, err = get("motor_tool", gate="motion")
    if not err:
        snap["motor_tool"] = v
        ok = v.get("md5") and v.get("has_targets")
        add("motor_tool", "ok" if ok else "fail", f"{v['path']} md5 {v['md5']}, has 'targets'" if ok else
            (v.get("error") or f"{v['path']} has no 'targets' signal: it would leave the spine and legs at zero "
             "stiffness during a neck move"), "motion")
    add("gantry", "info", "the operator confirms the robot hangs on the gantry before the motors are enabled (M5)",
        "motion")
    return Result(checks, snap, base)


# ---------------------------------------------------------------- the robot's probes

def _sh(argv, timeout=10):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, f"{type(e).__name__}: {e}"


class RobotProbes(dict):
    """the probes on the robot: kube (producer.Kube), capture (a running capture.Capture), lock_info () -> (ok, text),
    run (argv, timeout) -> (rc, text)"""

    def __init__(self, prof, kube, capture, lock_info, run=_sh, shm="/dev/shm", sysfs="/sys/bus/usb/devices",
                 head_json="/etc/phantom/head_camera.json", robot_env="/etc/phantom/robot.env"):
        super().__init__()
        from calibtool import motors, storage
        self.prof, self.kube, self.capture, self.run = prof, kube, capture, run
        self.update({
            "identity": lambda: (socket.gethostname(), Path("/etc/machine-id").read_text().strip()),
            "lock": lock_info,
            "clock": time.time,
            "boot_id": lambda: Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "k0s": self._k0s,
            "autonomy": self._autonomy,
            "pods": self._pods,
            "producer_spec": kube.get_yaml,
            "head_camera_json": lambda: Path(head_json).read_text(),
            "robot_env": lambda: Path(robot_env).read_text() if Path(robot_env).is_file() else None,
            "okvis": self._okvis,
            "usb": lambda: usb_speeds(sysfs),
            "frames": self._frames,
            "imu": lambda: self._ring_delta(Path(shm) / "raw_imu_actuals"),
            "master": lambda: motors.master(lambda a, timeout=5: run(a, timeout)),
            "motor_tool": lambda: motors.motor_tool(prof.motor_tool, lambda a, timeout=5: run(a, timeout)),
            "desired_still": lambda: motors.desired_still(lambda: motors.ring_head(Path(shm) / "desired")),
            "shm_count": lambda: len(os.listdir(shm)),
            "disk": lambda: storage.disk_check(prof.root),
            "load": lambda: (os.getloadavg()[0], os.cpu_count()),
        })

    def _k0s(self):
        try:
            return True, f"{len(self.kube.pods())} producer pod(s) listed"
        except Exception as e:  # noqa: BLE001
            return False, str(e)[:200]

    def _autonomy(self):
        rc, out = self.run(["nav-cli", "status"], 5)
        if rc == 127 and ("FileNotFoundError" in out or "No such file" in out):
            return None
        return out

    def _pods(self):
        rc, out = self.run(["k0s", "kubectl", "get", "pods", "-A", "--field-selector=status.phase=Running", "-o",
                            "jsonpath={range .items[*]}{.metadata.namespace}/{.metadata.name}{\"\\n\"}{end}"], 20)
        if rc != 0:
            raise RuntimeError(f"kubectl get pods failed: {out.strip()[:200]}")
        return [ln.strip() for ln in out.splitlines() if ln.strip()]

    def _okvis(self):
        rc, out = self.run(["k0s", "kubectl", "get", "pods", "-A", "--no-headers"], 20)
        lines = [ln for ln in out.splitlines() if "okvis2x" in ln]
        return {"running": any("Running" in ln for ln in lines),
                "detail": "; ".join(" ".join(ln.split()[:4]) for ln in lines) or "no okvis2x pod"}

    def _frames(self, n=3, timeout=6.0):
        c0 = {q: (k.none_reads, k.frames) for q, k in self.capture.counters.items()}
        prs = self.capture.grab(n, timeout)
        snap = self.capture.snapshot()
        none = sum(self.capture.counters[q].none_reads - c0[q][0] for q in c0)
        dts = [abs(p.dt_ns) for p in prs]
        last = prs[-1] if prs else None
        return {"pairs": len(prs), "wanted": n, "none_reads": none,
                "sync_median_ms": float(np.median(dts)) / 1e6 if dts else None,
                "size": None if last is None else (last.left.y.shape[1], last.left.y.shape[0]),
                "fps": {q: c["fps"] for q, c in snap["cams"].items()},
                "left": None if last is None else last.left.y, "right": None if last is None else last.right.y}

    @staticmethod
    def _ring_delta(path, wait=0.5):
        from calibtool import motors
        h0 = motors.ring_head(path)
        time.sleep(wait)
        return h0, motors.ring_head(path)
