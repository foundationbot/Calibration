"""Onboard a robot for calibtool: `python -m calibtool.onboard <host> [--dry-run] [--bot PNN] [--old-name NAME]...`

What a new robot needs before `calib <host>` can run on it (10 Nov 2026: phantom11 / P11 was set up so by hand):
  identity   its /etc/machine-id (first 12 hex digits) and host name in config/robots.yaml, with its rig file
  cameras    left / right queues from /etc/phantom/head_camera.json (ego_left / ego_right; calibd re-checks left/right
             by disparity in preflight anyway); the producer dma-video/producer
  rig file   tools/calibration/config/rig-<bot>.yaml from rig-phantom11.yaml (P10's), nominal focal / baseline from the
             robot's live OKVIS config (focal at the rig's width; baseline |inv(T_SC0) T_SC1 translation|)
  live cfg   robot.env's OKVIS_CALIBRATION (/etc/okvis/X = /etc/phantomos/okvis/X on the host); when it names none, the
             operator picks one of the robot's OKVIS configs (newest first) and the profile keeps it as live_config
  kit        a Python with numpy, scipy, OpenCV (cv2.aruco), PyYAML; the robot's own dma_ipc in /root/calib-kit/pydeps;
             the fixed dma_motor_motion (md5 FIXED_MOTOR_TOOL_MD5); /data/calib. A usable Python or tool already on the
             robot is kept; a missing one is copied from the reference robot's kit (P10's profile) by rsync run ON the
             reference robot (ssh -A), to the target's IP
  profile    config/profiles.yaml: python, pydeps, root, queues, producer, motor tool, stand, placement time limit,
             live_config, neck_kp

run(host): probe (read-only) -> ONE summary of what it found and every change it would make (each robot write, each
repo file) -> "Y" to apply -> apply -> verify (machine-id, the kit's import test, the motor tool's md5, the YAML files
load) -> read-only checks for the session (loco/policy pods: the operator stops them, never this tool; the EtherCAT
master; dma_cmd_client ping; the OKVIS pod) -> the next step. Nothing to do -> says what exists and asks nothing.

The host-name trap (10 Nov 2026): from phantom10, "phantom11" resolved to phantom10 itself (an old /etc/hosts name),
so a robot-to-robot copy by name wrote to the wrong robot. Here every robot command goes to the IP resolved ON THIS
LAPTOP; the robot at that IP must call itself by the name given (or the name is an IP) and its machine-id is read
before and after every write; the copy from the reference robot goes to that IP, after the reference robot read the
same machine-id there. Never robot-to-robot by host name.

Robot commands go through run_cmd(host, shell_line, timeout=None, agent=False) -> (rc, text) (ssh_run: ssh with
BatchMode, -A when agent), names through resolve(name) -> IP (socket.gethostbyname): tests pass fakes. YAML files are
edited as text (an entry block inserted at the end of its mapping), never re-dumped, so their comments stay. Stdlib +
PyYAML only."""
import argparse
import datetime
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibtool import paths, profile as prof_mod, robots  # noqa: E402
from calibtool.producer import KUBECONFIG  # noqa: E402

FIXED_MOTOR_TOOL_MD5 = "208231a29c88cc2385971a0b73e02925"     # dma_motor_motion with the 'targets' signal
KIT = "/root/calib-kit"
KIT_PYDEPS = f"{KIT}/pydeps"
DATA_ROOT = "/data/calib"
HEAD_CAMERA_JSON = "/etc/phantom/head_camera.json"
ROBOT_ENV = "/etc/phantomos/okvis/robot.env"
HOST_CFG_DIR = "/etc/phantomos/okvis"
CONTAINER_CFG_DIR = "/etc/okvis"
REF_BOT = "P10"
RIG_TEMPLATE = "rig-phantom11.yaml"
PRODUCER = ("dma-video", "producer")
STAND = "{raise_max_m: 0.30, lower_max_m: 0.40, auto_skip: false, steps: true}"
NECK_KP = 60
PICK_TRIES = 3
BASE_IMPORT = "import numpy, scipy, cv2, yaml"
KIT_IMPORT = "import numpy, scipy, cv2, yaml, dma_ipc"
# = preflight.POLICY_PATTERNS (kept here: preflight needs numpy, this file does not)
POLICY_PATTERNS = ("ghost-wbc", "omni-wbc", "locomotion", "loco-", "walker", "policy", "sonic", "foundation-vla")
ROBOT_SSH_OPTS = "-o BatchMode=yes -o StrictHostKeyChecking=accept-new"


class OnboardError(RuntimeError):
    """stop: what the operator reads"""


class Blocked(OnboardError):
    """a problem the operator fixes first; nothing changed"""


class Stopped(OnboardError):
    """the operator stopped it; nothing changed"""


# ---------------------------------------------------------------- runners

def ssh_run(host, cmd, timeout=None, agent=False):
    """run one shell line on host over ssh; (rc, stdout + stderr)"""
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "LogLevel=ERROR",       # 10 Nov: "Permanently added ... to known hosts" glued onto the output
            *(["-A"] if agent else []), host, cmd]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout or 60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, f"{type(e).__name__}: {e}"
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _today():
    d = datetime.date.today()
    return f"{d.day} {d.strftime('%b %Y')}"


# ---------------------------------------------------------------- pure helpers

def import_test(python, pydeps=None, dma=False) -> str:
    """the shell line that imports what calibd needs (with dma_ipc from pydeps when dma)"""
    return (f"PYTHONPATH={pydeps} " if pydeps else "") + \
        f"{python} -c '{KIT_IMPORT if dma else BASE_IMPORT}; cv2.aruco.ArucoDetector'"


def env_calibration(env_text):
    """the host path of robot.env's OKVIS_CALIBRATION (/etc/okvis/X -> /etc/phantomos/okvis/X); None when robot.env
    is missing or names none (absent, commented out, empty)"""
    val = None
    for ln in (env_text or "").splitlines():
        s = ln.strip()
        if s.startswith("#"):
            continue
        m = re.match(r"(?:export\s+)?OKVIS_CALIBRATION\s*=\s*(.*)$", s)
        if m:
            q = re.match(r'"([^"]*)"|\'([^\']*)\'|([^\s#]*)', m.group(1))
            val = next((g for g in q.groups() if g is not None), "").strip() or None
    if val and val.startswith(CONTAINER_CFG_DIR + "/"):
        val = HOST_CFG_DIR + val[len(CONTAINER_CFG_DIR):]
    return val


@dataclass
class Cams:
    left: int
    right: int
    left_socket: object
    right_socket: object
    mxid: object


def head_cameras(text) -> Cams:
    """left = ego_left's queue, right = ego_right's, from head_camera.json"""
    j = json.loads(text)
    cams = j.get("cameras") or {}
    for k in ("ego_left", "ego_right"):
        if k not in cams or cams[k].get("queue_id") is None:
            raise ValueError(f"head_camera.json has no {k} camera with a queue_id")
    left, right = cams["ego_left"], cams["ego_right"]
    board = str(left.get("board"))
    mxid = ((j.get("boards") or {}).get(board) or {}).get("mxid")
    return Cams(int(left["queue_id"]), int(right["queue_id"]), left.get("socket"), right.get("socket"), mxid)


def nominal_from_okvis(text, rig_width=None):
    """(focal px, baseline m, (W, H)) of an OKVIS config's cameras 0 and 1: focal = the mean of their fx, fy (brought
    to rig_width when given), baseline = |inv(T_SC0) T_SC1 translation| = |t1 - t0| (T_SC0's rotation is orthonormal).
    The `%YAML:1.0` file is read as text (PyYAML refuses it): comments dropped, the flow lists by key"""
    body = "\n".join(ln.split("#")[0] for ln in text.splitlines())

    def lists(key, n):
        out = []
        for m in re.finditer(rf"\b{key}\s*:\s*\[([^\]]*)\]", body):
            v = [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
            if len(v) != n:
                raise ValueError(f"{key}: {len(v)} numbers, want {n}")
            out.append(v)
        if len(out) < 2:
            raise ValueError(f"{key}: {len(out)} camera(s), want 2")
        return out[:2]
    T, f, d = lists("T_SC", 16), lists("focal_length", 2), lists("image_dimension", 2)
    if d[0] != d[1]:
        raise ValueError(f"cameras of different sizes {d[0]} / {d[1]}")
    size = (int(d[0][0]), int(d[0][1]))
    focal = sum(f[0] + f[1]) / 4
    if rig_width:
        focal *= rig_width / size[0]
    t0, t1 = (T[0][3], T[0][7], T[0][11]), (T[1][3], T[1][7], T[1][11])
    base = sum((a - b) ** 2 for a, b in zip(t0, t1)) ** 0.5
    return focal, base, size


def insert_entry(text, key, block) -> str:
    """text with block (indented lines) added as the last entry of the top-level mapping `key:`; every other line
    as it was. Column-0 comments just before the next top-level key (or the end) stay after the new entry"""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    start = next((i for i, ln in enumerate(lines) if re.match(rf"{re.escape(key)}\s*:\s*(#.*)?$", ln)), None)
    if start is None:
        raise ValueError(f"no top-level {key!r} mapping")
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"[A-Za-z_]", lines[i])), len(lines))
    while end > start + 1 and (lines[end - 1].startswith("#") or not lines[end - 1].strip()):
        end -= 1
    if not block.endswith("\n"):
        block += "\n"
    return "".join(lines[:end]) + block + "".join(lines[end:])


def _wrap_comment(text, width=118):
    return "".join(f"# {ln}\n" for ln in textwrap.wrap(text, width - 2))


def _sub_once(pattern, repl, text, what):
    new, n = re.subn(pattern, repl, text, count=0, flags=re.M)
    if n != 1:
        raise ValueError(f"rig template: {n} {what} lines (want 1)")
    return new


def rig_text(template, bot, thor, cams, focal, baseline, live_name, size, template_name, template_owner, today):
    """the new rig file: the template's lines with rig_name, okvis_yaml_name, nominal focal / baseline, the camera
    queues and robot.host replaced and a new header; everything else (gates, capture, imu) as in the template"""
    body = re.sub(r"\A(?:#[^\n]*\n)+", "", template)
    low = bot.lower()
    body = _sub_once(r"^rig_name:.*$", f"rig_name: {low}", body, "rig_name")
    body = _sub_once(r"^okvis_yaml_name:.*$", f"okvis_yaml_name: {thor}_{low}_stereo_kb4t-1280x800.yaml", body,
                     "okvis_yaml_name")
    body = _sub_once(r"^(\s+)nominal_focal_px:.*$", rf"\g<1>nominal_focal_px: {round(focal):.1f}", body,
                     "nominal_focal_px")
    body = _sub_once(r"^(\s+)nominal_baseline_m:.*$", rf"\g<1>nominal_baseline_m: {baseline:.4f}", body,
                     "nominal_baseline_m")
    for side, q in (("left", cams.left), ("right", cams.right)):
        body = _sub_once(rf"^(\s+)- \{{record_dir: cam\d+, dma_queue: \d+, side: {side}\}}$",
                         rf"\g<1>- {{record_dir: cam{q}, dma_queue: {q}, side: {side}}}", body, f"kalibr {side}")
    body = _sub_once(r"^(\s+)host:.*$", rf"\g<1>host: root@{thor}", body, "robot host")
    sock = lambda s: f", socket {s}" if s else ""  # noqa: E731
    head = (f"rig-{low}.yaml -- {bot} (Thor {thor}) head: queue {cams.left} = LEFT (ego_left{sock(cams.left_socket)}), "
            f"queue {cams.right} = RIGHT (ego_right{sock(cams.right_socket)})"
            + (f", board {cams.mxid}" if cams.mxid else "") + f". Written {today} by calibtool.onboard from "
            f"{template_name}" + (f" ({template_owner}'s, same head design)" if template_owner else "") +
            f"; nominal focal / baseline from the live config {live_name} ({focal:.1f} px at the rig's width, "
            f"{baseline * 1000:.1f} mm; the config is {size[0]}x{size[1]}).")
    return _wrap_comment(head) + body


def robots_block(bot, thor, old_names, mid, rig, today):
    olds = ", ".join(old_names)
    return (f"  {bot}:\n    thor: {thor}\n    old_names: [{olds}]\n"
            f"    machine_id_prefix: \"{mid[:robots.PREFIX_LEN]}\"     # read on the robot {today} "
            "(calibtool.onboard)\n"
            f"    rig: {rig}\n")


def profile_block(bot, kit, left, right, live_config, live_note, today):
    lines = [f"  {bot}:",
             f"    python: {kit['python']}" + (f"     # {kit['python_note']}" if kit.get("python_note") else ""),
             f"    pydeps: {kit['pydeps']}" + (f"     # {kit['pydeps_note']}" if kit.get("pydeps_note") else ""),
             f"    root: {kit['root']}",
             f"    left: {left}",
             f"    right: {right}",
             f"    producer: {{namespace: {PRODUCER[0]}, deployment: {PRODUCER[1]}}}",
             f"    motor_tool: {kit['motor_tool']}     # the fixed tool (md5 {FIXED_MOTOR_TOOL_MD5[:8]}...)",
             f"    stand: {STAND}   # the chair-shaped stand, as on P10 / P11",
             "    placement_timeout_s: null"]
    if live_config:
        lines.append(f"    # {live_note} ({today}, calibtool.onboard)")
        lines.append(f"    live_config: {live_config}")
    lines.append(f"    neck_kp: {NECK_KP}     # kp only, never kd")
    return "\n".join(lines) + "\n"


def _policy(names):
    return [n for n in names if any(p in n.split("/")[-1] for p in POLICY_PATTERNS)]


# ---------------------------------------------------------------- the run

@dataclass
class Result:
    status: str                     # done | nothing-to-do | dry-run | declined | refused | blocked | stopped | failed
    bot: object = None
    ip: object = None
    live_config: object = None
    changes: list = field(default_factory=list)


@dataclass
class Step:
    where: str                      # robot | repo
    text: str                       # what the operator reads
    do: object                      # callable


class Onboard:
    def __init__(self, host, run_cmd, resolve, ask, out, repo_root, dry_run, bot, old_names, ref, today):
        self.host, self.rc, self.resolve, self.ask, self.out = host, run_cmd, resolve, ask, out
        self.dry_run, self.bot_arg, self.old_names, self.ref_bot = dry_run, bot, list(old_names or ()), ref
        self.today = today or _today()
        root = Path(repo_root) if repo_root else paths.CAL.parents[1]
        self.cal = root / "tools" / "calibration"
        self.robots_yaml = self.cal / "calibtool" / "config" / "robots.yaml"
        self.profiles_yaml = self.cal / "calibtool" / "config" / "profiles.yaml"
        self.rig_dir = self.cal / "config"
        self.notes, self.problems, self.steps = [], [], []
        self.live_config = None

    # ------------------------------------------------ robot access
    def sh(self, cmd, timeout=None, ip=None, agent=False):
        return self.rc(f"root@{ip or self.ip}", cmd, timeout=timeout, agent=agent)

    def ok(self, cmd, **kw):
        return self.sh(cmd, **kw)[0] == 0

    def read(self, path):
        rc, out = self.sh(f"cat {shlex.quote(path)}")
        return out if rc == 0 else None

    def machine_id(self, ip=None):
        rc, out = self.sh("cat /etc/machine-id", ip=ip)
        m = re.search(r"(?m)^\s*([0-9a-f]{32})\s*$", out.lower())
        mid = m.group(1) if m else ""
        if rc != 0 or not mid:
            raise OnboardError(f"could not read /etc/machine-id at {ip or self.ip}: {out.strip()[:120]}")
        return mid

    def verify_target(self, when):
        mid = self.machine_id()
        if mid != self.mid:
            raise OnboardError(f"{when}: the machine-id at {self.ip} is now {mid[:12]}, it was {self.mid[:12]}: "
                               "another machine answers at that address. Stopped.")

    def robot_write(self, cmd, ip=None, agent=False, timeout=None):
        """one robot write, the target's machine-id read before and after it"""
        self.verify_target("before a write")
        rc, out = self.sh(cmd, ip=ip, agent=agent, timeout=timeout)
        self.verify_target("after a write")
        if rc != 0:
            raise OnboardError(f"failed ({rc}): {cmd}\n  {out.strip()[:300]}")

    # ------------------------------------------------ probe
    def identify(self):
        self.ip = self.resolve(self.host)
        self.mid = self.machine_id()
        rc, hn = self.sh("hostname")
        self.hostname = hn.strip().split(".")[0].lower()
        self.reg = robots.load(self.robots_yaml)
        host = str(self.host).split("@")[-1].split(".")[0].lower()
        is_ip = re.fullmatch(r"[\d.]+", str(self.host).split("@")[-1]) is not None
        known = [r for r in self.reg.values() if r.machine_id_prefix and self.mid.startswith(r.machine_id_prefix)]
        who = f" = {known[0].bot} by its machine-id" if known else ""
        if not is_ip and host != self.hostname:
            raise OnboardError(f"{self.host} resolves on this laptop to {self.ip}, but the robot there calls itself "
                               f"{self.hostname} (machine-id {self.mid[:12]}{who}). A stale host name? Nothing done: "
                               f"run onboard with the robot's own name, or fix the name lookup first.")
        self.entry = known[0] if known else None
        if self.entry:
            r = self.entry
            if self.bot_arg and self.bot_arg != r.bot:
                raise OnboardError(f"this robot is {r.bot} in robots.yaml (machine-id {self.mid[:12]}), not "
                                   f"{self.bot_arg}")
            if self.hostname != r.thor:
                raise OnboardError(f"robots.yaml has {r.bot} as Thor {r.thor}, but the robot calls itself "
                                   f"{self.hostname}: update robots.yaml by hand (a rename?), then run this again")
            self.bot, self.thor = r.bot, r.thor
            if self.old_names and tuple(self.old_names) != r.old_names:
                self.notes.append(f"robots.yaml keeps {r.bot}'s old names {list(r.old_names)} (--old-name ignored "
                                  "for a registered robot)")
            return
        m = re.fullmatch(r"phantom(\d+)", self.hostname)
        self.bot = self.bot_arg or (f"P{int(m.group(1)):02d}" if m else None)
        if not self.bot:
            raise OnboardError(f"cannot tell the bot name from the host name {self.hostname!r}: give it with --bot PNN")
        if not re.fullmatch(r"P\d+", self.bot):
            raise OnboardError(f"bot name {self.bot!r}: want P<number>")
        if self.bot in self.reg:
            r = self.reg[self.bot]
            raise OnboardError(f"{self.bot} is already in robots.yaml as Thor {r.thor}, machine-id "
                               f"{r.machine_id_prefix}; this robot's machine-id is {self.mid[:12]}: refusing")
        for r in self.reg.values():
            if self.hostname == r.thor:
                raise OnboardError(f"{self.hostname} is {r.bot}'s host name in robots.yaml, but this machine-id "
                                   f"({self.mid[:12]}) is not {r.bot}'s: refusing")
            if self.hostname in r.old_names:
                self.notes.append(f"{self.hostname} was {r.bot}'s name before; the machine-id tells them apart")
            for o in self.old_names:
                if o == r.thor:
                    raise OnboardError(f"old name {o} is {r.bot}'s current host name: refusing")
        self.thor = self.hostname

    def probe(self):
        self.identify()
        self.profiles_raw = self._profiles_raw()
        self.prof = self.profiles_raw.get(self.bot)
        hj = self.read(HEAD_CAMERA_JSON)
        if hj is None:
            raise OnboardError(f"could not read {HEAD_CAMERA_JSON} on {self.thor}")
        try:
            self.cams = head_cameras(hj)
        except (ValueError, KeyError, TypeError) as e:
            raise OnboardError(f"{HEAD_CAMERA_JSON}: {e}") from None
        if self.prof and (int(self.prof["left"]), int(self.prof["right"])) != (self.cams.left, self.cams.right):
            self.notes.append(f"the profile has queues {self.prof['left']} / {self.prof['right']}, head_camera.json "
                              f"{self.cams.left} / {self.cams.right} (calibd checks left/right in preflight)")
        self.producer_ok = self.ok(f"env KUBECONFIG={KUBECONFIG} k0s kubectl -n {PRODUCER[0]} get deployment "
                                   f"{PRODUCER[1]} -o name")
        self.env_text = self.read(ROBOT_ENV)
        self.env_cal = env_calibration(self.env_text)
        self.probe_rig()
        self.probe_live()
        self.probe_kit()
        self.plan_repo()

    def _profiles_raw(self):
        import yaml
        return (yaml.safe_load(self.profiles_yaml.read_text()) or {}).get("profiles") or {}

    def probe_rig(self):
        if self.entry:
            self.rig = self.entry.rig
            if not self.rig:
                self.rig_write = False
                self.notes.append(f"robots.yaml's {self.bot} names no rig file: add one by hand")
                return
        else:
            self.rig = f"rig-{self.bot.lower()}.yaml"
        self.rig_write = not (self.rig_dir / self.rig).is_file()

    def probe_live(self):
        """the live config: the profile's, robot.env's, or the operator's pick; read when the rig file needs it"""
        if self.prof:
            self.live_config = self.prof.get("live_config") or self.env_cal
            self.live_from = "the profile" if self.prof.get("live_config") else "robot.env"
            if self.prof.get("live_config") and not self.ok(f"test -e {shlex.quote(self.prof['live_config'])}"):
                self.problems.append(f"the profile's live_config {self.prof['live_config']} is not on the robot: "
                                     "fix the profile by hand")
        elif self.env_cal:
            self.live_config, self.live_from = self.env_cal, "robot.env"
        else:
            self.live_config, self.live_from = self.pick_live(), "your pick"
        self.nominal = None
        if self.rig_write:
            if not self.live_config:
                raise OnboardError("no live OKVIS config to take the rig file's nominal focal / baseline from")
            text = self.read(self.live_config)
            if text is None:
                raise OnboardError(f"could not read the live OKVIS config {self.live_config} on {self.thor}")
            tmpl = (self.rig_dir / RIG_TEMPLATE).read_text()
            m = re.search(r"resolution:\s*\[(\d+),", tmpl)
            try:
                self.nominal = nominal_from_okvis(text, int(m.group(1)) if m else None)
            except ValueError as e:
                raise OnboardError(f"{self.live_config}: {e}") from None

    def pick_live(self):
        rc, out = self.sh(f"ls -1t {HOST_CFG_DIR}/*.yaml")
        files = [ln.strip() for ln in out.splitlines() if ln.strip().endswith(".yaml")] if rc == 0 else []
        why = "robot.env is missing" if self.env_text is None else \
            "robot.env names no calibration (OKVIS_CALIBRATION missing or commented out)"
        if not files:
            raise Blocked(f"{why}, and there is no OKVIS config (*.yaml) in {HOST_CFG_DIR} on the robot")
        self.out(f"{why}. The OKVIS configs in {HOST_CFG_DIR} on {self.thor}, newest first:")
        for i, f in enumerate(files, 1):
            self.out(f"  {i:2d}  {f.split('/')[-1]}")
        self.out("The one you pick is the live calibration: a new fit is compared with it, and a rollback goes back "
                 "to it.")
        for _ in range(PICK_TRIES):
            a = str(self.ask("Pick the live calibration by its number: ")).strip()
            if a.isdigit() and 1 <= int(a) <= len(files):
                self.out(f"live calibration: {files[int(a) - 1]}")
                return files[int(a) - 1]
            self.out(f"{a!r} is not one of 1..{len(files)}")
        raise Stopped("no live calibration picked: nothing changed")

    def probe_kit(self):
        """what python / pydeps / motor tool / root the profile has (or will have), and the robot writes that make
        them work"""
        p = self.prof or {}
        self.kit = {"root": str(p.get("root") or DATA_ROOT)}
        self.copy_from_ref = []                 # (source on the reference robot, kit path it lands at)
        ref = None

        def ref_prof():
            nonlocal ref
            if ref is None:
                try:
                    ref = prof_mod.load(self.ref_bot, self.profiles_yaml, self.reg)
                except KeyError as e:
                    raise Blocked(f"the reference robot {self.ref_bot}: {e}") from None
            return ref

        # python
        if p.get("python"):
            py = str(p["python"])
            if not self.ok(import_test(py), timeout=60):
                r = ref_prof()
                venv = str(Path(r.python).parents[1])
                if py == f"{KIT}/{Path(venv).name}/bin/{Path(r.python).name}":
                    self.copy_from_ref.append((venv, py))
                else:
                    self.problems.append(f"the profile's python {py} does not import numpy/scipy/cv2/yaml; fix it "
                                         "by hand")
            self.kit["python"] = py
            self.kit["python_ok"] = not self.copy_from_ref and not self.problems
        else:
            rc, out = self.sh("ls -1d /root/*/venv/bin/python")
            cands = sorted({ln.strip() for ln in out.splitlines() if ln.strip()} if rc == 0 else set(),
                           key=lambda c: (not c.startswith(KIT + "/"), c))
            good = next((c for c in cands if self.ok(import_test(c), timeout=60)), None)
            if good:
                self.kit.update(python=good, python_ok=True, python_note="found on the robot")
            else:
                r = ref_prof()
                venv = str(Path(r.python).parents[1])
                py = f"{KIT}/{Path(venv).name}/bin/{Path(r.python).name}"
                self.copy_from_ref.append((venv, py))
                self.kit.update(python=py, python_ok=False,
                                python_note=f"copied from {r.bot}'s kit ({r.thor}) {self.today}")
        # dma_ipc
        pydeps = str(p.get("pydeps") or KIT_PYDEPS)
        self.kit["pydeps"] = pydeps
        self.dma_src = None
        if not self.ok(f"ls -1d {pydeps}/dma_ipc*"):
            rc, out = self.sh("ls -1d /root/*/pydeps/dma_ipc*")
            dirs = []
            for ln in out.splitlines() if rc == 0 else ():
                d = str(Path(ln.strip()).parent)
                if ln.strip() and d != pydeps and d not in dirs:
                    dirs.append(d)
            if self.kit["python_ok"]:
                dirs = [d for d in dirs if self.ok(import_test(self.kit["python"], d, dma=True), timeout=60)]
            if pydeps != KIT_PYDEPS:
                self.problems.append(f"the profile's pydeps {pydeps} holds no dma_ipc; fix it by hand")
            elif not dirs:
                self.problems.append("no dma_ipc for this robot under /root/*/pydeps" +
                                     (f" that imports with {self.kit['python']}" if self.kit["python_ok"] else "") +
                                     ": it is built per robot (DMA.common); put one there, then run this again")
            else:
                self.dma_src = dirs[0]
                self.kit["pydeps_note"] = f"this robot's own dma_ipc (from {self.dma_src})"
        # motor tool
        tool = p.get("motor_tool")
        if tool:
            md5 = self._md5s(str(tool)).get(str(tool))
            if md5 != FIXED_MOTOR_TOOL_MD5:
                r = ref_prof()
                if str(tool) == f"{KIT}/{Path(r.motor_tool).name}":
                    self.copy_from_ref.append((r.motor_tool, str(tool)))
                else:
                    self.problems.append(f"the profile's motor tool {tool} is not the fixed one (md5 {md5}); fix it "
                                         "by hand")
            self.kit["motor_tool"] = str(tool)
        else:
            md5s = self._md5s("/root/*/dma_motor_motion /usr/bin/dma_motor_motion")
            self.tool_md5s = md5s
            fixed = sorted((t for t, m in md5s.items() if m == FIXED_MOTOR_TOOL_MD5),
                           key=lambda t: (not t.startswith(KIT + "/"), t))
            if fixed:
                self.kit["motor_tool"] = fixed[0]
            else:
                r = ref_prof()
                dst = f"{KIT}/{Path(r.motor_tool).name}"
                self.copy_from_ref.append((r.motor_tool, dst))
                self.kit["motor_tool"] = dst
        self.ref = ref
        # the folders
        self.mkdirs = []
        if self.copy_from_ref or self.dma_src:
            self.mkdirs.append(KIT)
        if self.dma_src:
            self.mkdirs.append(KIT_PYDEPS)
        if not self.ok(f"test -d {shlex.quote(self.kit['root'])}"):
            self.mkdirs.append(self.kit["root"])
        self.plan_robot()

    def _md5s(self, args):
        rc, out = self.sh(f"md5sum {args}")
        got = {}
        for ln in out.splitlines():
            m = re.match(r"([0-9a-f]{32})\s+\*?(\S+)$", ln.strip())
            if m:
                got[m.group(2)] = m.group(1)
        return got

    def plan_robot(self):
        if self.mkdirs:
            self.steps.append(Step("robot", "create " + ", ".join(self.mkdirs),
                                   lambda d=tuple(self.mkdirs): self.robot_write("mkdir -p " + " ".join(d))))
        if self.copy_from_ref:
            r = self.ref
            srcs = " and ".join(s for s, _ in self.copy_from_ref)
            self.steps.append(Step("robot", f"copy {srcs} from {r.bot}'s kit ({r.thor}) to {KIT}/ here: rsync run on "
                                   f"{r.thor} (ssh -A) to {self.ip}, after {r.thor} reads machine-id "
                                   f"{self.mid[:12]} at {self.ip}", self.copy_kit))
        if self.dma_src:
            self.steps.append(Step("robot", f"copy this robot's own dma_ipc from {self.dma_src} into {KIT_PYDEPS}",
                                   lambda: self.robot_write(f"cp -a {self.dma_src}/dma_ipc* {KIT_PYDEPS}/")))

    def plan_repo(self):
        rel = lambda p: p.relative_to(self.cal.parents[1]).as_posix()  # noqa: E731
        if self.rig_write:
            f, b, size = self.nominal
            self.steps.append(Step("repo", f"write {rel(self.rig_dir / self.rig)} (from {RIG_TEMPLATE}: queues "
                                   f"{self.cams.left}/{self.cams.right}, focal {round(f):.1f} px, baseline "
                                   f"{b * 1000:.1f} mm, host root@{self.thor})", self.write_rig))
        if not self.entry:
            self.steps.append(Step("repo", f"add {self.bot} to {rel(self.robots_yaml)} (Thor {self.thor}, old names "
                                   f"{self.old_names or 'none'}, machine-id {self.mid[:12]}, rig {self.rig})",
                                   self.write_robots))
        if not self.prof:
            self.steps.append(Step("repo", f"add {self.bot} to {rel(self.profiles_yaml)} (python {self.kit['python']}, "
                                   f"pydeps {self.kit['pydeps']}, root {self.kit['root']}, queues "
                                   f"{self.cams.left}/{self.cams.right}, motor tool {self.kit['motor_tool']}"
                                   + (f", live_config {self.live_config.split('/')[-1]}"
                                      if self.live_from == "your pick" else "") + f", neck_kp {NECK_KP})",
                                   self.write_profile))

    # ------------------------------------------------ summary
    def summary(self):
        o = self.out
        o("")
        o(f"Onboarding {self.host}: resolved on this laptop to {self.ip}")
        o(f"  robot        {self.bot}, Thor {self.thor}, machine-id {self.mid[:12]} "
          + ("(in robots.yaml)" if self.entry else "(new: not in robots.yaml)"))
        c = self.cams
        o(f"  cameras      left = queue {c.left} (ego_left{', socket ' + str(c.left_socket) if c.left_socket else ''}),"
          f" right = queue {c.right} (ego_right{', socket ' + str(c.right_socket) if c.right_socket else ''})"
          + (f", board {c.mxid}" if c.mxid else "") + "; calibd checks left/right by disparity in preflight")
        o(f"  producer     {PRODUCER[0]}/{PRODUCER[1]} " + ("found" if self.producer_ok else
                                                           "NOT FOUND (calibd needs it)"))
        lc = self.live_config.split("/")[-1] if self.live_config else "none"
        o(f"  live config  {lc} ({'robot.env names none; ' if not self.env_cal else ''}from {self.live_from})"
          if self.live_config else "  live config  none")
        if self.nominal:
            f, b, size = self.nominal
            o(f"  nominal      focal {f:.1f} px (at the rig's 1280 width), baseline {b * 1000:.1f} mm, from "
              f"{lc} ({size[0]}x{size[1]})")
        k = self.kit
        o(f"  python       {k['python']}: " + ("works" if k.get("python_ok") else "not on the robot yet"))
        o(f"  dma_ipc      {k['pydeps']}: " + (f"to copy from {self.dma_src}" if self.dma_src else
                                               ("present" if not any("dma_ipc" in p for p in self.problems)
                                                else "MISSING")))
        tool_ok = not any(dst == k["motor_tool"] for _, dst in self.copy_from_ref)
        o(f"  motor tool   {k['motor_tool']}: " + ("the fixed one" if tool_ok else "to copy (the fixed one, md5 "
                                                    f"{FIXED_MOTOR_TOOL_MD5[:8]}...)"))
        for t, m in sorted(getattr(self, "tool_md5s", {}).items()):
            if m != FIXED_MOTOR_TOOL_MD5:
                o(f"               ({t} is another build, md5 {m[:8]}...: not used)")
        o(f"  rig file     {self.rig or 'none'}" + (": to write" if self.rig_write else " (exists, kept)"))
        o(f"  profile      {self.bot} " + ("exists in profiles.yaml (kept)" if self.prof else
                                            "not in profiles.yaml yet"))
        for n in self.notes:
            o(f"  note         {n}")
        for p in self.problems:
            o(f"  PROBLEM      {p}")
        if self.steps:
            rob = [s for s in self.steps if s.where == "robot"]
            rep = [s for s in self.steps if s.where == "repo"]
            o("")
            n = 0
            if rob:
                o(f"It will change, on the robot (machine-id {self.mid[:12]} at {self.ip} read before and after each "
                  "write):")
                for s in rob:
                    n += 1
                    o(f"  {n}. {s.text}")
            if rep:
                o("and in this repo:" if rob else "It will change, in this repo:")
                for s in rep:
                    n += 1
                    o(f"  {n}. {s.text}")
            o("then check: the machine-id again, the kit's import test "
              f"({KIT_IMPORT.replace('import ', '')} with {k['python']}), the motor tool's md5, {k['root']}, and that "
              "the YAML files load.")

    # ------------------------------------------------ apply
    def check_ref(self):
        """the reference robot is the one robots.yaml names (its machine-id at its IP), and it reaches THIS machine at
        self.ip (it reads the same machine-id there): before any write"""
        r = self.ref
        ref_ip = self.resolve(r.thor)
        if ref_ip == self.ip:
            raise OnboardError(f"{r.thor} resolves to {self.ip}, the robot being onboarded: stopped")
        ref_mid = self.machine_id(ref_ip)
        if not (r.machine_id_prefix and ref_mid.startswith(r.machine_id_prefix)):
            raise OnboardError(f"{r.thor} ({ref_ip}) has machine-id {ref_mid[:12]}, robots.yaml has "
                               f"{r.machine_id_prefix} for {r.bot}: not the reference robot. Stopped, nothing copied.")
        self.ref_ip = ref_ip
        self.ref_sees_target()
        self.out(f"ok  {r.bot} = {r.thor} ({ref_ip}, machine-id {ref_mid[:12]}) reaches machine-id {self.mid[:12]} at "
                 f"{self.ip}")

    def ref_sees_target(self):
        r, ref_ip = self.ref, self.ref_ip
        rc, out = self.sh(f"ssh {ROBOT_SSH_OPTS} -o ConnectTimeout=10 root@{self.ip} cat /etc/machine-id",
                          ip=ref_ip, agent=True, timeout=60)
        seen = out.strip().lower()
        if rc != 0 or seen != self.mid:
            raise OnboardError(f"from {r.thor}, {self.ip} answers with machine-id "
                               f"{seen[:12] if rc == 0 else '(none: ' + seen[:80] + ')'}, not {self.mid[:12]}: the "
                               "reference robot would copy to another machine. Stopped, nothing copied.")

    def copy_kit(self):
        self.verify_target("before the copy")
        self.ref_sees_target()
        srcs = " ".join(s for s, _ in self.copy_from_ref)
        self.robot_write(f"rsync -a -e 'ssh {ROBOT_SSH_OPTS}' {srcs} root@{self.ip}:{KIT}/", ip=self.ref_ip,
                         agent=True, timeout=900)

    def verify_kit(self):
        k = self.kit
        rc, out = self.sh(import_test(k["python"], k["pydeps"], dma=True), timeout=120)
        if rc != 0:
            raise OnboardError(f"the kit's import test failed: {import_test(k['python'], k['pydeps'], dma=True)}\n  "
                               f"{out.strip()[-300:]}")
        self.out(f"ok  kit: {k['python']} imports numpy, scipy, cv2 (aruco), yaml and dma_ipc from {k['pydeps']}")
        md5 = self._md5s(k["motor_tool"]).get(k["motor_tool"])
        if md5 != FIXED_MOTOR_TOOL_MD5:
            raise OnboardError(f"{k['motor_tool']} has md5 {md5}, not the fixed tool's {FIXED_MOTOR_TOOL_MD5}")
        self.out(f"ok  motor tool: {k['motor_tool']} md5 {md5[:8]}... (the fixed one)")
        if not self.ok(f"test -d {shlex.quote(k['root'])}"):
            raise OnboardError(f"{k['root']} is not on the robot")
        self.out(f"ok  {k['root']} exists")

    def _write_checked(self, path, text, check):
        tmp = path.with_name(path.name + ".onboard-tmp")
        tmp.write_text(text)
        try:
            check(tmp)
        except Exception as e:
            tmp.unlink()
            raise OnboardError(f"{path.name}: the new text does not check out ({type(e).__name__}: {e}); left as "
                               "it was") from None
        os.replace(tmp, path)

    def write_rig(self):
        import yaml
        f, b, size = self.nominal
        owner = next((r.bot for r in self.reg.values() if r.rig == RIG_TEMPLATE), None)
        text = rig_text((self.rig_dir / RIG_TEMPLATE).read_text(), self.bot, self.thor, self.cams, f, b,
                        self.live_config.split("/")[-1], size, RIG_TEMPLATE, owner, self.today)

        def check(p):
            y = yaml.safe_load(p.read_text())
            assert y["rig_name"] == self.bot.lower() and y["robot"]["host"] == f"root@{self.thor}"
            assert [k["dma_queue"] for k in y["cameras"]["kalibr"]] == [self.cams.left, self.cams.right]
        self._write_checked(self.rig_dir / self.rig, text, check)

    def write_robots(self):
        text = insert_entry(self.robots_yaml.read_text(), "robots",
                            robots_block(self.bot, self.thor, self.old_names, self.mid, self.rig, self.today))

        def check(p):
            r = robots.load(p)[self.bot]
            assert (r.thor, r.old_names, r.machine_id_prefix, r.rig) == (
                self.thor, tuple(self.old_names), self.mid[:robots.PREFIX_LEN], self.rig)
            assert robots.identify(self.thor, self.mid, robots.load(p)).robot.bot == self.bot
        self._write_checked(self.robots_yaml, text, check)

    def write_profile(self):
        live = self.live_config if self.live_from == "your pick" else None
        text = insert_entry(self.profiles_yaml.read_text(), "profiles",
                            profile_block(self.bot, self.kit, self.cams.left, self.cams.right, live,
                                          "robot.env names no calibration; the live one, picked by the operator",
                                          self.today))

        def check(p):
            pr = prof_mod.load(self.bot, p, robots.load(self.robots_yaml))
            assert (pr.python, pr.pydeps, pr.root, pr.left, pr.right, pr.motor_tool, pr.live_config) == (
                self.kit["python"], self.kit["pydeps"], self.kit["root"], self.cams.left, self.cams.right,
                self.kit["motor_tool"], live)
        self._write_checked(self.profiles_yaml, text, check)

    def apply(self):
        self.verify_target("before the changes")
        if self.copy_from_ref:
            self.check_ref()
        for i, s in enumerate(self.steps, 1):
            if s.where == "repo" and not getattr(self, "_kit_verified", False):
                self.verify_kit()
                self._kit_verified = True
            s.do()
            self.out(f"done {i}. {s.text}")
        if not getattr(self, "_kit_verified", False):
            self.verify_kit()
        self.verify_target("after the changes")
        self.out(f"ok  machine-id at {self.ip} still {self.mid[:12]}")

    # ------------------------------------------------ the end
    def end_checks(self):
        o = self.out
        o("")
        o("Before a session (read-only checks):")
        rc, out = self.sh(f"env KUBECONFIG={KUBECONFIG} k0s kubectl get pods -A --no-headers")
        pods = []
        for ln in out.splitlines() if rc == 0 else ():
            f = ln.split()
            if len(f) >= 4:
                pods.append((f"{f[0]}/{f[1]}", f[3]))
        if rc != 0:
            o(f"  pods         could not list them: {out.strip()[:120]}")
        pol = _policy([n for n, st in pods if st == "Running"])
        if pol:
            o(f"  loco/policy  RUNNING: {', '.join(pol)}. Loco is operator-only: stop it yourself before the "
              "session (this tool never stops it).")
        elif rc == 0:
            o("  loco/policy  no loco/policy pod running")
        rc, out = self.sh("systemctl is-active dma-ethercat")
        o(f"  master       dma-ethercat {out.strip() or f'(rc {rc})'}")
        rc, out = self.sh("dma_cmd_client ping")
        o(f"  ping         dma_cmd_client ping: {out.strip()[:120] or f'(rc {rc})'}")
        ok = [f"{n} {st}" for n, st in pods if "okvis2x" in n]
        o(f"  OKVIS        {'; '.join(ok) if ok else 'no okvis2x pod'}")

    def next_step(self):
        self.out("")
        self.out(f"Next: start a calibration: calib, choose {self.thor}")


def run(host, run_cmd=ssh_run, ask=input, out=print, repo_root=None, resolve=socket.gethostbyname, dry_run=False,
        bot=None, old_names=(), ref=REF_BOT, today=None) -> Result:
    """onboard the robot at host (module text); the Result says what happened"""
    ob = Onboard(host, run_cmd, resolve, ask, out, repo_root, dry_run, bot, old_names, ref, today)
    try:
        ob.probe()
    except (Blocked, Stopped) as e:
        out(str(e))
        return Result("blocked" if isinstance(e, Blocked) else "stopped", getattr(ob, "bot", None),
                      getattr(ob, "ip", None), ob.live_config)
    except OnboardError as e:
        out(f"refused: {e}")
        return Result("refused", getattr(ob, "bot", None), getattr(ob, "ip", None))
    res = Result("", ob.bot, ob.ip, ob.live_config, [s.text for s in ob.steps])
    ob.summary()
    if ob.problems:
        out("")
        out("Blocked: fix the PROBLEM line(s) above, then run this again. Nothing changed.")
        res.status = "blocked"
        return res
    if not ob.steps:
        out("")
        out(f"{ob.bot} ({ob.thor}) is already onboarded: robots.yaml, profiles.yaml, its rig file and the kit are all "
            "there. Nothing to change.")
        ob.end_checks()
        ob.next_step()
        res.status = "nothing-to-do"
        return res
    if dry_run:
        out("")
        out("Dry run: nothing changed.")
        ob.end_checks()
        res.status = "dry-run"
        return res
    out("")
    a = str(ask("Type Y to apply these changes (anything else: nothing changes): ")).strip()
    if a.lower() != "y":
        out("Nothing changed.")
        res.status = "declined"
        return res
    try:
        ob.apply()
    except OnboardError as e:
        out(f"FAILED: {e}")
        out("The repo files are written only after the robot side checks out; look at the robot before running "
            "this again.")
        res.status = "failed"
        return res
    out(f"{ob.bot} onboarded.")
    ob.end_checks()
    ob.next_step()
    res.status = "done"
    return res


def main(argv=None, run_cmd=ssh_run, resolve=socket.gethostbyname, repo_root=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m calibtool.onboard", description=__doc__.split("\n")[0])
    ap.add_argument("host", help="the robot's host name (as it calls itself) or its IP")
    ap.add_argument("--dry-run", action="store_true", help="probe and show the plan; change nothing")
    ap.add_argument("--bot", default=None, help="the bot name (P11); default: from the host name phantomNN")
    ap.add_argument("--old-name", action="append", default=[], help="a host name the Thor had before (repeatable)")
    ap.add_argument("--ref", default=REF_BOT, help="the robot whose kit is copied when this one has none "
                    f"(default {REF_BOT})")
    a = ap.parse_args(argv)
    res = run(a.host, run_cmd=run_cmd, ask=lambda p="": input(p), out=print, repo_root=repo_root, resolve=resolve,
              dry_run=a.dry_run, bot=a.bot, old_names=a.old_name, ref=a.ref)
    return 0 if res.status in ("done", "nothing-to-do", "dry-run") else 1


if __name__ == "__main__":
    sys.exit(main())
