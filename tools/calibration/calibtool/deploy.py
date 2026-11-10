"""Deploy a new calibration as one transaction with a rollback (design plan 6.3-6.4, decision D6/D7).

Everything on the robot goes through Remote: an injected runner run(argv, input=None, timeout=None) -> (rc, text)
(shell_runner() on the robot). Remote(dry_run=True) runs the reads and RECORDS every mutating command instead of
running it (Remote.text(): the exact commands, one per line). Commands carrying `--grace-period=0` / `--grace-period 0`
/ `--force` are refused before anything runs, dry run included (RUNBOOK 4a/7: a zero grace kills OKVIS's final BA).

The transaction (Deployer.deploy), steps in this order, each logged in DeployResult.steps:
  drift            the robot's robot.env, live calibration file, head_camera.json and OKVIS image are still what the
                   snapshot saw (someone changed them since -> DeployDrift, nothing written: ask the operator)
  checks           devel PR #930's startup checks on the new config at the deploy size (okvis_startup_checks:
                   distortion_type known with its exact coefficient count; image_dimension = the frames' size), the
                   consumer checks passed in (e.g. box_pick able to read KB4T), an OKVIS restart path known, no other
                   file of the new name in the way -> DeployRefused, nothing written
  backup           robot.env, the live config, head_camera.json, the producer YAML (cleaned, 0 replicas) and the
                   snapshot into the session's deploy folder
  rollback_script  <deploy>/rollback.sh written BEFORE the first change (rollback_script(): robot.env as saved, the
                   producer as saved through the no-overlap cycle, OKVIS restarted by the detected path)
  config           the new config installed in /etc/phantomos/okvis (= /etc/okvis in the pod)
  robot_env        OKVIS_CALIBRATION pointed at it; every other line kept as it was
  producer         producer_apply(size) (the lead wires producer.resize / restore; dry run: the cycle's commands)
  restart          OKVIS restarted by the path detected in the snapshot: `rollout restart ds/okvis2x` (the DaemonSet)
                   or the older robots' `okvis.sh restart`
  verify           robot.env and the config read back; the OKVIS log names the new file's sha256 prefix (the
                   entrypoint's `(<sha12>)`) and, for kb4t, the reader's `KB4T pinhole camera`; a crash loop is
                   reported (crash_loop), never hidden. Deciding to roll back is postcheck's job.
Deployer.rollback: first the robot is compared with what the deploy left (someone changed it since -> RollbackDrift:
ask); then robot.env as saved, producer_restore(), OKVIS restarted, verified (robot.env, the old sha in the log, no
crash loop). A failed rollback raises RollbackFailed with a loud message naming the rollback.sh to run by hand; nothing
is retried. If the rollback target itself failed on today's data, the result says so.

find_dependents: files naming the old calibration (e.g. a VLA RIG_YAML) -- REPORTED only, never changed.
Deploy size (D7): size_options (match the previous default by scaling, recommended, or 1280), config_for_size
(scaling.scale_config, then #930 at that size), check_verdict (the check grid's paired statistics, holds as the
bootstrap clusters: one placement cannot be resampled by placement)."""
import hashlib
import json
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath

from calibtool import decide, scaling
from calibtool.producer import KUBECONFIG

ROBOT_ENV = "/etc/phantomos/okvis/robot.env"
HOST_CFG_DIR = "/etc/phantomos/okvis"
CONTAINER_CFG_DIR = "/etc/okvis"
OKVIS_SH = "/data/okvis-perf/deployment/okvis.sh"
OKVIS_NS = "positronic"
OKVIS_DS = "okvis2x"
OKVIS_LABEL = "app.kubernetes.io/name=okvis2x"
OKVIS_CONTAINER = "okvis2x"
HEAD_CAMERA_JSON = "/etc/phantom/head_camera.json"
DEPENDENT_ROOTS = ("/etc/phantomos", "/etc/phantom", "/etc/okvis", "/opt/configs", "/data/configs")
DEPENDENT_INCLUDE = ("*.yaml", "*.yml", "*.env", "*.json", "*.sh", "*.conf", "*.txt", "*.toml")
NATIVE = (1280, 800)
UP_TIMEOUT_S = 180.0
POLL_S = 2.0
CRASH_RESTARTS = 2
CHECK_MIN_HOLDS = 8
# devel PR #930 (ViParametersReader kNumDistortionCoefficients): distortion_type -> exact coefficient count
COEFFS = {"equidistant": 4, "kb4t": 6, "radialtangential": 4, "plumb_bob": 4, "radialtangential8": 8,
          "plumb_bob8": 8}


class ForbiddenCommand(RuntimeError):
    pass


class RemoteError(RuntimeError):
    pass


class DeployRefused(RuntimeError):
    """a check failed before anything was written"""


class DeployDrift(RuntimeError):
    """the robot changed since the snapshot: nothing written, ask the operator"""


class DeployFailed(RuntimeError):
    """a step after the first write failed: roll back"""


class RollbackDrift(RuntimeError):
    """the robot changed since the deploy: not rolling back by itself, ask the operator"""


class RollbackFailed(RuntimeError):
    """loud: the robot is not back to its previous calibration; nothing is retried"""


def kubectl(*args) -> list:
    return ["env", f"KUBECONFIG={KUBECONFIG}", "k0s", "kubectl", *map(str, args)]


def sha12(text) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()[:12]


def _forbidden(argv):
    for i, a in enumerate(argv):
        a = str(a)
        if a.startswith("--grace-period=0") or a == "--force" or re.search(r"--grace-period[= ]0\b|\s--force\b", a):
            return a
        if a == "--grace-period" and i + 1 < len(argv) and str(argv[i + 1]).strip() == "0":
            return "--grace-period 0"
    return None


def shell_runner(default_timeout=60):
    """the robot's runner: subprocess, stdout + stderr"""
    def run(argv, input=None, timeout=None):
        try:
            r = subprocess.run(list(argv), input=input, capture_output=True, text=True,
                               timeout=timeout or default_timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 127, f"{type(e).__name__}: {e}"
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    return run


class Remote:
    _WRITE = 'umask 022; cat > "$1.calibtool-tmp" && chmod "$2" "$1.calibtool-tmp" && mv -f "$1.calibtool-tmp" "$1"'

    def __init__(self, run, dry_run=False):
        self.run, self.dry_run = run, dry_run
        self.planned = []           # (argv, input) of the mutating commands a dry run did not send
        self.sent = []              # argv of every command sent

    def call(self, argv, input=None, mutating=False, timeout=None):
        argv = [str(a) for a in argv]
        bad = _forbidden(argv)
        if bad:
            raise ForbiddenCommand(f"refused: {bad!r} in {shlex.join(argv)} (never a zero grace period or --force "
                                   "on the robot)")
        if mutating and self.dry_run:
            self.planned.append((argv, input))
            return 0, ""
        self.sent.append(argv)
        return self.run(argv, input=input, timeout=timeout) if input is not None else self.run(argv, timeout=timeout)

    def read(self, path):
        rc, out = self.call(["cat", str(path)])
        return out if rc == 0 else None

    def exists(self, path) -> bool:
        return self.call(["test", "-e", str(path)])[0] == 0

    def mkdir(self, path):
        rc, out = self.call(["mkdir", "-p", str(path)], mutating=True)
        if rc != 0:
            raise RemoteError(f"mkdir {path} failed: {out.strip()[:200]}")

    def write(self, path, text, mode="0644"):
        """atomic: a temp file next to it, chmod, mv"""
        rc, out = self.call(["sh", "-c", self._WRITE, "sh", str(path), mode], input=text, mutating=True)
        if rc != 0:
            raise RemoteError(f"writing {path} failed (rc {rc}): {out.strip()[:200]}")

    def text(self) -> str:
        lines = []
        for argv, inp in self.planned:
            line = shlex.join(argv)
            if inp is not None:
                line += f"  # <<< {len(inp.encode())} bytes on stdin (sha256 {sha12(inp)})"
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------- robot.env, config paths, #930


_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def parse_env(text) -> dict:
    out = {}
    for line in (text or "").splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        v = m.group(2)
        try:
            parts = shlex.split(v, comments=True)
        except ValueError:
            parts = [v.strip()]
        out[m.group(1)] = parts[0] if parts else ""
    return out


def set_env(text, key, value) -> str:
    """robot.env with key=value replaced in place (appended when absent); every other line kept byte for byte"""
    lines = (text or "").split("\n")
    pat = re.compile(rf"^(\s*(?:export\s+)?){re.escape(key)}=")
    hit = False
    for i, ln in enumerate(lines):
        if pat.match(ln) and not ln.lstrip().startswith("#"):
            lines[i] = pat.match(ln).group(1) + f"{key}={value}"
            hit = True
    out = "\n".join(lines)
    if not hit:
        if out and not out.endswith("\n"):
            out += "\n"
        out += f"{key}={value}\n"
    return out


def host_path(container_path) -> str:
    """/etc/okvis/<x> in the pod = /etc/phantomos/okvis/<x> on the host"""
    p = str(container_path or "")
    if p.startswith(CONTAINER_CFG_DIR + "/"):
        return HOST_CFG_DIR + p[len(CONTAINER_CFG_DIR):]
    return p


def container_path(name) -> str:
    return f"{CONTAINER_CFG_DIR}/{name}"


def _camera_items(text):
    """the camera list items of an OKVIS config (comments stripped), as text blocks"""
    code = "\n".join(scaling._code(ln)[0] for ln in (text or "").splitlines())
    m = re.search(r"^cameras\s*:\s*$", code, re.M)
    if not m:
        return []
    rest = code[m.end():]
    end = re.search(r"^\S", rest, re.M)
    body = rest[:end.start()] if end else rest
    return [b for b in re.split(r"^\s*-\s", body, flags=re.M)[1:] if b.strip()]


def okvis_startup_checks(text, frame_size) -> list:
    """the problems that would make OKVIS (devel PR #930) stop at startup with these frames; [] when none"""
    out = []
    items = _camera_items(text)
    if not items:
        return ["no camera blocks under 'cameras:'"]
    W, H = (int(v) for v in frame_size)
    for i, b in enumerate(items):
        model = re.search(r"\bcam_model\s*:\s*([A-Za-z0-9_]+)", b)
        if model and model.group(1) != "pinhole":
            if model.group(1) != "eucm":
                out.append(f"camera {i}: cam_model '{model.group(1)}' is not supported; use pinhole or eucm")
        else:
            t = re.search(r"\bdistortion_type\s*:\s*([A-Za-z0-9_]+)", b)
            c = re.search(r"\bdistortion_coefficients\s*:\s*\[([^\]]*)\]", b)
            typ = t.group(1) if t else None
            n = len([x for x in (c.group(1).split(",") if c else []) if x.strip()])
            if typ not in COEFFS:
                out.append(f"camera {i}: unknown distortion_type '{typ}'; supported: {' '.join(sorted(COEFFS))}")
            elif n != COEFFS[typ]:
                out.append(f"camera {i}: distortion_type {typ} takes exactly {COEFFS[typ]} distortion_coefficients, "
                           f"got {n}")
        d = re.search(r"\bimage_dimension\s*:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", b)
        if not d:
            out.append(f"camera {i}: no image_dimension")
        elif (int(float(d.group(1))), int(float(d.group(2)))) != (W, H):
            out.append(f"camera {i}: frames arrive at {W}x{H} but image_dimension is [{d.group(1)}, {d.group(2)}]")
    return out


# ---------------------------------------------------------------- OKVIS: restart path, pods, log


@dataclass
class RestartPath:
    kind: str                   # "daemonset" | "okvis_sh" | "none"
    argv: list
    detail: str

    def as_dict(self):
        return asdict(self)


def restart_path(remote: Remote, origin="calibtool") -> RestartPath:
    rc, out = remote.call(kubectl("-n", OKVIS_NS, "get", "ds", OKVIS_DS, "-o",
                                  "jsonpath={.status.desiredNumberScheduled}"))
    desired = int(out.strip()) if rc == 0 and out.strip().isdigit() else None
    if desired is not None and desired >= 1:
        return RestartPath("daemonset", kubectl("-n", OKVIS_NS, "rollout", "restart", f"ds/{OKVIS_DS}"),
                           f"DaemonSet {OKVIS_NS}/{OKVIS_DS} scheduled on {desired} node(s)")
    if remote.exists(OKVIS_SH):
        return RestartPath("okvis_sh", ["env", f"OKVIS_LAUNCH_ORIGIN={origin}", "bash", OKVIS_SH, "restart"],
                           f"{OKVIS_SH} (no scheduled {OKVIS_DS} DaemonSet)")
    why = "no DaemonSet" if desired is None else f"DaemonSet {OKVIS_DS} scheduled on 0 nodes"
    return RestartPath("none", [], f"{why} and no {OKVIS_SH}: OKVIS is not run on this robot")


def okvis_image(remote: Remote):
    rc, out = remote.call(kubectl("-n", OKVIS_NS, "get", "ds", OKVIS_DS, "-o",
                                  "jsonpath={.spec.template.spec.containers[0].image}"))
    return out.strip() if rc == 0 and out.strip() else None


def okvis_pods(remote: Remote) -> list:
    rc, out = remote.call(kubectl("get", "pods", "-A", "-l", OKVIS_LABEL, "-o", "json"))
    if rc != 0:
        return []
    pods = []
    for it in json.loads(out).get("items") or []:
        st = it.get("status") or {}
        cs = st.get("containerStatuses") or []
        waiting = [((c.get("state") or {}).get("waiting") or {}).get("reason") for c in cs]
        pods.append({"name": it["metadata"]["name"], "ns": it["metadata"].get("namespace", OKVIS_NS),
                     "phase": st.get("phase"), "restarts": sum(int(c.get("restartCount") or 0) for c in cs),
                     "waiting": next((w for w in waiting if w), None),
                     "ready": bool(cs) and all(c.get("ready") for c in cs)})
    return pods


def okvis_log(remote: Remote, pods=None, tail=600) -> list:
    pods = okvis_pods(remote) if pods is None else pods
    if not pods:
        return []
    p = pods[-1]
    rc, out = remote.call(kubectl("-n", p["ns"], "logs", p["name"], "-c", OKVIS_CONTAINER, f"--tail={tail}"))
    return out.splitlines() if rc == 0 else []


def crash_loop(pods) -> bool:
    return any(p.get("waiting") == "CrashLoopBackOff" or int(p.get("restarts") or 0) >= CRASH_RESTARTS for p in pods)


def log_has(lines, sha, kb4t) -> dict:
    txt = "\n".join(lines)
    return {"log_sha": bool(sha) and f"({sha})" in txt, "kb4t_line": "KB4T pinhole camera" in txt if kb4t else None}


# ---------------------------------------------------------------- dependents


@dataclass
class Dependent:
    path: str
    line: int
    text: str


def find_dependents(remote: Remote, name, roots=DEPENDENT_ROOTS, exclude=(ROBOT_ENV,)) -> list:
    """files naming the calibration `name` (report only; nothing is changed)"""
    if not name:
        return []
    inc = [f"--include={p}" for p in DEPENDENT_INCLUDE]
    rc, out = remote.call(["grep", "-rnIF", "-s", "-m", "3", *inc, "--", str(name), *roots], timeout=60)
    deps = []
    for ln in out.splitlines() if rc in (0, 2) else []:
        parts = ln.split(":", 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[0] not in exclude:
            deps.append(Dependent(parts[0], int(parts[1]), parts[2].strip()[:200]))
    return deps


# ---------------------------------------------------------------- the snapshot


@dataclass
class DeploySnapshot:
    robot_env: object
    calibration: object         # OKVIS_CALIBRATION (the pod's path)
    calibration_host: object
    config_text: object
    dma_cameras: object
    producer_yaml: object
    head_camera_json: object
    okvis_image: object
    restart: dict
    dependents: list = field(default_factory=list)
    taken_at: float = 0.0

    def expected(self) -> list:
        """[(label, kind, path, text)] of what drift() compares"""
        out = [("robot.env", "file", ROBOT_ENV, self.robot_env)]
        if self.calibration_host:
            out.append(("the live calibration", "file", self.calibration_host, self.config_text))
        out += [("head_camera.json", "file", HEAD_CAMERA_JSON, self.head_camera_json),
                ("the OKVIS image", "image", None, self.okvis_image)]
        return out

    def as_dict(self) -> dict:
        return asdict(self)

    def save(self, d):
        p = Path(d) / "deploy_snapshot.json"
        p.write_text(json.dumps(self.as_dict(), indent=1))
        return p

    @classmethod
    def load(cls, d):
        x = json.loads((Path(d) / "deploy_snapshot.json").read_text())
        x["dependents"] = [Dependent(**y) for y in x.get("dependents") or []]
        return cls(**x)


def take_snapshot(remote: Remote, producer_yaml, clock=time.time, origin="calibtool", roots=DEPENDENT_ROOTS):
    env = remote.read(ROBOT_ENV)
    e = parse_env(env)
    cal = e.get("OKVIS_CALIBRATION")
    host = host_path(cal) if cal else None
    text = remote.read(host) if host else None
    deps = find_dependents(remote, PurePosixPath(cal).name, roots, exclude=(ROBOT_ENV, host)) if cal else []
    return DeploySnapshot(env, cal, host, text, e.get("OKVIS_DMA_CAMERAS"), producer_yaml,
                          remote.read(HEAD_CAMERA_JSON), okvis_image(remote),
                          restart_path(remote, origin).as_dict(), deps, float(clock()))


def drift(expected, remote: Remote) -> list:
    """what changed since `expected` was read: one sentence per item"""
    out = []
    for label, kind, path, text in expected:
        now = okvis_image(remote) if kind == "image" else remote.read(path)
        if now != text:
            where = f" ({path})" if path else ""
            if text is None:
                out.append(f"{label}{where} appeared since the tool read it")
            elif now is None:
                out.append(f"{label}{where} is gone since the tool read it")
            else:
                out.append(f"{label}{where} changed since the tool read it" +
                           (f": {text} -> {now}" if kind == "image" else ""))
    return out


# ---------------------------------------------------------------- the producer's no-overlap cycle as commands


def producer_cycle_argv(ns, deploy, replicas, between=()) -> list:
    """the no-overlap restart (producer.py) as commands: scale 0, wait until no pod (fail otherwise), settle,
    `between`, scale back"""
    wait = (f'for i in $(seq 1 60); do [ -z "$(k0s kubectl -n {ns} get pods -l app={deploy} --no-headers '
            f'2>/dev/null)" ] && exit 0; sleep 1; done; echo "the producer pod did not go away: not scaling up"; '
            f'exit 1')
    return [kubectl("-n", ns, "scale", "deploy", deploy, "--replicas=0"),
            ["env", f"KUBECONFIG={KUBECONFIG}", "sh", "-c", wait], ["sleep", "3"], *[list(b) for b in between],
            kubectl("-n", ns, "scale", "deploy", deploy, f"--replicas={int(replicas)}")]


def _producer_spec(yaml_text):
    from calibtool import producer
    return producer.Spec.from_yaml(yaml_text)


def rollback_script(snap: DeploySnapshot, deploy_dir, new_name, producer_ns="dma-video",
                    producer_deploy="producer") -> str:
    q = shlex.quote
    dd = str(deploy_dir)
    old = PurePosixPath(snap.calibration or "?").name
    L = ["#!/bin/bash",
         f"# calibtool rollback: from {new_name} back to {old} (the calibration live before the session).",
         "# Order: robot.env as saved, the camera producer as saved (no two producer pods at once), OKVIS restarted",
         f"# by the path detected in preflight ({snap.restart.get('kind')}).",
         "set -euo pipefail",
         f"export KUBECONFIG={KUBECONFIG}",
         f'echo "ROLLBACK to {old}"',
         f"cp -a {q(dd + '/robot.env.before')} {q(ROBOT_ENV)}"]
    if snap.producer_yaml:
        n = _producer_spec(snap.producer_yaml).replicas
        K = f"k0s kubectl -n {q(producer_ns)}"
        pods = f"$({K} get pods -l app={q(producer_deploy)} --no-headers 2>/dev/null)"
        L += [f"{K} scale deploy {q(producer_deploy)} --replicas=0",
              f'for i in $(seq 1 60); do [ -z "{pods}" ] && break; sleep 1; done',
              f'if [ -n "{pods}" ]; then echo "ROLLBACK STOPPED: the producer pod did not go away (not starting a '
              f'second one)"; exit 1; fi',
              "sleep 3",
              f"{K} replace -f {q(dd + '/producer.before.yaml')}",
              f"{K} scale deploy {q(producer_deploy)} --replicas={n}",
              f'for i in $(seq 1 90); do {K} get pods -l app={q(producer_deploy)} --no-headers 2>/dev/null | '
              f'grep -q "1/1 *Running" && break; sleep 1; done']
    kind = snap.restart.get("kind")
    if kind == "daemonset":
        L.append(f"k0s kubectl -n {OKVIS_NS} rollout restart ds/{OKVIS_DS}")
    elif kind == "okvis_sh":
        L.append(f"OKVIS_LAUNCH_ORIGIN={q('calibtool rollback to ' + old)} bash {OKVIS_SH} restart")
    else:
        L.append(f'echo "no OKVIS restart path on this robot ({snap.restart.get("detail")})"')
    L.append(f'echo "rolled back: OKVIS_CALIBRATION={snap.calibration}; the OKVIS log should name ({sha12(snap.config_text)})"')
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- the transaction


@dataclass
class DeployPlan:
    name: str                   # file name in /etc/phantomos/okvis
    config_text: str
    size: tuple                 # (W, H) the producer runs at after the deploy


@dataclass
class DeployResult:
    ok: bool
    steps: list
    sha12: str
    rollback_script: str
    verify: dict = field(default_factory=dict)
    crash_loop: bool = False
    dry_run: bool = False
    dependents: list = field(default_factory=list)
    detail: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class RollbackResult:
    ok: bool
    steps: list
    verify: dict
    notes: list

    def as_dict(self) -> dict:
        return asdict(self)


class Deployer:
    """remote: Remote; snap: DeploySnapshot (preflight); deploy_dir: the session's deploy folder ON THE ROBOT;
    producer_apply(size) / producer_restore(): the producer at the deploy size / as saved (producer.resize /
    producer.restore with a writable Kube); checks: callables () -> (ok, detail) (consumer compatibility, identity,
    autonomy not armed)"""

    def __init__(self, remote: Remote, snap: DeploySnapshot, deploy_dir, producer_apply=None, producer_restore=None,
                 clock=time.monotonic, sleep=time.sleep, session="", checks=(), producer_ns="dma-video",
                 producer_deploy="producer", up_timeout_s=UP_TIMEOUT_S):
        self.remote, self.snap, self.deploy_dir = remote, snap, str(deploy_dir)
        self.producer_apply, self.producer_restore = producer_apply, producer_restore
        self.clock, self.sleep, self.session, self.checks = clock, sleep, session, list(checks)
        self.producer_ns, self.producer_deploy, self.up_timeout_s = producer_ns, producer_deploy, up_timeout_s
        self.left = None            # what the deploy left: drift() input for a rollback
        self.plan = None

    @property
    def rollback_path(self):
        return f"{self.deploy_dir}/rollback.sh"

    def _restart(self):
        argv = self.snap.restart.get("argv") or []
        if not argv:
            raise DeployFailed(f"no OKVIS restart path: {self.snap.restart.get('detail')}")
        rc, out = self.remote.call(argv, mutating=True, timeout=120)
        if rc != 0:
            raise DeployFailed(f"OKVIS restart failed (rc {rc}): {out.strip()[:300]}")

    def _wait_okvis(self, sha, kb4t):
        t0 = self.clock()
        while True:
            pods = okvis_pods(self.remote)
            crash = crash_loop(pods)
            lines = okvis_log(self.remote, pods)
            seen = log_has(lines, sha, kb4t)
            if crash or (seen["log_sha"] and seen["kb4t_line"] is not False):
                break
            if self.clock() - t0 >= self.up_timeout_s:
                break
            self.sleep(POLL_S)
        return {**seen, "crash_loop": crash, "pods": pods, "log_tail": lines[-20:]}

    def deploy(self, plan: DeployPlan) -> DeployResult:
        steps = []
        new_host = f"{HOST_CFG_DIR}/{plan.name}"
        sha = sha12(plan.config_text)

        def step(name, **kw):
            steps.append({"step": name, "t": self.clock(), **kw})

        d = drift(self.snap.expected(), self.remote)
        step("drift", changes=d)
        if d:
            raise DeployDrift("the robot changed since the tool read it; not deploying, ask the operator: "
                              + "; ".join(d))
        problems = [f"OKVIS would stop at startup ({plan.size[0]}x{plan.size[1]} frames): {p}"
                    for p in okvis_startup_checks(plan.config_text, plan.size)]
        for c in self.checks:
            ok, detail = c()
            if not ok:
                problems.append(detail)
        if self.snap.restart.get("kind") == "none":
            problems.append(f"no OKVIS restart path: {self.snap.restart.get('detail')}")
        cur = self.remote.read(new_host)
        if cur is not None and cur != plan.config_text:
            problems.append(f"{new_host} already exists with other contents: not overwriting it; pick another name")
        step("checks", problems=problems)
        if problems:
            raise DeployRefused("not deploying: " + "; ".join(problems))
        old = PurePosixPath(self.snap.calibration or "none").name
        try:
            self.remote.mkdir(self.deploy_dir)
            self.remote.write(f"{self.deploy_dir}/robot.env.before", self.snap.robot_env or "")
            if self.snap.config_text is not None:
                self.remote.write(f"{self.deploy_dir}/{old}.before", self.snap.config_text)
            if self.snap.head_camera_json is not None:
                self.remote.write(f"{self.deploy_dir}/head_camera.json.before", self.snap.head_camera_json)
            if self.snap.producer_yaml:
                self.remote.write(f"{self.deploy_dir}/producer.before.yaml",
                                  _producer_spec(self.snap.producer_yaml).clean_yaml(replicas=0))
            self.remote.write(f"{self.deploy_dir}/deploy_snapshot.json", json.dumps(self.snap.as_dict(), indent=1))
            step("backup", dir=self.deploy_dir)
            self.remote.write(self.rollback_path, rollback_script(self.snap, self.deploy_dir, plan.name,
                                                                  self.producer_ns, self.producer_deploy), "0755")
            step("rollback_script", path=self.rollback_path)
            if cur is None:
                self.remote.write(new_host, plan.config_text)
            step("config", path=new_host, sha12=sha)
            env = set_env(self.snap.robot_env or "", "OKVIS_CALIBRATION", container_path(plan.name))
            self.remote.write(ROBOT_ENV, env)
            step("robot_env", calibration=container_path(plan.name))
            self.left = [("robot.env", "file", ROBOT_ENV, env), ("the new calibration", "file", new_host,
                                                                 plan.config_text),
                         ("head_camera.json", "file", HEAD_CAMERA_JSON, self.snap.head_camera_json),
                         ("the OKVIS image", "image", None, self.snap.okvis_image)]
            self.plan = plan
            if self.remote.dry_run:
                if self.snap.producer_yaml:
                    from calibtool import producer
                    spec = _producer_spec(self.snap.producer_yaml)
                    args = producer.set_args(spec.args, width=plan.size[0], height=plan.size[1])
                    patch = json.dumps([{"op": "replace", "path": "/spec/template/spec/containers/0/args",
                                         "value": args}])
                    for a in producer_cycle_argv(self.producer_ns, self.producer_deploy, spec.replicas,
                                                 [kubectl("-n", self.producer_ns, "patch", "deploy",
                                                          self.producer_deploy, "--type", "json", "-p", patch)]):
                        self.remote.call(a, mutating=True)
            elif self.producer_apply is not None:
                self.producer_apply(tuple(plan.size))
            step("producer", size=list(plan.size))
            self._restart()
            step("restart", kind=self.snap.restart.get("kind"))
        except (RemoteError, DeployFailed) as e:
            raise DeployFailed(f"deploy stopped after step '{steps[-1]['step']}': {e}. Roll back "
                               f"(Deployer.rollback or bash {self.rollback_path})") from e
        if self.remote.dry_run:
            return DeployResult(False, steps, sha, self.rollback_path, dry_run=True,
                                dependents=[asdict(x) for x in self.snap.dependents],
                                detail="dry run: nothing sent; the commands are in Remote.text()")
        v = self._wait_okvis(sha, "kb4t" in plan.config_text)
        v["robot_env"] = parse_env(self.remote.read(ROBOT_ENV)).get("OKVIS_CALIBRATION") == container_path(plan.name)
        v["config"] = sha12(self.remote.read(new_host)) == sha
        step("verify", **{k: v[k] for k in ("log_sha", "kb4t_line", "crash_loop", "robot_env", "config")})
        ok = v["robot_env"] and v["config"] and v["log_sha"] and v["kb4t_line"] is not False and not v["crash_loop"]
        why = [] if ok else [k for k in ("robot_env", "config", "log_sha", "kb4t_line") if v[k] is False] + (
            ["crash_loop"] if v["crash_loop"] else [])
        return DeployResult(ok, steps, sha, self.rollback_path, v, v["crash_loop"],
                            dependents=[asdict(x) for x in self.snap.dependents],
                            detail="deployed and verified" if ok else "not verified: " + ", ".join(why))

    def rollback(self, reason, force=False, target_failed_today=False) -> RollbackResult:
        steps, notes = [], []
        expected = self.left or self.snap.expected()
        d = drift(expected, self.remote)
        steps.append({"step": "drift", "changes": d, "reason": reason})
        if d and not force:
            raise RollbackDrift("the robot changed since the deploy; not rolling back by itself, ask the operator: "
                                + "; ".join(d))
        if target_failed_today:
            notes.append(f"the rollback target {self.snap.calibration} FAILED on today's data: rolled back because "
                         f"{reason}, but the previous calibration is not good either")
        where = "robot.env"
        try:
            self.remote.write(ROBOT_ENV, self.snap.robot_env or "")
            steps.append({"step": "robot_env"})
            where = "producer"
            if self.remote.dry_run:
                if self.snap.producer_yaml:
                    spec = _producer_spec(self.snap.producer_yaml)
                    for a in producer_cycle_argv(self.producer_ns, self.producer_deploy, spec.replicas,
                                                 [kubectl("-n", self.producer_ns, "replace", "-f",
                                                          f"{self.deploy_dir}/producer.before.yaml")]):
                        self.remote.call(a, mutating=True)
            elif self.producer_restore is not None:
                self.producer_restore()
            steps.append({"step": "producer"})
            where = "restart"
            self._restart()
            steps.append({"step": "restart"})
            if self.remote.dry_run:
                return RollbackResult(False, steps, {}, notes + ["dry run: nothing sent"])
            where = "verify"
            v = self._wait_okvis(sha12(self.snap.config_text) if self.snap.config_text else None,
                                 "kb4t" in (self.snap.config_text or ""))
            v["robot_env"] = self.remote.read(ROBOT_ENV) == self.snap.robot_env
            steps.append({"step": "verify", **{k: v[k] for k in ("log_sha", "crash_loop", "robot_env")}})
            bad = [k for k in ("robot_env",) if not v[k]] + (["crash_loop"] if v["crash_loop"] else []) + (
                ["log_sha"] if self.snap.config_text and not v["log_sha"] else [])
            if bad:
                raise RollbackFailed(f"verification failed: {', '.join(bad)}")
        except Exception as e:  # noqa: BLE001  (every failure here is the loud one)
            raise RollbackFailed(
                f"ROLLBACK FAILED at {where}: {e}. The robot is NOT back to its previous calibration "
                f"({self.snap.calibration}). STOP; nothing is retried. By hand: bash {self.rollback_path}, then check "
                f"the OKVIS log names ({sha12(self.snap.config_text)}).") from e
        self.left = None
        return RollbackResult(True, steps, v, notes)


# ---------------------------------------------------------------- deploy size (D7) and the check grid


@dataclass
class SizeOption:
    key: str
    size: tuple
    scale: float
    label: str
    recommended: bool


def size_options(prev, native=NATIVE) -> list:
    """the choices at the end: match the previous default by scaling (recommended) or the native size"""
    pw, ph = int(prev[0]), int(prev[1])
    nat = f"{native[0]}x{native[1]}"
    if (pw, ph) == tuple(native):
        return [SizeOption("1", tuple(native), 1.0, f"{nat}, the previous default (no scaling)", True)]
    try:
        s = scaling.factor((pw, ph), native)
    except ValueError:
        return [SizeOption("1", tuple(native), 1.0, f"{nat} (the previous default {pw}x{ph} is not a scaling of "
                           f"it: every consumer must be set to {nat})", True)]
    return [SizeOption("1", (pw, ph), s, f"{pw}x{ph}, the previous default: the fit scaled by {s:g} (recommended)",
                       True),
            SizeOption("2", tuple(native), 1.0, f"{nat}: the producer and every camera consumer at {nat}", False)]


def config_for_size(text, size, native=NATIVE) -> str:
    """the fitted (native-size) config for frames of `size`; ValueError when OKVIS (#930) would refuse it there"""
    s = scaling.factor(tuple(size), native)
    out = text if s == 1.0 else scaling.scale_config(text, s)
    problems = okvis_startup_checks(out, size)
    if problems:
        raise ValueError(f"the config at {size[0]}x{size[1]} would stop OKVIS: " + "; ".join(problems))
    return out


@dataclass
class CheckVerdict:
    decision: str               # DEPLOY | KEEP_LIVE | NOT_ENOUGH_DATA
    reasons: list
    paired: object = None

    def as_dict(self) -> dict:
        return {"decision": self.decision, "reasons": list(self.reasons),
                "paired": self.paired.as_dict() if self.paired is not None else None}


def _f(s, unit):
    return (f"{s.new:.3f} vs live {s.live:.3f} {unit}, difference {s.diff:+.3f} (95 % {s.ci[0]:+.3f} .. "
            f"{s.ci[1]:+.3f})")


def check_verdict(new: decide.Scores, live: decide.Scores, gates=None) -> CheckVerdict:
    """the check grid at the deploy size (design 6.3): new vs live on the same pairs with 6.1's paired statistics, the
    bootstrap clustered by HOLD (the check grid is one placement). DEPLOY only when new is shown better on median
    |depth error| and on rows and no judged band shows it worse; otherwise keep live"""
    g = gates or decide.Gates(min_placements=CHECK_MIN_HOLDS)
    nc = replace(new, placement=new.hold.astype(str))
    lc = replace(live, placement=live.hold.astype(str))
    pr = decide.paired(nc, lc, g)
    depth, rows = pr.stat("depth"), pr.stat("rows")
    if not (depth.judged and rows.judged):
        return CheckVerdict("NOT_ENOUGH_DATA", [f"check grid not comparable: {depth.why or rows.why}"], pr)
    reasons, keep = [], False
    for s, unit, name, margin in ((depth, "%", "median |depth error|", g.depth_margin_pct),
                                  (rows, "px", "row error", g.rows_margin_px)):
        if s.ci[1] < -margin:
            reasons.append(f"at the deploy size, new better on {name}: " + _f(s, unit))
        else:
            keep = True
            reasons.append(f"at the deploy size, new not shown better on {name}: " + _f(s, unit))
    for s in pr.stats:
        if s.name.startswith("band ") and s.judged and s.ci[0] > g.band_worse_margin_pct:
            keep = True
            reasons.append(f"at the deploy size, new worse in {s.name} m: " + _f(s, "%"))
    return CheckVerdict("KEEP_LIVE" if keep else "DEPLOY", reasons, pr)
