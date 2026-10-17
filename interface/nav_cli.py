#!/usr/bin/env python3
"""nav-cli: the one command surface for the OKVIS node, its services and its apps (#610).

Library AND command line. An agent, a shell or a laptop runs it:

    nav-cli status                      nav-cli --robot phantom21 status
    nav-cli autonomy on|off             nav-cli mode explore|semantic|patrol|follow
    nav-cli verb reset-tracking|clear-goal
    nav-cli service list|start|stop|status [NAME]  nav-cli app list|start|stop|status [NAME]
    nav-cli follow prompt [TEXT] | select X Y | off
    nav-cli semantic TEXT [--top N] [--save DIR]
    nav-cli verbose on|off              nav-cli --schema
    nav-cli pose | frame cam0|cam1|follow|routegrid [--out P] | depth [--sectors N] [--out P.pgm]
    nav-cli grid [--radius M] [--ascii] | look X Y [Z] [--hold S] | look off
    nav-cli goto X Y [YAW_DEG] | semantic TEXT --go 0      # pose goals; the robot moves only while armed

Every command prints one JSON object. Exit 0 = done/accepted, 1 = the node refused,
2 = transport (nav API or ring missing, node not fresh, service did not come up, timeout).

COMMANDS GO THROUGH THE NAV API (interface/nav_api.py, 127.0.0.1:8897; env NAV_API_URL): it is the
one writer of every operator-command ring (/nav_verbs, /nav_goal, /nav_pose_goal, /nav_look,
/follow_cmd, /goto_object_cmd), so this file writes no ring and no command file
(nav-commands-to-dma-rings plan, 2026-10-17). What it still does itself is READ, and only through
DMA.common's dma_ipc (SOF-1559: no hand-written mmap): /nav_status and /route_grid
(interface/nav_rings.py, strict readers), the depth ring (interface/okvis_frames.py),
/goto_object_state. Without the binding a read raises RuntimeError naming why, and the command line
answers {"ok": false, "err": <why>} with exit 2.

The service/app start-stop helpers (pidfiles in /dev/shm) are a BENCH tool: on a robot the
perception apps are k0s Deployments started through the AI API server, and the nav API judges an
app running by its state ring (interface/ring_fresh.py), never by a pidfile.

Stdlib only for everything but the semantic query and the ring reads (dma_ipc, which the OKVIS image
ships): it runs inside the OKVIS image, and on a laptop as --robot (which pipes this very file over
ssh into the OKVIS pod, so the version you run is the one in your checkout).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- what the node shares
SHM = Path("/dev/shm")
# The shm queue name of the node's status ring (DMA.common NAV_STATUS_QUEUE_NAME). OKVIS_STATUS_SHM may
# still name it as its /dev/shm path.
STATUS_RING_NAME = "/" + Path(os.environ.get("OKVIS_STATUS_SHM", "/nav_status")).name
VIZ_DIR = Path(os.environ.get("OKVIS_VIZ_DIR", "/dev/shm/okvis-viz"))
VERBOSITY_FILE = Path("/dev/shm/okvis_verbosity")
NODE_FRESH_S = 5.0   # status older than this = the node is gone (the ring outlives it, #201)
NAV_API_URL = os.environ.get("NAV_API_URL", "http://127.0.0.1:8897")


# ---------------------------------------------------------------- the nav API: every command goes here
def api(path: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    """One call to the nav API: POST `body` as JSON (GET when None). Returns its answer plus "http" (the
    status code) and, when not ok, "err" (its reason) and "refused" (a rule or the node said no: 409/410),
    so _emit's exit codes hold: 0 ok, 1 refused, 2 transport (no API, 503, timeout)."""
    url = NAV_API_URL.rstrip("/") + path
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    try:   # inside a span (goto_object's command, a traced tool): the nav API continues its trace
        sys.path.insert(0, str(Path(__file__).resolve().parent) if Path(__file__).is_file() else str(REPO / "interface"))
        import okvis_otel
        headers.update(okvis_otel.headers_for_current())
    except Exception:  # noqa: BLE001 -- piped over ssh (no sibling modules), or no OTel: untraced
        pass
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read()
    except (urllib.error.URLError, OSError) as e:
        return {"ok": False, "http": None, "err": f"nav API at {NAV_API_URL}: {getattr(e, 'reason', e)}"}
    try:
        out = json.loads(raw or b"{}")
    except ValueError:
        out = {"ok": False, "reason": f"not JSON: {raw[:120]!r}"}
    if not isinstance(out, dict):
        out = {"ok": False, "reason": f"not a JSON object: {raw[:120]!r}"}
    out["http"] = code
    if not out.get("ok"):
        out.setdefault("err", out.get("reason") or f"HTTP {code}")
        out["refused"] = code in (400, 404, 409, 410)
    return out


# ---------------------------------------------------------------- the node's rings, through DMA.common
def _interface_import(mod: str):
    """A sibling module of interface/. Piped over ssh by --robot this file is "<stdin>", so interface/ is
    where the image keeps the repo (REPO, below)."""
    d = str(Path(__file__).resolve().parent) if Path(__file__).is_file() else str(REPO / "interface")
    if d not in sys.path:
        sys.path.insert(0, d)
    import importlib
    return importlib.import_module(mod)


_readers: dict = {}


def ring_reader(ring: str, name: str):
    """The strict nav_rings.Reader of `ring` on queue `name`, one per (ring, name): it keeps its queue, consumer
    and header view across polls. RuntimeError (nav_rings.RingError) naming why when this image's dma_ipc
    cannot read it."""
    key = (ring, name)
    r = _readers.get(key)
    if r is None:
        nav_rings = _interface_import("nav_rings")
        r = _readers[key] = nav_rings.Reader(ring, name=name, strict=True)
    return r


# ---------------------------------------------------------------- /nav_status (#205)
# The node publishes its state as ONE NavStatusData at ~100 Hz (DMA.common nav_ipc_types.h, written by
# StatusDMAPublisher.hpp), read here through dma_ipc.NavStatusQueue: no offsets in this file. Reading THIS
# instead of inferring state from log prose is the fix for the 2026-08-31 false alarm. Liveness comes from
# the ring header's last_write_ns (CLOCK_MONOTONIC, comparable here because console and node share the
# kernel).
STATUS_VERSION = 3   # NAV_STATUS_FORMAT_VERSION this reader knows: node, console and recorder ship in one image


def status_dict(m, age_s: float | None) -> dict | None:
    """One NavStatusData as the dict every caller uses; None for another format version."""
    if m is None or int(m.format_version) != STATUS_VERSION:
        return None
    pose_stamp_ns, stamp_ns = int(m.pose_stamp_ns), int(m.stamp_ns)
    p = [float(m.px), float(m.py), float(m.pz)]
    q = [float(m.qx), float(m.qy), float(m.qz), float(m.qw)]
    verb_seq = int(m.last_verb_seq)
    return {
        "age_s": round(age_s, 2) if age_s is not None else None, "seq": int(m.seq), "pid": int(m.pid),
        "pose_age_s": (round((stamp_ns - pose_stamp_ns) / 1e9, 1) if pose_stamp_ns > 0 else None),
        "pos": [round(v, 2) for v in p],
        "fsm": int(m.fsm_state),
        "goal": ([round(float(m.goal_x), 2), round(float(m.goal_y), 2), round(float(m.goal_z), 2)]
                 if int(m.goal_valid) else None),
        "goal_dist": (round(float(m.goal_dist_m), 2) if float(m.goal_dist_m) >= 0 else None),
        "goals_issued": int(m.goals_issued),
        "armed": bool(m.autonomy_armed), "yield": bool(m.pad_yield), "semantic": bool(m.semantic_mode),
        "verb": ({"seq": verb_seq, "ok": bool(m.last_verb_accepted), "reason": str(m.last_verb_reason)}
                 if verb_seq > 0 else None),
        "debug": {"mode": bool(m.debug_mode), "frozen": bool(m.debug_frozen), "label": str(m.debug_label)},
        # The full SLAM pose T_WS, unrounded: the depth cloud is placed with it.
        "pose": ({"stamp_ns": pose_stamp_ns, "p": p, "q": q} if pose_stamp_ns > 0 else None),
    }


def status_read(name: str | None = None) -> dict | None:
    """The newest NavStatusData, as status_dict; None if the ring is absent, never written, or another
    version. RuntimeError naming why when this image's dma_ipc cannot read /nav_status."""
    r = ring_reader("nav_status", name or STATUS_RING_NAME)
    m = r.latest()
    if m is None:
        return None
    age = r.age_s()
    return status_dict(m, 0.0 if age is None else age)


def status_poses(name: str | None = None) -> dict:
    """Every SLAM pose the /nav_status ring still holds, {pose_stamp_ns: {"stamp_ns", "p", "q"}}. The node
    republishes its newest state at every tick, so one stamp appears in several records; any is the same pose."""
    out: dict = {}
    for m in ring_reader("nav_status", name or STATUS_RING_NAME).records():
        if int(m.format_version) == STATUS_VERSION and int(m.pose_stamp_ns) > 0:
            t = int(m.pose_stamp_ns)
            out[t] = {"stamp_ns": t, "p": [float(m.px), float(m.py), float(m.pz)],
                      "q": [float(m.qx), float(m.qy), float(m.qz), float(m.qw)]}
    return out


def node_fresh(st: dict | None = None) -> dict | None:
    """The status if the node wrote it recently, else None. Existence of the ring proves
    nothing (it survives the node); only recency does."""
    st = status_read() if st is None else st
    return st if st and st["age_s"] < NODE_FRESH_S else None


def _wait(pred, timeout: float, period: float = 0.1):
    """Poll `pred` until it returns something truthy or `timeout` passes. Returns (value, waited_s)."""
    t0 = time.monotonic()
    while True:
        v = pred()
        if v:
            return v, round(time.monotonic() - t0, 2)
        if time.monotonic() - t0 >= timeout:
            return None, round(time.monotonic() - t0, 2)
        time.sleep(period)


# ---------------------------------------------------------------- /nav_goal: wire values
# A goal supersedes the previous goal, so this ring is read latest-wins. Its one writer is the nav API
# (operator: GO, STOP, EXPLORE, PATROL); follow_me's FOLLOW / STOP go on /follow_goal, its own ring.
NAV_GO, NAV_STOP, NAV_EXPLORE, NAV_PATROL = 1, 2, 3, 4   # okvis::NavCommand (mirrored by hand, see test)
NAV_FOLLOW = 10                              # lean follow; sent only by app/follow_me/follow_me.py
NAV_MODES = {"semantic": NAV_STOP, "explore": NAV_EXPLORE, "patrol": NAV_PATROL}
NAV_NAMES = {NAV_GO: "GO (operator goal)", NAV_STOP: "SEMANTIC (stop + await a goal)",
             NAV_EXPLORE: "EXPLORE (resume NBV)", NAV_PATROL: "PATROL (straight-line tour)"}
FT_TARGET_VALID, FT_RANGE_VALID, FT_LOST = 1, 2, 4  # okvis::FollowTargetFlag (/follow_target)


# ---------------------------------------------------------------- /nav_verbs (#204)
# Console -> node commands the node is allowed to REFUSE. The answer comes back on /nav_acks
# (kind verb, req_seq = the verb's seq; also /nav_status.last_verb_* for older readers). A verb
# supersedes nothing, so the node drains this ring in order. Its one writer is the nav API.
# WIRE FORMAT, shared with okvis_dma/include/okvis/VerbDMA.hpp: a console and a node from
# different builds share this ring, so a renumber turns one verb into another with nothing
# erroring. The test pins these against the header.
VERB_RESET_TRACKING, VERB_CLEAR_SEMANTIC_GOAL = 1, 2
VERB_ARM, VERB_DISARM, VERB_REC_START, VERB_REC_STOP = 3, 4, 5, 6
VERB_GS_STREAM_ON, VERB_GS_STREAM_OFF = 7, 8
VERB_SEMANTIC_GOAL = 9   # retired: pose goals come on /nav_pose_goal; the node refuses it (append-only)
VERB_NAMES = {VERB_RESET_TRACKING: "RESET_TRACKING", VERB_CLEAR_SEMANTIC_GOAL: "CLEAR_SEMANTIC_GOAL",
              VERB_ARM: "ARM", VERB_DISARM: "DISARM", VERB_REC_START: "REC_START", VERB_REC_STOP: "REC_STOP",
              VERB_GS_STREAM_ON: "GS_STREAM_ON", VERB_GS_STREAM_OFF: "GS_STREAM_OFF",
              VERB_SEMANTIC_GOAL: "SEMANTIC_GOAL"}
CLI_VERBS = {"reset-tracking": VERB_RESET_TRACKING, "clear-goal": VERB_CLEAR_SEMANTIC_GOAL}


# ---------------------------------------------------------------- tap demand
_want_viz_t = 0.0


def want_viz(viz_dir: Path | None = None, when=None) -> None:
    """Keep the app writing its viz taps: touch <tap>/.want_viz, at most once a second.

    dma_live writes rgbd.bin, the camera JPEGs and the map taps only while this marker is under
    10 s old (okvis::viztap::setDemandGated), so a robot nobody watches pays nothing for them.
    Call it every loop from a reader that is ACTIVE; `when` (a callable) is asked only when a
    touch is due, so a reader can make it conditional without paying for the check every loop."""
    global _want_viz_t
    now = time.monotonic()
    if now - _want_viz_t < 1.0:
        return
    _want_viz_t = now
    if when is not None and not when():
        return
    try:
        ((VIZ_DIR if viz_dir is None else viz_dir) / ".want_viz").touch()
    except OSError:
        pass


# ---------------------------------------------------------------- app state, read off the apps' rings
def goto_object_state() -> dict | None:
    """goto_object's newest answer on /goto_object_state (nav_rings.goto_state_dict), or None. Needs
    dma_ipc (interface/nav_rings.py); without it, None."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent) if Path(__file__).is_file() else str(REPO / "interface"))
        import nav_rings
        return nav_rings.goto_state_dict(nav_rings.Reader("goto_object_state").latest())
    except Exception:  # noqa: BLE001 -- no binding, no ring: no state
        return None


# ---------------------------------------------------------------- services and apps
# Two kinds of unit, both background processes that hold a GPU share and are off by default:
#   service  a capability anyone can use: services/<name>/service.json (semantic_memory, llm)
#   app      a script built on services:   app/<name>/app.json         (follow_me, semantic_obstacles)
# The manifest is the whole definition, so a new unit needs a folder, not an edit here:
#   {"run": ["bash", "x.sh"],                       argv; argv[1] is relative to the folder;
#                                                   "{rig}" becomes the node's rig yaml
#    "probe": "http://..." | "ring:/<ring>" | "file:<tap file>" | "shm:<file in /dev/shm>",
#                                                   ready = answers / pushed / written in the last 5 s
#    "up_s": 40,                                    how long a start waits for the probe
#    "requires": "perception" | "llama-server",     what the image must carry
#    "services": [...],  (apps only)                the services it uses; services never use each other
#    "label": "FOLLOW ME",  (apps only)             the console's button for it
#    "outputs": [...],  (apps only)                 its tap files: deleted at its start and stop (_app_reset)
#    "inset": "follow.jpg"}  (apps only)            a tap file shown over the 3D view while the app runs
# Units belong to NOBODY's process: whoever starts one records its pid in /dev/shm, anyone can
# see or stop it, and each runs in its own process group so a stop takes its children.
# Lifecycle: `service start` by hand PINS a service (only `service stop` ends it). `app start`
# starts the app's services unpinned, then the app under a parent that runs `reconcile` when the
# app exits, however it exits. reconcile stops every service no running app or pin still needs.
# Piped over ssh by --robot, this file runs as `python3 -` and __file__ is "<stdin>"; the
# repo is then where the image keeps it.
REPO = (Path(__file__).resolve().parent.parent if Path(__file__).is_file()
        else Path(os.environ.get("OKVIS_REPO", "/home/OKVIS2-X")))
UNIT_DIRS = {"service": REPO / "services", "app": REPO / "app"}
UNIT_FILE = {"service": "service.json", "app": "app.json"}
FOLLOW_RIG = ""
# What an app's parent runs once the app has exited.
RECONCILE_CMD = ("python3 "
                 + shlex.quote(str(Path(__file__).resolve() if Path(__file__).is_file()
                                   else REPO / "interface" / "nav_cli.py")) + " reconcile")


def units(kind: str) -> dict:
    """name -> manifest (with "dir") for every unit of this kind on disk."""
    out = {}
    for f in sorted(UNIT_DIRS[kind].glob(f"*/{UNIT_FILE[kind]}")):
        try:
            out[f.parent.name] = {**json.loads(f.read_text()), "dir": f.parent}
        except (OSError, ValueError):
            continue
    return out


def follow_rig() -> str:
    """The node's rig yaml, which follow_me.py needs: liveviz sets FOLLOW_RIG from its
    --follow-rig; anyone else reads it off the running node's own command line (argv[1] of
    the pid the status ring names). Not an env var: the entrypoint's exports never reach a
    `kubectl exec`, which is how --robot runs."""
    if FOLLOW_RIG:
        return FOLLOW_RIG
    st = status_read()
    try:
        argv = Path(f"/proc/{st['pid']}/cmdline").read_bytes().split(b"\0") if st else []
        return argv[1].decode() if len(argv) > 2 and argv[1].endswith(b".yaml") else ""
    except OSError:
        return ""


def _unit(kind: str, name: str) -> dict | None:
    return units(kind).get(name)


def unit_argv(kind: str, name: str) -> list[str]:
    m = _unit(kind, name)
    argv = [str(m["dir"] / a) if i == 1 else a for i, a in enumerate(m["run"])]
    return [follow_rig() if a == "{rig}" else a for a in argv]


def unit_available(kind: str, name: str) -> bool:
    m = _unit(kind, name)
    if m is None or not (m["dir"] / m["run"][1]).is_file():
        return False
    if "{rig}" in m["run"] and not follow_rig():
        return False
    req = m.get("requires")
    if req == "perception":
        return bool(os.environ.get("PERCEPTION_MODELS"))
    if req == "llama-server":
        import shutil
        return shutil.which("llama-server") is not None
    return True


def _pidfile(kind: str, name: str) -> Path:
    return SHM / f"okvis_{kind}_{name}.pid"


def _pinfile(name: str) -> Path:
    return SHM / f"okvis_service_{name}.pin"


def unit_pid(kind: str, name: str) -> int | None:
    """The unit's pid if it is alive AND still the process we started. /dev/shm outlives the
    pod, so a stale pid file can name an unrelated host process; the cmdline check is what
    keeps a stop from killing it."""
    m = _unit(kind, name)
    try:
        pid = int(_pidfile(kind, name).read_text().strip())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, ValueError):
        return None
    return pid if m is not None and Path(m["run"][1]).name in cmd else None


def unit_running(kind: str, name: str) -> bool:
    return unit_pid(kind, name) is not None


def unit_up(kind: str, name: str) -> bool:
    """Liveness beyond "the pid exists": the thing the unit is FOR is answering."""
    probe = (_unit(kind, name) or {}).get("probe")
    if not probe:
        return unit_running(kind, name)
    if probe.startswith("ring:"):   # a DMA ring pushed in the last 5 s (interface/ring_fresh.py)
        sys.path.insert(0, str(REPO / "interface"))
        import ring_fresh
        return ring_fresh.fresh(probe[len("ring:"):], 5.0, SHM)
    for prefix, base in (("file:", VIZ_DIR), ("shm:", SHM)):   # a file written in the last 5 s
        if probe.startswith(prefix):
            try:
                return time.time() - (base / probe[len(prefix):]).stat().st_mtime < 5.0
            except OSError:
                return False
    try:
        return urllib.request.urlopen(probe, timeout=2).status < 500
    except Exception:
        return False


def _spawn(kind: str, name: str, argv: list[str]) -> int:
    log = open(SHM / f"{name}.log", "ab")
    p = subprocess.Popen(argv, cwd=str(_unit(kind, name)["dir"]), stdin=subprocess.DEVNULL,
                         stdout=log, stderr=log, start_new_session=True)
    log.close()
    _pidfile(kind, name).write_text(f"{p.pid}\n")
    # Between fork and exec the child's cmdline is still ours, so unit_pid would not know it
    # yet; wait for the exec (microseconds, but a 1-in-25 test flake without this).
    _wait(lambda: unit_pid(kind, name) is not None, 2.0, period=0.01)
    return p.pid


def _await_up(kind: str, name: str, pid: int, wait_s: float) -> dict:
    out = {"ok": True, kind: name, "pid": pid, "running": True}
    if wait_s > 0:
        up, waited = _wait(lambda: unit_up(kind, name), wait_s, period=0.5)
        out.update(up=bool(up), waited_s=waited)
        if not up:
            out.update(ok=False, err=f"{name} started (pid {pid}) but did not answer within {wait_s:.0f} s;"
                                     f" see {SHM / (name + '.log')}")
    return out


def _missing(kind: str, name: str) -> dict:
    m = _unit(kind, name)
    if m is None:
        return {"ok": False, "err": f"unknown {kind} {name!r}; known: {', '.join(units(kind))}"}
    why = {"perception": "the perception layer", "llama-server": "llama-server (ML-deps layer, #619)"}
    return {"ok": False, "err": f"{kind} {name} cannot run in this image (needs "
                                + why.get(m.get("requires"), "its script")
                                + (" and the rig" if "{rig}" in m["run"] else "") + ")"}


def service_start(name: str, wait_s: float = 0.0, pin: bool = True) -> dict:
    """Start a service. By hand (pin) it stays on until `service stop`."""
    if not unit_available("service", name):
        return _missing("service", name)
    if pin:
        _pinfile(name).touch()
    pid = unit_pid("service", name) or _spawn("service", name, unit_argv("service", name))
    return _await_up("service", name, pid, wait_s)


def _kill(kind: str, name: str) -> None:
    pid = unit_pid(kind, name)
    if pid is not None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        gone, _ = _wait(lambda: not unit_running(kind, name), 10.0)
        if not gone:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.waitpid(pid, 0)   # our own child (it is dead or dying): reap it, or it lingers as a zombie
        except ChildProcessError:
            pass                 # someone else's: the kernel reaps it
    _pidfile(kind, name).unlink(missing_ok=True)


def service_stop(name: str) -> dict:
    """Unpin and stop it, unless a running app still needs it (then it stays, unpinned)."""
    if _unit("service", name) is None:
        return _missing("service", name)
    _pinfile(name).unlink(missing_ok=True)
    if name in needed():
        return {"ok": True, "service": name, "running": unit_running("service", name),
                "kept": "a running app needs it"}
    _kill("service", name)
    reconcile()
    return {"ok": True, "service": name, "running": False}


def _app_reset(name: str, viz_dir: Path | None = None) -> None:
    """Nothing from a previous run, at an app's start and at its stop: the tap files its manifest lists
    as "outputs" are deleted, and the operator inputs it reads go back to blank through the nav API
    (follow_me: out of FOLLOW; semantic_obstacles: no list, which has the app clear its cells).
    Best effort: with no nav API there is nothing to reset."""
    d = VIZ_DIR if viz_dir is None else viz_dir
    for f in _unit("app", name).get("outputs", []):
        (d / f).unlink(missing_ok=True)
    if name == "follow_me":
        api("/follow", {"mode": "off"}, timeout=3.0)
    elif name == "semantic_obstacles":
        api("/follow/obstacles", {"text": ""}, timeout=3.0)


def app_start(name: str, wait_s: float = 0.0, viz_dir: Path | None = None) -> dict:
    """BENCH TOOL. Start the app's services (unpinned), then the app under a parent that reconciles when
    the app exits -- a crash included, which never calls nav-cli itself. An app runs from its start
    until its stop, idle while it has nothing to do. On a robot the apps are k0s Deployments (the AI
    API server starts them); this is for a bench box without one. Entering FOLLOW is the nav API's
    (POST /follow {"mode": "follow"}), once follow_me's /follow_state is live."""
    if not unit_available("app", name):
        return _missing("app", name)
    for s in _unit("app", name).get("services", []):
        r = service_start(s, pin=False)
        if not r["ok"]:
            return r
    pid = unit_pid("app", name)
    if pid is None:
        _app_reset(name, viz_dir)   # nothing from a previous run
        parent = '"$@"; rm -f ' + shlex.quote(str(_pidfile("app", name))) + "; " + RECONCILE_CMD + " >/dev/null 2>&1"
        pid = _spawn("app", name, ["sh", "-c", parent, "sh", *unit_argv("app", name)])
    return _await_up("app", name, pid, wait_s)


def app_stop(name: str, viz_dir: Path | None = None) -> dict:
    """BENCH TOOL. follow_me's stop leaves FOLLOW first (the nav API: STOP if following, target blank).
    Then, as at its start, nothing of the app is left behind (_app_reset)."""
    if _unit("app", name) is None:
        return _missing("app", name)
    if name == "follow_me":
        api("/follow", {"mode": "off"}, timeout=3.0)
    _kill("app", name)
    _app_reset(name, viz_dir)
    reconcile()
    return {"ok": True, "app": name, "running": False}


def needed() -> list[str]:
    """Services something still needs: a running app, or a pin."""
    want = [s for a, m in units("app").items() if unit_running("app", a) for s in m.get("services", [])]
    want += [s for s in units("service") if _pinfile(s).exists()]
    return sorted(set(want))


def reconcile() -> dict:
    """Stop every running service nothing needs. Safe to run at any time, by anyone."""
    keep = needed()
    stopped = [s for s in units("service") if s not in keep and unit_running("service", s)]
    for s in stopped:
        _kill("service", s)
    return {"ok": True, "stopped": stopped, "needed": keep}


def app_started_at(name: str) -> float | None:
    """When the running app started (its pid file is written at spawn), else None."""
    try:
        return _pidfile("app", name).stat().st_mtime if unit_running("app", name) else None
    except OSError:
        return None


def unit_status(kind: str, name: str) -> dict:
    pid = unit_pid(kind, name)
    out = {"available": unit_available(kind, name), "running": pid is not None, "pid": pid,
           "up": unit_up(kind, name) if pid is not None else False}
    if kind == "service":
        out["pinned"] = _pinfile(name).exists()
    return out


def okvis_mode() -> str | None:
    """The running node's OKVIS_MODE: our own env inside the OKVIS pod."""
    return os.environ.get("OKVIS_MODE") or None


# ---------------------------------------------------------------- verbosity
def verbose_get(path: Path | None = None) -> bool:
    """Is the app in verbose mode? Absent/garbage file = no."""
    try:
        return int((path or VERBOSITY_FILE).read_text().strip() or 0) > 0
    except (OSError, ValueError):
        return False


def verbose_set(on: bool, path: Path | None = None) -> dict:
    try:
        (path or VERBOSITY_FILE).write_text(f"{2 if on else 0}\n")
    except OSError as e:
        return {"ok": False, "err": str(e)}
    return {"ok": True, "verbose": on}


# ---------------------------------------------------------------- semantic query, through goto_object
def goto_ask(text: str, top: int = 5, drive: bool = False, frame: int | None = None,
             timeout: float = 30.0) -> dict:
    """POST /goto_object/cmd, then goto_object's answer to it on /goto_object_state (answered_seq ==
    the command's seq), or {"ok": False, "err"}."""
    r = api("/goto_object/cmd", {"text": text, "top": int(top), "drive": bool(drive), "frame": frame})
    if not r.get("ok"):
        return r
    seq = r["seq"]
    st, waited = _wait(lambda: (s := goto_object_state()) and s.get("answered_seq") == seq and s, timeout, 0.1)
    if not st:
        return {"ok": False, "seq": seq, "waited_s": waited,
                "err": f"goto_object did not answer seq {seq} on /goto_object_state within {timeout} s"}
    return {**st, "ok": not st.get("err"), "waited_s": waited}


def semantic_query(text: str, top: int = 5, timeout: float = 30.0) -> dict:
    """Ranked keyframes for `text`, from the goto_object app (which asks semantic_memory): score, the camera
    position at that keyframe, its stamp and the bearing to the matched grid cell. NOT the object's
    position: the index stores no depth, so a hit is a camera pose plus a ray. The thumbnails are
    goto_object's taps, <tap dir>/goto_hit<k>.jpg, where it writes them."""
    st = goto_ask(text, top, False, None, timeout)
    if not st.get("ok") or not st.get("hits"):
        return {"ok": False, "err": st.get("err") or "no hits", **({"seq": st["seq"]} if "seq" in st else {})}
    hits = [{"score": round(h["score"], 4), "frame": h["frame"], "pos": [h["x"], h["y"], h["z"]],
             "bearing": h["bearing_deg"], "stamp_ns": h["stamp_ns"], "image": str(VIZ_DIR / f"goto_hit{k}.jpg")}
            for k, h in enumerate(st["hits"][:top])]
    return {"ok": True, "text": text, "seq": st.get("answered_seq"), "hits": hits}


# ---------------------------------------------------------------- perception reads, for the agent
# What the node already publishes, read back as numbers an LLM can use: the camera taps as
# files, the metric depth ring as nearest-return per bearing sector, the route grid as a map
# around the robot, and the pose with the grid's yaw (the status yaw is the IMU frame, 90 deg
# off the heading). Everything here is read-only.
FRAMES = {"cam0": "rect_cam0.jpg", "cam1": "rect_cam1.jpg", "follow": "follow.jpg",
          "routegrid": "routegrid.jpg", "feat0": "feat_rgb0.jpg", "feat1": "feat_rgb1.jpg"}


def frame_read(name: str, out: str | None = None, viz_dir: Path | None = None) -> dict:
    """Copy one camera tap to `out` (a path, or "-" for base64 inline). The taps are JPEGs the
    node writes tmp+rename, so a read never sees a torn file."""
    if name not in FRAMES:
        return {"ok": False, "err": f"frame must be one of {sorted(FRAMES)}"}
    p = (VIZ_DIR if viz_dir is None else viz_dir) / FRAMES[name]
    try:
        data = p.read_bytes()
        age = time.time() - p.stat().st_mtime
    except OSError as e:
        return {"ok": False, "err": f"{p}: {e}"}
    r = {"ok": True, "frame": name, "bytes": len(data), "age_s": round(age, 2), "source": str(p)}
    if out == "-":
        import base64
        r["jpeg_b64"] = base64.b64encode(data).decode()
    elif out:
        Path(out).write_bytes(data)
        r["path"] = out
    return r


# --- /okvis_depth_{meta,frames}: metric stereo depth (#510), read through DMA.common's dma_ipc
# (okvis_frames.FrameRing: DepthFrameMetaQueue + OkvisFramePool, AI-1184). No offsets here: the wire is
# dma_common/okvis_frame_ipc.hpp's. Without the binding, depth_read() raises RuntimeError naming why.
DEPTH_META_NAME = "/okvis_depth_meta"     # dma_ipc.OKVIS_DEPTH_META_QUEUE_NAME
DEPTH_POOL_NAME = "/okvis_depth_frames"   # dma_ipc.OKVIS_DEPTH_FRAMES_POOL_NAME
_depth_rings: dict = {}


def depth_read(meta_name: str | None = None, pool_name: str | None = None) -> dict | None:
    """The newest depth frame: {"age_s", "meta" (okvis_frames.meta_dict), "depth": float32 LE bytes,
    "sigma": the same or None}; None if absent, or if its pool slot was recycled / rewritten while
    copied (DMA.common's seqlock read)."""
    import okvis_frames
    key = (meta_name or DEPTH_META_NAME, pool_name or DEPTH_POOL_NAME)
    ring = _depth_rings.get(key)
    if ring is None:
        ring = _depth_rings[key] = okvis_frames.FrameRing("depth", *key)
    newest = ring.newest()
    if newest is None or newest[0].format_version < 1:
        return None
    m = newest[0]
    try:
        got = ring.pixels(m)
    except ValueError:   # bytes do not match the geometry: not a frame this reader knows
        return None
    if got is None:
        return None
    age = ring.age_s()
    return {"age_s": round(age, 2) if age is not None else None, "meta": okvis_frames.meta_dict(m),
            "depth": got[0].tobytes(), "sigma": None if got[1] is None else got[1].tobytes()}


def depth_sectors(d: dict, sectors: int = 9, band: float = 1 / 3, min_m: float = 0.3) -> dict:
    """Nearest return per horizontal sector, in metres, over the middle `band` of rows: the 5th
    percentile of valid depths in each column strip, so one speckle cannot claim a wall. Sector
    0 is the image's left edge; the camera's horizontal field of view is fx/width."""
    import array
    import math
    w, h = d["meta"]["width"], d["meta"]["height"]
    a = array.array("f"); a.frombytes(d["depth"])
    r0, r1 = int(h * (0.5 - band / 2)), int(h * (0.5 + band / 2))
    edges = [round(k * w / sectors) for k in range(sectors + 1)]
    near, valid_total = [], 0
    for k in range(sectors):
        vals = [a[r * w + c] for r in range(r0, r1) for c in range(edges[k], edges[k + 1])
                if a[r * w + c] >= min_m]
        valid_total += len(vals)
        if vals:
            vals.sort()
            near.append(round(vals[len(vals) // 20], 2))
        else:
            near.append(None)
    fx = d["meta"]["fx"]
    hfov = round(math.degrees(2 * math.atan(w / (2 * fx))), 1) if fx > 0 else None
    # bearing of each sector's centre, degrees, positive = left (robot convention)
    bearings = [round(-math.degrees(math.atan(((edges[k] + edges[k + 1]) / 2 - d["meta"]["cx"]) / fx)), 1)
                if fx > 0 else None for k in range(sectors)]
    return {"sectors": sectors, "nearest_m": near, "bearing_deg": bearings, "hfov_deg": hfov,
            "valid_frac": round(valid_total / max(1, (r1 - r0) * w), 3),
            "band_rows": [r0, r1]}


def depth_write_pgm(d: dict, out: str) -> dict:
    """16-bit PGM in millimetres (0 = no return): readable by any image tool, no encoder needed."""
    import array
    w, h = d["meta"]["width"], d["meta"]["height"]
    a = array.array("f"); a.frombytes(d["depth"])
    mm_ = array.array("H", (min(65535, int(v * 1000)) if v > 0 else 0 for v in a))
    if sys.byteorder == "little":
        mm_.byteswap()   # PGM is big-endian
    with open(out, "wb") as fh:
        fh.write(f"P5\n{w} {h}\n65535\n".encode())
        fh.write(mm_.tobytes())
    return {"path": out, "format": "pgm16-mm"}


# --- /route_grid: the 2D traversability map the planner drives on, read through dma_ipc.RouteGridQueue
# (DMA.common nav_ipc_types.h RouteGridData) and handed on as interface/route_grid_mcap's RouteGrid.
ROUTEGRID_RING_NAME = "/" + Path(os.environ.get("OKVIS_ROUTEGRID_SHM", "/route_grid")).name


def _route_grid_mcap():
    return _interface_import("route_grid_mcap")


def grid_read(name: str | None = None):
    """The newest RouteGrid on the ring and its age in seconds, or None (absent, never written, or a record
    this reader does not know). (RouteGrid has __slots__, so the age rides beside it.) RuntimeError naming
    why when this image's dma_ipc cannot read /route_grid."""
    r = ring_reader("route_grid", name or ROUTEGRID_RING_NAME)
    m = r.latest()
    if m is None:
        return None
    try:
        g = _route_grid_mcap().from_record(m)
    except ValueError:
        return None
    age = r.age_s()
    return g, round(age, 2) if age is not None else 0.0


COMPASS = ("+x", "+x+y", "+y", "-x+y", "-x", "-x-y", "-y", "+x-y")


def grid_summary(g, radius_m: float = 4.0, ascii_map: bool = False, age_s: float | None = None) -> dict:
    """The grid as an LLM can use it: robot cell and heading, goal, cell counts, the nearest
    blocked cell along 8 world-frame compass rays (metres, None = clear to radius_m or the
    edge), and optionally an ASCII window around the robot: '#' blocked, '.' observed free,
    ' ' unobserved, 'R' robot, 'G' goal; +x is up, +y is left (a map, not a camera)."""
    import math
    ri, rj = g.cell_of(g.robot_x, g.robot_y)
    steps = int(radius_m / g.res)
    rays = {}
    for name, (dx, dy) in zip(COMPASS, ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))):
        hit = None
        for k in range(1, steps + 1):
            i, j = ri + k * dx, rj + k * dy
            if not (0 <= i < g.nx and 0 <= j < g.ny):
                break
            if g.blocked(i, j):
                hit = round(k * g.res * math.hypot(dx, dy), 2)
                break
        rays[name] = hit
    out = {
        "age_s": age_s, "seq": g.seq, "res_m": g.res, "size": [g.nx, g.ny],
        "robot": {"x": round(g.robot_x, 2), "y": round(g.robot_y, 2), "yaw_rad": round(g.robot_yaw, 3),
                  "yaw_deg": round(math.degrees(g.robot_yaw), 1), "cell": [ri, rj]},
        "goal": ({"x": round(g.goal_x, 2), "y": round(g.goal_y, 2)} if g.flags & 2 else None),
        "map_consistent": bool(g.map_consistent),
        "blocked_cells": g.blocked_cells, "observed_cells": g.observed_cells,
        "nearest_blocked_m": rays,
    }
    if ascii_map:
        gi, gj = g.cell_of(g.goal_x, g.goal_y) if g.flags & 2 else (None, None)
        rows = []
        for i in range(ri + steps, ri - steps - 1, -1):          # +x up
            row = []
            for j in range(rj + steps, rj - steps - 1, -1):      # +y left
                if (i, j) == (ri, rj):
                    row.append("R")
                elif (i, j) == (gi, gj):
                    row.append("G")
                elif not (0 <= i < g.nx and 0 <= j < g.ny):
                    row.append("~")
                elif g.blocked(i, j):
                    row.append("#")
                elif g.observed(i, j):
                    row.append(".")
                else:
                    row.append(" ")
            rows.append("".join(row))
        out["ascii"] = rows
        out["ascii_legend"] = "# blocked . free (observed) ' ' unobserved ~ off-grid R robot G goal; +x up, +y left, %.2f m/cell" % g.res
    return out


def pose() -> dict:
    """Position from the status ring, heading from the route grid (the status yaw is the IMU
    frame, 90 deg off the robot's heading)."""
    st = status_read()
    if not node_fresh(st):
        return {"ok": False, "err": "node not fresh"}
    got = grid_read()
    g = got[0] if got else None
    r = {"ok": True, "x": st["pos"][0], "y": st["pos"][1], "z": st["pos"][2], "pose_age_s": st["pose_age_s"],
         "goal": st["goal"], "goal_dist_m": st["goal_dist"], "fsm": st["fsm"], "armed": st["armed"]}
    if g is not None:
        import math
        r["yaw_rad"], r["yaw_deg"] = round(g.robot_yaw, 3), round(math.degrees(g.robot_yaw), 1)
        r["yaw_source"] = "route_grid"
    else:
        r["yaw_rad"] = r["yaw_deg"] = None
        r["yaw_source"] = None
    return r


# ---------------------------------------------------------------- pose goals: the camera yaw offset
# A pose goal (/nav_pose_goal, written by the nav API) faces yaw(q) + camera_yaw_offset + bearing; the
# nav API's GO with a heading needs the offset to cancel it. The robot still only moves while ARMED.
LIVE_LOGS = (Path("/work/live.log"), Path("/data/okvis-perf/live.log"))


def camera_yaw_offset_deg() -> float | None:
    """The yaw of the semantic camera's optical axis in the body frame, as the node derived it
    from T_SC at startup (its "[semantic-goal] pose-index ingress armed ... cam_yaw_deg=" line).
    Read, never guessed: a wrong constant turns every goal by that constant. Without that log file
    (a pod with no /work), the same value computed from the node's rig yaml by the node's own
    formula (camera_yaw_offset_from_rig)."""
    import re
    for p in LIVE_LOGS:
        try:
            # The line is logged at startup and live.log is truncated on restart, so it sits
            # near the top: stream from the start and stop at the first match.
            with open(p, "rb") as fh:
                for ln in fh:
                    if b"pose-index ingress armed" in ln:
                        m = re.search(rb"cam_yaw_deg=(-?[\d.]+)", ln)
                        if m:
                            return float(m.group(1))
                        break
        except OSError:
            continue
    rig = follow_rig()
    return camera_yaw_offset_from_rig(rig) if rig else None


def _leading_number(s: str, cast):
    """C's strtod / atoi on a string: the longest numeric prefix, 0 when there is none."""
    import re
    m = re.match(r"\s*[-+]?(\d+\.?\d*([eE][-+]?\d+)?|\.\d+([eE][-+]?\d+)?)" if cast is float else r"\s*[-+]?\d+", s)
    return cast(m.group(0)) if m else cast(0)


def rig_cameras(path) -> list[dict]:
    """The `cameras:` entries of an OKVIS rig yaml (OpenCV FileStorage YAML, flow or block style):
    [{"T_SC": 16 floats row-major or None, "camera_type": str or None}], in file order. Stdlib only: the
    okvis2x image's python has neither cv2 nor PyYAML. [] when the file cannot be read."""
    import re
    try:
        lines = Path(path).read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    lines = [ln.split("#", 1)[0] for ln in lines]   # no '#' inside a rig yaml's values
    try:
        start = next(i for i, ln in enumerate(lines) if re.match(r"cameras\s*:", ln))
    except StopIteration:
        return []
    body = []
    for ln in lines[start + 1:]:
        if ln[:1] not in ("", " ", "\t", "-") and ln.strip():   # the next top-level key
            break
        body.append(ln)
    out = []
    for item in re.split(r"(?m)^\s*-\s", "\n".join(body))[1:]:
        m = re.search(r"T_SC\s*:\s*\[([^\]]*)\]", item)
        t = re.search(r"camera_type\s*:\s*([^,}\s]+)", item)
        try:
            T = [float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()] if m else None
        except ValueError:
            T = None
        out.append({"T_SC": T if T and len(T) == 16 else None, "camera_type": t.group(1) if t else None})
    return out


def _quat_from_matrix(m) -> tuple[float, float, float, float]:
    """Eigen's Quaternion<double>(Matrix3d) (Shepperd's method, quaternionbase_assign_impl), as (w, x, y, z).
    Like Eigen it assumes nothing about orthonormality; the caller normalizes."""
    import math as _m
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        t = _m.sqrt(t + 1.0)
        w = 0.5 * t
        t = 0.5 / t
        return w, (m[2][1] - m[1][2]) * t, (m[0][2] - m[2][0]) * t, (m[1][0] - m[0][1]) * t
    i = 0
    if m[1][1] > m[0][0]:
        i = 1
    if m[2][2] > m[i][i]:
        i = 2
    j, k = (i + 1) % 3, (i + 2) % 3
    t = _m.sqrt(m[i][i] - m[j][j] - m[k][k] + 1.0)
    q = [0.0, 0.0, 0.0]
    q[i] = 0.5 * t
    t = 0.5 / t
    w = (m[k][j] - m[j][k]) * t
    q[j] = (m[j][i] + m[i][j]) * t
    q[k] = (m[k][i] + m[i][k]) * t
    return w, q[0], q[1], q[2]


def camera_yaw_offset_from_rig(path, env=None) -> float | None:
    """The semantic camera's yaw in the body frame [deg], computed from the rig yaml EXACTLY as dma_live
    computes the value its "[semantic-goal] pose-index ingress armed ... cam_yaw_deg=" line prints
    (okvis2x_app_dma_live.cpp, the pose-goal ingress block):
      - the camera: the first colour one (ViParametersReader: camera_type not starting with "gray"; no
        camera_type = colour, the CameraType default), or OKVIS_SEMANTIC_POSE_CAM_INDEX (atoi);
      - its T_SC as the node holds it: ViParametersReader builds Transformation(r, q.normalized()) with
        q = Eigen::Quaterniond(R) of the yaml's rotation, so R is re-orthonormalized through that quaternion;
      - the optical axis is camera +z; in the IMU/body frame S it is R_SC * z (R's third column), and the
        offset is atan2(axis.y, axis.x);
      - an index with no camera: 0 ("assumed body +x"); OKVIS_SEMANTIC_POSE_CAM_YAW_DEG (strtod) overrides.
    None when the rig cannot be read or names no camera."""
    import math as _m
    env = os.environ if env is None else env
    if env.get("OKVIS_SEMANTIC_POSE_CAM_YAW_DEG") is not None:
        return float(_leading_number(env["OKVIS_SEMANTIC_POSE_CAM_YAW_DEG"], float))
    cams = rig_cameras(path)
    if not cams:
        return None
    idx = next((i for i, c in enumerate(cams) if not (c["camera_type"] or "").startswith("gray")), -1)
    if env.get("OKVIS_SEMANTIC_POSE_CAM_INDEX") is not None:
        idx = _leading_number(env["OKVIS_SEMANTIC_POSE_CAM_INDEX"], int)
    if not 0 <= idx < len(cams) or cams[idx]["T_SC"] is None:
        return 0.0
    T = cams[idx]["T_SC"]
    w, x, y, z = _quat_from_matrix([T[0:3], T[4:7], T[8:11]])
    n = _m.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    # Eigen toRotationMatrix, third column: (2xz + 2wy, 2yz - 2wx, 1 - 2xx - 2yy)
    ax, ay = 2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x
    return _m.degrees(_m.atan2(ay, ax))


# ---------------------------------------------------------------- status, all in one
def status() -> dict:
    """The node (its status ring, read here) and the nav API's view of follow and the apps (their state
    rings), in one JSON. "autonomy" is the node's armed flag; an app falls back to its bench pidfile
    status when the nav API does not answer."""
    st = status_read()
    nav = api("/status", timeout=3.0)
    napps = (nav.get("apps") or {}) if nav.get("ok") else {}
    return {
        "ok": True,
        "node": {"fresh": bool(node_fresh(st)), **(st or {})} if st else {"fresh": False},
        "autonomy": bool(st and st["armed"]),
        "okvis_mode": okvis_mode(),
        "follow": nav.get("follow") if nav.get("ok") else {"err": nav.get("err")},
        "services": {n: unit_status("service", n) for n in units("service")},
        "apps": {n: napps.get(n) or unit_status("app", n) for n in units("app")},
        "verbose": verbose_get(),
    }


# ---------------------------------------------------------------- command line
SCHEMA = [
    {"name": "status", "args": {"--watch": "bool"}, "help": "node, autonomy, modes, services; one JSON"},
    {"name": "autonomy", "args": {"state": ["on", "off"]}, "help": "arm or disarm; the robot moves only while armed"},
    {"name": "mode", "args": {"mode": ["explore", "semantic", "patrol", "follow"]},
     "help": "goal source: explore = frontier NBV, patrol = straight-line tour, semantic = stop and await a goal, follow = follow-me"},
    {"name": "verb", "args": {"verb": sorted(CLI_VERBS)}, "help": "a command the node may refuse; waits for its answer"},
    {"name": "service", "args": {"action": ["list", "start", "stop", "status"], "name": list(units("service"))},
     "help": "capabilities anyone can use: semantic_memory = CLIP index + text encoder, llm = orchestrator llama-server :8911. start pins it on until stop"},
    {"name": "app", "args": {"action": ["list", "start", "stop", "status"], "name": list(units("app"))},
     "help": "BENCH ONLY (robots start apps through the AI API server): scripts built on services, follow_me = person follower. start brings its services; stop drops the ones nothing else needs"},
    {"name": "reconcile", "args": {}, "help": "stop every service that no running app and no pin needs; safe at any time"},
    {"name": "follow", "args": {"action": ["prompt", "select", "off"], "value": "TEXT | X Y"},
     "help": "follow-me target: prompt = what to look for (no TEXT = blank; outside follow mode only shown, entering follow mode fills a blank one with person), select = pick the match at image fraction X Y in any mode (default in follow mode: the nearest), off = leave follow mode"},
    {"name": "semantic", "args": {"text": "str", "--top": "int", "--go": "hit index, drive to where it was seen from, facing it"},
     "help": "where was TEXT seen: ranked keyframes with camera pose and bearing; --go N commits hit N as the goal"},
    {"name": "verbose", "args": {"state": ["on", "off"]}, "help": "node log detail"},
    {"name": "pose", "args": {}, "help": "where the robot is: x y z, heading yaw_deg, current goal and distance"},
    {"name": "frame", "args": {"frame": sorted(FRAMES), "--out": "PATH or - for base64"},
     "help": "latest camera image (JPEG): cam0/cam1 rectified, follow = follow-me view, routegrid = map picture"},
    {"name": "depth", "args": {"--sectors": "int", "--out": "PATH.pgm"},
     "help": "metric depth: nearest obstacle per bearing sector (m, left to right), field of view, valid fraction"},
    {"name": "grid", "args": {"--radius": "metres", "--ascii": "bool"},
     "help": "traversability map: robot cell + heading, goal, nearest blocked cell on 8 compass rays, ASCII map"},
    {"name": "look", "args": {"xyz": "X Y [Z] | off", "--hold": "seconds"},
     "help": "point the head at a world point (idle in semantic/follow mode only); --hold S keeps it there S seconds (one push, released after)"},
    {"name": "goto", "args": {"x": "metres", "y": "metres", "yaw_deg": "optional, world frame"},
     "help": "commit a pose goal: drive to x y and face yaw_deg (default: keep heading); moves only while armed; waits for the node's answer"},
]


def _emit(r: dict) -> int:
    print(json.dumps(r), flush=True)
    if r.get("ok"):
        return 0
    return 1 if r.get("refused") else 2


def _need_node() -> dict | None:
    st = status_read()
    if node_fresh(st):
        return None
    if st is None:
        return {"ok": False, "err": f"{STATUS_RING_NAME}: no status ring (node never ran here)"}
    return {"ok": False, "err": f"{STATUS_RING_NAME}: node not fresh (age {st['age_s']} s)"}


def cmd_autonomy(a) -> dict:
    return api("/autonomy", {"on": a.state == "on"})


def cmd_mode(a) -> dict:
    if a.mode == "follow":
        return api("/follow", {"mode": "follow"})
    return api("/mode", {"mode": a.mode})   # the nav API ends FOLLOW first


def cmd_verb(a) -> dict:
    return api("/verb", {"verb": a.verb})


def cmd_unit(a) -> dict:
    kind = a.cmd
    if a.action == "list":
        return {"ok": True, kind + "s": {n: unit_status(kind, n) for n in units(kind)}}
    if not a.name:
        return {"ok": False, "err": f"{kind} {a.action} needs a NAME: {', '.join(units(kind))}"}
    wait = a.timeout if a.timeout is not None else float((_unit(kind, a.name) or {}).get("up_s", 30))
    if _unit(kind, a.name) is None:
        return _missing(kind, a.name)
    if a.action == "status":
        return {"ok": True, kind: a.name, **unit_status(kind, a.name)}
    if kind == "service":
        return service_start(a.name, wait_s=wait) if a.action == "start" else service_stop(a.name)
    return app_start(a.name, wait_s=wait) if a.action == "start" else app_stop(a.name)


def cmd_reconcile(a) -> dict:
    return reconcile()


def cmd_follow(a) -> dict:
    v = a.value
    if a.action == "off":
        return api("/follow", {"mode": "off"})
    if a.action == "prompt":   # perception only: matches shown, no mode change
        return api("/follow/prompt", {"text": " ".join(v)[:79]})
    if a.action == "select":
        try:
            x, y = float(v[0]), float(v[1])
        except (IndexError, ValueError):
            return {"ok": False, "err": "select needs X Y as image fractions in [0,1]"}
        return api("/follow/select", {"x": x, "y": y})   # picks a target; never the mode
    return {"ok": False, "err": "unknown follow action"}


def cmd_semantic(a) -> dict:
    text = " ".join(a.text)
    r = semantic_query(text, min(5, max(a.top, (a.go or 0) + 1)))
    if r.get("ok") and a.go is not None:   # the app drives: to where hit N was seen from, facing it
        if a.go >= len(r["hits"]):
            return {"ok": False, "err": f"only {len(r['hits'])} hits, no hit {a.go}", "hits": r["hits"]}
        st = goto_ask(text, 5, True, r["hits"][a.go]["frame"])
        d = st.get("drive") or {"ok": False, "reason": st.get("err") or "no drive"}
        return {"ok": bool(d.get("ok")), "refused": not d.get("ok"), "reason": d.get("reason"), "hits": r["hits"]}
    if r.get("ok") and a.save:
        import shutil
        Path(a.save).mkdir(parents=True, exist_ok=True)
        for h in r["hits"]:
            try:
                shutil.copy2(h["image"], a.save)
                h["saved"] = str(Path(a.save) / Path(h["image"]).name)
            except OSError as e:
                h["saved"] = None
                h["save_err"] = str(e)
    return r


def cmd_status(a) -> dict:
    if not a.watch:
        return status()
    try:
        while True:
            print(json.dumps(status()), flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    return {"ok": True}


def cmd_verbose(a) -> dict:
    return verbose_set(a.state == "on")


def cmd_pose(a) -> dict:
    return pose()


def cmd_frame(a) -> dict:
    return frame_read(a.frame, a.out)


def cmd_depth(a) -> dict:
    d = depth_read()
    if d is None:
        return {"ok": False, "err": f"{DEPTH_META_NAME}: no depth frame (ring absent, torn, or never written)"}
    r = {"ok": True, "age_s": d["age_s"], "stamp_ns": d["meta"]["stamp_ns"],
         "size": [d["meta"]["width"], d["meta"]["height"]], **depth_sectors(d, a.sectors)}
    if a.out:
        r.update(depth_write_pgm(d, a.out))
    return r


def cmd_grid(a) -> dict:
    got = grid_read()
    if got is None:
        return {"ok": False, "err": f"{ROUTEGRID_RING_NAME}: no route grid (ring absent or never written)"}
    return {"ok": True, **grid_summary(got[0], a.radius, a.ascii, got[1])}


def cmd_goto(a) -> dict:
    return api("/goal", {"cmd": "go", "x": a.x, "y": a.y, "yaw_deg": a.yaw_deg})


def cmd_look(a) -> dict:
    if a.xyz and a.xyz[0] == "off":
        return api("/look", {"off": True})
    try:
        x, y = float(a.xyz[0]), float(a.xyz[1])
        z = float(a.xyz[2]) if len(a.xyz) > 2 else 0.0
    except (IndexError, ValueError):
        return {"ok": False, "err": "look needs X Y [Z] in the world frame, or off"}
    return api("/look", {"x": x, "y": y, "z": z, "hold_s": a.hold})


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nav-cli", description=__doc__.split("\n", 1)[0])
    ap.add_argument("--robot", metavar="HOST", help="run on that robot over ssh (inside its OKVIS pod)")
    ap.add_argument("--schema", action="store_true", help="print the command tree as JSON tool definitions")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--timeout", type=float, default=None,
                        help="seconds to wait for the node's answer (default 2; service/app start: the unit's up_s)")
    sub = ap.add_subparsers(dest="cmd")
    add = lambda name: sub.add_parser(name, parents=[common])
    p = add("status"); p.add_argument("--watch", action="store_true")
    p = add("autonomy"); p.add_argument("state", choices=["on", "off"])
    p = add("mode"); p.add_argument("mode", choices=["explore", "semantic", "patrol", "follow"])
    p = add("verb"); p.add_argument("verb", choices=sorted(CLI_VERBS))
    for kind in ("service", "app"):   # NAME unchecked here: --robot validates it on the robot, against its manifests
        p = add(kind); p.add_argument("action", choices=["list", "start", "stop", "status"]); p.add_argument("name", nargs="?")
    add("reconcile")
    p = add("follow"); p.add_argument("action", choices=["prompt", "select", "off"]); p.add_argument("value", nargs="*")
    p = add("semantic"); p.add_argument("text", nargs="+"); p.add_argument("--top", type=int, default=5)
    p.add_argument("--save", metavar="DIR", help="copy the hit images here")
    p.add_argument("--go", type=int, metavar="N", help="commit hit N (0 = best) as the goal: drive to where it was seen from, facing it")
    p = add("verbose"); p.add_argument("state", choices=["on", "off"])
    add("pose")
    p = add("frame"); p.add_argument("frame", choices=sorted(FRAMES)); p.add_argument("--out", metavar="PATH", help="write the JPEG here; - = base64 in the JSON")
    p = add("depth"); p.add_argument("--sectors", type=int, default=9); p.add_argument("--out", metavar="PATH.pgm", help="also write a 16-bit PGM in mm")
    p = add("grid"); p.add_argument("--radius", type=float, default=4.0, help="metres around the robot for the rays and the ASCII map"); p.add_argument("--ascii", action="store_true")
    p = add("look"); p.add_argument("xyz", nargs="+"); p.add_argument("--hold", type=float, default=0.0)
    p = add("goto"); p.add_argument("x", type=float); p.add_argument("y", type=float)
    p.add_argument("yaw_deg", type=float, nargs="?", default=None, help="facing, world frame; default keeps the current heading")
    return ap


# Where --robot finds OKVIS: the PKO DaemonSet okvis2x (manifests/base/okvis2x).
OKVIS_NS = "positronic"
OKVIS_POD_LABEL = "app.kubernetes.io/name=okvis2x"


def run_remote(host: str, argv: list[str]) -> int:
    """Run THIS file on the robot: inside the running OKVIS pod when there is one (the services
    and their paths live there), else on the host (the rings are the same /dev/shm). The pod is
    the DaemonSet okvis2x's in namespace positronic, named okvis2x-<hash>, so it is found by its
    label. stdout and the exit code come straight back."""
    args = " ".join(shlex.quote(a) for a in argv)
    script = (f"pod=$(k0s kubectl -n {OKVIS_NS} get pod -l {OKVIS_POD_LABEL} --field-selector=status.phase=Running "
              f"-o jsonpath='{{.items[0].metadata.name}}' 2>/dev/null); "
              f"if [ -n \"$pod\" ]; then exec k0s kubectl -n {OKVIS_NS} exec -i \"$pod\" -c okvis2x -- python3 - {args}; "
              f"else exec python3 - {args}; fi")
    with open(__file__, "rb") as src:
        return subprocess.run(["ssh", host, script], stdin=src).returncode


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.schema:
        print(json.dumps(SCHEMA, indent=1))
        return 0
    if a.robot:
        rest = [x for x in argv if x not in ("--robot", a.robot)]
        rc = run_remote(a.robot, rest)
        return rc
    if not a.cmd:
        ap.print_help()
        return 2
    if a.timeout is None and a.cmd not in ("service", "app"):
        a.timeout = 2.0
    handler = {"status": cmd_status, "autonomy": cmd_autonomy, "mode": cmd_mode, "verb": cmd_verb,
               "service": cmd_unit, "app": cmd_unit, "reconcile": cmd_reconcile, "follow": cmd_follow,
               "semantic": cmd_semantic, "verbose": cmd_verbose, "pose": cmd_pose, "frame": cmd_frame,
               "depth": cmd_depth, "grid": cmd_grid, "look": cmd_look, "goto": cmd_goto}[a.cmd]
    try:
        return _emit(handler(a))
    except RuntimeError as e:   # a ring this image's dma_ipc cannot read (nav_rings.RingError, okvis_frames)
        return _emit({"ok": False, "err": str(e)})


if __name__ == "__main__":
    sys.exit(main())
