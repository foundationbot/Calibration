#!/usr/bin/env python3
"""nav-cli: the one command surface for the OKVIS node and its sidecar services (#610).

Library AND command line. liveviz imports this module; an agent, a shell or a laptop runs it:

    nav-cli status                      nav-cli --robot phantom21 status
    nav-cli autonomy on|off             nav-cli mode explore|semantic|patrol|follow
    nav-cli verb reset-tracking|clear-goal
    nav-cli svc start|stop|status clip|follow-me|llm
    nav-cli follow prompt [TEXT] | select X Y | off
    nav-cli slam restart [--mode M] [--autonomy A] [--follow-standoff M] [--app dma_live|replay] | stop
    nav-cli record start|stop|status    nav-cli semantic TEXT [--top N] [--save DIR]
    nav-cli verbose on|off              nav-cli --schema
    nav-cli pose | frame cam0|cam1|follow|routegrid [--out P] | depth [--sectors N] [--out P.pgm]
    nav-cli grid [--radius M] [--ascii] | look X Y [Z] [--hold S] | look off
    nav-cli goto X Y [YAW_DEG] | semantic TEXT --go 0      # pose goals; the robot moves only while armed

Every command prints one JSON object. Exit 0 = done/accepted, 1 = the node refused,
2 = transport (ring missing, node not fresh, service did not come up, timeout).

Wire formats are mirrored BY HAND from the C++ headers (StatusDMAPublisher.hpp, VerbDMA.hpp,
NavGoalDMA) -- this file is the one copy of them on the Python side. test_nav_cli.py pins
them against the headers.

Stdlib only: it runs inside the OKVIS image, on the robot host, and on a laptop (--robot
pipes this very file over ssh, so the version you run is the one in your checkout).
"""
from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import shlex
import signal
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- files the node and the agent share
SHM = Path("/dev/shm")
AUTONOMY_GATE = Path(os.environ.get("OKVIS_AUTONOMY_GATE", "/dev/shm/okvis_autonomy"))
NAVGOAL_PATH = Path(os.environ.get("OKVIS_NAV_GOAL_SHM", "/dev/shm/nav_goal"))
VERBS_PATH = Path(os.environ.get("OKVIS_VERBS_SHM", "/dev/shm/okvis_verbs"))
STATUS_RING_PATH = Path(os.environ.get("OKVIS_STATUS_SHM", "/dev/shm/okvis_status"))
VIZ_DIR = Path(os.environ.get("OKVIS_VIZ_DIR", "/dev/shm/okvis-viz"))
VERBOSITY_FILE = Path("/dev/shm/okvis_verbosity")
# The console agent (deployment/okvis-console-agent.sh, the okvis-agent pod): heartbeat, and the
# request files it consumes. Nothing inside the OKVIS pod can recreate that pod.
RESTART_AGENT = Path("/dev/shm/okvis_restart_agent")
RESTART_AGENT_STALE_S = 10.0
RESTART_REQUEST = Path("/dev/shm/okvis_restart_request")
STOP_REQUEST = Path("/dev/shm/okvis_stop_request")
# Services for a client outside the OKVIS pod (the VLM agent pod), the recorder pod, and the
# recorder's state the agent republishes for `record status`.
SVC_REQUEST = Path("/dev/shm/okvis_svc_request")
SVC_RESULT = Path("/dev/shm/okvis_svc_result")
RECORD_REQUEST = Path("/dev/shm/okvis_record_request")
RECORD_RESULT = Path("/dev/shm/okvis_record_result")
RECORD_STATE = Path("/dev/shm/okvis_record")
LIVEVIZ_URL = os.environ.get("OKVIS_LIVEVIZ_URL", "http://127.0.0.1:8899")
NODE_FRESH_S = 5.0   # status older than this = the node is gone (the ring outlives it, #201)

# ---------------------------------------------------------------- dma::ShmRingBuffer, by hand
SHM_MAGIC = 0x444D415F51554555          # "DMA_QUEU"


def ring_push(path: Path, fmt: str, fields, stamp_ns=None) -> dict:
    """Push one record onto a dma::ShmRingBuffer, following its push() exactly.
    stamp_ns: record timestamp (ns); default is the push time.

    Shared by /nav_goal and /okvis_verbs: both mirror the same 192-byte DMA_QUEU header and
    slot geometry, and two copies of those offsets are how they drift apart.

    Write order is the writer's whole contract: payload, then the slot's sequence, then head.
    A reader takes head, steps back one slot and checks that slot's sequence, so publishing
    head before the slot is filled would hand it a stale or half-written record. msync
    between the steps because Python gives us no release ordering of its own.

    `fields` are the values for `fmt` EXCLUDING the leading timestamp_ns and sequence, which
    this function stamps itself -- the ring's sequence and the record's must agree.
    """
    try:
        with open(path, "r+b") as fh:
            mm = mmap.mmap(fh.fileno(), 0)
    except OSError as e:
        return {"ok": False, "err": f"{path}: {e}"}
    try:
        magic, _version = struct.unpack_from("<QQ", mm, 0)
        slot_size, capacity = struct.unpack_from("<QQ", mm, 16)
        if magic != SHM_MAGIC:
            return {"ok": False, "err": f"bad magic 0x{magic:016x} — not a dma_common ring"}
        if capacity == 0 or (capacity & (capacity - 1)) != 0:
            # push() indexes with `head & (Capacity - 1)`, which is only modulo for a power
            # of two. Refuse rather than write to a slot the reader never looks at.
            return {"ok": False, "err": f"capacity {capacity} is not a power of two"}
        header_size = len(mm) - capacity * slot_size
        if header_size < 72 or slot_size < 8 + struct.calcsize(fmt):
            return {"ok": False, "err": f"unexpected geometry: header {header_size}, slot {slot_size}"}
        head = struct.unpack_from("<Q", mm, 64)[0]
        off = header_size + (head & (capacity - 1)) * slot_size
        struct.pack_into(fmt, mm, off + 8, time.time_ns() if stamp_ns is None else stamp_ns, (head + 1) & 0xFFFFFFFF, *fields)
        mm.flush()
        struct.pack_into("<Q", mm, off, head + 1)              # slot sequence marks it valid
        mm.flush()
        struct.pack_into("<Q", mm, 64, head + 1)               # publish
        struct.pack_into("<q", mm, 72, time.monotonic_ns())    # last_write_ns, a recency hint
        mm.flush()
    except (struct.error, ValueError) as e:
        return {"ok": False, "err": str(e)}
    finally:
        mm.close()
    return {"ok": True, "seq": head + 1}


def ring_read_latest(path: Path, payload: int) -> tuple[bytes, int] | None:
    """The newest slot's payload and the ring's last_write_ns; None if absent/invalid/torn.

    Reader half of dma::ShmRingBuffer. Protocol: take head, step back one slot, check that
    slot's sequence == head, copy the payload, re-check the sequence — a torn or lapped read
    returns None and the next poll gets a clean one.
    """
    try:
        with open(path, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
    except (OSError, ValueError):
        return None
    try:
        if struct.unpack_from("<Q", mm, 0)[0] != SHM_MAGIC:
            return None
        slot_size, capacity = struct.unpack_from("<QQ", mm, 16)
        if capacity == 0 or (capacity & (capacity - 1)) != 0:
            return None
        header_size = len(mm) - capacity * slot_size
        if header_size < 80 or slot_size < 8 + payload:
            return None
        head, = struct.unpack_from("<Q", mm, 64)
        if head == 0:
            return None  # created but never written
        last_write_ns, = struct.unpack_from("<q", mm, 72)
        off = header_size + ((head - 1) & (capacity - 1)) * slot_size
        seq1, = struct.unpack_from("<Q", mm, off)
        raw = bytes(mm[off + 8: off + 8 + payload])
        seq2, = struct.unpack_from("<Q", mm, off)
        if seq1 != head or seq2 != seq1:
            return None
        return raw, last_write_ns
    except (struct.error, ValueError):
        return None
    finally:
        mm.close()


# ---------------------------------------------------------------- /okvis_status (#205)
# The node publishes its state as ONE struct at ~20 Hz (StatusDMAPublisher.hpp documents
# every byte offset mirrored here). Reading THIS instead of inferring state from log prose
# is the fix for the 2026-08-31 false alarm. Liveness comes from the header's last_write_ns
# (CLOCK_MONOTONIC, comparable here because console and node share the kernel).
STATUS_VERSION = 3
STATUS_STRUCT = "<IIqqii3f4f4fI4BIB3x64s2B50s"   # OkvisStatusData v3, 208 bytes


def status_read(path: Path | None = None) -> dict | None:
    """The newest OkvisStatusData, decoded; None if absent/invalid/torn/wrong version."""
    got = ring_read_latest(path or STATUS_RING_PATH, struct.calcsize(STATUS_STRUCT))
    if got is None:
        return None
    raw, last_write_ns = got
    try:
        (ver, seq, stamp_ns, pose_stamp_ns, pid, fsm,
         px, py, pz, qx, qy, qz, qw, gx, gy, gz, gdist,
         goals_issued, goal_valid, armed, pyield, semantic,
         verb_seq, verb_ok, reason, dmode, dfrozen, dlabel) = struct.unpack(STATUS_STRUCT, raw)
    except struct.error:
        return None
    if ver != STATUS_VERSION:
        return None
    age_s = max(0.0, (time.monotonic_ns() - last_write_ns) / 1e9)
    return {
        "age_s": round(age_s, 2), "seq": seq, "pid": pid,
        "pose_age_s": (round((stamp_ns - pose_stamp_ns) / 1e9, 1) if pose_stamp_ns > 0 else None),
        "pos": [round(px, 2), round(py, 2), round(pz, 2)],
        "fsm": fsm,
        "goal": ([round(gx, 2), round(gy, 2), round(gz, 2)] if goal_valid else None),
        "goal_dist": (round(gdist, 2) if gdist >= 0 else None),
        "goals_issued": goals_issued,
        "armed": bool(armed), "yield": bool(pyield), "semantic": bool(semantic),
        "verb": ({"seq": verb_seq, "ok": bool(verb_ok),
                  "reason": reason.split(b"\0", 1)[0].decode(errors="replace")}
                 if verb_seq > 0 else None),
        "debug": {"mode": bool(dmode), "frozen": bool(dfrozen),
                  "label": dlabel.split(b"\0", 1)[0].decode(errors="replace")},
        # The full SLAM pose T_WS, unrounded: the depth cloud is placed with it.
        "pose": ({"stamp_ns": pose_stamp_ns, "p": [px, py, pz], "q": [qx, qy, qz, qw]}
                 if pose_stamp_ns > 0 else None),
    }


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


# ---------------------------------------------------------------- /nav_goal: the mode ring
# A goal supersedes the previous goal, so this ring is read latest-wins. GO with a
# coordinate is deliberately NOT sent from here (only follow_me.py does, in follow mode).
NAV_STOP, NAV_EXPLORE, NAV_PATROL = 2, 3, 4   # okvis::NavCommand (mirrored by hand, see test)
NAV_FOLLOW = 10                              # lean follow; sent only by app/follow_me/follow_me.py
NAV_STRUCT = "<qIHHfffI"                     # NavGoalData, 32 bytes, as its header documents
NAV_MODES = {"semantic": NAV_STOP, "explore": NAV_EXPLORE, "patrol": NAV_PATROL}
NAV_NAMES = {NAV_STOP: "SEMANTIC (stop + await a goal)", NAV_EXPLORE: "EXPLORE (resume NBV)",
             NAV_PATROL: "PATROL (straight-line tour)"}


def navgoal_push(command: int, path: Path | None = None) -> dict:
    """Push one positionless command onto the /nav_goal ring."""
    r = ring_push(path or NAVGOAL_PATH, NAV_STRUCT,
                  (1, command,          # format_version, command
                   0.0, 0.0, 0.0,       # goal x, y, yaw — unused by STOP/EXPLORE/PATROL
                   0))                  # reserved
    if not r.get("ok"):
        return r
    return {"ok": True, "command": command, "name": NAV_NAMES.get(command, str(command)), "seq": r["seq"]}


# /follow_target: follow-me's target measurement in the OKVIS cam0 frame, one per camera frame
# (okvis::FollowTargetData in NavCommandDMA.hpp). No SLAM pose: the app maps it to the pelvis
# with the neck FK at the capture stamp.
FOLLOWTARGET_PATH = Path("/dev/shm/follow_target")
FT_STRUCT = "<qIHHffffqb7x"             # FollowTargetData, 48 bytes, as its header documents
FT_TARGET_VALID, FT_RANGE_VALID, FT_LOST = 1, 2, 4  # okvis::FollowTargetFlag


def follow_target_push(capture_ns: int, flags: int, xyz, similarity: float = math.nan,
                       depth_ns: int = 0, edge: int = 0) -> dict:
    """Push one measurement: xyz in OKVIS cam0 [m], a unit ray unless FT_RANGE_VALID."""
    return ring_push(FOLLOWTARGET_PATH, FT_STRUCT,
                     (1, flags, *xyz, similarity, depth_ns, edge), stamp_ns=capture_ns)


# ---------------------------------------------------------------- /okvis_verbs (#204)
# Console -> node commands the node is allowed to REFUSE. The answer comes back as
# last_verb_seq / last_verb_accepted / last_verb_reason in /okvis_status -- there is
# deliberately no second reply channel. A verb supersedes nothing, so the node drains this
# ring in order, and the ring's sequence IS the verb's sequence: the node echoes it.
VERB_STRUCT = "<qIHH2d64s"   # OkvisVerbData, 96 bytes, as VerbDMA.hpp documents
# WIRE FORMAT, shared with okvis_dma/include/okvis/VerbDMA.hpp: a console and a node from
# different builds share this ring, so a renumber turns one verb into another with nothing
# erroring. The test pins these against the header.
VERB_RESET_TRACKING, VERB_CLEAR_SEMANTIC_GOAL = 1, 2
VERB_ARM, VERB_DISARM, VERB_REC_START, VERB_REC_STOP = 3, 4, 5, 6
VERB_GS_STREAM_ON, VERB_GS_STREAM_OFF = 7, 8
VERB_SEMANTIC_GOAL = 9   # "a pose request is waiting in the goal file"; the seq is what the file never had
VERB_NAMES = {VERB_RESET_TRACKING: "RESET_TRACKING", VERB_CLEAR_SEMANTIC_GOAL: "CLEAR_SEMANTIC_GOAL",
              VERB_ARM: "ARM", VERB_DISARM: "DISARM", VERB_REC_START: "REC_START", VERB_REC_STOP: "REC_STOP",
              VERB_GS_STREAM_ON: "GS_STREAM_ON", VERB_GS_STREAM_OFF: "GS_STREAM_OFF",
              VERB_SEMANTIC_GOAL: "SEMANTIC_GOAL"}
CLI_VERBS = {"reset-tracking": VERB_RESET_TRACKING, "clear-goal": VERB_CLEAR_SEMANTIC_GOAL}


def verb_push(verb: int, arg0: float = 0.0, arg1: float = 0.0, text: str = "",
              path: Path | None = None) -> dict:
    """Send one verb. The returned seq is what the ACK will carry. The node treats
    last_verb_seq == 0 as "no verb yet" and ring_push stamps head+1, so the first verb on a
    fresh ring is seq 1 and never collides with that sentinel."""
    r = ring_push(path or VERBS_PATH, VERB_STRUCT,
                  (1, verb, float(arg0), float(arg1), text.encode()[:63]))   # 63 + NUL
    name = VERB_NAMES.get(verb, f"verb {verb}")
    if not r.get("ok"):
        return r
    return {"ok": True, "verb": verb, "name": name, "seq": r["seq"]}


def verb_wait(seq: int, timeout: float) -> dict | None:
    """The node's ACK for verb `seq` from the status ring, or None on timeout."""
    def acked():
        st = status_read()
        v = st and st.get("verb")
        return v if v and v["seq"] == seq else None
    return _wait(acked, timeout)


# ---------------------------------------------------------------- autonomy gate
def autonomy_get() -> bool:
    return AUTONOMY_GATE.exists()


def autonomy_set(on: bool) -> dict:
    try:
        if on:
            AUTONOMY_GATE.touch()
        else:
            AUTONOMY_GATE.unlink(missing_ok=True)
    except OSError as e:
        return {"ok": False, "err": str(e)}
    return {"ok": True, "autonomy": autonomy_get()}


# ---------------------------------------------------------------- follow-me intent file
# app/follow_me/follow_me.py reads the operator's intent from follow_cmd.json in the tap dir:
# the mode and the last click (a new click bumps
# seq, which is how follow_me.py tells a fresh pick from the same one). It writes follow.jpg
# and follow_state.json back.
def follow_cmd(viz_dir: Path | None = None, **upd) -> dict:
    d = VIZ_DIR if viz_dir is None else viz_dir
    p = d / "follow_cmd.json"
    try:
        cur = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        cur = {"seq": 0, "mode": "off", "click": None}
    if upd:
        if "click" in upd:
            cur["seq"] = int(cur.get("seq", 0)) + 1
        cur.update(upd)
        tmp = p.with_name(".follow_cmd.json.tmp")
        tmp.write_text(json.dumps(cur))
        os.replace(tmp, p)
    return cur


def follow_state(viz_dir: Path | None = None) -> dict:
    d = VIZ_DIR if viz_dir is None else viz_dir
    out = {"cmd": follow_cmd(d), "state": None, "age": None}
    p = d / "follow_state.json"
    try:
        out["state"] = json.loads(p.read_text())
        out["age"] = time.time() - p.stat().st_mtime
    except (OSError, json.JSONDecodeError):
        pass
    return out


def follow_needed(c: dict) -> bool:
    """follow_me.py is the shared segmentation backbone: it runs while SEMANTIC OBSTACLES or the
    target prompt is set, or while in FOLLOW, else not at all."""
    return bool(c.get("obstacles") or c.get("prompt") or c.get("mode", "off") != "off")


def follow_sync(viz_dir: Path | None = None) -> None:
    """Start or stop follow_me.py to match follow_needed. Never writes the mode, a prompt or
    /nav_goal: obstacles and FOLLOW do not reach into each other through here. A (re)start
    drops a stale click, so in FOLLOW it takes the largest match, as at startup."""
    if not svc_available("follow-me"):
        return
    c = follow_cmd(viz_dir)
    if follow_needed(c) and not svc_running("follow-me"):
        if c.get("click"):
            follow_cmd(viz_dir, click=None)
        svc_start("follow-me")
    elif not follow_needed(c) and svc_running("follow-me"):
        svc_stop("follow-me")   # nothing to segment, not following: free its GPU share


def follow_enter(viz_dir: Path | None = None) -> dict:
    """Into FOLLOW (the FOLLOW button, mode follow): if FOLLOW was off, STOP on /nav_goal (suspend
    NBV + drop any held goal, as SEMANTIC does), a blank target becomes the visible "person", and
    follow_me.py (started if needed) follows the target clicked before, else the nearest match.
    The robot moves only while ARMED."""
    c = follow_cmd(viz_dir)
    if c["mode"] == "off":
        navgoal_push(NAV_STOP)
        follow_cmd(viz_dir, mode="follow", since=time.time(), prompt=c.get("prompt") or "person")
    follow_sync(viz_dir)
    return {"ok": True, "running": svc_running("follow-me")}


def follow_leave(viz_dir: Path | None = None) -> dict:
    """Out of FOLLOW, by any mode change: the target goes back to blank, and follow_me.py stops
    at once unless SEMANTIC OBSTACLES still need it. Sends nothing: the caller's mode does."""
    c = follow_cmd(viz_dir)
    if c.get("mode", "off") != "off":
        c = follow_cmd(viz_dir, mode="off", prompt="")
    follow_sync(viz_dir)
    return c


def follow_off(viz_dir: Path | None = None) -> dict:
    """Leave FOLLOW. If following, STOP first: follow_me.py sends its own STOP on "off", but a
    following robot must not coast on the app's 1.5 s dead man if it does not."""
    if follow_cmd(viz_dir).get("mode") == "follow":
        navgoal_push(NAV_STOP)
    return follow_leave(viz_dir)


# ---------------------------------------------------------------- sidecar services
# follow-me and the CLIP indexer each hold a CUDA context on the GPU SLAM shares, so they are
# off by default and started on demand. They belong to NOBODY's process: whoever starts one
# (liveviz, the agent, a shell) records its pid in /dev/shm, and anyone can see or stop it.
# Each runs in its own process group so a stop takes its children (the indexer is a shell
# script running two pythons). Only where the image carries the ML-deps layer
# (PERCEPTION_MODELS, docker/okvis_apps.Dockerfile).
# Piped over ssh by --robot, this file runs as `python3 -` and __file__ is "<stdin>"; the
# repo is then where the image keeps it.
REPO = (Path(__file__).resolve().parent.parent if Path(__file__).is_file()
        else Path(os.environ.get("OKVIS_REPO", "/home/OKVIS2-X")))
FOLLOW_SRC = REPO / "app" / "follow_me" / "follow_me.py"
CLIP_SRC = REPO / "app" / "clip_indexer" / "clip_indexer.sh"
CLIP_PROBE = "http://127.0.0.1:8901/embed?text=ping"
# The orchestrator LLM (#619): llama-server, OpenAI-compatible on :8911. Its own binary, not the
# perception layer's python, so it is available wherever llama-server is on PATH.
LLM_SRC = REPO / "app" / "orchestrator" / "llm_server.sh"
LLM_PROBE = "http://127.0.0.1:8911/health"
SERVICES = ("clip", "follow-me", "llm")
# #567 measured 16 s and 12 s on phantom21; llm is a guess (6.6 GB of weights off NVMe), unmeasured.
SVC_UP_S = {"clip": 40.0, "follow-me": 30.0, "llm": 90.0}


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


FOLLOW_RIG = ""


def svc_argv(name: str) -> list[str]:
    if name == "clip":
        return ["bash", str(CLIP_SRC)]
    if name == "follow-me":
        return ["python3", str(FOLLOW_SRC), follow_rig()]
    if name == "llm":
        return ["bash", str(LLM_SRC)]
    raise ValueError(f"unknown service {name!r}")


def svc_available(name: str) -> bool:
    if name == "llm":
        import shutil
        return LLM_SRC.is_file() and shutil.which("llama-server") is not None
    if not os.environ.get("PERCEPTION_MODELS"):
        return False
    if name == "clip":
        return CLIP_SRC.is_file()
    if name == "follow-me":
        return FOLLOW_SRC.is_file() and bool(follow_rig())
    return False


def _pidfile(name: str) -> Path:
    return SHM / f"okvis_svc_{name}.pid"


def svc_pid(name: str) -> int | None:
    """The service's pid if it is alive AND still the process we started. /dev/shm outlives
    the pod, so a stale pid file can name an unrelated host process; the cmdline check
    is what keeps a stop from killing it."""
    try:
        pid = int(_pidfile(name).read_text().strip())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, ValueError):
        return None
    src = {"clip": CLIP_SRC, "follow-me": FOLLOW_SRC, "llm": LLM_SRC}[name]
    return pid if src.name in cmd else None


def svc_running(name: str) -> bool:
    return svc_pid(name) is not None


def svc_up(name: str) -> bool:
    """Liveness beyond "the pid exists": the thing the service is FOR is answering."""
    if name in ("clip", "llm"):
        try:
            return urllib.request.urlopen(CLIP_PROBE if name == "clip" else LLM_PROBE, timeout=2).status < 500
        except Exception:
            return False
    if name == "follow-me":
        age = follow_state().get("age")
        return age is not None and age < 5.0
    return False


def svc_start(name: str, wait_s: float = 0.0) -> dict:
    if name not in SERVICES:
        return {"ok": False, "err": f"unknown service {name!r}"}
    if not svc_available(name):
        if name == "llm":
            return {"ok": False, "err": "no llama-server in this image (needs the ML-deps layer, #619)"}
        return {"ok": False, "err": f"no {name} in this image (needs the perception layer"
                                    + (" and the rig" if name == "follow-me" else "") + ")"}
    pid = svc_pid(name)
    if pid is None:
        log = open(SHM / f"{name}.log", "ab")
        p = subprocess.Popen(svc_argv(name), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=True)
        log.close()
        _pidfile(name).write_text(f"{p.pid}\n")
        pid = p.pid
        # Between fork and exec the child's cmdline is still ours, so svc_pid would not know it
        # yet; wait for the exec (microseconds, but a 1-in-25 test flake without this).
        _wait(lambda: svc_pid(name) is not None, 2.0, period=0.01)
    out = {"ok": True, "svc": name, "pid": pid, "running": True}
    if wait_s > 0:
        up, waited = _wait(lambda: svc_up(name), wait_s, period=0.5)
        out.update(up=bool(up), waited_s=waited)
        if not up:
            out.update(ok=False, err=f"{name} started (pid {pid}) but did not answer within {wait_s:.0f} s;"
                                     f" see {SHM / (name + '.log')}")
    return out


def svc_stop(name: str) -> dict:
    if name not in SERVICES:
        return {"ok": False, "err": f"unknown service {name!r}"}
    pid = svc_pid(name)
    if pid is not None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        gone, _ = _wait(lambda: not svc_running(name), 10.0)
        if not gone:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.waitpid(pid, 0)   # our own child (it is dead or dying): reap it, or it lingers as a zombie
        except ChildProcessError:
            pass                 # someone else's: the kernel reaps it
    _pidfile(name).unlink(missing_ok=True)
    return {"ok": True, "svc": name, "running": False}


def svc_status(name: str) -> dict:
    pid = svc_pid(name)
    return {"available": svc_available(name), "running": pid is not None, "pid": pid,
            "up": svc_up(name) if pid is not None else False}


# ---------------------------------------------------------------- SLAM lifecycle, via the agent
def agent_live() -> bool:
    """Is the console agent (the okvis-agent pod) watching? Without one, asking for a restart
    would stop SLAM and nothing would bring it back -- worse than not offering it."""
    try:
        return time.time() - RESTART_AGENT.stat().st_mtime < RESTART_AGENT_STALE_S
    except OSError:
        return False


def agent_why() -> str:
    """WHY the restart path is unavailable: a STALE heartbeat means the agent pod is behind or
    being replaced; only a MISSING one means nobody is watching."""
    try:
        age = time.time() - RESTART_AGENT.stat().st_mtime
    except OSError:
        return ("no console agent -- the okvis-agent pod is not running "
                "(k0s kubectl -n okvis get pods; runbook §4a)")
    return (f"console agent is alive but {age:.0f}s behind (its pod restarting?); "
            "try again in a moment")


def agent_ask(request: Path, result: Path, line: str, timeout: float) -> dict:
    """Hand one request to the agent and wait for its JSON answer. A request nobody takes
    within 5 s means no agent is running."""
    result.unlink(missing_ok=True)
    request.write_text(line + "\n")
    taken, _ = _wait(lambda: not request.exists(), 5.0)
    if not taken:
        request.unlink(missing_ok=True)
        return {"ok": False, "err": f"nobody took {request.name}: only the k0s agent answers it "
                                    "and none is running (k0s kubectl -n okvis get pods)"}
    done, _ = _wait(lambda: result.exists() and result.read_text().strip(), timeout, period=0.2)
    if not done:
        return {"ok": False, "err": f"no answer to {line!r} within {timeout:.0f} s"}
    try:
        return json.loads(result.read_text())
    except ValueError as e:
        return {"ok": False, "err": f"bad answer from the agent: {e}"}


def svc_here() -> bool:
    """Do the services run HERE? Yes inside the OKVIS pod: when the agent execs us there
    (OKVIS_SVC_HERE, so a request can never bounce back to the agent), and wherever the node's
    pid from the status ring is visible as dma_live -- i.e. its own PID namespace. Anywhere else
    (the VLM agent pod, the host) the agent starts them in the OKVIS pod."""
    if os.environ.get("OKVIS_SVC_HERE"):
        return True
    st = status_read()
    try:
        return bool(st) and b"okvis2x_app_dma_live" in Path(f"/proc/{st['pid']}/cmdline").read_bytes()
    except OSError:
        return False


OKVIS_MODES = ("map", "plan", "drive", "calibrate", "depth")
AUTONOMY_MODES = ("patrol", "explore", "semantic", "follow")
OKVIS_APPS = ("dma_live", "replay")


def slam_restart(mode: str | None = None, autonomy: str | None = None, follow_standoff: float | None = None,
                 app: str | None = None) -> dict:
    """Ask the agent to recreate the OKVIS pod from okvis2x:on-device. The request file carries
    KEY=VALUE overrides, which the agent patches into the ConfigMap okvis-env (they persist)."""
    if not agent_live():
        return {"ok": False, "err": agent_why()}
    if mode is not None and mode not in OKVIS_MODES:
        return {"ok": False, "err": f"mode must be one of {OKVIS_MODES}"}
    if autonomy is not None and autonomy not in AUTONOMY_MODES:
        return {"ok": False, "err": f"autonomy must be one of {AUTONOMY_MODES}"}
    if app is not None and app not in OKVIS_APPS:
        return {"ok": False, "err": f"app must be one of {OKVIS_APPS}"}
    # No OKVIS_APP line = the agent appends OKVIS_APP=dma_live (#648).
    lines = [f"OKVIS_MODE={mode}" if mode else None, f"OKVIS_AUTONOMY_MODE={autonomy}" if autonomy else None,
             f"OKVIS_FOLLOW_STANDOFF_M={follow_standoff}" if follow_standoff is not None else None,
             f"OKVIS_APP={app}" if app else None]
    body = "\n".join(l for l in lines if l) or str(time.time())
    try:
        RESTART_REQUEST.write_text(body + "\n")
    except OSError as e:
        return {"ok": False, "err": f"could not ask for a restart: {e}"}
    return {"ok": True, "requested": "restart", "overrides": [l for l in lines if l]}


def slam_stop() -> dict:
    """Stop the OKVIS pod and LEAVE it stopped (via the agent). Anything running inside it,
    liveviz included, goes down with it."""
    if not agent_live():
        return {"ok": False, "err": agent_why()}
    try:
        STOP_REQUEST.write_text(str(time.time()))
    except OSError as e:
        return {"ok": False, "err": f"could not ask the agent: {e}"}
    return {"ok": True, "requested": "stop"}


def okvis_mode() -> str | None:
    """The running node's OKVIS_MODE: our own env inside the OKVIS pod (the ConfigMap's)."""
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


# ---------------------------------------------------------------- recording, via the agent
# A standalone recorder pod, okvis-record (deployment/k0s/record.yaml: one session.mcap of raw
# sensors + believed-state channels, until stop finalises the file), independent of the SLAM
# process and recording with okvis2x:on-device, i.e. THE BUILD BEING TESTED. The agent creates
# and deletes it and republishes its state to RECORD_STATE.
RECORD_DIR = Path("/data/okvis-recordings")


def _record_files(out: dict) -> dict:
    """Add the session.mcap size and the free disk to a recorder state."""
    if out.get("name"):
        try:
            out["mcap_mb"] = round(os.stat(RECORD_DIR / out["name"] / "session.mcap").st_size / 1e6)
        except OSError:
            out["mcap_mb"] = None
    try:
        sv = os.statvfs(RECORD_DIR)
        out["free_gb"] = round(sv.f_bavail * sv.f_frsize / 1e9, 1)
    except OSError:
        pass
    return out


def record_status() -> dict:
    """The recorder pod as the agent last saw it, plus the file it writes."""
    try:
        st = json.loads(RECORD_STATE.read_text())
    except (OSError, ValueError):
        return {"ok": False, "err": "no recorder state from the agent"}
    return _record_files({"ok": True, "recording": bool(st.get("recording")), "name": st.get("name"),
                          "started_at": st.get("started_at")})


def record_start(name: str | None = None) -> dict:
    name = name or "rec-" + time.strftime("%Y%m%d-%H%M%S")
    if not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
        return {"ok": False, "err": f"bad recording name {name!r}"}
    return agent_ask(RECORD_REQUEST, RECORD_RESULT, f"start {name}", 90.0)


def record_stop() -> dict:
    return agent_ask(RECORD_REQUEST, RECORD_RESULT, "stop", 60.0)


# ---------------------------------------------------------------- semantic query, via liveviz
# The CLIP index (several MB of frame features) lives in liveviz's memory, and loading it per
# call would cost seconds, so this one command asks liveviz. It is always up beside the node.
def _liveviz_get(path: str, timeout: float = 20.0) -> dict:
    with urllib.request.urlopen(LIVEVIZ_URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def semantic_query(text: str, top: int = 5) -> dict:
    """Ranked keyframes for `text`: score, image path, camera pose at that keyframe
    (x y z, quaternion, stamp) and the bearing to the matched grid cell. NOT the object's
    position: the index stores no depth, so a hit is a camera pose plus a ray."""
    try:
        st = _liveviz_get("/semantic/state", timeout=5)
        r = _liveviz_get("/semantic?text=" + urllib.parse.quote_plus(text) + f"&top={int(top)}")
    except Exception as e:
        return {"ok": False, "err": f"liveviz at {LIVEVIZ_URL}: {e}"}
    if not r.get("ok"):
        return r
    images = st.get("images")
    hits = []
    for h in r.get("topk", [])[:top]:
        e = {"score": round(h["score"], 4), "frame": h["i"],
             "image": str(Path(images) / h["img"]) if images else h["img"],
             "pos": [h["x"], h["y"], h["z"]]}
        for k in ("q", "stamp_ns", "bearing", "cell"):
            if k in h:
                e[k] = h[k]
        hits.append(e)
    return {"ok": True, "text": text, "n_frames": r.get("n"), "hits": hits}



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


# --- /okvis_depth_{meta,frames}: metric stereo depth (#510), DepthDMAPublisher.hpp by hand.
DEPTH_META_PATH = Path("/dev/shm/okvis_depth_meta")
DEPTH_POOL_PATH = Path("/dev/shm/okvis_depth_frames")
DEPTH_STRUCT = "<IIqqQIIHHB3x4f16f"
DEPTH_KEYS = ("format_version", "seq", "stamp_ns", "publish_ns", "frame_seq", "pool_slot", "bytes",
              "width", "height", "has_sigma", "fx", "fy", "cx", "cy")
POOL_MAGIC = 0x4652414D45504F4F  # "FRAMEPOO"


def depth_read(meta_path: Path | None = None, pool_path: Path | None = None) -> dict | None:
    """The newest depth frame: {"age_s", "meta", "depth": float32 LE bytes, "sigma": the same or
    None}; None if absent/torn.
    The pool slot is copied between two checks of its sequence against the meta's frame_seq,
    since the producer zeroes it before rewriting the slot."""
    got = ring_read_latest(meta_path or DEPTH_META_PATH, struct.calcsize(DEPTH_STRUCT))
    if got is None:
        return None
    raw, last_write_ns = got
    f = struct.unpack(DEPTH_STRUCT, raw)
    meta = dict(zip(DEPTH_KEYS, f[:14]))
    meta["has_sigma"] = bool(meta["has_sigma"])
    meta["T_SC"] = list(f[14:30])
    n = meta["width"] * meta["height"] * 4
    if meta["format_version"] < 1 or meta["bytes"] != n * (2 if meta["has_sigma"] else 1):
        return None
    try:
        with open(pool_path or DEPTH_POOL_PATH, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
    except (OSError, ValueError):
        return None
    try:
        magic, _ver, num_slots, _max, stride = struct.unpack_from("<5Q", mm, 0)
        slot = meta["pool_slot"]
        if magic != POOL_MAGIC or slot >= num_slots or meta["bytes"] > stride - 64:
            return None
        off = 64 + slot * stride
        s1 = struct.unpack_from("<Q", mm, off)[0]
        data = bytes(mm[off + 64: off + 64 + meta["bytes"]])
        if s1 != meta["frame_seq"] or struct.unpack_from("<Q", mm, off)[0] != s1:
            return None
    except (struct.error, ValueError):
        return None
    finally:
        mm.close()
    return {"age_s": round(max(0.0, (time.monotonic_ns() - last_write_ns) / 1e9), 2),
            "meta": meta, "depth": data[:n], "sigma": data[n:] or None}


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


# --- /route_grid: the 2D traversability map the planner drives on, decoded by interface/route_grid_mcap.
ROUTEGRID_RING_PATH = Path(os.environ.get("OKVIS_ROUTEGRID_SHM", "/dev/shm/route_grid"))


def _route_grid_mcap():
    sys.path.insert(0, str(REPO / "interface"))
    import route_grid_mcap
    return route_grid_mcap


def grid_read(path: Path | None = None):
    """The newest RouteGrid on the ring and its age in seconds, or None. (RouteGrid has
    __slots__, so the age rides beside it.)"""
    rg = _route_grid_mcap()
    for n in (rg.RECORD_SIZE, rg.RECORD_SIZE_V2, rg.RECORD_SIZE_V1):
        got = ring_read_latest(path or ROUTEGRID_RING_PATH, n)
        if got is None:
            continue
        try:
            g = rg.decode(got[0][:n])
        except ValueError:
            continue
        return g, round(max(0.0, (time.monotonic_ns() - got[1]) / 1e9), 2)
    return None


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


# --- look_at.txt: the gaze target. The node points the head at "x y z" (world) while the file
# is younger than 1 s and the robot is idle in a latched mode (semantic/follow); a stopped
# writer releases the head. So a one-shot write is a glance, and --hold rewrites it.
def look_at(x: float, y: float, z: float = 0.0, hold_s: float = 0.0, viz_dir: Path | None = None) -> dict:
    d = VIZ_DIR if viz_dir is None else viz_dir
    p = d / "look_at.txt"
    t0 = time.monotonic()
    n = 0
    while True:
        tmp = p.with_name(".look_at.txt.tmp")
        tmp.write_text("%.4f %.4f %.4f\n" % (x, y, z))
        os.replace(tmp, p)
        n += 1
        if time.monotonic() - t0 >= hold_s:
            break
        time.sleep(0.2)
    return {"ok": True, "look_at": [x, y, z], "writes": n, "held_s": round(time.monotonic() - t0, 1),
            "note": "the head follows only while the robot is idle in semantic/follow mode; the file ages out after 1 s"}


def look_off(viz_dir: Path | None = None) -> dict:
    p = (VIZ_DIR if viz_dir is None else viz_dir) / "look_at.txt"
    p.unlink(missing_ok=True)
    return {"ok": True, "look_at": None}



# ---------------------------------------------------------------- pose goals: goto, and drive to a semantic hit
# The node's pose-goal ingress (SemanticPoseGoal.hpp): a request FILE with one line
#     <ts_ns> <x> <y> <z> <qx> <qy> <qz> <qw> <bearing_deg> [score]
# plus verb SEMANTIC_GOAL to say "a request is waiting"; the ACK comes back on the status ring.
# The goal is the pose itself, facing yaw(q) + camera_yaw_offset + bearing. With ts != 0 the node
# re-resolves the pose from the trajectory as it stands now (loop closure moves old poses), so a
# semantic hit is sent with ITS timestamp; a raw goto sends ts 0 and is driven to as written.
# The robot still only moves while ARMED; committing a goal puts it on the ledger either way.
def goal_file() -> Path:
    """OKVIS_SEMANTIC_GOAL_POSE inside the pod (/output/goal.txt); the same file on the host."""
    return Path(os.environ.get("OKVIS_SEMANTIC_GOAL_POSE") or "/data/okvis/goal.txt")


LIVE_LOGS = (Path("/work/live.log"), Path("/data/okvis-perf/live.log"))


def camera_yaw_offset_deg() -> float | None:
    """The yaw of the semantic camera's optical axis in the body frame, as the node derived it
    from T_SC at startup (its "[semantic-goal] pose-index ingress armed ... cam_yaw_deg=" line).
    Read, never guessed: a wrong constant turns every goal by that constant."""
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
    return None


def goal_request(ts_ns: int, p, q, bearing_deg: float, score: float, comment: str,
                 timeout: float = 2.0) -> dict:
    """Write the request file (tmp+rename: the node polls it and must never see a torn line),
    ring the verb, and return the node's answer."""
    path = goal_file()
    line = "%d %.6f %.6f %.6f %.9f %.9f %.9f %.9f %.3f %.6f\n" % (ts_ns, *p, *q, bearing_deg, score)
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("# " + comment + "\n" + line)
        os.replace(tmp, path)
    except OSError as e:
        return {"ok": False, "err": f"{path}: {e}"}
    r = verb_push(VERB_SEMANTIC_GOAL)
    if not r["ok"]:
        return r
    ack, waited = verb_wait(r["seq"], timeout)
    if ack is None:
        return {**r, "ok": False, "waited_s": waited, "err": f"no ACK for seq {r['seq']} within {timeout} s"}
    return {**r, "ok": ack["ok"], "refused": not ack["ok"], "accepted": ack["ok"], "reason": ack["reason"],
            "waited_s": waited, "request": line.strip(), "file": str(path)}


def goto(x: float, y: float, yaw_deg: float | None, timeout: float = 2.0) -> dict:
    """Drive to (x, y) facing yaw_deg (world frame; omitted = keep the current heading). Written
    with the heading as the pose's quaternion and bearing = -camera_yaw_offset, so the node's
    yaw(q) + offset + bearing comes out as the yaw asked for."""
    import math
    if (e := _need_node()):
        return e
    off = camera_yaw_offset_deg()
    if off is None:
        return {"ok": False, "err": "camera yaw offset not found in the node log (no 'pose-index ingress armed' "
                                    "line): is OKVIS_SEMANTIC_GOAL_POSE set for this run?"}
    if yaw_deg is None:
        got = grid_read()
        if got is None:
            return {"ok": False, "err": "no yaw given and no route grid to take the current heading from"}
        yaw_deg = math.degrees(got[0].robot_yaw)
    st = status_read()
    z = st["pos"][2] if st else 0.0
    h = math.radians(yaw_deg) / 2
    q = (0.0, 0.0, math.sin(h), math.cos(h))
    r = goal_request(0, (x, y, z), q, -off, 0.0, "nav-cli goto x=%.2f y=%.2f yaw=%.1f" % (x, y, yaw_deg), timeout)
    r.update(goal={"x": x, "y": y, "yaw_deg": round(yaw_deg, 1)}, armed=bool(st and st["armed"]))
    if r.get("ok") and not r["armed"]:
        r["note"] = "goal committed; the robot moves only after `nav-cli autonomy on`"
    return r


def semantic_go(hit: dict, text: str, timeout: float = 2.0) -> dict:
    """Drive to where a semantic hit was seen from, facing the matched cell: the hit's own
    timestamp, pose, quaternion and bearing, so the node re-resolves the pose against the
    current trajectory (SemanticPoseGoal P1.3), exactly as liveviz's DRIVE button does."""
    if (e := _need_node()):
        return e
    if "q" not in hit or "stamp_ns" not in hit:
        return {"ok": False, "err": "this hit has no pose quaternion/stamp (flat index); cannot drive to it"}
    r = goal_request(int(hit["stamp_ns"]), hit["pos"], hit["q"], float(hit.get("bearing", 0.0)),
                     float(hit.get("score", 0.0)),
                     "nav-cli semantic goal for %r (frame %s, %s)" % (text, hit.get("frame"), hit.get("image")), timeout)
    st = status_read()
    r.update(hit={k: hit[k] for k in ("frame", "score", "pos", "bearing") if k in hit}, armed=bool(st and st["armed"]))
    return r


# ---------------------------------------------------------------- status, all in one
def status() -> dict:
    st = status_read()
    return {
        "ok": True,
        "node": {"fresh": bool(node_fresh(st)), **(st or {})} if st else {"fresh": False},
        "autonomy": autonomy_get(),
        "okvis_mode": okvis_mode(),
        "follow": follow_cmd(),
        "services": {n: svc_status(n) for n in SERVICES},
        "agent": agent_live(),
        "verbose": verbose_get(),
        "record": record_status(),
    }


# ---------------------------------------------------------------- command line
SCHEMA = [
    {"name": "status", "args": {"--watch": "bool"}, "help": "node, autonomy, modes, services, recorder; one JSON"},
    {"name": "autonomy", "args": {"state": ["on", "off"]}, "help": "arm or disarm; the robot moves only while armed"},
    {"name": "mode", "args": {"mode": ["explore", "semantic", "patrol", "follow"]},
     "help": "goal source: explore = frontier NBV, patrol = straight-line tour, semantic = stop and await a goal, follow = follow-me"},
    {"name": "verb", "args": {"verb": sorted(CLI_VERBS)}, "help": "a command the node may refuse; waits for its answer"},
    {"name": "svc", "args": {"action": ["start", "stop", "status"], "svc": list(SERVICES)},
     "help": "sidecar services: clip = semantic indexer + text encoder, follow-me = person follower, llm = orchestrator llama-server :8911"},
    {"name": "follow", "args": {"action": ["prompt", "select", "off"], "value": "TEXT | X Y"},
     "help": "follow-me target: prompt = what to look for (no TEXT = blank; outside follow mode only shown, entering follow mode fills a blank one with person), select = pick the match at image fraction X Y in any mode (default in follow mode: the nearest), off = leave follow mode"},
    {"name": "slam", "args": {"action": ["restart", "stop"], "--mode": list(OKVIS_MODES), "--autonomy": list(AUTONOMY_MODES),
                              "--follow-standoff": "METRES (follow-me hold distance, default 0.8)", "--app": list(OKVIS_APPS)},
     "help": "restart recreates the node from the newest on-device image (map and trajectory are lost); stop leaves it stopped"},
    {"name": "record", "args": {"action": ["start", "stop", "status"]}, "help": "one session.mcap per start..stop"},
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
     "help": "point the head at a world point (idle in semantic/follow mode only); --hold keeps it there"},
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
        return {"ok": False, "err": f"{STATUS_RING_PATH}: no status ring (node never ran here)"}
    return {"ok": False, "err": f"{STATUS_RING_PATH}: node not fresh (age {st['age_s']} s)"}


def cmd_autonomy(a) -> dict:
    if (e := _need_node()):
        return e
    r = autonomy_set(a.state == "on")
    if not r["ok"]:
        return r
    want = a.state == "on"
    got, waited = _wait(lambda: (s := node_fresh()) and s["armed"] == want and s, a.timeout)
    return {**r, "armed": bool(got and got["armed"]), "waited_s": waited} if got else \
        {"ok": False, "err": f"gate set but status.armed did not follow within {a.timeout} s", "waited_s": waited}


def cmd_mode(a) -> dict:
    if (e := _need_node()):
        return e
    if a.mode == "follow":
        return follow_enter()
    follow_off()   # any other mode ends follow-me
    r = navgoal_push(NAV_MODES[a.mode])
    if not r["ok"]:
        return r
    # The node echoes the mode as a [nav-cmd] log line and in status.semantic for STOP; the
    # ring's own head advancing past our seq is what says it was consumed.
    want_sem = a.mode == "semantic"
    got, waited = _wait(lambda: (s := node_fresh()) and s["semantic"] == want_sem and s, a.timeout)
    return {**r, "mode": a.mode, "waited_s": waited} if got else \
        {**r, "ok": False, "mode": a.mode, "waited_s": waited,
         "err": f"sent, but status did not reflect {a.mode} within {a.timeout} s"}


def cmd_verb(a) -> dict:
    if (e := _need_node()):
        return e
    r = verb_push(CLI_VERBS[a.verb])
    if not r["ok"]:
        return r
    ack, waited = verb_wait(r["seq"], a.timeout)
    if ack is None:
        return {**r, "ok": False, "waited_s": waited, "err": f"no ACK for seq {r['seq']} within {a.timeout} s"}
    return {**r, "ok": ack["ok"], "refused": not ack["ok"], "accepted": ack["ok"],
            "reason": ack["reason"], "waited_s": waited}


def cmd_svc(a) -> dict:
    if not svc_here():
        wait = a.timeout if a.timeout is not None else SVC_UP_S[a.svc]
        return agent_ask(SVC_REQUEST, SVC_RESULT, f"{a.action} {a.svc}", wait + 15)
    if a.action == "start":
        r = svc_start(a.svc, wait_s=a.timeout if a.timeout is not None else SVC_UP_S[a.svc])
        if a.svc == "follow-me" and r["ok"]:
            follow_enter()   # follow-me on = autonomy mode follow, as liveviz's FOLLOW mode button does
        return r
    if a.action == "stop":
        if a.svc == "follow-me":
            follow_off()
        return svc_stop(a.svc)
    return {"ok": True, "svc": a.svc, **svc_status(a.svc)}


def cmd_follow(a) -> dict:
    v = a.value
    if a.action == "off":
        return {"ok": True, "follow": follow_off()}
    if a.action == "prompt":
        c = follow_cmd(prompt=" ".join(v)[:80])   # perception only: matches shown, no mode change
        follow_sync()
        return {"ok": True, "follow": c}
    if a.action == "select":
        try:
            x, y = float(v[0]), float(v[1])
        except (IndexError, ValueError):
            return {"ok": False, "err": "select needs X Y as image fractions in [0,1]"}
        return {"ok": True, "follow": follow_cmd(click=[x, y])}   # picks a target; never the mode
    return {"ok": False, "err": "unknown follow action"}


def cmd_slam(a) -> dict:
    return slam_restart(a.mode, a.autonomy, a.follow_standoff, a.app) if a.action == "restart" else slam_stop()


def cmd_record(a) -> dict:
    return {"start": record_start, "stop": record_stop, "status": record_status}[a.action]()


def cmd_semantic(a) -> dict:
    r = semantic_query(" ".join(a.text), max(a.top, (a.go or 0) + 1))
    if r.get("ok") and a.go is not None:
        if a.go >= len(r["hits"]):
            return {"ok": False, "err": f"only {len(r['hits'])} hits, no hit {a.go}", "hits": r["hits"]}
        g = semantic_go(r["hits"][a.go], " ".join(a.text), a.timeout)
        return {**g, "hits": r["hits"]}
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
        return {"ok": False, "err": f"{DEPTH_META_PATH}: no depth frame (ring absent, torn, or never written)"}
    r = {"ok": True, "age_s": d["age_s"], "stamp_ns": d["meta"]["stamp_ns"],
         "size": [d["meta"]["width"], d["meta"]["height"]], **depth_sectors(d, a.sectors)}
    if a.out:
        r.update(depth_write_pgm(d, a.out))
    return r


def cmd_grid(a) -> dict:
    got = grid_read()
    if got is None:
        return {"ok": False, "err": f"{ROUTEGRID_RING_PATH}: no route grid (ring absent or never written)"}
    return {"ok": True, **grid_summary(got[0], a.radius, a.ascii, got[1])}


def cmd_goto(a) -> dict:
    return goto(a.x, a.y, a.yaw_deg, a.timeout)


def cmd_look(a) -> dict:
    if a.xyz and a.xyz[0] == "off":
        return look_off()
    try:
        x, y = float(a.xyz[0]), float(a.xyz[1])
        z = float(a.xyz[2]) if len(a.xyz) > 2 else 0.0
    except (IndexError, ValueError):
        return {"ok": False, "err": "look needs X Y [Z] in the world frame, or off"}
    return look_at(x, y, z, a.hold)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nav-cli", description=__doc__.split("\n", 1)[0])
    ap.add_argument("--robot", metavar="HOST", help="run on that robot over ssh (inside its OKVIS pod)")
    ap.add_argument("--schema", action="store_true", help="print the command tree as JSON tool definitions")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--timeout", type=float, default=None,
                        help="seconds to wait for the node's answer (default 2; svc start: per service)")
    sub = ap.add_subparsers(dest="cmd")
    add = lambda name: sub.add_parser(name, parents=[common])
    p = add("status"); p.add_argument("--watch", action="store_true")
    p = add("autonomy"); p.add_argument("state", choices=["on", "off"])
    p = add("mode"); p.add_argument("mode", choices=["explore", "semantic", "patrol", "follow"])
    p = add("verb"); p.add_argument("verb", choices=sorted(CLI_VERBS))
    p = add("svc"); p.add_argument("action", choices=["start", "stop", "status"]); p.add_argument("svc", choices=SERVICES)
    p = add("follow"); p.add_argument("action", choices=["prompt", "select", "off"]); p.add_argument("value", nargs="*")
    p = add("slam"); p.add_argument("action", choices=["restart", "stop"])
    p.add_argument("--mode", choices=OKVIS_MODES); p.add_argument("--autonomy", choices=AUTONOMY_MODES)
    p.add_argument("--follow-standoff", type=float); p.add_argument("--app", choices=OKVIS_APPS)
    p = add("record"); p.add_argument("action", choices=["start", "stop", "status"])
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


def run_remote(host: str, argv: list[str]) -> int:
    """Run THIS file on the robot: inside the OKVIS pod when it is up (the services and their
    paths live there), else on the host (the rings and request files are the same /dev/shm, so
    `slam restart` still works). stdout and the exit code come straight back."""
    args = " ".join(shlex.quote(a) for a in argv)
    script = (f"if k0s kubectl -n okvis get pod okvis2x -o jsonpath='{{.status.phase}}' 2>/dev/null | grep -q Running; "
              f"then exec k0s kubectl -n okvis exec -i okvis2x -- python3 - {args}; "
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
    if a.timeout is None and a.cmd != "svc":
        a.timeout = 2.0
    handler = {"status": cmd_status, "autonomy": cmd_autonomy, "mode": cmd_mode, "verb": cmd_verb,
               "svc": cmd_svc, "follow": cmd_follow, "slam": cmd_slam, "record": cmd_record,
               "semantic": cmd_semantic, "verbose": cmd_verbose, "pose": cmd_pose, "frame": cmd_frame,
               "depth": cmd_depth, "grid": cmd_grid, "look": cmd_look, "goto": cmd_goto}[a.cmd]
    return _emit(handler(a))


if __name__ == "__main__":
    sys.exit(main())
