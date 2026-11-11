"""Motor-side status, READ-ONLY in M2 (design section 7; enabling is M5 and lives elsewhere).

master(run): the EtherCAT master service (`systemctl is-active dma-ethercat`), its version (`dma_cmd_client -V`),
`dma_cmd_client ping` (uptime, motor_state: 0 = motors off) and `estop-status`. desired_still: /dev/shm/desired's head
counter unchanged over a second (nothing is commanding the motors). motor_tool: md5 and whether the binary's strings
name the 'targets' signal (read, never executed) (step 08's refusal: a tool without it leaves the spine and legs at zero stiffness during a neck move).
`dma_cmd_client state` prints zeros on the 2.1.x master, so it is not used. run(argv, timeout) -> (rc, text) is
injected; run_cmd() is the subprocess one."""
import hashlib
import re
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

SERVICE = "dma-ethercat"
_PING = re.compile(r"PONG\s+uptime=([\d.]+)s\s+motor_state=(\d+)")
_VER = re.compile(r"dma_cmd_client\s+(\d+\.\d+\.\d+)")
HEAD_OFFSET = 64


def run_cmd(argv, timeout=5):
    r = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def parse_ping(text):
    m = _PING.search(text or "")
    return None if m is None else {"uptime_s": float(m.group(1)), "motor_state": int(m.group(2))}


def parse_version(text):
    m = _VER.search(text or "")
    return None if m is None else m.group(1)


@dataclass
class MasterStatus:
    active: bool
    version: object
    uptime_s: object
    motor_state: object
    estop: object
    error: object

    @property
    def motors_on(self):
        return None if self.motor_state is None else self.motor_state != 0

    def as_dict(self):
        return {"active": self.active, "version": self.version, "uptime_s": self.uptime_s,
                "motor_state": self.motor_state, "motors_on": self.motors_on, "estop": self.estop, "error": self.error}


def _try(run, argv):
    try:
        return run(argv, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"{type(e).__name__}"


PING_GAP_S = 0.25
PING_TRIES = 4


def ping(run=run_cmd, sleep=time.sleep, tries=PING_TRIES, gap_s=PING_GAP_S):
    """the master's ping, fresh. 10 Nov, phantom10 (dma_cmd_client 2.1.1): each `dma_cmd_client ping` prints the
    reply to the PREVIOUS request -- a blank one (uptime 0.0, motor_state 0) when none came since the master started,
    else an old one (uptime 450.9 s repeated while the master ran on). A single ping is no reading: this pings until a
    reply with uptime > 0 follows another whose uptime it exceeds (the reply to a request made gap_s earlier) and
    returns it; None when no two such replies came in `tries`"""
    prev = None
    for i in range(tries):
        if i:
            sleep(gap_s)
        rc, out = _try(run, ["dma_cmd_client", "ping"])
        p = parse_ping(out) if rc == 0 else None
        if p is None or p["uptime_s"] <= 0:
            continue
        if prev is not None and p["uptime_s"] > prev["uptime_s"]:
            return p
        prev = p
    return None


def estop(run=run_cmd, sleep=time.sleep, gap_s=PING_GAP_S):
    """estop-status read twice (the reply is the previous request's, like ping), the second answer"""
    val = None
    for i in range(2):
        if i:
            sleep(gap_s)
        rc, out = _try(run, ["dma_cmd_client", "estop-status"])
        m = re.search(r"(true|false)", out or "", re.I) if rc == 0 else None
        val = None if m is None else m.group(1).lower() == "true"
    return val


def master(run=run_cmd, sleep=time.sleep) -> MasterStatus:
    errs = []
    rc, out = _try(run, ["systemctl", "is-active", SERVICE])
    active = rc == 0 and (out or "").strip() == "active"
    if not active:
        errs.append(f"{SERVICE} is {(out or '').strip() or 'not answering'}")
    rc, out = _try(run, ["dma_cmd_client", "-V"])
    version = parse_version(out) if rc == 0 else None
    p = ping(run, sleep) if active else None
    if p is None:
        errs.append("dma_cmd_client ping gave no fresh reply")
    e = estop(run, sleep) if active else None
    return MasterStatus(active, version, p and p["uptime_s"], p and p["motor_state"], e, "; ".join(errs) or None)


def ring_head(path) -> int:
    """the head counter of a /dev/shm ring (UEUQ header: <Q at byte 64)"""
    with open(path, "rb") as f:
        b = f.read(HEAD_OFFSET + 8)
    return struct.unpack_from("<Q", b, HEAD_OFFSET)[0]


def desired_still(read_head, sleep=time.sleep, seconds=1.0) -> bool:
    """True when /dev/shm/desired's head did not move over `seconds` (nothing commanding the motors)"""
    h0 = read_head()
    sleep(seconds)
    return read_head() == h0


def motor_tool(path, run=None) -> dict:
    """md5 and whether the tool's strings name the 'targets' signal; the tool is only read, never run (review 7: a
    `--help` run is still running a motion tool). run is unused (kept for the callers' signature)"""
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "md5": None, "has_targets": False, "error": f"{p} not found"}
    data = p.read_bytes()
    return {"path": str(p), "md5": hashlib.md5(data).hexdigest(), "has_targets": b"targets" in data, "error": None}
