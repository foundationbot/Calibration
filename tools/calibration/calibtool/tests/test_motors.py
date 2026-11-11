"""calibtool.motors in M2: READ-ONLY status checks of the motor side (design section 7): the EtherCAT master service
and its version, `dma_cmd_client ping` (motor_state), /dev/shm/desired still (nobody commanding), the motor tool's md5
and its 'targets' signal. Enabling motors is M5; nothing here may call enable/disable."""
import hashlib
import re
import struct
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from calibtool import motors as mo  # noqa: E402

PING = "  PONG  uptime=6065.3s  motor_state=0\n"
VERSION = "dma_cmd_client 2.1.1 (build V-2-1-1-20261023-025557)\n"


class OneBehindPing:
    """dma_cmd_client ping as the 2.1.1 master answers (10 Nov, phantom10): each call prints the reply to the
    PREVIOUS request; the first one blank (uptime 0.0, motor_state 0)"""

    def __init__(self, motor_state=0, t0=6065.3):
        self.motor_state, self.t, self.prev = motor_state, t0, None

    def __call__(self):
        out = self.prev or "  PONG  uptime=0.0s  motor_state=0\n"
        self.t += 0.5
        self.prev = f"  PONG  uptime={self.t:.1f}s  motor_state={self.motor_state}\n"
        return 0, out


class Run:
    """a subprocess stand-in: argv[0..1] -> (rc, out) (or a callable giving it)"""

    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, argv, timeout=5):
        self.calls.append(list(argv))
        for k, v in self.table.items():
            if tuple(argv[:len(k)]) == k:
                if isinstance(v, Exception):
                    raise v
                return v() if callable(v) else v
        return 127, "not found"


def good_run():
    return Run({("systemctl", "is-active"): (0, "active\n"), ("dma_cmd_client", "-V"): (0, VERSION),
                ("dma_cmd_client", "ping"): OneBehindPing(),
                ("dma_cmd_client", "estop-status"): (0, "Emergency stop: false\n")})


def test_parse_ping():
    assert mo.parse_ping(PING) == {"uptime_s": 6065.3, "motor_state": 0}
    assert mo.parse_ping("garbage") is None


def test_parse_version():
    assert mo.parse_version(VERSION) == "2.1.1"
    assert mo.parse_version("") is None


def test_master_status_ok():
    s = mo.master(good_run())
    assert (s.active, s.version, s.motor_state, s.estop, s.error) == (True, "2.1.1", 0, False, None)
    assert s.uptime_s > 6065.3
    assert s.motors_on is False


def test_master_inactive():
    r = good_run()
    r.table[("systemctl", "is-active")] = (3, "inactive\n")
    s = mo.master(r)
    assert not s.active and "inactive" in s.error


def test_ping_timeout_is_an_error_not_a_crash():
    import subprocess
    r = good_run()
    r.table[("dma_cmd_client", "ping")] = subprocess.TimeoutExpired("dma_cmd_client", 5)
    s = mo.master(r)
    assert s.error and "ping" in s.error and s.motor_state is None


def _ring(path, head):
    b = bytearray(128 + 64)
    b[:8] = b"UEUQ_AMD"
    struct.pack_into("<QQ", b, 16, 64, 1)
    struct.pack_into("<Q", b, 64, head)
    Path(path).write_bytes(bytes(b))


def test_ring_head(tmp_path):
    _ring(tmp_path / "desired", 7)
    assert mo.ring_head(tmp_path / "desired") == 7


def test_desired_still_and_advancing():
    heads = iter([5, 5])
    assert mo.desired_still(lambda: next(heads), sleep=lambda s: None, seconds=1.0) is True
    heads = iter([5, 6])
    assert mo.desired_still(lambda: next(heads), sleep=lambda s: None, seconds=1.0) is False


def test_motor_tool_md5_and_targets_from_its_strings_never_run(tmp_path):
    """review 7: the check ran `dma_motor_motion --help`, i.e. executed a motion tool; it reads the file's strings"""
    marker = tmp_path / "ran"
    tool = tmp_path / "dma_motor_motion"
    body = f"#!/bin/sh\ntouch {marker}\n# \x00usage: signals: targets position\x00\n".encode()
    tool.write_bytes(body)
    tool.chmod(0o755)
    r = Run({})
    t = mo.motor_tool(tool, r)
    assert t == {"path": str(tool), "md5": hashlib.md5(body).hexdigest(), "has_targets": True, "error": None}
    assert r.calls == [] and not marker.exists()
    tool.write_bytes(b"\x7fELF usage ... position\x00")
    assert mo.motor_tool(tool, r)["has_targets"] is False
    assert r.calls == []


def test_motor_tool_missing(tmp_path):
    t = mo.motor_tool(tmp_path / "nope", Run({}))
    assert t["md5"] is None and t["error"]


def test_no_enable_or_disable_anywhere_in_m2():
    src = Path(mo.__file__).read_text()
    code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
    assert not re.search(r"enable-motors|disable-motors|estop-clear", code)
    assert not re.search(r'"dma_cmd_client",\s*"estop"\s*\]', code)



def test_ping_reads_past_the_blank_and_the_stale_reply():
    """10 Nov, phantom10: one ping printed 'uptime=0.0s motor_state=0' with the motors on, the next ones repeated an
    old uptime; motors.ping asks until a reply follows a fresher one"""
    p = OneBehindPing(motor_state=2)
    run = Run({("dma_cmd_client", "ping"): p})
    got = mo.ping(run, sleep=lambda s: None)
    assert got["motor_state"] == 2 and got["uptime_s"] > 6065.3
    assert len(run.calls) == 3                           # blank, first real reply, a fresher one


def test_ping_gives_none_when_the_replies_never_freshen():
    run = Run({("dma_cmd_client", "ping"): (0, "  PONG  uptime=450.9s  motor_state=2\n")})
    assert mo.ping(run, sleep=lambda s: None) is None
