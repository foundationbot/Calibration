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


class Run:
    """a subprocess stand-in: argv[0..1] -> (rc, out)"""

    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, argv, timeout=5):
        self.calls.append(list(argv))
        for k, v in self.table.items():
            if tuple(argv[:len(k)]) == k:
                if isinstance(v, Exception):
                    raise v
                return v
        return 127, "not found"


def good_run():
    return Run({("systemctl", "is-active"): (0, "active\n"), ("dma_cmd_client", "-V"): (0, VERSION),
                ("dma_cmd_client", "ping"): (0, PING),
                ("dma_cmd_client", "estop-status"): (0, "Emergency stop: false\n")})


def test_parse_ping():
    assert mo.parse_ping(PING) == {"uptime_s": 6065.3, "motor_state": 0}
    assert mo.parse_ping("garbage") is None


def test_parse_version():
    assert mo.parse_version(VERSION) == "2.1.1"
    assert mo.parse_version("") is None


def test_master_status_ok():
    s = mo.master(good_run())
    assert (s.active, s.version, s.uptime_s, s.motor_state, s.estop, s.error) == (True, "2.1.1", 6065.3, 0, False, None)
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


def test_motor_tool_md5_and_targets(tmp_path):
    tool = tmp_path / "dma_motor_motion"
    tool.write_bytes(b"binary")
    r = Run({(str(tool), "--help"): (0, "usage ... signals: targets position\n")})
    t = mo.motor_tool(tool, r)
    assert t == {"path": str(tool), "md5": hashlib.md5(b"binary").hexdigest(), "has_targets": True, "error": None}
    r = Run({(str(tool), "--help"): (0, "usage ... position\n")})
    assert mo.motor_tool(tool, r)["has_targets"] is False


def test_motor_tool_missing(tmp_path):
    t = mo.motor_tool(tmp_path / "nope", Run({}))
    assert t["md5"] is None and t["error"]


def test_no_enable_or_disable_anywhere_in_m2():
    src = Path(mo.__file__).read_text()
    code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
    assert not re.search(r"enable-motors|disable-motors|estop-clear", code)
    assert not re.search(r'"dma_cmd_client",\s*"estop"\s*\]', code)
