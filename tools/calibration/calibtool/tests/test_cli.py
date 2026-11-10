"""calibtool.cli: `calib <robot>` on the laptop: copies the kit to the robot, starts calibd there (niced, at most 2 BLAS
threads, the profile's python and dma_ipc folder) or re-attaches to the running session, stops it, and runs the TUI
over the ssh attach link. ssh, rsync and the speaker are injected (tests use fakes)."""
import json
import shlex

import pytest

pytest.importorskip("yaml")

from calibtool import cli, profile  # noqa: E402


def P():
    return profile.load("P10")


def test_remote_command_line():
    cmd = cli.remote(P(), "start", "--trial")
    w = shlex.split(cmd)
    assert w[:2] == ["cd", "/tmp/calibtool-m2/kit/tools/calibration"]
    assert "OMP_NUM_THREADS=2" in cmd and "OPENBLAS_NUM_THREADS=2" in cmd
    assert "PYTHONPATH=/tmp/calibtool-m2/kit/tools/calibration:/root/p11-cal/pydeps" in cmd
    assert ("nice -n 10 /root/p11-cal/venv/bin/python -m calibtool.daemon start --root /tmp/calibtool-m2 "
            "--profile P10 --trial") in cmd


def test_ssh_argv_keeps_the_link_alive_check_short():
    a = cli.ssh_argv(P(), "echo hi")
    assert a[0] == "ssh" and a[-2:] == ["root@phantom10", "echo hi"]
    assert "ServerAliveInterval=2" in a and "BatchMode=yes" in a


def test_bootstrap_rsyncs_code_only():
    calls = []
    cli.bootstrap(P(), run=lambda argv, **kw: calls.append(argv) or type("R", (), {"returncode": 0})())
    assert calls[0][:3] == ["ssh", "-o", "BatchMode=yes"] and "mkdir -p /tmp/calibtool-m2/kit/tools/calibration" in \
        calls[0][-1]
    r = calls[1]
    assert r[0] == "rsync" and r[-1] == "root@phantom10:/tmp/calibtool-m2/kit/tools/calibration/"
    for d in ("calibtool", "scripts", "config"):
        assert any(x.endswith(f"tools/calibration/{d}") for x in r), d
    assert "--exclude=__pycache__" in r and "--exclude=tests" in r


class FakeRun:
    def __init__(self, outs):
        self.outs, self.calls = list(outs), []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        out = self.outs.pop(0) if self.outs else ""
        return type("R", (), {"returncode": 0, "stdout": out, "stderr": ""})()


def test_start_returns_the_session():
    r = FakeRun([json.dumps({"session": "S1", "pid": 42, "sock": "/x"})])
    assert cli.start(P(), trial=True, run=r) == {"session": "S1", "pid": 42, "sock": "/x"}
    assert "daemon start" in r.calls[0][-1] and "--trial" in r.calls[0][-1]


K = "kit-here"


def _sa(r, session="S1", trial=False, replace=False):
    synced = []
    sid = cli.start_or_attach(P(), session=session, trial=trial, run=r, kit=K, replace=replace,
                              sync=lambda: synced.append(len(r.calls)))
    return sid, synced


def test_existing_session_is_attached_not_restarted():
    r = FakeRun([json.dumps({"phase": "HOLD", "pid": 42, "session": "S1", "live": True, "kit": K})])
    sid, synced = _sa(r)
    assert sid == "S1" and synced == []
    assert len(r.calls) == 1 and "daemon status" in r.calls[0][-1]


def test_dead_session_is_restarted_under_its_id():
    """liveness is the session's socket (review 6: kill -0 on a stale pid could hit another process)"""
    r = FakeRun([json.dumps({"phase": "TRIAL", "pid": 42, "session": "S1", "live": False}),
                 json.dumps({"session": "S1", "pid": 43})])
    sid, synced = _sa(r, trial=True)
    assert sid == "S1" and "daemon start" in r.calls[1][-1] and "--session S1" in r.calls[1][-1]
    assert synced == [1]                        # the kit is copied after the check, before the start


def test_a_live_daemon_running_other_code_is_refused():
    """review 6: bootstrap rsynced new code and the CLI attached to the old daemon still running the old code"""
    r = FakeRun([json.dumps({"phase": "TRIAL", "pid": 42, "session": "S1", "live": True, "kit": "kit-old"})])
    with pytest.raises(RuntimeError, match="--replace") as e:
        _sa(r)
    assert "kit-old" in str(e.value) and len(r.calls) == 1


def test_replace_stops_the_old_daemon_then_copies_and_starts():
    r = FakeRun([json.dumps({"phase": "TRIAL", "pid": 42, "session": "S1", "live": True, "kit": "kit-old"}),
                 json.dumps({"session": "S1", "stopped": True}), json.dumps({"session": "S1", "pid": 43})])
    sid, synced = _sa(r, replace=True)
    assert sid == "S1" and synced == [2]
    assert "daemon stop" in r.calls[1][-1] and "--session S1" in r.calls[1][-1] and "daemon start" in r.calls[2][-1]


def test_a_new_session_while_another_runs_is_refused_with_the_way_out():
    r = FakeRun([json.dumps({"phase": "TRIAL", "pid": 42, "session": "S0", "live": True, "kit": K})])
    with pytest.raises(RuntimeError, match="--session S0") as e:
        _sa(r, session=None)
    assert "--replace" in str(e.value)


def test_a_failed_start_is_an_error():
    r = FakeRun([json.dumps({"live": False}), json.dumps({"session": "S1", "pid": 43, "error": "calibd exited rc 3"})])
    with pytest.raises(RuntimeError, match="rc 3"):
        _sa(r)


def test_stop_command():
    r = FakeRun([json.dumps({"session": "S1", "stopped": True, "pid": 42})])
    assert cli.stop(P(), "S1", run=r)["stopped"]
    assert "daemon stop" in r.calls[0][-1] and "--session S1" in r.calls[0][-1]


class FakeLink:
    def __init__(self, lines):
        self._lines, self.sent, self.closed = list(lines), [], False

    def lines(self):
        yield from self._lines

    def send(self, s):
        self.sent.append(s)

    def close(self):
        self.closed = True


def test_headless_renders_n_states_heartbeats_and_closes():
    st = {"session": "S1", "robot": "P10", "phase": "READY", "banner": [], "checks": [], "capture": {}, "say": None,
          "updated": 1.0}
    link = FakeLink([json.dumps(st)] * 5)
    out = []
    n = cli.run_tui(link, speaker=None, headless=3, out=out.append)
    assert n == 3 and len(out) == 3 and link.closed
    assert any('"hb"' in s for s in link.sent)
