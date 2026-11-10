"""calibtool.daemon (calibd): the session state machine with state.json + events.jsonl after every step, the TUI
heartbeat deadman (5 s without a heartbeat -> HOLD; only key G with a TUI attached leaves HOLD; nothing resumes by
itself), abort from any phase, resume from disk after a crash or reboot (an active phase comes back as HOLD), the
Unix-socket link a TUI attaches to over ssh, and the one-job-per-robot lock."""
import json
import os
import socket
import threading
import time
from concurrent.futures import Future

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

from calibtool import capture as cp, daemon as dm, detect, place, preflight as pf  # noqa: E402


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def machine(tmp_path, clock=None, **kw):
    st = dm.Store(tmp_path / "s1")
    return dm.Machine(st, clock or Clock(), session="s1", robot="P10", **kw)


def kinds(store):
    return [e["kind"] for e in store.events()]


# ------------------------------------------------------------------ Store

def test_store_writes_state_atomically(tmp_path):
    s = dm.Store(tmp_path / "s")
    s.write_state({"phase": "READY", "n": 1})
    (s.dir / "state.json.tmp").write_text('{"phase": "TRU')           # a crash mid-write leaves only the tmp file
    assert s.load_state() == {"phase": "READY", "n": 1}
    s.write_state({"phase": "TRIAL", "n": 2})
    assert json.loads((s.dir / "state.json").read_text())["phase"] == "TRIAL"


def test_store_events_append_in_order(tmp_path):
    s = dm.Store(tmp_path / "s")
    s.event("a", t=1.0, x=1)
    s.event("b", t=2.0)
    lines = (s.dir / "events.jsonl").read_text().splitlines()
    assert [json.loads(x)["kind"] for x in lines] == ["a", "b"]
    assert json.loads(lines[0]) == {"t": 1.0, "kind": "a", "x": 1}
    assert [e["kind"] for e in s.events()] == ["a", "b"]


def test_store_survives_a_torn_last_event_line(tmp_path):
    s = dm.Store(tmp_path / "s")
    s.event("a", t=1.0)
    with open(s.dir / "events.jsonl", "a") as f:
        f.write('{"t": 2.0, "ki')
    assert [e["kind"] for e in s.events()] == ["a"]


# ------------------------------------------------------------------ Machine: heartbeat, keys, abort

def test_starts_in_preflight_and_persists(tmp_path):
    m = machine(tmp_path)
    assert m.phase == "PREFLIGHT"
    assert m.store.load_state()["phase"] == "PREFLIGHT"
    assert kinds(m.store)[0] == "session_start"


def test_heartbeat_deadman_holds_after_5_s(tmp_path):
    c = Clock()
    m = machine(tmp_path, c)
    m.to("READY", "preflight green")
    m.heartbeat()
    m.key("T")
    assert m.phase == "TRIAL"
    c.t += 4.9
    m.tick()
    assert m.phase == "TRIAL"
    c.t += 0.2
    m.tick()
    assert m.phase == "HOLD" and m.state["held_from"] == "TRIAL"
    assert "heartbeat_lost" in kinds(m.store)
    assert m.store.load_state()["phase"] == "HOLD"


def test_heartbeat_back_does_not_resume_by_itself(tmp_path):
    c = Clock()
    m = machine(tmp_path, c)
    m.to("READY", "x")
    m.heartbeat()
    m.key("T")
    c.t += 6
    m.tick()
    m.heartbeat()
    m.tick()
    assert m.phase == "HOLD"
    m.key("G")
    assert m.phase == "TRIAL"
    assert "resumed_by_operator" in kinds(m.store)


def test_g_without_a_live_tui_is_ignored(tmp_path):
    c = Clock()
    m = machine(tmp_path, c)
    m.to("READY", "x")
    m.heartbeat()
    m.key("T")
    c.t += 6
    m.tick()
    m.key("G")                                  # heartbeat is 6 s old: no TUI attached
    assert m.phase == "HOLD"


def test_abort_from_any_phase(tmp_path):
    for start in ("PREFLIGHT", "READY", "TRIAL", "HOLD"):
        m = machine(tmp_path / start)
        if start != "PREFLIGHT":
            m.to(start, "test")
        m.key("A")
        assert m.phase == "ABORTED" and m.state["status"] == "aborted"
        m.key("G")
        m.key("T")
        assert m.phase == "ABORTED"


def test_no_heartbeat_while_idle_is_not_a_hold(tmp_path):
    c = Clock()
    m = machine(tmp_path, c)
    m.to("READY", "x")
    c.t += 60
    m.tick()
    assert m.phase == "READY"
    assert m.state["attached"] is False


def test_keys_are_logged(tmp_path):
    m = machine(tmp_path)
    m.key("x")
    assert any(e["kind"] == "key" and e["key"] == "X" for e in m.store.events())


# ------------------------------------------------------------------ resume

def test_resume_brings_an_active_phase_back_as_hold(tmp_path):
    c = Clock()
    m = machine(tmp_path, c)
    m.to("READY", "x")
    m.heartbeat()
    m.key("T")
    m.state["trial"] = {"index": 3, "of": 6}
    m.save()
    del m                                       # the daemon dies (crash, reboot)
    c.t += 100
    m2 = dm.Machine.resume(dm.Store(tmp_path / "s1"), c)
    assert m2.phase == "HOLD" and m2.state["held_from"] == "TRIAL"
    assert m2.state["trial"] == {"index": 3, "of": 6}
    assert m2.state["session"] == "s1"
    assert kinds(m2.store)[-1] == "resumed"
    assert kinds(m2.store).count("session_start") == 1


def test_resume_of_a_finished_session_stays_finished(tmp_path):
    m = machine(tmp_path)
    m.key("A")
    m2 = dm.Machine.resume(dm.Store(tmp_path / "s1"), Clock())
    assert m2.phase == "ABORTED"


# ------------------------------------------------------------------ the socket link

def _client(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(path))
    s.settimeout(3.0)
    return s, s.makefile("r")


def test_socket_round_trip_hold_and_reattach(tmp_path):
    c = Clock()
    m = machine(tmp_path, c)
    m.to("READY", "x")
    sock = tmp_path / "calibd.sock"
    srv = dm.Server(sock, m, period_s=0.05)
    srv.start()
    try:
        s, f = _client(sock)
        st = json.loads(f.readline())
        assert st["phase"] == "READY" and st["session"] == "s1"
        s.sendall(b'{"hb": 1}\n{"key": "T"}\n')
        for _ in range(100):
            if m.phase == "TRIAL":
                break
            time.sleep(0.02)
        assert m.phase == "TRIAL" and m.state["attached"]
        s.close()                               # the link drops (ssh killed)
        c.t += 6
        m.tick()
        assert m.phase == "HOLD"
        s2, f2 = _client(sock)                  # a new TUI attaches
        assert json.loads(f2.readline())["phase"] == "HOLD"
        s2.sendall(b'{"hb": 1}\n{"key": "G"}\n')
        for _ in range(100):
            if m.phase == "TRIAL":
                break
            time.sleep(0.02)
        assert m.phase == "TRIAL"
        s2.close()
    finally:
        srv.stop()
    ks = kinds(m.store)
    assert ks.index("heartbeat_lost") < ks.index("resumed_by_operator")
    assert ks.count("client_attached") == 2


def test_stop_command_over_the_socket(tmp_path):
    m = machine(tmp_path)
    stopped = threading.Event()
    sock = tmp_path / "calibd.sock"
    srv = dm.Server(sock, m, period_s=0.05, on_stop=stopped.set)
    srv.start()
    try:
        s, f = _client(sock)
        f.readline()
        s.sendall(b'{"cmd": "stop"}\n')
        assert stopped.wait(3.0)
    finally:
        srv.stop()


# ------------------------------------------------------------------ the lock

def test_one_daemon_per_robot_root(tmp_path):
    a = dm.RobotLock(tmp_path / "calibd.lock")
    assert a.acquire("s1")
    b = dm.RobotLock(tmp_path / "calibd.lock")
    assert not b.acquire("s2")
    assert "s1" in b.holder()
    a.release()
    assert b.acquire("s2")
    b.release()


# ------------------------------------------------------------------ one wired loop step (fakes)

def _done(v):
    f = Future()
    f.set_result(v)
    return f


class FakeSource:
    def __init__(self):
        self.i = 0

    def read(self, cam, timeout):
        self.i += 1
        t = 1_000_000_000 + (self.i // 2) * 40_000_000 + (0 if cam == 3 else 14_000)
        return cp.Frame(cam, self.i // 2, t, t, np.zeros((8, 8), np.uint8))

    def reattach(self, cam):
        pass


def _obs(z):
    c = np.array([640.0, 400.0])
    return detect.BoardObs(c, z, 35, 140, False, (), c - 100, c + 100)


def test_daemon_step_runs_trial_health_and_persists(tmp_path):
    c = Clock()
    m = machine(tmp_path, c)
    cap = cp.Capture(FakeSource(), 3, 0, clock=c)
    for _ in range(4):
        cap.step()
    checks = [pf.Check("frames", "ok", "3 pairs"), pf.Check("policy_pods", "fail", "ghost", "motion")]
    pre = pf.Result(checks, {"producer_yaml": "x"}, {"fps": 25.0, "load": 9.0, "boot_id": "b1", "head_serials": [],
                                                     "head_ports": [], "producer": {}})
    zs = iter([0.52] * 100)
    d = dm.Daemon(m, cap, detector=lambda pair: _done((None, None)), to_obs=lambda det: _obs(next(zs)),
                  sampler=lambda: {"boot_id": "b2", "clock_s": c.t, "load": 9.0},
                  trial=place.Trial([place.target("0.5 m")], tmp_path / "trial.jsonl"), clock=c)
    d.preflight_done(pre)
    assert m.phase == "READY"
    assert m.state["checks"][1]["gate"] == "motion"
    m.heartbeat()
    m.key("T")
    for _ in range(6):
        c.t += 0.5
        m.heartbeat()
        cap.step()
        cap.step()
        d.step()
    st = m.store.load_state()
    assert st["capture"]["pairs"] >= 2
    assert st["board"]["z_m"] == pytest.approx(0.52)
    ev = m.store.events()
    assert [e["say"]["kind"] for e in ev if e["kind"] == "say"] == ["intro", "good", "end"]
    assert st["say"]["kind"] == "end"
    assert any(e["kind"] == "health" and e["code"] == "reboot" for e in ev)
    assert any(e["kind"] == "say" for e in ev)
    assert st["banner"]
    assert st["trial"]["summary"]["placed"] == 1


def test_trial_pauses_in_hold(tmp_path):
    c = Clock()
    m = machine(tmp_path, c)
    cap = cp.Capture(FakeSource(), 3, 0, clock=c)
    cap.step()
    cap.step()
    tr = place.Trial([place.target("0.5 m")], tmp_path / "trial.jsonl")
    d = dm.Daemon(m, cap, detector=lambda pair: _done((None, None)), to_obs=lambda det: _obs(0.52),
                  sampler=lambda: {}, trial=tr, clock=c)
    d.preflight_done(pf.Result([], {}, {}))
    m.heartbeat()
    m.key("T")
    c.t += 6
    m.tick()
    assert m.phase == "HOLD"
    for _ in range(6):
        c.t += 0.5
        cap.step()
        cap.step()
        d.step()
    assert tr.records == []                     # the guide did not run while holding
