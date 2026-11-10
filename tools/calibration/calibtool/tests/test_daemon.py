"""calibtool.daemon (calibd): the session state machine with state.json + events.jsonl after every step, the TUI
heartbeat deadman (5 s without a heartbeat -> HOLD; only key G with a TUI attached leaves HOLD; nothing resumes by
itself), abort from any phase, resume from disk after a crash or reboot (an active phase comes back as HOLD), the
Unix-socket link a TUI attaches to over ssh, and the one-job-per-robot lock."""
import json
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


def test_every_say_of_a_step_reaches_the_state_and_the_trial_line_is_current(tmp_path):
    """9 Nov trial: a timeout's "Skipping this one." and the next intro come in one step; both must reach the TUI.
    And the trial line must name the placement being placed, also outside TRIAL (no stale 'placement 2 of 6')"""
    c = Clock()
    m = machine(tmp_path, c)
    cap = cp.Capture(FakeSource(), 3, 0, clock=c)
    cap.step()
    cap.step()
    tr = place.Trial([place.target("1.6 m"), place.target("1.1 m")], tmp_path / "trial.jsonl", timeout_s=3.0)
    d = dm.Daemon(m, cap, detector=lambda pair: _done((None, None)), to_obs=lambda det: None, sampler=lambda: {},
                  trial=tr, clock=c)
    d.preflight_done(pf.Result([], {}, {}))
    assert m.state["trial"]["index"] == 1 and m.state["trial"]["started"] is False
    m.heartbeat()
    m.key("T")
    for _ in range(30):
        c.t += 0.15
        m.heartbeat()
        cap.step()
        cap.step()
        d.step()
    texts = [x["text"] for x in m.state["says"]]
    assert "Skipping this one." in texts and any(x.startswith("Placement 2 of 2.") for x in texts)
    ids = [x["id"] for x in m.state["says"]]
    assert ids == sorted(ids)
    assert m.state["trial"]["index"] == 2 and m.state["trial"]["target"] == "1.1 m"


def test_step_period_is_fast_enough_for_the_guide():
    assert dm.STEP_S <= 0.15


# ------------------------------------------------------------------ review fixes (9 Nov night)

class SaidSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


def _trial_daemon(tmp_path, m, c, targets, to_obs=lambda det: None, timeout_s=3.0, **kw):
    cap = cp.Capture(FakeSource(), 3, 0, clock=c)
    cap.step()
    cap.step()
    tr = place.Trial([place.target(n) for n in targets], tmp_path / "trial.jsonl", timeout_s=timeout_s)
    d = dm.Daemon(m, cap, detector=lambda pair: _done((None, None)), to_obs=to_obs, sampler=lambda: {}, trial=tr,
                  clock=c, **kw)
    return d, cap, tr


def _run(d, cap, c, seconds, tui_=None, hb=True, dt=0.15):
    for _ in range(int(round(seconds / dt))):
        c.t += dt
        if hb:
            d.m.heartbeat()
        cap.step()
        cap.step()
        d.step()
        if tui_ is not None:
            tui_.step(json.dumps(d.m.snapshot()))


def test_a_restarted_calibd_keeps_talking_to_the_tui(tmp_path):
    """review 1: after a calibd restart the old prompts stayed in state.json and the new trial numbered its prompts from
    1 again, so the TUI (which skips ids it has spoken) never spoke again"""
    from calibtool import tui
    c = Clock()
    m = machine(tmp_path, c)
    d, cap, tr = _trial_daemon(tmp_path, m, c, ["1.6 m", "1.1 m", "far"])
    d.preflight_done(pf.Result([], {}, {}))
    sp = SaidSpeaker()
    t = tui.Tui(lambda s: None, sp, out=lambda s: None)
    m.heartbeat()
    m.key("T")
    _run(d, cap, c, 3.5, t)                     # intro 1, timeout: skip + intro 2
    assert sp.said[-1].startswith("Placement 2 of 3.")
    n_said = len(sp.said)
    del d, m                                    # calibd dies
    c.t += 30
    m2 = dm.Machine.resume(dm.Store(tmp_path / "s1"), c)
    assert m2.state["say"] is None and not m2.state["says"] and m2.state["board"] is None
    d2, cap2, tr2 = _trial_daemon(tmp_path, m2, c, ["1.6 m", "1.1 m", "far"])
    m2.heartbeat()
    _run(d2, cap2, c, 0.3, t)
    assert m2.phase == "HOLD"
    m2.key("G")
    _run(d2, cap2, c, 0.3, t)
    assert len(sp.said) > n_said and sp.said[-1].startswith("Placement 2 of 3."), sp.said
    old = [e["say"]["id"] for e in m2.store.events() if e["kind"] == "say"]
    assert len(old) == len(set(old))            # prompt ids unique across both runs


def _hold(d, cap, c, seconds):
    """no heartbeat: the deadman holds, then `seconds` pass in HOLD"""
    _run(d, cap, c, dm.DEADMAN_S + 0.3, hb=False)
    assert d.m.phase == "HOLD"
    _run(d, cap, c, seconds, hb=False)


def test_keys_pressed_outside_the_trial_are_dropped_not_replayed_after_g(tmp_path):
    """review 4: two C pressed in HOLD were queued and replayed after G, skipping two placements"""
    c = Clock()
    m = machine(tmp_path, c)
    d, cap, tr = _trial_daemon(tmp_path, m, c, ["1.6 m", "1.1 m", "far"], timeout_s=180.0)
    d.preflight_done(pf.Result([], {}, {}))
    m.heartbeat()
    m.key("T")
    _run(d, cap, c, 1.0)
    _hold(d, cap, c, 1.0)
    m.key("C")
    m.key("C")
    m.heartbeat()
    m.key("G")
    _run(d, cap, c, 1.0)
    assert m.phase == "TRIAL" and tr.records == [] and tr.state()["index"] == 1
    assert [e["key"] for e in m.store.events() if e["kind"] == "key_dropped"] == ["C", "C"]
    m.key("C")                                  # in TRIAL it still works
    _run(d, cap, c, 0.3)
    assert [r["result"] for r in tr.records] == ["unreachable"]


def test_hold_time_does_not_count_against_the_placement_and_g_repeats_the_intro(tmp_path):
    """review 4: a 200 s HOLD -> 'Skipping this one.' at once after G; and nobody said again what to place"""
    c = Clock()
    m = machine(tmp_path, c)
    d, cap, tr = _trial_daemon(tmp_path, m, c, ["1.6 m", "1.1 m"], timeout_s=20.0)
    d.preflight_done(pf.Result([], {}, {}))
    m.heartbeat()
    m.key("T")
    _run(d, cap, c, 2.0)
    _hold(d, cap, c, 200.0)
    m.heartbeat()
    m.key("G")
    _run(d, cap, c, 1.0)
    texts = [x["text"] for x in m.state["says"]]
    assert "Skipping this one." not in texts
    assert texts[-1].startswith("Placement 1 of 2."), texts
    assert sum(x.startswith("Placement 1 of 2.") for x in texts) == 2
    assert m.state["trial"]["since_intro_s"] < 9.0          # 2 s + the 5 s deadman + 1 s: not the 200 s
    _run(d, cap, c, 18.0)                       # the placement's own 20 s still run out
    assert tr.records and tr.records[0]["result"] == "timeout"
    assert tr.records[0]["paused_s"] == pytest.approx(200.3, abs=0.5)


def test_a_stalled_camera_is_shown_said_once_and_pauses_the_trial(tmp_path):
    """review 2: a frozen or dropped camera stalled the trial silently (latest() never expires; the freeze warning
    never reached the banner)"""
    c = Clock()
    m = machine(tmp_path, c)
    d, cap, tr = _trial_daemon(tmp_path, m, c, ["1.6 m", "1.1 m"], to_obs=lambda det: _obs(1.3), timeout_s=10.0)
    d.preflight_done(pf.Result([], {}, {}))
    m.heartbeat()
    m.key("T")
    _run(d, cap, c, 1.0)
    assert m.state["board"] is not None and m.state["pair_age_s"] < 1.0
    for _ in range(int(30 / 0.15)):             # 30 s without a new pair (the reader is not stepped)
        c.t += 0.15
        m.heartbeat()
        d.step()
    st = m.state
    assert st["board"] is None and st["pair_age_s"] > 25
    assert any("CAMERA STALLED" in b for b in st["banner"]), st["banner"]
    said = [x["text"] for x in st["says"]]
    assert said.count("The camera stopped. Please wait.") == 1, said
    assert tr.records == []                     # 30 s stalled is no 10 s timeout
    _run(d, cap, c, 1.0)                        # frames again
    assert not any("CAMERA STALLED" in b for b in m.state["banner"])
    after = [x["text"] for x in m.state["says"]]
    after = after[after.index("The camera stopped. Please wait.") + 1:]
    assert after and after[0].startswith("Placement 1 of 2."), after          # what to place, again
    kinds_ = kinds(m.store)
    assert "camera_stalled" in kinds_ and "camera_back" in kinds_


def test_the_trial_times_out_without_fresh_detections(tmp_path):
    """review 2: the timeout ran only when a detection finished; a detector that never answers stalled it"""
    c = Clock()
    m = machine(tmp_path, c)
    cap = cp.Capture(FakeSource(), 3, 0, clock=c)
    tr = place.Trial([place.target("1.6 m"), place.target("1.1 m")], tmp_path / "trial.jsonl", timeout_s=3.0)
    d = dm.Daemon(m, cap, detector=lambda pair: Future(), to_obs=lambda det: None, sampler=lambda: {}, trial=tr,
                  clock=c)
    d.preflight_done(pf.Result([], {}, {}))
    m.heartbeat()
    m.key("T")
    _run(d, cap, c, 4.0)
    assert [r["result"] for r in tr.records] == ["timeout"]
    assert "Skipping this one." in [x["text"] for x in m.state["says"]]


class FlakyCapture:
    """a capture whose snapshot raises the first `fail` times"""

    def __init__(self, fail=2, exc=RuntimeError("shm read failed")):
        self.fail, self.exc, self.calls = fail, exc, 0
        self.inner = cp.Capture(FakeSource(), 3, 0)

    def snapshot(self):
        self.calls += 1
        if self.calls <= self.fail:
            raise self.exc
        self.inner.step()
        self.inner.step()
        return self.inner.snapshot()

    def latest(self):
        return self.inner.latest()

    def latest_age(self):
        return self.inner.latest_age()


def _wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_an_exception_in_a_step_does_not_stop_the_loop(tmp_path):
    """review 3: one exception in step() ended the loop thread silently; the machine looked alive"""
    m = dm.Machine(dm.Store(tmp_path / "s1"), session="s1", robot="P10")
    d = dm.Daemon(m, FlakyCapture(fail=3), detector=lambda pair: _done((None, None)), to_obs=lambda det: None,
                  sampler=lambda: {}, clock=time.time)
    d.preflight_done(pf.Result([], {}, {}))
    th = threading.Thread(target=d.loop, kwargs={"period_s": 0.01}, daemon=True)
    th.start()
    try:
        assert _wait_for(lambda: int(m.state.get("step_n") or 0) >= 10)
    finally:
        d.stop_flag.set()
        th.join(2.0)
    errs = [e for e in m.store.events() if e["kind"] == "step_error"]
    assert errs and "shm read failed" in errs[0]["error"] and "Traceback" in errs[0]["trace"]
    assert len(errs) == 1                       # the same error repeated is logged once, then counted
    st = m.store.load_state()
    assert st["step_errors"] == 3 and st["step_error_run"] == 0


def test_a_step_that_keeps_failing_says_so_in_the_banner(tmp_path):
    m = dm.Machine(dm.Store(tmp_path / "s1"), session="s1", robot="P10")
    d = dm.Daemon(m, FlakyCapture(fail=10 ** 6), detector=lambda pair: _done((None, None)), to_obs=lambda det: None,
                  sampler=lambda: {}, clock=time.time)
    for _ in range(3):
        d.safe_step()
    st = m.store.load_state()
    assert st["step_n"] == 3 and st["step_error_run"] == 3
    assert any("calibd step failing" in b and "shm read failed" in b for b in st["banner"]), st["banner"]


def test_a_broken_detection_pool_is_rebuilt(tmp_path):
    """review 3: BrokenProcessPool from submit() went uncaught and killed the loop"""
    from concurrent.futures.process import BrokenProcessPool
    c = Clock()
    m = machine(tmp_path, c)
    cap = cp.Capture(FakeSource(), 3, 0, clock=c)
    calls = {"broken": 0, "good": 0, "rebuilt": 0}

    def broken(pair):
        calls["broken"] += 1
        raise BrokenProcessPool("a worker died")

    def good(pair):
        calls["good"] += 1
        return _done((None, None))

    def rebuild():
        calls["rebuilt"] += 1
        return good
    d = dm.Daemon(m, cap, detector=broken, to_obs=lambda det: _obs(1.1), sampler=lambda: {}, clock=c,
                  rebuild_detector=rebuild)
    d.preflight_done(pf.Result([], {}, {}))
    for _ in range(6):
        c.t += 0.15
        cap.step()
        cap.step()
        d.safe_step()
    assert calls["rebuilt"] == 1 and calls["good"] >= 3
    assert m.state["board"] is not None and m.state["step_error_run"] == 0
    assert "detect_pool_rebuilt" in kinds(m.store)


class _Die(BaseException):
    pass


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_the_main_thread_restarts_a_dead_loop(tmp_path):
    m = dm.Machine(dm.Store(tmp_path / "s1"), session="s1", robot="P10")
    cap = FlakyCapture(fail=1, exc=_Die())
    d = dm.Daemon(m, cap, detector=lambda pair: _done((None, None)), to_obs=lambda det: None, sampler=lambda: {},
                  clock=time.time)
    d.start_loop(period_s=0.01)
    try:
        assert _wait_for(lambda: not d.loop_alive())
        assert d.watch_loop() is True
        assert _wait_for(lambda: int(m.state.get("step_n") or 0) >= 5)
        assert d.watch_loop() is False
    finally:
        d.stop_flag.set()
    assert "loop_died" in kinds(m.store) and m.state["loop_restarts"] == 1


def _held_after_restart(tmp_path, c):
    m = machine(tmp_path, c)
    m.to("READY", "x")
    m.heartbeat()
    m.key("T")
    del m
    c.t += 30
    m2 = dm.Machine.resume(dm.Store(tmp_path / "s1"), c)
    assert m2.phase == "HOLD"
    cap = cp.Capture(FakeSource(), 3, 0, clock=c)
    return m2, dm.Daemon(m2, cap, detector=lambda pair: _done((None, None)), to_obs=lambda det: None,
                         sampler=lambda: {}, clock=c)


def test_preflight_after_a_restart_does_not_lift_hold(tmp_path):
    """review 7: resume -> HOLD, then cmd_run's preflight_done moved it to READY: HOLD left without G"""
    c = Clock()
    m, d = _held_after_restart(tmp_path, c)
    d.preflight_done(pf.Result([pf.Check("frames", "ok", "3 pairs")], {}, {}))
    assert m.phase == "HOLD" and m.state["held_from"] == "TRIAL"
    m.heartbeat()
    m.key("G")
    assert m.phase == "TRIAL"


def test_g_is_refused_while_a_failed_preflight_holds(tmp_path):
    c = Clock()
    m, d = _held_after_restart(tmp_path, c)
    d.preflight_done(pf.Result([pf.Check("frames", "fail", "0 pairs")], {}, {}))
    assert m.phase == "HOLD"
    assert any("preflight failed" in b for b in m.state["banner"])
    m.heartbeat()
    m.key("G")
    assert m.phase == "HOLD" and "g_refused" in kinds(m.store)


def test_the_deadman_runs_on_a_monotonic_clock(tmp_path):
    """review 7: liveness on time.time: an NTP step back kept a dead link 'live' (or a step forward held a live one)"""
    wall, mono = Clock(1000.0), Clock(50.0)
    m = dm.Machine(dm.Store(tmp_path / "s1"), wall, session="s1", robot="P10", mono=mono)
    m.to("READY", "x")
    m.heartbeat()
    m.key("T")
    wall.t += 3600.0                            # the wall clock steps forward; 1 s really passed
    mono.t += 1.0
    m.tick()
    assert m.phase == "TRIAL"
    wall.t -= 7200.0                            # ... and back; 6 s really passed since the heartbeat
    mono.t += 5.0
    m.tick()
    assert m.phase == "HOLD"


def _until(cond, timeout=3.0):
    assert _wait_for(cond, timeout)


def _last_line(f, n=3):
    st = None
    for _ in range(n):
        st = json.loads(f.readline())
    return st


def test_only_the_newest_tui_controls_but_any_can_abort(tmp_path):
    """review 7: with two TUIs attached, a stale one's heartbeat kept the deadman satisfied and its keys counted"""
    c = Clock()
    m = machine(tmp_path, c)
    m.to("READY", "x")
    sock = tmp_path / "calibd.sock"
    srv = dm.Server(sock, m, period_s=0.02)
    srv.start()
    try:
        s1, f1 = _client(sock)
        s1.sendall(b'{"hb": 1}\n')
        _until(lambda: m.state["attached"])
        s2, f2 = _client(sock)
        s2.sendall(b'{"hb": 1}\n')
        _until(lambda: srv.controller == 2)
        assert _last_line(f1)["client"] == {"id": 1, "claimed": True, "controlling": False}
        assert _last_line(f2)["client"]["controlling"] is True
        s1.sendall(b'{"key": "T"}\n')            # the old TUI's key does not count
        _until(lambda: "key_ignored" in kinds(m.store))
        assert m.phase == "READY"
        s2.sendall(b'{"key": "T"}\n')
        _until(lambda: m.phase == "TRIAL")
        c.t += 6
        s1.sendall(b'{"hb": 2}\n')               # only the old TUI still beats: that is no live link
        time.sleep(0.1)
        m.tick()
        assert m.phase == "HOLD"
        f2.close()
        s2.close()                              # the controlling TUI leaves: the old one may take over
        _until(lambda: srv.controller is None)
        s1.sendall(b'{"hb": 3}\n')
        _until(lambda: srv.controller == 1)
        s3, f3 = _client(sock)
        s3.sendall(b'{"hb": 1}\n')
        _until(lambda: srv.controller == 3)
        s1.sendall(b'{"key": "A"}\n')            # abort counts from any TUI
        _until(lambda: m.phase == "ABORTED")
        s1.close()
        s3.close()
    finally:
        srv.stop()


# ------------------------------------------------------------------ start / attach: the daemon that answers is ours

def _kit(base, files):
    for rel, text in files.items():
        f = base / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)


def test_kit_hash_follows_the_code_copied_to_the_robot(tmp_path):
    _kit(tmp_path, {"calibtool/a.py": "x = 1", "calibtool/config/profiles.yaml": "p: 1", "scripts/s.py": "s",
                    "config/rig.yaml": "r", "calibtool/tests/test_a.py": "t", "calibtool/__pycache__/a.pyc": "c",
                    "calibtool/results/r.json": "{}", "calibtool/x.npz": "n"})
    h = dm.kit_hash(tmp_path)
    _kit(tmp_path, {"calibtool/tests/test_a.py": "t2", "calibtool/__pycache__/a.pyc": "c2",
                    "calibtool/results/r.json": "[]", "calibtool/x.npz": "n2"})
    assert dm.kit_hash(tmp_path) == h           # what bootstrap does not copy does not count
    _kit(tmp_path, {"calibtool/a.py": "x = 2"})
    assert dm.kit_hash(tmp_path) != h


def test_the_state_names_the_kit_and_drops_the_pid_on_a_clean_stop(tmp_path):
    m = machine(tmp_path)
    assert m.state["kit"] == dm.kit_hash() and m.state["pid"]
    m.stopped()
    st = m.store.load_state()
    assert st["pid"] is None and st["attached"] is False and kinds(m.store)[-1] == "daemon_stop"
    m2 = dm.Machine.resume(dm.Store(tmp_path / "s1"), Clock())
    assert m2.state["pid"] and m2.state["kit"] == dm.kit_hash()


def test_start_waits_for_its_own_child_not_an_old_daemon(tmp_path):
    """review 6: start connected to the still-running old daemon's socket and reported success"""
    m = machine(tmp_path)
    sock = tmp_path / "calibd.sock"
    srv = dm.Server(sock, m, period_s=0.05)
    srv.start()
    try:
        assert dm.read_state_line(sock)["pid"] == m.state["pid"]
        rcs = iter([None, None, 3])
        r = dm.wait_started(sock, child_pid=-1, child_rc=lambda: next(rcs), sleep=lambda s: None)
        assert not r["ok"] and "rc 3" in r["error"] and f"pid {m.state['pid']}" in r["error"]
        assert "--replace" in r["error"]
        assert dm.wait_started(sock, child_pid=m.state["pid"], child_rc=lambda: None)["ok"]
    finally:
        srv.stop()
    assert dm.read_state_line(sock) is None     # liveness is the socket: nothing answers now


def test_status_says_live_only_when_the_socket_answers(tmp_path, capsys):
    import argparse
    m = machine(tmp_path)                       # state.json with a pid, but no daemon serving the socket
    a = argparse.Namespace(root=str(tmp_path), session="s1")
    import calibtool.storage as storage
    d = storage.SessionDirs(tmp_path, "s1")
    dm.Store(d.root).write_state(m.state)
    dm.cmd_status(a)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["live"] is False and out["pid"] == m.state["pid"]
    srv = dm.Server(d.root / "calibd.sock", m, period_s=0.05)
    srv.start()
    try:
        dm.cmd_status(a)
        out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert out["live"] is True and out["kit"] == dm.kit_hash()
    finally:
        srv.stop()


# ------------------------------------------------------------------ full session mode (M5): phases and keys

def full(tmp_path, clock=None, motion_green=True):
    m = machine(tmp_path, clock, mode="full", read_only=False)
    m.state["motion_gate_green"] = motion_green
    m.to("READY", "preflight green")
    m.heartbeat()
    return m


def test_m_enters_motors_only_with_the_motion_gate_green(tmp_path):
    m = full(tmp_path / "a", motion_green=False)
    m.key("M")
    assert m.phase == "READY"
    assert "motors_refused" in kinds(m.store)
    m2 = full(tmp_path / "b")
    m2.key("M")
    assert m2.phase == "MOTORS"


def test_m_and_t_belong_to_their_modes(tmp_path):
    m = full(tmp_path / "a")
    m.key("T")
    assert m.phase == "READY"                     # no placement trial inside a full session
    w = machine(tmp_path / "b")
    w.state["motion_gate_green"] = True
    w.to("READY", "x")
    w.key("M")
    assert w.phase == "READY"                     # a watch/trial session never enters MOTORS


@pytest.mark.parametrize("phase,key", [("MOTORS", "Y"), ("DEPLOY_SIZE", "1"), ("DEPLOY_SIZE", "2"),
                                       ("DEPLOY_SIZE", "?"), ("PLACE", "C"), ("POSTCHECK", "R"), ("FINISH", "D"),
                                       ("FINISH", "K"), ("FINISH", "E"), ("FINISH", "X"), ("DEPLOY", "Y")])
def test_flow_keys_count_only_in_their_phase(tmp_path, phase, key):
    m = full(tmp_path)
    m.to(phase, "test")
    m.key(key)
    assert m.take_keys() == [key]
    m.to("FIT", "test")
    m.key(key)
    assert m.take_keys() == []
    assert kinds(m.store)[-1] == "key_dropped"


def test_y_outside_motors_never_reaches_the_flow(tmp_path):
    m = full(tmp_path)
    for ph in ("READY", "PLACE", "GRID", "IMU_SWEEP", "DECIDE", "POSTCHECK"):
        m.to(ph, "test")
        m.key("Y")
        assert m.take_keys() == [], ph


def test_keys_pending_are_dropped_on_any_phase_change(tmp_path):
    m = full(tmp_path)
    m.to("DEPLOY_SIZE", "x")
    m.key("2")
    m.to("FIT", "x")
    m.to("DEPLOY_SIZE", "x")
    assert m.take_keys() == []


def test_o_opts_out_of_the_okvis_check_at_any_time(tmp_path):
    m = full(tmp_path)
    m.to("GRID", "x")
    m.key("O")
    assert m.state["okvis_check"] is False
    assert "okvis_check_skipped" in kinds(m.store)


def test_d_counts_after_the_session_ended(tmp_path):
    for end in ("DONE", "ABORTED", "STOPPED_LOUD"):
        m = full(tmp_path / end)
        m.to(end, "x")
        m.key("D")
        m.key("G")
        assert m.take_keys() == ["D"], end
        assert m.phase == end


def test_the_new_motion_phases_need_a_heartbeat(tmp_path):
    for ph in ("MOTORS", "PLACE", "GRID", "IMU_SWEEP", "CHECK_GRID", "DEPLOY", "POSTCHECK"):
        c = Clock()
        m = full(tmp_path / ph, c)
        m.to(ph, "x")
        c.t += 5.1
        m.tick()
        assert m.phase == "HOLD" and m.state["held_from"] == ph
    for ph in ("DECIDE", "DEPLOY_SIZE", "FIT"):
        c = Clock()
        m = full(tmp_path / ph, c)
        m.to(ph, "x")
        c.t += 60
        m.tick()
        assert m.phase == ph


def test_hold_by_the_flow_keeps_where_it_came_from(tmp_path):
    m = full(tmp_path)
    m.to("GRID", "x")
    m.hold("another program commands the motors")
    assert m.phase == "HOLD" and m.state["held_from"] == "GRID"
    assert m.state["hold_reason"] == "another program commands the motors"
    m.key("G")
    assert m.phase == "GRID" and m.state["hold_reason"] is None


@pytest.mark.parametrize("phase", ["MOTORS", "SETUP", "PLACE", "GRID", "FIT", "IMU_SWEEP", "DECIDE", "DEPLOY_SIZE",
                                   "DEPLOY", "POSTCHECK", "FINISH"])
def test_a_full_session_resumes_every_flow_phase_in_hold(tmp_path, phase):
    m = full(tmp_path)
    m.to(phase, "x")
    m2 = dm.Machine.resume(dm.Store(tmp_path / "s1"), Clock())
    assert m2.phase == "HOLD" and m2.state["held_from"] == phase


# ------------------------------------------------------------------ the full session in the loop (M5 wiring)

class FakeFlow:
    def __init__(self):
        self.terminal_calls = 0

    def handle_terminal_keys(self):
        self.terminal_calls += 1


def full_daemon(tmp_path, c):
    m = machine(tmp_path, c, mode="full", read_only=False)
    cap = cp.Capture(FakeSource(), 3, 0, clock=c)
    for _ in range(4):
        cap.step()
    calls = []

    def detector(pair):
        calls.append(pair.t0)
        return _done((None, None))
    box = dm.ObsBox()
    flow = FakeFlow()
    d = dm.Daemon(m, cap, detector=detector, to_obs=lambda det: _obs(0.75), sampler=lambda: {}, clock=c, flow=flow,
                  obs_box=box)
    return m, cap, d, box, flow, calls


def test_obs_box_hands_each_detection_out_once():
    b = dm.ObsBox()
    assert b.take() == (False, None)
    b.put("o1")
    assert b.take() == (True, "o1")
    assert b.take() == (False, None)
    b.put(None)
    assert b.take() == (True, None)


def test_obs_box_keeps_why_the_newest_detection_found_no_board():
    b = dm.ObsBox()
    assert b.last_miss() is None
    m = detect.Miss("dark", 0, 0.0, 20.0, ())
    b.put(None, m)
    assert b.take() == (True, None) and b.last_miss() is m
    b.put("o2")
    assert b.last_miss() is None


def test_the_trial_hears_why_the_board_is_not_seen(tmp_path):
    """10 Nov (phantom10): the detection's reason reaches the guide (and the state) through calibd"""
    c = Clock()
    m = machine(tmp_path, c)
    got = []

    def to_miss(dl, info):
        got.append(info)
        return detect.board_miss(dl or {}, (1280, 800), brightness=(info or {}).get("brightness"))
    cap = cp.Capture(FakeSource(), 3, 0, clock=c)
    cap.step()
    cap.step()
    tr = place.Trial([place.target("1.1 m")], tmp_path / "trial.jsonl", timeout_s=None)
    d = dm.Daemon(m, cap, detector=lambda pair: _done(({}, {"brightness": 15.0})), to_obs=lambda det: None,
                  sampler=lambda: {}, trial=tr, clock=c, to_miss=to_miss)
    d.preflight_done(pf.Result([], {}, {}))
    m.heartbeat()
    m.key("T")
    _run(d, cap, c, 4.0)
    assert got and got[0] == {"brightness": 15.0}
    assert "It is too dark for me to see the board." in [x["text"] for x in m.state["says"]]
    assert m.state["board"] is None and m.state["board_miss"]["reason"] == "dark"


def test_full_mode_hands_the_reason_to_the_obs_box(tmp_path):
    c = Clock()
    m, cap, d, box, flow, calls = full_daemon(tmp_path, c)
    d.to_obs = lambda det: None
    d.to_miss = lambda dl, info: detect.Miss("far", 3, 9.0, 120.0, ())
    m.to("READY", "x")
    m.heartbeat()
    m.to("PLACE", "flow")
    for _ in range(3):
        c.t += 0.2
        m.heartbeat()
        cap.step()
        cap.step()
        d.step()
    assert box.take() == (True, None) and box.last_miss().reason == "far"


def test_full_mode_detects_for_the_guidance_only(tmp_path):
    c = Clock()
    m, cap, d, box, flow, calls = full_daemon(tmp_path, c)
    m.to("READY", "x")
    m.heartbeat()
    m.to("PLACE", "flow")
    for _ in range(3):
        c.t += 0.2
        m.heartbeat()
        cap.step()
        cap.step()
        d.step()
    fresh, obs = box.take()
    assert fresh and obs.z == pytest.approx(0.75)
    n = len(calls)
    m.to("GRID", "flow")
    for _ in range(3):
        c.t += 0.2
        m.heartbeat()
        cap.step()
        cap.step()
        d.step()
    assert len(calls) <= n + 1                     # the grid owns the detection pool: no live detection there


def test_full_mode_hands_d_to_the_flow_after_the_end(tmp_path):
    c = Clock()
    m, cap, d, box, flow, calls = full_daemon(tmp_path, c)
    m.to("ABORTED", "x")
    d.step()
    assert flow.terminal_calls == 1


def test_full_mode_banner_says_what_the_flow_needs_said(tmp_path):
    c = Clock()
    m, cap, d, box, flow, calls = full_daemon(tmp_path, c)
    m.state["flow_banner"] = ["ROLLBACK FAILED at verify: ..."]
    m.state["hold_reason"] = "another program commands the motors"
    m.to("HOLD", "x")
    b = d.banner()
    assert not any("READ-ONLY" in x for x in b)
    assert "ROLLBACK FAILED at verify: ..." in b
    assert any("another program commands the motors" in x for x in b)


def test_daemon_say_numbers_after_the_last_prompt(tmp_path):
    c = Clock()
    m, cap, d, box, flow, calls = full_daemon(tmp_path, c)
    d.say(place.Say(next(d.ids), c.t, "countdown", "Step back. Starting in 5."))
    assert m.state["says"][-1]["text"] == "Step back. Starting in 5." and m.state["say_id_max"] >= 1


def test_the_session_mode_and_dry_run_reach_the_daemon_child():
    a = dm.main.__globals__["argparse"].Namespace(root="/r", session="S", profile="P10", trial=False,
                                                  detect_workers=0, session_mode="full", dry_run=True)
    argv = dm.run_argv(a)
    assert argv[argv.index("--session-mode") + 1] == "full" and "--dry-run" in argv
    a.session_mode, a.dry_run = "watch", False
    argv = dm.run_argv(a)
    assert "--session-mode" not in argv and "--dry-run" not in argv


def test_every_key_gets_an_answer_in_the_state(tmp_path):
    """10 Nov, phantom11: "i press Y and nothing is shown" -- each key says whether it was taken and what it does"""
    m = full(tmp_path)
    m.to("MOTORS", "test")
    m.key("y")
    lk = m.state["last_key"]
    assert lk["key"] == "Y" and lk["phase"] == "MOTORS" and lk["accepted"] and "motors" in lk["why"]
    m.to("FIT", "test")
    m.key("Y")
    assert not m.state["last_key"]["accepted"] and "does nothing in FIT" in m.state["last_key"]["why"]
    m.state["busy"] = {"what": "Deploying (up to 3 min)", "since": 0.0}
    m.to("DEPLOY", "test")
    m.key("K")
    assert not m.state["last_key"]["accepted"] and "busy" in m.state["last_key"]["why"]
    m.key("G")
    assert not m.state["last_key"]["accepted"] and "HOLD" in m.state["last_key"]["why"]
    m.key("A")
    assert m.state["last_key"]["accepted"] and "abort" in m.state["last_key"]["why"]
