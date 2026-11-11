"""calibtool.menu: `calib` with no arguments asks which robot, looks at it (one ssh call), and offers only what makes
sense: continue an unfinished session, start a new calibration (restoring the robot first when a dead session still
holds the producer lease), the placement trial, status, stop. ssh, the kit copy, the TUI and input() are fakes."""
import json

import pytest

pytest.importorskip("yaml")

from calibtool import menu, profile  # noqa: E402


@pytest.fixture(autouse=True)
def _piper_installed(monkeypatch):
    """the tests run on a box without the Piper files; a test about a PC without Piper says so itself"""
    monkeypatch.setattr(menu, "piper_available", lambda: True)

KIT = "kit-laptop"
NOW = 1_762_800_000.0          # robot clock of the fake probe
LIVE_SID, DEAD_SID = "20261110-140500-P10", "20261110-123252-P10"


def profiles():
    return [profile.load("P10"), profile.load("P11")]


def sess(sid, phase, live=False, kit=KIT, mode="full", created=NOW - 3600):
    return {"id": sid, "phase": phase, "mode": mode, "created": created, "updated": created + 60, "live": live,
            "pid": 42 if live else None, "kit": kit}


def probe(sessions=(), lease=None):
    return {"now": NOW, "lease": lease, "lock": None, "current": None, "sessions": list(sessions)}


class Robot:
    """the fake robot behind ssh / rsync: answers by what the remote line runs, records every call"""

    def __init__(self, looks, restore_rc=0, start=None, ssh_fails=0):
        self.looks = list(looks)
        self.calls, self.restore_rc = [], restore_rc
        self.start_out = start or {"session": "20261110-150000-P10", "pid": 77}
        self.ssh_fails = ssh_fails

    def kinds(self):
        return [k for k, _ in self.calls]

    def __call__(self, argv, **kw):
        line = argv[-1]
        rc, out, err = 0, "", ""
        if argv[0] == "rsync":
            kind = "rsync"
        elif menu.PROBE_MARK in line:
            kind = "look"
            if self.ssh_fails:
                self.ssh_fails -= 1
                rc, err = 255, "ssh: connect to host phantom10 port 22: No route to host"
            else:
                out = json.dumps(self.looks.pop(0) if len(self.looks) > 1 else self.looks[0])
        elif "calibtool restore" in line:
            kind = "restore"
            rc = self.restore_rc
            out = ("calibtool restore: session 20261110-123252-P10 restored: producer restored as saved (1280 800); "
                   "OKVIS started (label); lease released\n" if rc == 0 else
                   "calibtool restore: FAILED, the robot is NOT restored: producer: ProducerError: boom\n")
        elif "daemon start" in line:
            kind, out = "start", json.dumps(self.start_out)
        elif "daemon stop" in line:
            kind, out = "stop", json.dumps({"stopped": True})
        elif "daemon status" in line:
            kind = "status"
            st = next((s for s in self.looks[0]["sessions"] if s["live"]), None)
            out = json.dumps(dict(st, session=st["id"]) if st else {"live": False})
        elif "mkdir -p" in line:
            kind = "mkdir"
        else:
            kind = "other"
        self.calls.append((kind, line))
        return type("R", (), {"returncode": rc, "stdout": out, "stderr": err})()


class Op:
    """the operator: answers input() in order and keeps the whole transcript"""

    def __init__(self, *answers):
        self.answers, self.text = list(answers), []

    def input(self, prompt):
        self.text.append(prompt)
        if not self.answers:
            raise EOFError
        a = self.answers.pop(0)
        self.text.append(a + "\n")
        return a

    def out(self, s=""):
        self.text.append(str(s) + "\n")

    @property
    def transcript(self):
        return "".join(self.text)


def run(op, fake, tmp_path, cfg=None, attached=None, **kw):
    attached = attached if attached is not None else []
    cfg_path = tmp_path / "laptop.json"
    if cfg is not None:
        cfg_path.write_text(json.dumps(cfg))
    rc = menu.main(input_fn=op.input, out=op.out, run=fake, cfg_path=cfg_path, kit=KIT, profiles=profiles(),
                   attach=lambda p, sid, voice: attached.append((p.bot, sid, voice)) or 0,
                   sync=lambda p: fake.calls.append(("sync", p.bot)), **kw)
    return rc, attached


# ------------------------------------------------------------------ which robot


def test_the_robots_are_listed_by_host_and_bot_name_and_any_of_number_host_bot_is_accepted(tmp_path):
    for answer, bot in (("1", "P10"), ("phantom11", "P11"), ("p10", "P10"), ("P11", "P11")):
        op = Op(answer, "q")
        r = Robot([probe()])
        run(op, r, tmp_path)
        assert "1) phantom10 (P10)" in op.transcript and "2) phantom11 (P11)" in op.transcript
        assert "new robot (setup)" in op.transcript
        assert f"Looking at {'phantom10' if bot == 'P10' else 'phantom11'} ({bot})" in op.transcript, answer


def test_an_unknown_answer_asks_again_and_enter_takes_last_time_robot(tmp_path):
    op = Op("phantom99", "", "q")
    r = Robot([probe()])
    run(op, r, tmp_path, cfg={"robot": "phantom11"})
    t = op.transcript
    assert "phantom99" in t and "not in the list" in t
    assert "Enter = phantom11" in t and "Looking at phantom11 (P11)" in t


def test_new_robot_setup_calls_onboard_when_it_exists(tmp_path):
    seen = []
    res = type("Result", (), {"status": "done"})()
    fake = type("M", (), {"run": staticmethod(lambda host, ask=None, out=None: seen.append(host) or res)})
    op = Op("3", "phantom13")
    rc, _ = run(op, Robot([probe()]), tmp_path, import_module=lambda name: fake)
    assert rc == 0 and seen == ["phantom13"] and "setup: done" in op.transcript


def test_new_robot_setup_says_not_available_without_the_module(tmp_path):
    def missing(name):
        raise ImportError(name)
    op = Op("new robot", "phantom13")
    rc, _ = run(op, Robot([probe()]), tmp_path, import_module=missing)
    assert rc == 1 and "setup is not available yet" in op.transcript


# ------------------------------------------------------------------ looking at the robot


def test_one_ssh_call_reads_status_lease_and_sessions():
    r = Robot([probe([sess(DEAD_SID, "GRID")], lease={"session": DEAD_SID, "heartbeat": NOW - 7200})])
    lk = menu.look(profile.load("P10"), run=r)
    assert r.kinds() == ["look"] and "/data/calib" in r.calls[0][1]
    assert lk.error is None and lk.live == [] and [s["id"] for s in lk.unfinished] == [DEAD_SID]
    assert lk.dead_lease["session"] == DEAD_SID


def test_the_probe_is_read_only_python_stdlib():
    """runs on the robot without the kit (a new or wiped robot), changes nothing"""
    src = menu.PROBE
    for bad in ("write", "unlink", "remove", "rmtree", "flock", "kill", "subprocess", "open(r + '/producer.lease', 'w"):
        assert bad not in src, bad
    compile(src, "probe", "exec")


def test_the_probe_reads_a_fake_root(tmp_path):
    """the probe itself, on a session tree like calibd leaves: unfinished sessions, newest first, finished ones left
    out unless they hold the lease; no socket = not live"""
    import subprocess
    import sys
    root = tmp_path / "root"
    for sid, phase, t in ((DEAD_SID, "GRID", 2.0), ("20261109-100000-P10", "DONE", 1.0),
                          ("20261110-090000-P10", "READY", 1.5)):
        d = root / "sessions" / sid
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"session": sid, "phase": phase, "mode": "full", "created": t,
                                                  "kit": "k"}))
    (root / "producer.lease").write_text(json.dumps({"session": DEAD_SID, "heartbeat": 5.0}))
    r = subprocess.run([sys.executable, "-c", menu.PROBE, str(root)], capture_output=True, text=True, timeout=30)
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert [s["id"] for s in d["sessions"]] == [DEAD_SID, "20261110-090000-P10"]
    assert d["lease"]["session"] == DEAD_SID and not any(s["live"] for s in d["sessions"])


def test_an_unreachable_robot_offers_try_again(tmp_path):
    op = Op("1", "1", "5")
    r = Robot([probe([sess(DEAD_SID, "GRID")])], ssh_fails=1)
    run(op, r, tmp_path)
    t = op.transcript
    assert "Could not look at phantom10" in t and "No route to host" in t and "1) try again" in t
    assert r.kinds() == ["look", "look"]


# ------------------------------------------------------------------ what it offers


def test_a_dead_session_holding_the_lease_restores_the_robot_then_starts_a_new_calibration(tmp_path):
    lk = probe([sess(DEAD_SID, "GRID")], lease={"session": DEAD_SID, "heartbeat": NOW - 7200})
    op = Op("1", "2", "", "")
    r = Robot([lk])
    rc, attached = run(op, r, tmp_path)
    t = op.transcript
    assert "continue session 20261110-123252-P10 (phase GRID" in t
    assert "start a new calibration (first restores the robot" in t
    assert "stop the running session" not in t
    assert r.kinds() == ["look", "sync", "restore", "status", "start"]
    restore = r.calls[2][1]
    assert "-m calibtool restore --if-stale-hours 0 --root /data/calib" in restore
    assert "producer restored as saved" in t                   # its output is shown
    assert "--session-mode full" in r.calls[4][1]
    assert attached == [("P10", "20261110-150000-P10", "piper")] and rc == 0


def test_a_failed_restore_starts_nothing_and_goes_back_to_the_menu(tmp_path):
    lk = probe([sess(DEAD_SID, "GRID")], lease={"session": DEAD_SID, "heartbeat": NOW - 7200})
    op = Op("1", "2", "", "5")
    r = Robot([lk], restore_rc=1)
    rc, attached = run(op, r, tmp_path)
    assert "FAILED, the robot is NOT restored" in op.transcript and "nothing was started" in op.transcript
    assert "start" not in r.kinds() and attached == []
    assert r.kinds() == ["look", "sync", "restore", "look"]


def test_a_live_session_running_this_kit_is_attached_as_it_is(tmp_path):
    op = Op("1", "1", "2")
    r = Robot([probe([sess(LIVE_SID, "HOLD", live=True)], lease={"session": LIVE_SID, "heartbeat": NOW - 30})])
    rc, attached = run(op, r, tmp_path)
    t = op.transcript
    assert "calibd: running session 20261110-140500-P10, phase HOLD, the same kit as this laptop" in t
    assert "start a new calibration" not in t and "placement trial" not in t          # never a second session
    assert "stop the running session 20261110-140500-P10" in t
    assert r.kinds() == ["look"] and attached == [("P10", LIVE_SID, "text")]


def test_a_live_session_running_other_code_is_restarted_with_this_kit(tmp_path):
    op = Op("1", "1", "")
    r = Robot([probe([sess(LIVE_SID, "GRID", live=True, kit="kit-old")])], start={"session": LIVE_SID, "pid": 9})
    rc, attached = run(op, r, tmp_path)
    assert "runs other code" in op.transcript
    assert r.kinds() == ["look", "status", "stop", "sync", "start"]
    assert f"--session {LIVE_SID}" in r.calls[4][1] and attached == [("P10", LIVE_SID, "piper")]


def test_a_dead_session_is_continued_by_restarting_calibd_under_its_id_after_copying_the_kit(tmp_path):
    op = Op("1", "1", "")
    r = Robot([probe([sess(DEAD_SID, "PLACE")])], start={"session": DEAD_SID, "pid": 9})
    run(op, r, tmp_path)
    assert r.kinds() == ["look", "status", "sync", "start"] and f"--session {DEAD_SID}" in r.calls[3][1]
    assert "comes back in HOLD" in op.transcript


def test_the_placement_trial_starts_a_trial_session(tmp_path):
    op = Op("1", "2", "")
    r = Robot([probe()])
    run(op, r, tmp_path)
    assert r.kinds() == ["look", "sync", "status", "start"]
    assert "--trial" in r.calls[3][1] and "--session-mode" not in r.calls[3][1]


def test_stop_asks_first_then_looks_again(tmp_path):
    lv = probe([sess(LIVE_SID, "HOLD", live=True)])
    after = probe([sess(LIVE_SID, "HOLD")])
    op = Op("1", "3", "y", "5")
    r = Robot([lv, after])
    run(op, r, tmp_path)
    assert "Stop session 20261110-140500-P10?" in op.transcript
    assert r.kinds() == ["look", "stop", "look"]
    assert "start a new calibration" in op.transcript.split("Stop session")[1]


def test_stop_answered_no_stops_nothing(tmp_path):
    op = Op("1", "3", "n", "4")
    r = Robot([probe([sess(LIVE_SID, "HOLD", live=True)])])
    run(op, r, tmp_path)
    assert "stop" not in r.kinds()


def test_status_only_prints_the_state_and_comes_back(tmp_path):
    op = Op("1", "2", "4")
    r = Robot([probe([sess(LIVE_SID, "HOLD", live=True)])])
    run(op, r, tmp_path)
    assert r.kinds() == ["look", "status", "look"]
    assert "phase HOLD" in op.transcript


# ------------------------------------------------------------------ voice


def test_voice_is_asked_remembered_and_offered_as_the_default(tmp_path, monkeypatch):
    monkeypatch.setattr(menu, "piper_available", lambda: True)
    lv = probe([sess(LIVE_SID, "HOLD", live=True)])
    op = Op("1", "1", "2")
    run(op, Robot([lv]), tmp_path)
    assert "1) Piper voice (recommended)" in op.transcript and "2) text on screen" in op.transcript
    assert "espeak" not in op.transcript                  # 10 Nov: not offered
    cfg = json.loads((tmp_path / "laptop.json").read_text())
    assert cfg["voice"] == "text" and cfg["robot"] == "phantom10"
    op2 = Op("", "1", "")
    _, attached = run(op2, Robot([lv]), tmp_path)
    assert "Enter = 2" in op2.transcript and attached[0][2] == "text"


def test_a_pc_without_piper_shows_the_prompts_as_text(tmp_path, monkeypatch):
    monkeypatch.setattr(menu, "piper_available", lambda: False)
    lv = probe([sess(LIVE_SID, "HOLD", live=True)])
    op = Op("1", "1")
    _, attached = run(op, Robot([lv]), tmp_path)
    assert "Piper is not installed" in op.transcript and attached[0][2] == "text"


def test_the_menu_works_without_a_writable_config(tmp_path):
    bad = tmp_path / "file"
    bad.write_text("x")
    op = Op("1", "1", "1")
    attached = []
    menu.main(input_fn=op.input, out=op.out, run=Robot([probe([sess(LIVE_SID, "HOLD", live=True)])]),
              cfg_path=bad / "sub" / "laptop.json", kit=KIT, profiles=profiles(),
              attach=lambda p, sid, voice: attached.append(voice) or 0, sync=lambda p: None)
    assert attached == ["piper"]


def test_eof_or_q_leaves_quietly(tmp_path):
    assert run(Op(), Robot([probe()]), tmp_path)[0] == 0
    assert run(Op("q"), Robot([probe()]), tmp_path)[0] == 0


def test_robot_given_on_the_command_line_skips_the_question(tmp_path):
    op = Op("q")
    run(op, Robot([probe()]), tmp_path, robot="P11")
    assert "Which robot?" not in op.transcript and "Looking at phantom11 (P11)" in op.transcript
