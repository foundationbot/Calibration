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
    assert w[:2] == ["cd", "/data/calib/kit/tools/calibration"]
    assert "OMP_NUM_THREADS=2" in cmd and "OPENBLAS_NUM_THREADS=2" in cmd
    assert "PYTHONPATH=/data/calib/kit/tools/calibration:/root/p11-cal/pydeps" in cmd
    assert ("nice -n 10 /root/p11-cal/venv/bin/python -m calibtool.daemon start --root /data/calib "
            "--profile P10 --trial") in cmd


def test_ssh_argv_keeps_the_link_alive_check_short():
    a = cli.ssh_argv(P(), "echo hi")
    assert a[0] == "ssh" and a[-2:] == ["root@phantom10", "echo hi"]
    assert "ServerAliveInterval=2" in a and "BatchMode=yes" in a


def test_bootstrap_rsyncs_code_only():
    calls = []
    cli.bootstrap(P(), run=lambda argv, **kw: calls.append(argv) or type("R", (), {"returncode": 0})())
    assert calls[0][:3] == ["ssh", "-o", "BatchMode=yes"] and "mkdir -p /data/calib/kit/tools/calibration" in \
        calls[0][-1]
    r = calls[1]
    assert r[0] == "rsync" and r[-1] == "root@phantom10:/data/calib/kit/tools/calibration/"
    for d in ("calibtool", "scripts", "config"):
        assert any(x.endswith(f"tools/calibration/{d}") for x in r), d
    assert "--exclude=__pycache__" in r and "--exclude=tests" in r
    # 10 Nov: the planner's neck chain reads the robot URDF; the kit lacked it and calibd died at start
    u = calls[2]
    assert u[0] == "rsync" and u[-2].endswith("config/foundation_rig/20250206_phantom_mk1.urdf")
    assert u[-1] == "root@phantom10:/data/calib/kit/config/foundation_rig/"
    assert "/data/calib/kit/config/foundation_rig" in calls[0][-1] and "mkdir -p" in calls[0][-1]


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
    assert n == 3 and link.closed
    assert len(out) == 1                    # 10 Nov: an unchanged state is not drawn again
    assert any('"hb"' in s for s in link.sent)


def test_headless_draws_every_changed_state_and_stops_at_n():
    sts = [{"session": "S1", "robot": "P10", "phase": ph, "banner": [], "checks": [], "capture": {}, "say": None,
            "updated": 1.0} for ph in ("READY", "MOTORS", "MOTORS", "PLACE", "GRID")]
    link = FakeLink([json.dumps(st) for st in sts])
    out = []
    n = cli.run_tui(link, speaker=None, headless=4, out=out.append)
    assert n == 4 and [x.splitlines()[0].split("phase ")[1].split()[0] for x in out] == ["READY", "MOTORS", "PLACE"]


def test_a_full_session_starts_calibd_in_full_mode():
    r = FakeRun([json.dumps({"session": "S1", "pid": 42, "sock": "/x"})])
    cli.start(P(), mode="full", dry_run=True, run=r)
    line = r.calls[0][-1]
    assert "--session-mode full" in line and "--dry-run" in line and "--trial" not in line
    r = FakeRun([json.dumps({"live": False}), json.dumps({"session": "S2", "pid": 7, "sock": "/y"})])
    assert cli.start_or_attach(P(), None, run=r, kit=K, mode="full") == "S2"
    assert "--session-mode full" in r.calls[-1][-1] and "--dry-run" not in r.calls[-1][-1]


def test_full_and_trial_do_not_mix():
    with pytest.raises(SystemExit):
        cli.main(["P10", "--full", "--trial", "--no-sync", "--headless", "1"])


def test_text_voice_is_offered_and_say_prints_instead_of_speaking(capsys):
    assert cli.main(["P10", "--voice", "text", "--say", "Placement 2 of 6."]) == 0
    assert "Placement 2 of 6." in capsys.readouterr().out


# ------------------------------------------------------------------ the link drops: reconnect, never exit (10 Nov)


def _st(phase="PLACE", n=1):
    return json.dumps({"session": "S1", "robot": "P10", "phase": phase, "mode": "full", "banner": [], "checks": [],
                       "capture": {}, "say": None, "updated": 1.0, "step_n": n, "attached": True})


class Links:
    """reconnect(): the next fake link (an empty one = the attach failed)"""

    def __init__(self, *line_lists):
        self.made = [FakeLink(x) for x in line_lists]
        self.i = 0

    def __call__(self):
        lk = self.made[self.i]
        self.i += 1
        return lk


def test_a_dropped_link_reconnects_with_backoff_and_says_so():
    first = FakeLink([_st("PLACE", 1), _st("PLACE", 2)])
    again = Links([], [], [_st("HOLD", 3), _st("HOLD", 4)])
    waits, out = [], []
    n = cli.run_tui(first, speaker=None, headless=4, out=out.append, reconnect=again, robot="phantom10",
                    wait=lambda s: waits.append(s) or False)
    assert n == 4 and waits == [1, 2, 4]
    screens = "\n".join(out)
    for k in (1, 2, 3):
        assert f"NOT NORMAL: Connection lost to phantom10: reconnecting (attempt {k})" in screens
    assert "press G when ready" in screens and "keys pressed now are not sent" in screens.lower()
    last = out[-1]
    assert "Connection back" in last and "the robot is holding; press G when ready" in last
    assert "Connection lost" not in last


def test_the_backoff_stops_growing_at_10_s():
    waits = []
    links = Links(*([[]] * 7 + [[_st()]]))
    cli.run_tui(FakeLink([_st()]), speaker=None, headless=2, out=lambda s: None, reconnect=links,
                wait=lambda s: waits.append(s) or False)
    assert waits == [1, 2, 4, 8, 10, 10, 10, 10]


def test_q_while_reconnecting_leaves():
    waits = []
    n = cli.run_tui(FakeLink([_st()]), speaker=None, headless=5, out=lambda s: None, reconnect=Links([], []),
                    wait=lambda s: waits.append(s) or len(waits) == 2)     # True = Q pressed during the wait
    assert n == 1 and waits == [1, 2]


def test_without_reconnect_a_dropped_link_still_leaves():
    assert cli.run_tui(FakeLink([_st()]), speaker=None, headless=5, out=lambda s: None) == 1


def test_keys_and_heartbeats_are_not_sent_or_queued_while_the_link_is_down():
    from calibtool import tui
    sent, out = [], []
    t = tui.Tui(sent.append, None, out=out.append, clock=lambda: 100.0, min_draw_s=0.0)
    t.step(_st("PLACE", 1))
    t.link_lost("phantom10", 1)
    sent.clear()
    t.heartbeat()
    t.key("g")
    assert sent == []
    assert "G NOT sent" in out[-1] and "not queued" in out[-1]
    assert "link down" in out[-1].splitlines()[0]
    t.link_back()
    t.step(_st("HOLD", 2))
    t.heartbeat()
    assert sent and '"hb"' in sent[-1]
    assert "Connection back" in out[-1] and "NOT sent" in out[-1]       # the unsent key's line stays a few seconds


def test_connection_back_goes_after_a_few_seconds():
    from calibtool import tui
    now = [100.0]
    out = []
    t = tui.Tui(lambda s: None, None, out=out.append, clock=lambda: now[0], min_draw_s=0.0)
    t.link_lost("phantom10", 1)
    t.link_back()
    t.step(_st("PLACE", 1))
    assert "Connection back" in out[-1]
    now[0] += tui.LINK_BACK_S + 1
    t.step(_st("PLACE", 2))
    assert "Connection back" not in out[-1]


def test_the_attach_link_runs_the_daemon_attach_and_a_quiet_one_hides_ssh_errors():
    seen = []

    def popen(argv, **kw):
        seen.append((argv, kw))
        return type("P", (), {"stdout": iter(()), "stdin": None, "terminate": lambda self: None})()
    cli.Link(P(), "S1", popen=popen)
    cli.Link(P(), "S1", popen=popen, quiet=True)
    assert "daemon attach" in seen[0][0][-1] and "--session S1" in seen[0][0][-1]
    assert "stderr" not in seen[0][1] and seen[1][1]["stderr"] == cli.subprocess.DEVNULL


# ------------------------------------------------------------------ `calib` alone = the menu


def test_no_arguments_opens_the_menu(monkeypatch):
    from calibtool import menu
    seen = []
    monkeypatch.setattr(menu, "main", lambda robot=None, voice=None, **kw: seen.append((robot, voice)) or 0)
    assert cli.main([]) == 0 and seen == [(None, None)]
    assert cli.main(["phantom10"]) == 0 and seen[-1] == ("phantom10", None)
    assert cli.main(["phantom10", "--voice", "text"]) == 0 and seen[-1] == ("phantom10", "text")


def test_scripted_flags_do_not_open_the_menu(monkeypatch):
    from calibtool import menu
    monkeypatch.setattr(menu, "main", lambda **kw: pytest.fail("menu opened"))
    monkeypatch.setattr(cli, "status", lambda p, s=None, run=None: {"live": False})
    assert cli.main(["P10", "--status"]) == 0


# ------------------------------------------------------------------ the single-file executable


def test_install_builds_a_zipapp_whose_kit_hash_matches_the_worktree(tmp_path, capsys):
    import os
    import subprocess
    import sys
    import zipfile
    from calibtool import daemon, paths
    dest = tmp_path / "bin"
    assert cli.main(["--install", "--install-dir", str(dest)]) == 0
    calib = dest / "calib"
    said = capsys.readouterr().out
    assert str(calib) in said and "calib phantom10 --full" in said
    assert calib.read_bytes().startswith(b"#!/usr/bin/env python3\n")
    assert os.stat(calib).st_mode & 0o777 == 0o755
    names = zipfile.ZipFile(calib).namelist()
    assert "__main__.py" in names
    assert "kit/tools/calibration/calibtool/cli.py" in names and "kit/tools/calibration/calibtool/menu.py" in names
    assert "kit/config/foundation_rig/20250206_phantom_mk1.urdf" in names
    assert "kit/tools/calibration/calibtool/config/profiles.yaml" in names
    assert not [n for n in names if "/tests/" in n or "__pycache__" in n or n.endswith(".npz")]
    env = dict(os.environ, XDG_CACHE_HOME=str(tmp_path / "cache"), HOME=str(tmp_path))
    want = daemon.kit_hash(paths.CAL)
    for _ in range(2):                          # the second run reuses the extracted kit
        r = subprocess.run([sys.executable, str(calib), "--version"], capture_output=True, text=True, env=env,
                           timeout=60)
        assert r.returncode == 0, r.stderr
        assert f"kit {want}" in r.stdout
    kits = list((tmp_path / "cache" / "calibtool").glob("kit-*"))
    assert len(kits) == 1 and str(kits[0]) in r.stdout
    assert daemon.kit_hash(kits[0] / "tools" / "calibration") == want
    assert (kits[0] / "config/foundation_rig/20250206_phantom_mk1.urdf").is_file()       # bootstrap's URDF


def test_bootstrap_copies_from_wherever_the_kit_is():
    """inside the zipapp paths.CAL is the extracted kit: bootstrap copies that tree and its URDF"""
    from calibtool import paths
    calls = []
    cli.bootstrap(P(), run=lambda argv, **kw: calls.append(argv))
    assert calls[1][-4].startswith(str(paths.CAL)) and calls[2][-2] == str(paths.CAL.parents[1] / cli.URDF_REL)


def test_version_prints_the_kit_hash(capsys):
    from calibtool import daemon
    assert cli.main(["--version"]) == 0
    assert f"kit {daemon.kit_hash()}" in capsys.readouterr().out
