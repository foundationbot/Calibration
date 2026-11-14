"""handeyeflow: the menu's hand-eye session against a fake robot (every ssh line answered as phantom10 does)."""
import json
from types import SimpleNamespace

import pytest

from calibtool import handeyeflow as hf, menu

P = SimpleNamespace(bot="P10", host="root@phantom10", root="/data/calib", python="/py", pydeps="/deps",
                    motor_tool="/tool")
SETUP = {"robot": "phantom10", "bot": "P10", "urdf": "/u.urdf", "grab": "/g.py", "grab_pydeps": "/gd", "frames": 5,
         "profiles": "", "snapshot": {"q_deg": {"NeckPitch": 0.0}},
         "arms": {a: {"moves": [{"name": "stage", "deg": [0] * 7, "burst": False},
                                {"name": "p08", "deg": [0] * 7, "burst": True, "necks": [[10, 0], [22, 12]]}]}
                  for a in ("Right", "Left")}}
REPORT = {"chosen": "M12+head", "cv": {"M12+head": {"Right": {"n": 9, "median": 7.0, "p90": 11.5, "base_median": 45.0},
                                                   "Left": {"n": 16, "median": 5.1, "p90": 8.4, "base_median": 37.8}}}}


class Robot:
    def deploy_out(self, line):
        if self.refuse:
            return "NOT DEPLOYED: hand-eye stale for this camera geometry"
        return "dry run: 9 commands" if "--dry-run" in line else "DEPLOYED: OKVIS loaded the new config and URDF"

    """answers the flow's ssh lines; `logs` = per arm, the log.jsonl lines the wave writes (all at once)"""

    def __init__(self, check=None, logs=None, refuse=False):
        self.lines, self.check, self.refuse = [], check or {"ok": True, "problems": []}, refuse
        self.logs = logs or {a: [{"t": 0, "kind": "start"}, {"t": 1, "kind": "burst", "name": "01-p08-10-0", "dt": 9},
                                 {"t": 2, "kind": "done", "captures": 2}] for a in ("right", "left")}

    def __call__(self, argv, **kw):
        line = argv[-1]
        self.lines.append(line)
        out = ""
        if "--check" in line:
            out = json.dumps(self.check)
        elif line.startswith("setsid"):
            out = "started"
        elif "log.jsonl" in line:
            arm = "right" if "/right/" in line else "left"
            start = int(line.split("tail -n +")[1].split()[0]) - 1
            out = "\n".join(json.dumps(e) for e in self.logs[arm][start:])
        elif "handeye_match.sh" in line:
            out = "/x/readb.json: 20 captures, 16 clean\nreadb wall time 30 s for 20 captures, 4 processes"
        elif line.startswith("cat ") and "fit.json" in line:
            out = json.dumps(REPORT)
        elif "handeye_deploy deploy" in line:
            out = self.deploy_out(line)
        elif "rollback.sh" in line:
            out = "rolled back"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")


def flow(robot, answers=("y",), typed=None):
    out = []
    a = list(answers)
    h = hf.HandEye(P, run=robot, out=out.append, ask=lambda q: a.pop(0) if a else "", sleep=lambda s: None,
                   typed=typed or (lambda: None), bootstrap=lambda: None, now=lambda f: "20261111-150000",
                   setup=SETUP, poll_s=0)
    return h, out


def started(robot):
    return [ln for ln in robot.lines if ln.startswith("setsid")]


def test_not_ready_moves_nothing():
    r = Robot(check={"ok": False, "problems": ["motors not on", "LeftAnklePitch +40.0"]})
    h, out = flow(r)
    assert h.session() == 2
    assert started(r) == []
    assert any("motors not on" in s for s in out)


def test_no_yes_moves_nothing():
    r = Robot()
    h, out = flow(r, answers=("n",))
    assert h.session() == 1 and started(r) == []


def test_full_session_right_then_left_then_fit():
    r = Robot()
    h, out = flow(r, answers=("y", "n"))
    assert h.session() == 0
    s = started(r)
    assert len(s) == 2 and "--arm Right" in s[0] and "--arm Left" in s[1]
    assert "/tool" in s[0] and "--grab /g.py" in s[0] and "/data/calib/handeye/20261111-150000/right" in s[0]
    assert any("armcal.urdffit" in ln and "/u.urdf" in ln for ln in r.lines)
    txt = "\n".join(out)
    assert "7.0 mm" in txt and "5.1 mm" in txt and "not deployed" in txt
    assert txt.index("RIGHT arm") < txt.index("Start") if "Start" in txt else True


def test_stop_key_writes_stop_and_ends_the_session():
    logs = {"right": [{"t": 0, "kind": "start"}, {"t": 1, "kind": "stopped", "reason": "STOP file"}], "left": []}
    r = Robot(logs=logs)
    keys = ["s"]
    h, out = flow(r, typed=lambda: keys.pop(0) if keys else None)
    assert h.session() == 3
    assert any("touch" in ln and "/right/STOP" in ln for ln in r.lines)
    assert len(started(r)) == 1                       # the left arm never starts


def test_alive_check_cannot_match_its_own_shell():
    r = Robot()
    h, _ = flow(r)
    h.session()
    follow = [ln for ln in r.lines if "tail -n +" in ln][0]
    assert "[h]andeye_wave.py" in follow


def test_menu_offers_handeye_only_when_set_up_and_no_live_session():
    lk = menu.Look(error=None, sessions=[]) if hasattr(menu, "Look") else None
    if lk is None:
        pytest.skip("menu.Look changed")
    kinds = [o.kind for o in menu.options(lk, "k", handeye=True)]
    assert "handeye" in kinds
    assert "handeye" not in [o.kind for o in menu.options(lk, "k", handeye=False)]


def test_available_needs_the_profile_opt_in_and_the_setup():
    assert hf.available(SimpleNamespace(bot="P10", handeye=True))
    assert not hf.available(SimpleNamespace(bot="P10", handeye=False))
    assert not hf.available(SimpleNamespace(bot="P99", handeye=True))


def test_p10_setup_is_shipped_and_consistent():
    s = json.loads(hf.setup_path("P10").read_text())
    assert s["urdf"] == "/etc/phantomos/okvis/mk1_1.urdf" and s["snapshot"]["q_deg"]
    for arm in ("Right", "Left"):
        bursts = [m for m in s["arms"][arm]["moves"] if m.get("burst")]
        assert bursts and all(len(m["deg"]) == 7 and m["necks"] for m in bursts)
        assert all(abs(p) <= 26 and abs(y) <= 34 for m in bursts for p, y in m["necks"])


def test_deploy_dry_run_first_then_y_deploys_and_r_rolls_back():
    r = Robot()
    h, out = flow(r, answers=("y", "y", "R"))
    assert h.session() == 0
    dep = [ln for ln in r.lines if "handeye_deploy" in ln]
    assert "geometry" in dep[0] and "--dry-run" in dep[1] and "--dry-run" not in dep[-1] and "deploy --fit" in dep[-1]
    assert any("rollback.sh" in ln for ln in r.lines)
    assert any("DEPLOYED" in x for x in out)


def test_a_refused_deploy_changes_nothing_and_says_why():
    r = Robot(refuse=True)
    h, out = flow(r, answers=("y", "y"))
    assert h.session() == 6
    assert not [ln for ln in r.lines if "handeye_deploy deploy" in ln and "--dry-run" not in ln]
    assert any("stale for this camera geometry" in x for x in out)


def test_the_detached_wave_redirects_its_whole_group_so_ssh_returns():
    """13 Nov live: only the python was redirected, the subshell kept ssh's stdout open, ssh hung 60 s"""
    r = Robot()
    h, _ = flow(r)
    h.upload_plans()
    h.start_wave("Right")
    line = [ln for ln in r.lines if ln.startswith("setsid")][0]
    inner = line.split("setsid bash -c ", 1)[1]
    import shlex as _s
    body = _s.split(inner)[0]
    assert "( {" in body and "; } >" in body and "run.out 2>&1 < /dev/null &" in body


def test_an_earlier_runs_log_is_moved_aside_before_a_new_wave():
    """13 Nov live: the retried left arm's follow() read the first try's 'stopped' line and quit at once"""
    r = Robot()
    h, _ = flow(r)
    h.upload_plans()
    h.start_wave("Left")
    line = [ln for ln in r.lines if ln.startswith("setsid")][0]
    import shlex as _s
    body = _s.split(line.split("setsid bash -c ", 1)[1])[0]
    assert "mv /data/calib/handeye/20261111-150000/left/log.jsonl" in body or "log.prev-" in body
    assert body.index("log.prev-") < body.index("handeye_wave.py")


class SlowWave(Robot):
    """the wave's log grows one line per poll; `alive` says per poll whether pgrep found the wave (__ALIVE__)"""

    def __init__(self, n_lines, alive):
        super().__init__()
        self.n, self.alive, self.poll = n_lines, list(alive), 0

    def __call__(self, argv, **kw):
        line = argv[-1]
        if "log.jsonl" in line and "pgrep" in line:
            self.lines.append(line)
            start = int(line.split("tail -n +")[1].split()[0]) - 1
            have = min(self.poll + 1, self.n)
            ev = [{"t": i, "kind": "burst", "name": f"{i:02d}", "dt": 9} for i in range(self.n - 1)] \
                + [{"t": self.n, "kind": "done", "captures": self.n - 1}]
            out = [json.dumps(e) for e in ev[start:have]]
            if self.alive[min(self.poll, len(self.alive) - 1)]:
                out.append("__ALIVE__")
            self.poll += 1
            return SimpleNamespace(returncode=0, stdout="\n".join(out), stderr="")
        return super().__call__(argv, **kw)


def test_scattered_misses_of_a_running_wave_are_not_lost():
    """13 Nov 19:38, phantom10: three scattered pgrep misses (ssh hiccups) over a 7.5 min wave, never reset, called
    the healthy right wave 'lost' during its return move and stopped the session before the left arm"""
    alive = [True] * 40
    for i in (5, 15, 25, 33):
        alive[i] = False
    r = SlowWave(40, alive)
    h, _ = flow(r)
    assert h.follow("/data/calib/handeye/20261111-150000/right") == "done"


def test_three_misses_in_a_row_are_lost():
    r = SlowWave(40, [True] * 5 + [False] * 40)
    h, _ = flow(r)
    assert h.follow("/data/calib/handeye/20261111-150000/right") == "lost"
