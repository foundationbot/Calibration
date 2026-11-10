"""calibtool.tui (laptop side): renders calibd's state (header with the robot, motors, the NOT NORMAL banner and the
abort key in every phase), sends the heartbeat at 1 Hz and the operator's keys, and speaks each new prompt exactly once
through spd-say (injected: tests use a fake speaker)."""
import json

import pytest

pytest.importorskip("numpy")

from calibtool import tui  # noqa: E402


def state(phase="TRIAL", say=None, banner=("READ-ONLY session: the tool changes nothing on the robot",), updated=100.0,
          **kw):
    s = {"session": "20261109-190507-P10", "robot": "P10", "phase": phase, "status": "running", "banner": list(banner),
         "checks": [{"name": "frames", "status": "ok", "detail": "3 pairs", "gate": "collect"},
                    {"name": "policy_pods", "status": "fail", "detail": "running: ghost", "gate": "motion"},
                    {"name": "motors_off", "status": "ok", "detail": "motors are off", "gate": "motion"}],
         "capture": {"pairs": 120, "sync_median_ms": 0.014, "cams": {"3": {"fps": 25.0, "none_reads": 0},
                                                                    "0": {"fps": 25.0, "none_reads": 0}}},
         "board": {"z_m": 1.08, "tags": 35, "cut": []}, "health": [], "say": say, "updated": updated,
         "trial": {"index": 2, "of": 6, "target": "0.75 m", "records": [], "summary": {}}, "attached": True}
    s.update(kw)
    return s


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(**kw):
    sent, out, sp, c = [], [], FakeSpeaker(), Clock()
    t = tui.Tui(sent.append, sp, clock=c, out=out.append, **kw)
    return t, sent, out, sp, c


@pytest.mark.parametrize("phase", ["PREFLIGHT", "READY", "BLOCKED", "TRIAL", "HOLD", "DONE", "ABORTED"])
def test_render_has_banner_and_abort_key_in_every_phase(phase):
    r = tui.render(state(phase=phase))
    assert "NOT NORMAL" in r and "READ-ONLY" in r
    assert "A = abort" in r
    assert phase in r and "P10" in r


def test_render_shows_motors_checks_capture_board_and_prompt():
    r = tui.render(state(say={"id": 3, "t": 99.0, "kind": "correction", "text": "Move the board closer."}))
    assert "motors off" in r
    assert "policy_pods" in r and "FAIL" in r
    assert "25.0" in r and "0.014" in r
    assert "1.08 m" in r and "35 tags" in r
    assert "Move the board closer." in r
    assert "2 of 6" in r


def test_render_without_banner_says_normal():
    r = tui.render(state(banner=()))
    assert "NOT NORMAL" not in r


def test_each_new_prompt_is_spoken_once():
    t, sent, out, sp, c = make()
    s3 = state(say={"id": 3, "t": 99.5, "kind": "intro", "text": "Placement 2 of 6."})
    t.step(json.dumps(s3))
    t.step(json.dumps(s3))
    assert sp.said == ["Placement 2 of 6."]
    t.step(json.dumps(state(say={"id": 4, "t": 100.0, "kind": "good", "text": "Good. Step back."})))
    assert sp.said == ["Placement 2 of 6.", "Good. Step back."]
    assert len(out) == 3


def test_an_old_prompt_is_not_spoken_on_reattach():
    t, sent, out, sp, c = make()
    t.step(json.dumps(state(say={"id": 9, "t": 10.0, "kind": "correction", "text": "old"}, updated=100.0)))
    assert sp.said == []


def test_keys_and_heartbeat_lines():
    t, sent, out, sp, c = make()
    t.key("a")
    assert json.loads(sent[-1]) == {"key": "A"}
    t.heartbeat()
    c.t = 0.5
    t.heartbeat()
    c.t = 1.0
    t.heartbeat()
    hb = [json.loads(x) for x in sent if "hb" in json.loads(x)]
    assert len(hb) == 2


def test_spd_say_argv():
    calls = []
    sp = tui.SpdSay(path="/usr/bin/spd-say", run=lambda argv, **kw: calls.append(argv))
    sp.say("Good. Step back.")
    assert calls == [["/usr/bin/spd-say", "-w", "-x", "-l", "en-us", "-t", "male1", "-r", "-85",
                      "<speak>Good. <break time='450ms'/> Step <break time='150ms'/> back. <break time='450ms'/></speak>"]]


def test_ssml_pauses_escape_and_count():
    s = tui.ssml_with_pauses("Lower the board. About 10 cm & more", 250, 700)
    assert s.startswith("<speak>") and s.endswith("</speak>")
    assert s.count("<break time='250ms'/>") == 7 and s.count("<break time='700ms'/>") == 1
    assert "&amp;" in s and " & " not in s


def test_silent_speaker():
    tui.Silent().say("x")
