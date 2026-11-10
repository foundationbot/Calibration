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


def test_every_prompt_of_a_step_is_spoken_in_order():
    """9 Nov trial: a placement that ended and the next intro came in one step; only the last one reached the TUI"""
    t, sent, out, sp, c = make()
    says = [{"id": 4, "t": 99.0, "kind": "correction", "text": "Lower the board."},
            {"id": 5, "t": 100.0, "kind": "skip", "text": "Skipping this one."},
            {"id": 6, "t": 100.0, "kind": "intro", "text": "Placement 3 of 6."}]
    t.step(json.dumps(state(say=says[0], says=says[:1])))
    t.step(json.dumps(state(say=says[2], says=says)))
    t.step(json.dumps(state(say=says[2], says=says)))
    assert sp.said == ["Lower the board.", "Skipping this one.", "Placement 3 of 6."]


def test_a_restarted_daemon_numbering_from_1_is_spoken_again():
    t, sent, out, sp, c = make()
    t.step(json.dumps(state(say={"id": 40, "t": 99.0, "kind": "correction", "text": "old"},
                            says=[{"id": 40, "t": 99.0, "kind": "correction", "text": "old"}])))
    new = {"id": 1, "t": 200.0, "kind": "intro", "text": "Placement 4 of 6."}
    t.step(json.dumps(state(say=new, says=[new], updated=200.0)))
    assert sp.said == ["old", "Placement 4 of 6."]


def test_render_names_the_current_placement_gravity_and_height():
    tr = {"started": True, "index": 3, "of": 6, "target": "1.1 m", "since_intro_s": 45.2, "timeout_s": 180.0,
          "finished": False, "records": [{"target": "0.5 m", "result": "unreachable", "seconds": 1.2, "prompts": 1}],
          "summary": {}, "gravity": {"ok": True, "pitch_down_deg": 65.9, "note": "IMU slot 0, 250 samples"},
          "board": {"z_m": 1.13, "dh_m": -1.1, "dxy_m": 0.45, "tags": 35, "cut": []}}
    r = tui.render(state(trial=tr))
    assert "NOW placement 3 of 6 (1.1 m), 45 s of 180 s" in r
    assert "camera 65.9 deg below the horizon" in r
    assert "110 cm below the camera, 45 cm along the floor" in r
    tr2 = dict(tr, started=False, index=4, target="1.6 m", since_intro_s=None,
               gravity={"ok": False, "note": "IMU not read (FileNotFoundError); placement guidance uses the camera "
                                                 "distance only"})
    r = tui.render(state(trial=tr2))
    assert "not started: next is placement 4 of 6 (1.6 m)" in r and "NOW" not in r
    assert "gravity: NOT AVAILABLE" in r and "camera distance only" in r


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
    assert calls == [["/usr/bin/spd-say", "-w", "-x", "-l", "en-us", "-t", "male1", "-r", "-95",
                      "<speak>Good. <break time='450ms'/> Step <break time='150ms'/> back. <break time='450ms'/></speak>"]]


def test_ssml_pauses_escape_and_count():
    s = tui.ssml_with_pauses("Lower the board. About 10 cm & more", 250, 700)
    assert s.startswith("<speak>") and s.endswith("</speak>")
    assert s.count("<break time='250ms'/>") == 7 and s.count("<break time='700ms'/>") == 1
    assert "&amp;" in s and " & " not in s


def test_silent_speaker():
    tui.Silent().say("x")


def test_a_daemon_that_stopped_stepping_is_flagged():
    """review 3: the socket kept sending the same state after the loop died; nothing on screen said so"""
    t, sent, out, sp, c = make()
    t.step(json.dumps(state(step_n=10)))
    c.t = 1.0
    t.step(json.dumps(state(step_n=17)))
    assert "not stepping" not in out[-1]
    c.t = 2.0
    t.step(json.dumps(state(step_n=17)))
    assert "not stepping" not in out[-1]
    c.t = 1.0 + tui.NOT_STEPPING_S + 0.5
    t.step(json.dumps(state(step_n=17)))
    assert "NOT NORMAL: calibd not stepping" in out[-1]
    c.t += 0.2
    t.step(json.dumps(state(step_n=18)))
    assert "not stepping" not in out[-1]


def test_a_tui_that_lost_control_says_so():
    t, sent, out, sp, c = make()
    t.step(json.dumps(state(client={"id": 1, "claimed": True, "controlling": True})))
    assert "control" not in out[-1].split("\n")[2]
    t.step(json.dumps(state(client={"id": 1, "claimed": True, "controlling": False})))
    assert "NOT NORMAL: another TUI attached later and has control" in out[-1]


def _until(cond, timeout=3.0):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_queued_corrections_are_superseded_but_intros_skips_and_ends_never_dropped():
    """review 8: the speech queue was unbounded: while spd-say spoke slowly, stale corrections piled up and were all
    read out after the board had moved on"""
    import threading
    gate = threading.Event()
    said = []

    class Slow:
        def say(self, text):
            said.append(text)
            if text == "first":
                gate.wait(5)
    b = tui.Background(Slow())
    b.say_prompt("first", "correction")
    assert _until(lambda: said == ["first"])
    for text, kind in (("Placement 2 of 6.", "intro"), ("Lower the board.", "correction"),
                       ("Raise the board.", "correction"), ("I cannot see the board.", "lost"),
                       ("Skipping this one.", "skip"), ("This one is out of reach. Skipping.", "unreachable"),
                       ("Good. That one is done. Step back.", "good"), ("The camera stopped. Please wait.", "camera"),
                       ("Move the board closer.", "correction"), ("That was the last placement.", "end")):
        b.say_prompt(text, kind)
    gate.set()
    assert _until(lambda: said[-1] == "That was the last placement.")
    assert said == ["first", "Placement 2 of 6.", "Skipping this one.", "This one is out of reach. Skipping.",
                    "Good. That one is done. Step back.", "The camera stopped. Please wait.", "Move the board closer.",
                    "That was the last placement."]


def test_the_tui_hands_the_prompt_kind_to_the_speaker():
    got = []

    class K:
        def say_prompt(self, text, kind):
            got.append((text, kind))
    t = tui.Tui(lambda s: None, K(), clock=Clock(), out=lambda s: None)
    t.step(json.dumps(state(say={"id": 3, "t": 99.5, "kind": "intro", "text": "Placement 2 of 6."})))
    assert got == [("Placement 2 of 6.", "intro")]


# ---------------------------------------------------------------- the full session's screens (M5)


def full(phase, **kw):
    return state(phase=phase, mode="full", read_only=False, banner=(), trial=None, **kw)


def test_full_session_keys_follow_the_phase():
    assert "M = motors" in tui.render(full("READY")) and "T = start" not in tui.render(full("READY"))
    assert "Y = " in tui.render(full("MOTORS"))
    assert "1 / 2 = deploy size" in tui.render(full("DEPLOY_SIZE"))
    assert "D = disable" in tui.render(full("FINISH")) and "K = keep" in tui.render(full("FINISH"))
    assert "O = skip the OKVIS check" in tui.render(full("GRID"))
    assert "G = continue" in tui.render(full("HOLD", held_from="GRID"))
    assert "A = abort" in tui.render(full("GRID"))


def test_motors_screen_lists_the_pre_enable_checks_and_the_result():
    r = tui.render(full("MOTORS", motors={"checks": [{"name": "estop", "status": "fail", "detail": "e-stop engaged",
                                                      "gate": "motion"}], "detail": "not enabling: estop",
                                          "refused": True}))
    assert "estop" in r and "e-stop engaged" in r and "not enabling: estop" in r


def test_grid_screen_shows_the_countdown_and_the_pose():
    r = tui.render(full("GRID", countdown=3, placement={"target": "0.75 m", "kind": "default"},
                        grid={"placement": "0.75 m", "pose": 4, "of": 49,
                              "last": {"command": [8.0, -11.0], "measured": [8.4, -11.0], "tags_l": 30,
                                       "found": True}}))
    assert "STARTING IN 3" in r and "pose 4 of 49" in r and "0.75 m" in r


def test_deploy_size_screen_offers_the_options_and_the_facts():
    ds = {"prev": [960, 600], "options": [{"key": "1", "label": "960x600, the previous default (recommended)"},
                                          {"key": "2", "label": "1280x800"}],
          "help": ["Configs on this robot naming the live calibration: /etc/phantom/vla/rig.env"], "chosen": None}
    r = tui.render(full("DEPLOY_SIZE", deploy_size=ds, decision={"decision": "DEPLOY", "reasons": ["new better"]}))
    assert "1: 960x600" in r and "2: 1280x800" in r and "vla/rig.env" in r and "DEPLOY" in r


def test_postcheck_deploy_and_finish_screens():
    r = tui.render(full("FINISH", deploy={"ok": True, "detail": "deployed and verified", "rollback_script": "/x/rb.sh"},
                        postcheck={"decision": {"rollback": False, "reasons": ["post-check passed"]}},
                        motors_prompt="D = disable the motors (the warning comes first), K = keep them on",
                        motors_on=True))
    assert "deployed and verified" in r and "/x/rb.sh" in r and "post-check passed" in r
    assert "MOTORS ON" in r and "the warning comes first" in r
