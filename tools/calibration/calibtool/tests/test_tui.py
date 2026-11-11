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
    c.t = 1.0
    t.step(json.dumps(state(say={"id": 4, "t": 100.0, "kind": "good", "text": "Good. Step back."})))
    assert sp.said == ["Placement 2 of 6.", "Good. Step back."]
    assert len(out) == 2                    # the repeated state was not drawn again


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
    c.t += tui.MIN_DRAW_S
    t.step(json.dumps(state(step_n=18)))
    assert "not stepping" not in out[-1]


def test_a_tui_that_lost_control_says_so():
    t, sent, out, sp, c = make()
    t.step(json.dumps(state(client={"id": 1, "claimed": True, "controlling": True})))
    assert "control" not in out[-1].split("\n")[2]
    c.t = 1.0
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


def test_queued_status_and_keep_still_prompts_are_superseded_too():
    """10 Nov: "Still waiting." and "Keep the board still." describe the board at one moment, like a correction"""
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
    b.say_prompt("Still waiting. Board 110 centimetres away; I need about 50.", "status")
    b.say_prompt("Keep the board still.", "still")
    b.say_prompt("Lower the board.", "correction")
    gate.set()
    assert _until(lambda: said[-1] == "Lower the board.")
    assert said == ["first", "Lower the board."]


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
    kw = {"banner": (), **kw}
    return state(phase=phase, mode="full", read_only=False, trial=None, **kw)


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
    assert "STARTING IN 3" in r and "take 0.75 m: pose 4/49" in r and "last pose +8/-11 deg" in r


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


def test_piper_voice_synthesises_then_plays_and_falls_back_when_missing(tmp_path):
    """10 Nov: the operator found espeak 'ridiculously artificial'; Piper (a local neural voice) is the opt-in
    replacement. It writes a wav with piper, plays it, and is unavailable when its binary or model is missing."""
    binary, model = tmp_path / "piper", tmp_path / "voice.onnx"
    calls = []
    sp = tui.PiperSay(binary=binary, model=model, player="/usr/bin/pw-play",
                      run=lambda argv, **kw: calls.append((argv, kw.get("input"))))
    assert not sp.available()
    binary.write_text(""); model.write_text("")
    assert sp.available()
    sp.say("Good. Step back.")
    (synth, text), (play, _) = calls
    assert synth[0] == str(binary) and "--model" in synth and str(model) in synth and "--length_scale" in synth
    assert text == b"Good. Step back.\n"
    assert play[0] == "/usr/bin/pw-play" and play[1] == synth[synth.index("--output_file") + 1]


def test_make_speaker_prefers_the_asked_voice_and_falls_back(tmp_path):
    assert isinstance(tui.make_speaker("espeak"), tui.SpdSay)
    sp = tui.make_speaker("piper", piper_dir=tmp_path)             # nothing installed there
    assert isinstance(sp, tui.SpdSay)


# ---------------------------------------------------------------- text mode (10 Nov: an alternative to the voice)


def make_text(**kw):
    sent, out, c = [], [], Clock()
    sp = tui.make_speaker("text")
    t = tui.Tui(sent.append, sp, clock=c, out=out.append, **kw)
    step = t.step

    def stepped(line):                      # a second apart: every new screen is drawn
        c.t += 1.0
        return step(line)
    t.step = stepped
    return t, out, sp


def test_text_voice_is_a_screen_panel_and_runs_nothing(tmp_path):
    sp = tui.make_speaker("text", piper_dir=tmp_path)
    assert isinstance(sp, tui.TextPrompts)
    assert not hasattr(sp, "run")                   # nothing to launch: no spd-say, no piper


def test_text_mode_shows_the_newest_prompt_first_and_the_previous_ones_with_their_age():
    t, out, sp = make_text()
    says = [{"id": 1, "t": 80.0, "kind": "intro", "text": "Placement 2 of 6."},
            {"id": 2, "t": 92.0, "kind": "good", "text": "Good."},
            {"id": 3, "t": 99.0, "kind": "correction", "text": "Not yet. 30 more centimetres closer."}]
    for i in range(3):
        t.step(json.dumps(state(say=says[i], says=says[:i + 1], updated=100.0)))
    lines = out[-1].splitlines()
    i = next(k for k, x in enumerate(lines) if x.startswith("PROMPT"))
    assert "Not yet. 30 more centimetres closer." in lines[i]
    assert "8 s ago: Good." in lines[i + 1] and "20 s ago: Placement 2 of 6." in lines[i + 2]
    assert i < next(k for k, x in enumerate(lines) if x.startswith("preflight"))   # above the details


def test_text_mode_keeps_the_last_four_and_wraps_long_prompts():
    t, out, sp = make_text()
    says = [{"id": k, "t": 90.0 + k, "kind": "status", "text": f"Prompt {k}."} for k in range(1, 7)]
    t.step(json.dumps(state(say=says[-1], says=says, updated=100.0)))
    txt = out[-1]
    assert "Prompt 6." in txt and "Prompt 3." in txt and "Prompt 2." not in txt
    long = {"id": 7, "t": 100.0, "kind": "correction", "text": "Move the board " + "much " * 30 + "closer."}
    t.step(json.dumps(state(say=long, says=[long], updated=100.0)))
    lines = out[-1].splitlines()
    i = next(k for k, x in enumerate(lines) if x.startswith("PROMPT"))
    panel = lines[i:lines.index("-" * 100, i)]
    assert len(panel) > 2 and all(len(x) <= 100 for x in panel)
    assert "closer." in " ".join(panel)
    assert not any(x.startswith('>> "') for x in lines)            # the panel replaces the one-line prompt


def test_text_mode_shows_the_current_prompt_even_when_attaching_late():
    """voice skips an old prompt on re-attach (it would be replayed); on screen it is just what is being asked now"""
    t, out, sp = make_text()
    t.step(json.dumps(state(say={"id": 9, "t": 10.0, "kind": "correction", "text": "Lower the board."}, updated=100.0)))
    assert any(x.startswith("PROMPT") and "Lower the board." in x for x in out[-1].splitlines())


def test_text_mode_highlights_the_newest_prompt_only_on_a_terminal():
    t, out, sp = make_text()
    s = state(say={"id": 1, "t": 99.0, "kind": "good", "text": "Good."})
    t.step(json.dumps(s))
    assert "\033[" not in out[-1]
    t2, out2, _ = make_text(ansi=True)
    t2.step(json.dumps(s))
    assert "\033[1;7m" in out2[-1] and "Good." in out2[-1]


def test_voice_modes_show_no_panel():
    t, sent, out, sp, c = make()
    t.step(json.dumps(state(say={"id": 1, "t": 99.0, "kind": "good", "text": "Good."})))
    assert not any(x.startswith("PROMPT") for x in out[-1].splitlines())



def test_the_finish_screen_shows_e_and_x_when_offered():
    st = state(phase="FINISH", mode="full", finish_options=["E = more placements (the planner asks)",
                                                             "X = decide again accepting 5 judged regions"])
    txt = tui.render(st)
    assert "E = more placements" in txt and "X = decide again" in txt
    assert "E = more placements" in tui.keys_for(st) and "X = decide again" in tui.keys_for(st)
    assert "E =" not in tui.keys_for(state(phase="FINISH", mode="full"))


# ---------------------------------------------------------------- 10 Nov: a calmer screen (the operator: "why the hell is
# tool printing massive chunk 20 times over and over", "i press Y and nothing is shown")


def test_an_unchanged_screen_is_not_drawn_again():
    t, sent, out, sp, c = make()
    s = state(say={"id": 3, "t": 99.5, "kind": "intro", "text": "Placement 2 of 6."})
    for k in range(5):
        c.t = float(k)
        t.step(json.dumps(s))
    assert len(out) == 1


def test_the_screen_is_drawn_at_most_twice_a_second_and_a_waiting_change_comes_at_the_next_state():
    t, sent, out, sp, c = make()
    t.step(json.dumps(state(step_n=1, board={"z_m": 1.00, "tags": 35, "cut": []})))
    c.t = 0.2
    t.step(json.dumps(state(step_n=2, board={"z_m": 1.10, "tags": 35, "cut": []})))
    c.t = 0.4
    t.step(json.dumps(state(step_n=3, board={"z_m": 1.20, "tags": 35, "cut": []},
                            say={"id": 4, "t": 100.0, "kind": "good", "text": "Good."})))
    assert len(out) == 1 and "1.00 m" in out[0]
    assert sp.said == ["Good."]                       # speech does not wait for the screen
    c.t = 0.6
    t.step(json.dumps(state(step_n=4, board={"z_m": 1.20, "tags": 35, "cut": []},
                            say={"id": 4, "t": 100.0, "kind": "good", "text": "Good."})))
    assert len(out) == 2 and "1.20 m" in out[1]
    c.t = 0.7
    t.step(json.dumps(state(step_n=5, board={"z_m": 1.30, "tags": 35, "cut": []})))
    assert len(out) == 2
    t.flush()                                          # leaving: the last screen is drawn
    assert len(out) == 3 and "1.30 m" in out[2]
    t.flush()
    assert len(out) == 3


def _lines(r):
    return r.splitlines()


def _at(lines, text):
    return next(k for k, x in enumerate(lines) if text in x)


def test_the_current_prompt_comes_right_after_the_banner_then_progress_then_details():
    r = tui.render(full("GRID", banner=("motors were on when calibd restarted",), countdown=3,
                        say={"id": 5, "t": 99.0, "kind": "status", "text": "Step back from the head."},
                        placement={"target": "0.75 m", "kind": "default"},
                        grid={"placement": "0.75 m", "pose": 4, "of": 49},
                        health=[{"level": "warn", "message": "camera 3 slow"}]))
    L = _lines(r)
    assert "A = abort" in L[0] and "GRID" in L[0]
    assert _at(L, "NOT NORMAL") < _at(L, "Step back from the head.") < _at(L, "STARTING IN 3") \
        < _at(L, "pose 4/49") < _at(L, "preflight") < _at(L, "camera 3 slow")
    assert L[-1] == tui.keys_for(full("GRID"))


CHECKS = [{"name": f"c{k}", "status": "ok", "detail": f"fine {k}", "gate": "collect"} for k in range(21)] + [
    {"name": "info1", "status": "info", "detail": "i1", "gate": "collect"},
    {"name": "info2", "status": "info", "detail": "i2", "gate": "collect"},
    {"name": "slowcam", "status": "warn", "detail": "camera 3 at 20 fps", "gate": "collect"},
    {"name": "estop", "status": "fail", "detail": "e-stop engaged", "gate": "motion"}]


@pytest.mark.parametrize("phase", ["PREFLIGHT", "READY", "BLOCKED"])
def test_the_whole_preflight_list_is_shown_only_in_the_preflight_phases(phase):
    r = tui.render(full(phase, checks=CHECKS))
    assert "fine 0" in r and "fine 20" in r and "i2" in r and "camera 3 at 20 fps" in r and "e-stop engaged" in r


@pytest.mark.parametrize("phase", ["MOTORS", "PLACE", "GRID", "HOLD", "DEPLOY", "FINISH"])
def test_elsewhere_the_preflight_is_one_summary_line_plus_its_warnings_and_failures(phase):
    r = tui.render(full(phase, checks=CHECKS))
    assert "preflight: 21 ok, 2 info, 1 warn, 1 fail" in r
    assert "fine 0" not in r and "i2" not in r
    assert "camera 3 at 20 fps" in r and "e-stop engaged" in r and "FAIL" in r and "WARN" in r


MOTOR_CHECKS = [{"name": "estop", "status": "ok", "detail": "e-stop released"},
                {"name": "gantry", "status": "ok", "detail": "on the gantry"},
                {"name": "temps", "status": "warn", "detail": "neck pitch 61 C"}]


def test_the_motor_check_list_is_whole_in_motors_and_only_its_problems_in_hold():
    mo = {"checks": MOTOR_CHECKS, "detail": "ready to enable"}
    r = tui.render(full("MOTORS", motors=mo))
    assert "e-stop released" in r and "on the gantry" in r and "neck pitch 61 C" in r and "ready to enable" in r
    r = tui.render(full("HOLD", held_from="GRID", motors=mo))
    assert "e-stop released" not in r and "on the gantry" not in r and "neck pitch 61 C" in r
    r = tui.render(full("GRID", motors=mo))
    assert "e-stop released" not in r and "neck pitch 61 C" not in r


def test_health_shows_each_message_once_with_a_count_newest_first_at_most_three():
    h = [{"level": "warn", "message": "old one"}, {"level": "warn", "message": "lens hot"},
         {"level": "info", "message": "third"}] + [{"level": "warn", "message": "camera 3 slow"}] * 5 + [
        {"level": "fail", "message": "frames stalled"}]
    L = [x for x in _lines(tui.render(full("GRID", health=h))) if x.startswith("health")]
    assert L == ["health FAIL: frames stalled", "health WARN: camera 3 slow x5", "health INFO: third"]


def test_busy_is_shown_as_working_with_the_seconds_since_it_began():
    what = "Deploying: writing files, restarting OKVIS, waiting for it to load the new calibration (up to 3 min)"
    r = tui.render(full("DEPLOY", busy={"what": what, "since": 88.0}, updated=100.0))
    assert f"WORKING: {what} (12 s)" in r
    assert "WORKING" not in tui.render(full("DEPLOY"))


def test_a_pressed_key_is_acknowledged_on_screen_at_once_then_calibd_answer_shown():
    t, sent, out, sp, c = make()
    old = {"key": "Y", "phase": "READY", "accepted": False, "why": "Y does nothing in READY", "t": 90.0}
    t.step(json.dumps(full("MOTORS", last_key=old)))
    assert "Y does nothing" not in out[-1]             # an answer to a key pressed before is not this key's answer
    c.t = 0.1
    t.key("y")
    assert json.loads(sent[-1]) == {"key": "Y"}
    assert len(out) == 2 and "you pressed Y; waiting for calibd" in out[-1]   # drawn at once, no throttle
    assert _at(_lines(out[-1]), "you pressed Y") == 1                          # right under the header
    c.t = 0.7
    t.step(json.dumps(full("MOTORS", last_key=old)))
    assert "you pressed Y; waiting for calibd" in out[-1]
    c.t = 1.0
    ans = {"key": "Y", "phase": "MOTORS", "accepted": True,
           "why": "enabling the motors (or taking them over when already on)", "t": 100.5}
    t.step(json.dumps(full("MOTORS", last_key=ans)))
    assert "Y: enabling the motors (or taking them over when already on)" in out[-1]
    assert "waiting for calibd" not in out[-1]
    c.t = 10.5
    t.step(json.dumps(full("MOTORS", last_key=ans)))
    assert "Y: enabling the motors" in out[-1]
    c.t = 11.5
    t.step(json.dumps(full("MOTORS", last_key=ans)))
    assert "Y: enabling" not in out[-1] and "you pressed" not in out[-1]


def test_a_refused_key_says_why():
    t, sent, out, sp, c = make()
    t.step(json.dumps(full("GRID")))
    t.key("Y")
    c.t = 0.6
    t.step(json.dumps(full("GRID", last_key={"key": "Y", "phase": "GRID", "accepted": False,
                                             "why": "Y does nothing in GRID", "t": 100.0})))
    assert "Y ignored: Y does nothing in GRID" in out[-1]


def test_a_key_without_an_answer_in_3_s_says_so():
    t, sent, out, sp, c = make()
    t.step(json.dumps(full("MOTORS")))
    t.key("Y")
    c.t = 2.5
    t.step(json.dumps(full("MOTORS")))
    assert "you pressed Y; waiting for calibd" in out[-1]
    c.t = 3.2
    t.step(json.dumps(full("MOTORS", last_key={"key": "G", "phase": "MOTORS", "accepted": False, "why": "x",
                                               "t": 100.0})))      # another key's answer is not this one's
    assert "Y sent, no answer from calibd yet" in out[-1] and "you pressed" not in out[-1]


@pytest.mark.parametrize("phase", sorted(tui.FULL_KEYS))
def test_every_full_phase_keeps_abort_in_the_header_and_the_key_help_last(phase):
    L = _lines(tui.render(full(phase, banner=("DRY RUN",), checks=CHECKS)))
    assert "A = abort" in L[0] and phase in L[0]
    assert L[-1] == tui.keys_for(full(phase))
    assert "NOT NORMAL: DRY RUN" in L


# ------------------------------------------------------------------ progress per take and quality per section (10 Nov)


def test_the_take_has_a_progress_bar_and_its_time_left():
    r = tui.render(full("GRID", grid={"placement": "0.75 m", "pose": 23, "of": 49, "eta_s": 74}))
    line = next(x for x in r.splitlines() if x.startswith("take "))
    assert line == "take 0.75 m: pose 23/49 [#########-----------] 47 %  about 1 min left"
    r = tui.render(full("GRID", grid={"placement": "0.75 m", "pose": 45, "of": 49, "eta_s": 12}))
    assert "pose 45/49 [##################--] 92 %  about 12 s left" in r
    r = tui.render(full("CHECK_GRID", grid={"placement": "1.2 m", "pose": 49, "of": 49, "eta_s": 0}))
    assert "pose 49/49 [####################] 100 %" in r and "left" not in r.split("100 %")[1].splitlines()[0]


@pytest.mark.parametrize("phase", ["GRID", "CHECK_GRID", "HOLD", "IMU_SWEEP"])
def test_the_bar_shows_in_the_moving_phases_and_the_imu_sweep(phase):
    assert "pose 3/10 [######--------------] 30 %" in tui.render(full(phase, grid={"placement": "imu", "pose": 3,
                                                                                    "of": 10}))


def test_no_bar_outside_them():
    assert "take " not in tui.render(full("FIT", grid={"placement": "0.75 m", "pose": 3, "of": 10}))


def quality(stop=False):
    return {"cams": {"0": {"sections": [[249, 473, 164], [370, 600, 328], [90, 210, 120]],
                           "pct": [[62, 118, 41], [92, 150, 82], [22, 52, 30]],
                           "edges": {"left": 1, "right": 2, "top": 3, "bottom": 4}, "bottom_strip_pct": 40},
                     "1": {"sections": [[284, 528, 182], [1, 2, 3], [4, 5, 6]],
                           "pct": [[71, 132, 45], [100, 160, 99], [30, 60, 40]], "edges": {}}},
            "third_min": 400,
            "heldout": {"sections": [[[0, 2, 8], [0, 5, 10], [0, 1, 6]], [[3, 10, 10], [10, 10, 10], [1, 0, 0]],
                                     [[0, 0, 0], [2, 3, 4], [9, 9, 9]]],
                        "bands": ["0.4-0.6", "0.6-0.9", "0.9-1.3"], "want": 10, "judged": 5, "judged_min": 6,
                        "far": {"1.3-1.7": 51, "1.7-2.3": 12}, "far_min": 21},
            "blocking": [] if stop else ["more corners in the bottom-left of the left camera",
                                         "the held-out check needs the top-right section at 0.9-1.3 m"],
            "stop_met": stop}


def _q_lines(r):
    L = r.splitlines()
    k = next(i for i, x in enumerate(L) if x.startswith("image sections"))
    return L[k:k + 7]


@pytest.mark.parametrize("phase", ["PLACE", "GRID", "HOLD", "IMU_PLACE", "FIT", "DECIDE"])
def test_quality_per_section_both_cameras_beside_the_held_out_check(phase):
    r = tui.render(full(phase, quality=quality()))
    q = _q_lines(r)
    assert "held-out check: judged 5 of 6 needed" in q[0]
    assert "left camera" in q[1] and "right camera" in q[1] and "0.4-0.6/0.6-0.9/0.9-1.3" in q[1]
    assert q[2].split()[:7] == ["top", "62", "ok", "41", "71", "ok", "45"]
    assert "TL -/2/8" in q[2] and "T  -/5/ok" in q[2] and "TR -/1/6" in q[2]
    assert q[3].split()[:7] == ["middle", "92", "ok", "82", "ok", "ok", "99"] and "C  ok/ok/ok" in q[3]
    assert q[4].startswith("bottom") and "BL -/-/-" in q[4] and "BR 9/9/9" in q[4]
    assert q[5] == "far bands: 1.3-1.7 m 51/21 ok, 1.7-2.3 m 12/21"
    assert q[6].startswith("still needed: more corners in the bottom-left of the left camera; the held-out")
    assert all(len(x) <= 100 for x in q)
    assert "\033[" not in r                                  # plain text unless on a terminal


def test_quality_says_enough_when_the_stop_rule_is_met():
    r = tui.render(full("GRID", quality=quality(stop=True)))
    assert "coverage: enough to stop" in r and "still needed" not in r


def test_quality_is_coloured_only_on_a_terminal():
    r = tui.render(full("GRID", quality=quality()), ansi=True)
    assert f"{tui.GREEN}  ok{tui.PLAIN}" in r and f"{tui.YELLOW}  62{tui.PLAIN}" in r and f"{tui.RED}  41{tui.PLAIN}" in r
    import re
    plain = re.sub(r"\033\[[0-9;]*m", "", r)
    assert all(len(x) <= 100 for x in _q_lines(plain))
    assert _q_lines(plain)[2].index("TL") == _q_lines(tui.render(full("GRID", quality=quality())))[2].index("TL")


def test_quality_shows_in_the_trial_and_not_in_other_phases():
    assert "image sections" in tui.render(state("TRIAL", quality=quality()))
    assert "image sections" not in tui.render(full("DEPLOY", quality=quality()))
    assert "image sections" not in tui.render(full("READY", quality=quality()))


def test_quality_with_missing_keys_skips_them_silently():
    assert "image sections" not in tui.render(full("GRID", quality={}))
    only_cams = tui.render(full("GRID", quality={"cams": {"0": {"pct": [[10, 20, 30]]}}}))
    assert "image sections" in only_cams and "held-out" not in only_cams and "far bands" not in only_cams
    only_held = tui.render(full("GRID", quality={"heldout": {"sections": [[[1, 2, 3]]]}}))
    assert "held-out check" in only_held and "image sections" not in only_held
    assert "still needed: x" in tui.render(full("GRID", quality={"blocking": ["x"]}))
    tui.render(full("GRID", quality={"cams": {"0": {"pct": "junk"}}, "heldout": {"sections": "junk", "far": {}}}))


def test_the_quality_block_is_at_most_8_lines():
    r = tui.render(full("GRID", quality=quality()))
    L = r.splitlines()
    k = next(i for i, x in enumerate(L) if x.startswith("image sections"))
    assert sum(1 for x in L[k:k + 9] if x and not x.startswith(("preflight", "-", "board", "cameras"))) <= 8
