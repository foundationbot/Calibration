"""The laptop side's screen and voice (design plan sections 3, 4, 4.1): renders calibd's state, sends the operator's
keys and a heartbeat, speaks each new prompt once. Stdlib only (the laptop needs nothing installed).

render(state) -> text: a header that is always there (robot, session, phase, motor state, "A = abort"), the NOT NORMAL
banner (calibd's banner lines), the preflight checks, capture counters, the board as detected, the placement prompt,
recent health events and the key help. Tui(send, speaker): step(line) handles one state line from calibd (speaks a
prompt whose id is new and whose age is under SPEAK_MAX_AGE_S, so a re-attach does not replay an old one), heartbeat()
sends {"hb": t} at most once a second, key(k) sends {"key": K}. SpdSay speaks through spd-say (-w: wait until spoken;
the caller runs it off the render thread); Silent is the no-voice speaker; TextPrompts (--voice text) speaks nothing and
keeps the last prompts for a PROMPT panel at the top of the screen (the newest highlighted, the previous ones with
their age), for rooms where a voice is not wanted."""
import collections
import json
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path
import threading
import time

SPEAK_MAX_AGE_S = 5.0
HB_S = 1.0
NOT_STEPPING_S = 3.0     # calibd's step counter unchanged this long (laptop clock) = its loop stopped
KEYS = "A = abort | G = continue after HOLD | T = start the placement trial | C = can't put it there | " \
       "Q = leave (calibd keeps running)"
MARK = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL", "info": "info"}


LEAVE = "Q = leave (calibd keeps running)"
OKVIS_OPT = "O = skip the OKVIS check"
_GUIDE = f"C = can't put it there | {OKVIS_OPT} | A = abort"
_MOVING = f"A = abort (the pose in progress finishes, then nothing more moves) | {OKVIS_OPT}"
_AFTER = "D = disable the motors (twice: the warning first)"
FULL_KEYS = {"PREFLIGHT": "A = abort", "BLOCKED": "A = abort",
             "READY": f"M = motors (the enable screen) | {OKVIS_OPT} | A = abort",
             "MOTORS": "Y = the robot hangs on the gantry and nobody is at its head: enable the motors | A = abort",
             "SETUP": f"{OKVIS_OPT} | A = abort", "PLACE": _GUIDE, "IMU_PLACE": _GUIDE, "CHECK_PLACE": _GUIDE,
             "POST_PLACE": _GUIDE, "GRID": _MOVING, "IMU_SWEEP": _MOVING, "CHECK_GRID": _MOVING,
             "HOLD": "G = continue (after a countdown) | A = abort", "FIT": f"{OKVIS_OPT} | A = abort",
             "DECIDE": "A = abort", "DEPLOY_SIZE": "1 / 2 = deploy size | ? = this robot's facts | A = abort",
             "DEPLOY": "Y = deploy (when asked) | R = roll back | A = abort",
             "POSTCHECK": "R = roll back now | A = abort", "REPORT": "R = roll back now | A = abort",
             "FINISH": "D = disable the motors (twice: the warning first) | K = keep them on | R = roll back now",
             "DONE": _AFTER, "ABORTED": _AFTER, "STOPPED_LOUD": _AFTER}


def keys_for(state) -> str:
    s = state or {}
    if s.get("mode") != "full":
        return KEYS
    more = "".join(f" | {o.split(' (')[0]}" for o in s.get("finish_options") or ()) if s.get("phase") == "FINISH" \
        else ""
    return f"{FULL_KEYS.get(s.get('phase'), 'A = abort')}{more} | {LEAVE}"


def _motors(state):
    if state.get("motors_on") is True:
        return "MOTORS ON"
    if state.get("motors_on") is False:
        return "motors off"
    for c in state.get("checks") or ():
        if c.get("name") == "motors_off":
            return "motors off" if c.get("status") == "ok" else "MOTORS ON"
    return "motors ?"


PANEL_N = 4                 # text mode: the newest prompt and the 3 before it
HILITE, PLAIN = "\033[1;7m", "\033[0m"


def _panel(prompts, width=100, ansi=False) -> list:
    """text mode's PROMPT panel: prompts = [(age_s, text)], newest first"""
    L = []
    for k, (age, text) in enumerate(prompts[:PANEL_N]):
        if k == 0:
            for j, w in enumerate(textwrap.wrap(text, width - 8) or [""]):
                w = f"{w:<{width - 8}}"
                L.append(("PROMPT  " if j == 0 else "        ") + (f"{HILITE}{w}{PLAIN}" if ansi else w.rstrip()))
        else:
            L.append("        " + textwrap.shorten(f"{max(0, round(age)):.0f} s ago: {text}", width - 8,
                                                   placeholder=" ..."))
    return L


def render(state, width=100, notes=(), prompts=None, ansi=False) -> str:
    """notes: the laptop's own NOT NORMAL lines (e.g. calibd not stepping), shown before calibd's banner; prompts (text
    mode): [(age_s, text)] newest first, shown as the PROMPT panel under the banner, above everything else"""
    s = state or {}
    L = []
    L.append(f"calib  {s.get('robot', '?')}  session {s.get('session', '?')}  phase {s.get('phase', '?')}  "
             f"{_motors(s)}  link {'up' if s.get('attached') else 'down'}   [A = abort]")
    L.append("-" * width)
    banner = list(notes) + list(s.get("banner") or ())
    for b in banner:
        L.append(f"NOT NORMAL: {b}")
    if banner:
        L.append("-" * width)
    if prompts:
        L += _panel(prompts, width, ansi)
        L.append("-" * width)
    checks = s.get("checks") or []
    if checks:
        L.append("preflight (collect gate / motion gate, M5):")
        for c in checks:
            L.append(f"  {MARK.get(c.get('status'), '?   ')} {c.get('gate', ''):7s} {c.get('name', ''):16s} "
                     f"{str(c.get('detail', ''))[:width - 34]}")
    cap = s.get("capture") or {}
    if cap:
        cams = cap.get("cams") or {}
        per = ", ".join(f"queue {q}: {c.get('fps', 0):.1f} fps, {c.get('none_reads', 0)} empty reads"
                        for q, c in sorted(cams.items()))
        sync = cap.get("sync_median_ms")
        L.append(f"cameras: {cap.get('pairs', 0)} pairs; {per}; L/R sync "
                 f"{'?' if sync is None else f'{sync:.3f}'} ms")
    b = s.get("board")
    if b:
        cut = f", cut by the {'/'.join(b['cut'])} edge" if b.get("cut") else ""
        L.append(f"board: {b.get('z_m', 0):.2f} m, {b.get('tags', 0)} tags{cut}")
    elif cap:
        L.append("board: not seen")
    tr = s.get("trial")
    if tr:
        L.append("placement trial: " + _trial_line(tr))
        g = tr.get("gravity") or {}
        if g:
            L.append(f"  gravity: camera {g.get('pitch_down_deg')} deg below the horizon ({g.get('note', '')})"
                     if g.get("ok") else f"  gravity: NOT AVAILABLE -- {g.get('note', '')}")
        tb = tr.get("board") or {}
        if tb.get("dh_m") is not None:
            dh = float(tb["dh_m"])
            L.append(f"  board centre {abs(round(dh * 100)):.0f} cm {'below' if dh < 0 else 'above'} the camera, "
                     f"{round(float(tb['dxy_m']) * 100):.0f} cm along the floor")
        for r in tr.get("records") or ():
            L.append(f"  {r['target']:7s} {r['result']:11s} {r['seconds']:6.1f} s, {r.get('prompts', 0)} prompts")
    if s.get("mode") == "full":
        L += _full_lines(s, width)
    say = s.get("say")
    if say and not prompts:
        L.append(f'>> "{say.get("text")}"')
    for e in (s.get("health") or [])[-5:]:
        L.append(f"health {e.get('level', '').upper()}: {e.get('message')}")
    L.append("-" * width)
    L.append(keys_for(s))
    return "\n".join(L)


def _reasons(rs, n=4, width=100):
    return [f"    - {str(r)[:width - 6]}" for r in list(rs or [])[:n]]


def _full_lines(s, width=100) -> list:
    """the full session's screen: what the current phase needs the operator to see"""
    L = []
    mo = s.get("motors")
    if mo and s.get("phase") in ("MOTORS", "HOLD"):
        L.append(f"motors: {mo.get('detail', '')}"[:width])
        for c in mo.get("checks") or ():
            L.append(f"  {MARK.get(c.get('status'), '?   ')} {c.get('name', ''):16s} "
                     f"{str(c.get('detail', ''))[:width - 24]}")
    if s.get("countdown"):
        L.append(f">>> STARTING IN {s['countdown']} -- step back from the head <<<")
    pl = s.get("placement")
    if pl and s.get("phase") in ("PLACE", "GRID", "IMU_PLACE", "IMU_SWEEP", "CHECK_PLACE", "CHECK_GRID", "POST_PLACE",
                                 "HOLD"):
        L.append(f"placement: {pl.get('target')} ({pl.get('kind')})")
    g = s.get("grid")
    if g and g.get("pose") is not None and s.get("phase") in ("GRID", "CHECK_GRID", "HOLD"):
        last = g.get("last") or {}
        cmd, meas = last.get("command") or [], last.get("measured") or []
        tail = "" if not cmd else (f"; last pose {cmd[0]:+.0f}/{cmd[1]:+.0f} deg, measured {meas[0]:+.1f}/"
                                   f"{meas[1]:+.1f}, {last.get('tags_l', 0)} tags, "
                                   f"{'board found' if last.get('found') else 'NO BOARD'}")
        L.append(f"grid {g.get('placement')}: pose {g.get('pose')} of {g.get('of')}{tail}"[:width])
    sw = s.get("sweep")
    if sw:
        L.append(f"IMU sweep: {'apply the rotation' if sw.get('deploy') else 'keep the live rotation'}")
        L += _reasons(sw.get("reasons"), 2, width)
    d = s.get("decision")
    if d:
        L.append(f"lens verdict: {d.get('decision')}")
        L += _reasons(d.get("reasons"), 4, width)
    ds = s.get("deploy_size")
    if ds:
        if ds.get("chosen"):
            L.append(f"deploy size: {ds['chosen'].get('label')}")
        else:
            L.append(f"deploy size (the previous default {ds.get('prev', ['?', '?'])[0]}x"
                     f"{ds.get('prev', ['?', '?'])[1]}):")
            L += [f"  {o.get('key')}: {o.get('label')}" for o in ds.get("options") or ()]
        L += [f"    {h}"[:width] for h in ds.get("help") or ()]
    ck = s.get("check")
    if ck:
        L.append(f"check grid: {ck.get('decision') or 'skipped: ' + str(ck.get('skipped'))}"[:width])
        L += _reasons(ck.get("reasons"), 3, width)
    dep = s.get("deploy")
    if dep:
        if dep.get("interrupted"):
            L.append("deploy INTERRUPTED by a calibd restart: " + ("the robot changed: " + "; ".join(dep["drift"])
                                                                 if dep.get("drift") else "nothing changed yet"))
        elif dep.get("drift") or dep.get("refused"):
            L.append(f"deploy: {dep.get('drift') or dep.get('refused')}"[:width * 2])
        else:
            L.append(f"deploy: {dep.get('detail', '')}; rollback script {dep.get('rollback_script')}"[:width * 2])
    pc_ = s.get("postcheck")
    if pc_:
        dec = pc_.get("decision") or {}
        L.append(f"post-check: {'ROLL BACK' if dec.get('rollback') else 'keep the new calibration'}")
        L += _reasons(dec.get("reasons"), 4, width)
    rb = s.get("rollback")
    if rb:
        L.append(f"rollback: {'done' if rb.get('ok') or rb.get('dry_run') else 'NOT DONE'}"
                 f" ({rb.get('reason') or rb.get('drift') or rb.get('message')})"[:width * 2])
    rs = s.get("restore")
    if rs:
        L.append(f"restored: {'; '.join(rs.get('done') or ['nothing to restore'])}"
                 + (f"; ERRORS: {'; '.join(rs['errors'])}" if rs.get("errors") else ""))
    if s.get("finish_options") and s.get("phase") == "FINISH":
        L.append("not enough data to decide; you can also:")
        L += [f"  {o}"[:width * 2] for o in s["finish_options"]]
    if s.get("motors_prompt") and s.get("phase") in ("FINISH", "DONE", "ABORTED", "STOPPED_LOUD"):
        L.append(f"motors: {s['motors_prompt']}"[:width * 2])
    if s.get("dry_run"):
        L.append("DRY RUN: nothing is sent to the robot; the commands go to commands.txt in the session folder")
    return L


def ssml_with_pauses(text, word_ms=150, sentence_ms=450) -> str:
    """the text as SSML with a pause after every word and a longer one after every sentence (9 Nov trial: the
    operator asked for more pause between words; spd-say has no word-gap option, but -x takes SSML breaks)"""
    from xml.sax.saxutils import escape
    out = []
    for w in text.split():
        end = w[-1] in ".!?"
        out.append(escape(w) + f" <break time='{sentence_ms if end else word_ms}ms'/>")
    return "<speak>" + " ".join(out) + "</speak>"


class SpdSay:
    def __init__(self, path="/usr/bin/spd-say", run=subprocess.run, rate=-95, lang="en-us", voice="male1",
                 word_ms=150, sentence_ms=450):
        # 9 Nov trial: rate -10 was too fast and unclear; the operator asked for slower twice (-45, -70 still fast),
        # then for pauses between words
        self.path, self.run, self.rate, self.lang, self.voice = path, run, rate, lang, voice
        self.word_ms, self.sentence_ms = word_ms, sentence_ms

    def say(self, text):
        self.run([self.path, "-w", "-x", "-l", self.lang, "-t", self.voice, "-r", str(self.rate),
                  ssml_with_pauses(text, self.word_ms, self.sentence_ms)], timeout=90)


PIPER_DIR = Path.home() / ".local" / "share" / "calib-voice"


class PiperSay:
    """a local neural voice (Piper, rhasspy; voice en_US-lessac-medium) in place of espeak, which the operator found
    'ridiculously artificial' (10 Nov). Text -> wav with piper (length_scale > 1 = slower), then played. Opt-in:
    make_speaker("piper") falls back to SpdSay when the binary or the model is missing."""

    def __init__(self, binary=None, model=None, player="/usr/bin/pw-play", run=subprocess.run, length_scale=1.3,
                 sentence_silence=0.45, wav=None):
        self.binary = Path(binary or PIPER_DIR / "piper" / "piper")
        self.model = Path(model or PIPER_DIR / "en_US-lessac-medium.onnx")
        self.player, self.run = player, run
        self.length_scale, self.sentence_silence = length_scale, sentence_silence
        self.wav = str(wav or Path(tempfile.gettempdir()) / f"calibtool-say-{os.getpid()}.wav")

    def available(self) -> bool:
        return self.binary.is_file() and self.model.is_file()

    def say(self, text):
        self.run([str(self.binary), "--model", str(self.model), "--length_scale", str(self.length_scale),
                  "--sentence_silence", str(self.sentence_silence), "--output_file", self.wav],
                 input=(text.strip() + "\n").encode(), timeout=60, capture_output=True)
        self.run([self.player, self.wav], timeout=90)


def make_speaker(voice="espeak", piper_dir=None):
    """espeak (spd-say, the default), piper when asked and installed, text = the screen's PROMPT panel instead of a
    voice; anything else falls back to spd-say"""
    if voice == "text":
        return TextPrompts()
    if voice == "piper":
        d = Path(piper_dir) if piper_dir is not None else PIPER_DIR
        sp = PiperSay(binary=d / "piper" / "piper", model=d / "en_US-lessac-medium.onnx")
        if sp.available():
            return sp
    return SpdSay()


class Silent:
    def say(self, text):
        pass


class TextPrompts:
    """--voice text: nothing is spoken; the TUI shows the last PANEL_N prompts at the top of the screen. Instant, so it
    needs no Background thread; it also shows the prompt current when the TUI attaches (voice skips it: a replay)"""
    shows_stale = True

    def __init__(self, n=PANEL_N):
        self.history = collections.deque(maxlen=n)      # (calibd time, text), oldest first

    def show(self, x):
        self.history.append((float(x.get("t") or 0), str(x.get("text", ""))))

    def say(self, text):
        self.history.append((0.0, str(text)))

    def prompts(self, now) -> list:
        return [(now - t, text) for t, text in reversed(self.history)]


SUPERSEDED = frozenset({"correction", "lost", "status", "still"})   # a newer one of these makes the queued ones stale


class Background:
    """a speaker that never blocks the caller: texts queue up and are spoken in order by one thread. A correction (or
    "I cannot see the board") still waiting when a newer one arrives is dropped: the board has moved on (review 8: the
    queue was unbounded). Intros, skips, ends, "out of reach", "Good" and the camera prompt are never dropped."""

    def __init__(self, speaker):
        self.speaker = speaker
        self._q = collections.deque()
        self._cv = threading.Condition()
        self.dropped = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            with self._cv:
                while not self._q:
                    self._cv.wait()
                text, _ = self._q.popleft()
            try:
                self.speaker.say(text)
            except Exception:  # noqa: BLE001  (a speech failure must not take the TUI down)
                pass

    def say_prompt(self, text, kind=None):
        with self._cv:
            if kind in SUPERSEDED:
                keep = [x for x in self._q if x[1] not in SUPERSEDED]
                self.dropped += len(self._q) - len(keep)
                self._q = collections.deque(keep)
            self._q.append((text, kind))
            self._cv.notify()

    def say(self, text):
        self.say_prompt(text, None)


def _trial_line(tr) -> str:
    """the CURRENT placement, said plainly (9 Nov trial: a stale "placement 2 of 6" line misled everyone)"""
    where = f"placement {tr.get('index')} of {tr.get('of')} ({tr.get('target')})"
    if tr.get("finished"):
        return f"finished, {tr.get('of')} placements"
    if not tr.get("started"):
        return f"not started: next is {where} (press T)"
    since, limit = tr.get("since_intro_s"), tr.get("timeout_s")
    clock = "" if since is None else f", {since:.0f} s" + ("" if limit is None else f" of {limit:.0f} s")
    return f"NOW {where}{clock}"


class Tui:
    def __init__(self, send, speaker, clock=time.monotonic, out=print, speak_max_age_s=SPEAK_MAX_AGE_S, ansi=False):
        self.send_line, self.speaker, self.clock, self.out = send, speaker or Silent(), clock, out
        self.ansi = ansi
        self.max_age = speak_max_age_s
        self.spoken = None
        self._fresh = True              # nothing seen yet: old prompts (attaching late) are not spoken
        self._hb = None
        self.state = None
        self._step_n = None             # calibd's step counter and when (laptop clock) it last changed
        self._step_seen = None

    def step(self, line):
        try:
            st = json.loads(line)
        except json.JSONDecodeError:
            return None
        self.state = st
        say = st.get("say")
        says = [x for x in (st.get("says") or ([say] if say else [])) if x and x.get("id") is not None]
        if says:
            ids = [x["id"] for x in says]
            if self.spoken is not None and max(ids) < self.spoken:
                self.spoken, self._fresh = None, True      # calibd restarted: its prompt numbers began again
            for x in sorted(says, key=lambda x: x["id"]):
                if self.spoken is not None and x["id"] <= self.spoken:
                    continue
                age = float(st.get("updated") or x.get("t") or 0) - float(x.get("t") or 0)
                if not self._fresh or age <= self.max_age or getattr(self.speaker, "shows_stale", False):
                    self._speak(x)
                self.spoken = x["id"]
            self._fresh = False
        pf = getattr(self.speaker, "prompts", None)
        self.out(render(st, notes=self.notes(st), ansi=self.ansi,
                        prompts=pf(float(st.get("updated") or 0)) if pf else None))
        return st

    def _speak(self, x):
        show = getattr(self.speaker, "show", None)
        if show is not None:
            return show(x)                      # text mode: the prompt with calibd's time (for its age on screen)
        f = getattr(self.speaker, "say_prompt", None)
        if f is not None:
            f(x["text"], x.get("kind"))         # the kind lets a queued, stale correction be dropped
        else:
            self.speaker.say(x["text"])

    def notes(self, st) -> list:
        """the laptop's own NOT NORMAL lines: this TUI not in control (another attached later), calibd's step counter
        not advancing while its state keeps coming (its loop stopped: review 3)"""
        out = []
        cl = st.get("client") or {}
        if cl.get("claimed") and cl.get("controlling") is False:
            out.append("another TUI attached later and has control: keys here are ignored (A still aborts)")
        n, now = st.get("step_n"), self.clock()
        if n is None:
            return out
        if n != self._step_n:
            self._step_n, self._step_seen = n, now
            return out
        idle = now - self._step_seen
        if idle > NOT_STEPPING_S:
            out.append(f"calibd not stepping: its loop has not advanced for {idle:.0f} s (the prompts and the trial "
                       "are stopped; check calibd.log)")
        return out

    def heartbeat(self):
        t = self.clock()
        if self._hb is None or t - self._hb >= HB_S:
            self._hb = t
            self.send_line(json.dumps({"hb": t}))

    def key(self, k):
        self.send_line(json.dumps({"key": str(k).upper()}))
