"""The laptop side's screen and voice (design plan sections 3, 4, 4.1): renders calibd's state, sends the operator's
keys and a heartbeat, speaks each new prompt once. Stdlib only (the laptop needs nothing installed).

render(state) -> text: a header that is always there (robot, session, phase, motor state, "A = abort"), the NOT NORMAL
banner (calibd's banner lines), the preflight checks, capture counters, the board as detected, the placement prompt,
recent health events and the key help. Tui(send, speaker): step(line) handles one state line from calibd (speaks a
prompt whose id is new and whose age is under SPEAK_MAX_AGE_S, so a re-attach does not replay an old one), heartbeat()
sends {"hb": t} at most once a second, key(k) sends {"key": K}. SpdSay speaks through spd-say (-w: wait until spoken;
the caller runs it off the render thread); Silent is the no-voice speaker."""
import json
import queue
import subprocess
import threading
import time

SPEAK_MAX_AGE_S = 5.0
HB_S = 1.0
KEYS = "A = abort | G = continue after HOLD | T = start the placement trial | C = can't put it there | " \
       "Q = leave (calibd keeps running)"
MARK = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL", "info": "info"}


def _motors(state):
    for c in state.get("checks") or ():
        if c.get("name") == "motors_off":
            return "motors off" if c.get("status") == "ok" else "MOTORS ON"
    return "motors ?"


def render(state, width=100) -> str:
    s = state or {}
    L = []
    L.append(f"calib  {s.get('robot', '?')}  session {s.get('session', '?')}  phase {s.get('phase', '?')}  "
             f"{_motors(s)}  link {'up' if s.get('attached') else 'down'}   [A = abort]")
    L.append("-" * width)
    for b in s.get("banner") or ():
        L.append(f"NOT NORMAL: {b}")
    if s.get("banner"):
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
        L.append(f"placement trial: {tr.get('index')} of {tr.get('of')} ({tr.get('target')})"
                 + (" -- finished" if tr.get("finished") else ""))
        for r in tr.get("records") or ():
            L.append(f"  {r['target']:7s} {r['result']:11s} {r['seconds']:6.1f} s, {r.get('prompts', 0)} prompts")
    say = s.get("say")
    if say:
        L.append(f'>> "{say.get("text")}"')
    for e in (s.get("health") or [])[-5:]:
        L.append(f"health {e.get('level', '').upper()}: {e.get('message')}")
    L.append("-" * width)
    L.append(KEYS)
    return "\n".join(L)


def ssml_with_pauses(text, word_ms=300, sentence_ms=600) -> str:
    """the text as SSML with a pause after every word and a longer one after every sentence (9 Nov trial: the
    operator asked for more pause between words; spd-say has no word-gap option, but -x takes SSML breaks)"""
    from xml.sax.saxutils import escape
    out = []
    for w in text.split():
        end = w[-1] in ".!?"
        out.append(escape(w) + f" <break time='{sentence_ms if end else word_ms}ms'/>")
    return "<speak>" + " ".join(out) + "</speak>"


class SpdSay:
    def __init__(self, path="/usr/bin/spd-say", run=subprocess.run, rate=-85, lang="en-us", voice="male1",
                 word_ms=300, sentence_ms=600):
        # 9 Nov trial: rate -10 was too fast and unclear; the operator asked for slower twice (-45, -70 still fast),
        # then for pauses between words
        self.path, self.run, self.rate, self.lang, self.voice = path, run, rate, lang, voice
        self.word_ms, self.sentence_ms = word_ms, sentence_ms

    def say(self, text):
        self.run([self.path, "-w", "-x", "-l", self.lang, "-t", self.voice, "-r", str(self.rate),
                  ssml_with_pauses(text, self.word_ms, self.sentence_ms)], timeout=90)


class Silent:
    def say(self, text):
        pass


class Background:
    """a speaker that never blocks the caller: texts queue up and are spoken in order by one thread"""

    def __init__(self, speaker):
        self.speaker = speaker
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            t = self.q.get()
            try:
                self.speaker.say(t)
            except Exception:  # noqa: BLE001  (a speech failure must not take the TUI down)
                pass

    def say(self, text):
        self.q.put(text)


class Tui:
    def __init__(self, send, speaker, clock=time.monotonic, out=print, speak_max_age_s=SPEAK_MAX_AGE_S):
        self.send_line, self.speaker, self.clock, self.out = send, speaker or Silent(), clock, out
        self.max_age = speak_max_age_s
        self.spoken = None
        self._hb = None
        self.state = None

    def step(self, line):
        try:
            st = json.loads(line)
        except json.JSONDecodeError:
            return None
        self.state = st
        say = st.get("say")
        if say and say.get("id") != self.spoken:
            age = float(st.get("updated") or say.get("t") or 0) - float(say.get("t") or 0)
            if self.spoken is not None or age <= self.max_age:
                self.speaker.say(say["text"])
            self.spoken = say.get("id")
        self.out(render(st))
        return st

    def heartbeat(self):
        t = self.clock()
        if self._hb is None or t - self._hb >= HB_S:
            self._hb = t
            self.send_line(json.dumps({"hb": t}))

    def key(self, k):
        self.send_line(json.dumps({"key": str(k).upper()}))
