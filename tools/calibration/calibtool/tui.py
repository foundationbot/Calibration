"""The laptop side's screen and voice (design plan sections 3, 4, 4.1): renders calibd's state, sends the operator's
keys and a heartbeat, speaks each new prompt once. Stdlib only (the laptop needs nothing installed).

render(state) -> text, most important first (10 Nov: the operator glances at it from the robot): a header that is
always there (robot, session, phase, motor state, "A = abort"), the key acknowledgement and calibd's WORKING line, the
NOT NORMAL banner (calibd's banner lines), the current prompt, the progress (countdown, placement, the take's progress
bar with its time left, trial), the per-section data status (both cameras' image sections beside the held-out check),
this phase's details (the whole preflight list only in PREFLIGHT/READY/BLOCKED, else one summary line and its problems;
the motor checks whole only in MOTORS), the health events once each (newest first, at most 3) and the key help.
Tui(send, speaker): step(line) handles one state line from calibd (speaks a prompt whose id is new and whose age is
under SPEAK_MAX_AGE_S, so a re-attach does not replay an old one) and draws the screen only when its text changed, at
most twice a second; heartbeat() sends {"hb": t} at most once a second, key(k) sends {"key": K} and shows it at once,
then calibd's answer (state["last_key"]); link_lost(robot, n) / link_back(): the attach link dropped (a NOT NORMAL
line with the attempt, what calibd does meanwhile; heartbeats and keys not sent and not queued) and came back
("Connection back" for LINK_BACK_S). SpdSay speaks through spd-say (-w: wait until spoken;
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
MIN_DRAW_S = 0.5         # at most 2 screens a second (10 Nov: the whole screen was re-printed for every state line)
KEY_NO_ANSWER_S = 3.0    # a key calibd has not answered this long: said so on screen
KEY_SHOW_S = 10.0        # calibd's answer to a key stays on screen this long
KEY_GIVE_UP_S = 30.0     # an unanswered key's line goes after this long
LINK_BACK_S = 5.0        # "Connection back" stays on screen this long after a re-attach
KEYS = "A = abort | G = continue after HOLD | T = start the placement trial | C = can't put it there | " \
       "Q = leave (calibd keeps running)"
MARK = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL", "info": "info"}


LEAVE = "Q = leave (calibd keeps running)"
OKVIS_OPT = "O = skip the OKVIS check"
_GUIDE = f"C = can't put it there | {OKVIS_OPT} | A = abort"
_MOVING = f"A = abort (the pose in progress finishes, then nothing more moves) | {OKVIS_OPT}"
_AFTER = "D = disable the motors (twice: the warning first)"
FULL_KEYS = {"PREFLIGHT": "A = abort",
             "BLOCKED": "S = start the motor master (when it is not running) | the checks run again every 10 s | "
                        "A = abort",
             "READY": f"M = motors (the enable screen) | {OKVIS_OPT} | A = abort",
             "MOTORS": "Y = the robot hangs on the gantry and nobody is at its head: enable the motors | "
                       "R = restart the motor master (motors off) | A = abort",
             "SETUP": f"{OKVIS_OPT} | A = abort",
             "PLACE": f"C = can't put it there | U = use the board where it is | {OKVIS_OPT} | A = abort", "IMU_PLACE": _GUIDE, "CHECK_PLACE": _GUIDE,
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


PREFLIGHT_PHASES = ("PREFLIGHT", "READY", "BLOCKED")      # the whole preflight list; elsewhere one summary line
BOARD_PHASES = ("PLACE", "GRID", "IMU_PLACE", "IMU_SWEEP", "CHECK_PLACE", "CHECK_GRID", "POST_PLACE", "HOLD")
HEALTH_N = 3


def render(state, width=100, notes=(), prompts=None, ansi=False, ack=()) -> str:
    """notes: the laptop's own NOT NORMAL lines (e.g. calibd not stepping), shown before calibd's banner; prompts (text
    mode): [(age_s, text)] newest first, shown as the PROMPT panel under the banner, above everything else; ack: the
    key acknowledgement lines (under the header). 10 Nov: most important first -- header, NOT NORMAL, the current
    prompt, progress, then the details of this phase only (the operator glances at it from the robot)."""
    s = state or {}
    sep = "-" * width
    L = [f"calib  {s.get('robot', '?')}  session {s.get('session', '?')}  phase {s.get('phase', '?')}  "
         f"{_motors(s)}  link {'up' if s.get('attached') else 'down'}   [A = abort]"]
    L += [str(a)[:width * 2] for a in ack or ()]
    w = _working(s)
    if w:
        L.append(w[:width * 2])
    L.append(sep)
    banner = list(notes) + list(s.get("banner") or ())
    for b in banner:
        L.append(f"NOT NORMAL: {b}")
    if banner:
        L.append(sep)
    say = s.get("say")
    if prompts:
        L += _panel(prompts, width, ansi)
        L.append(sep)
    elif say and say.get("text"):
        for x in textwrap.wrap(f'>> "{say.get("text")}"', width, subsequent_indent="   ") or [""]:
            L.append(f"\033[1m{x}{PLAIN}" if ansi else x)
        L.append(sep)
    L += _progress_lines(s, width)
    L += _quality_lines(s, width, ansi)
    L += _detail_lines(s, width)
    L += _health_lines(s.get("health"), width)
    L.append(sep)
    L.append(keys_for(s))
    return "\n".join(L)


def _working(s):
    """calibd's busy (a long step under way, e.g. the deploy): what, and for how long (calibd's clock)"""
    b = s.get("busy")
    if not b or not b.get("what"):
        return None
    try:
        el = f" ({max(0.0, float(s.get('updated')) - float(b.get('since'))):.0f} s)"
    except (TypeError, ValueError):
        el = ""
    return f"WORKING: {b['what']}{el}"


def _check_line(c, width, gate=True):
    g = f"{c.get('gate', ''):7s} " if gate else ""
    return f"  {MARK.get(c.get('status'), '?   ')} {g}{c.get('name', ''):16s} {str(c.get('detail', ''))[:width - 34]}"


def _preflight_lines(s, width):
    checks = s.get("checks") or []
    if not checks:
        return []
    if s.get("phase") in PREFLIGHT_PHASES:
        return ["preflight (collect gate / motion gate, M5):"] + [_check_line(c, width) for c in checks]
    n = collections.Counter(c.get("status") for c in checks)
    parts = [f"{n[k]} {k}" for k in ("ok", "info", "warn", "fail") if n[k]]
    parts += [f"{v} {k}" for k, v in n.items() if k not in ("ok", "info", "warn", "fail")]
    return [f"preflight: {', '.join(parts)}"] + [_check_line(c, width) for c in checks
                                                 if c.get("status") not in ("ok", "info")]


def _health_lines(health, width):
    """each message once (with a count), newest first, at most HEALTH_N (10 Nov: 5 lines repeated one warning)"""
    seen, order = collections.Counter(), []
    for e in reversed(list(health or ())):
        k = (str(e.get("level", "")).upper(), str(e.get("message")))
        if k not in seen:
            order.append(k)
        seen[k] += 1
    return [f"health {lv}: {msg}{f' x{seen[(lv, msg)]}' if seen[(lv, msg)] > 1 else ''}"[:width * 2]
            for lv, msg in order[:HEALTH_N]]


def _progress_lines(s, width=100) -> list:
    """where the session is: the countdown, the placement, the grid pose, the placement trial"""
    L = []
    ph = s.get("phase")
    full = s.get("mode") == "full"
    if full and s.get("countdown"):
        L.append(f">>> STARTING IN {s['countdown']} -- step back from the head <<<")
    pl = s.get("placement")
    if full and pl and ph in BOARD_PHASES:
        L.append(f"placement: {pl.get('target')} ({pl.get('kind')})")
    g = s.get("grid")
    if full and g and g.get("pose") is not None and ph in GRID_PHASES:
        L += _grid_lines(g, width)
    tr = s.get("trial")
    if tr:
        L.append("placement trial: " + _trial_line(tr))
        gr = tr.get("gravity") or {}
        if gr:
            L.append(f"  gravity: camera {gr.get('pitch_down_deg')} deg below the horizon ({gr.get('note', '')})"
                     if gr.get("ok") else f"  gravity: NOT AVAILABLE -- {gr.get('note', '')}")
        tb = tr.get("board") or {}
        if tb.get("dh_m") is not None:
            dh = float(tb["dh_m"])
            L.append(f"  board centre {abs(round(dh * 100)):.0f} cm {'below' if dh < 0 else 'above'} the camera, "
                     f"{round(float(tb['dxy_m']) * 100):.0f} cm along the floor")
        for r in tr.get("records") or ():
            L.append(f"  {r['target']:7s} {r['result']:11s} {r['seconds']:6.1f} s, {r.get('prompts', 0)} prompts")
    return L


GRID_PHASES = ("GRID", "CHECK_GRID", "HOLD", "IMU_SWEEP")       # the take's progress bar
QUALITY_PHASES = ("PLACE", "GRID", "HOLD", "IMU_PLACE", "FIT", "DECIDE")   # the per-section data status
BAR_W = 20
GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"


def _bar(done, of, w=BAR_W) -> str:
    f = 0 if not of else max(0, min(w, round(w * float(done) / float(of))))
    return "[" + "#" * f + "-" * (w - f) + "]"


def _eta(eta_s) -> str:
    e = float(eta_s)
    return f"about {max(1, round(e)):.0f} s left" if e < 60 else f"about {round(e / 60):.0f} min left"


def _grid_lines(g, width=100) -> list:
    """the take's progress: `take 0.75 m: pose 23/49 [#########-----------] 47 %  about 1 min left`, then the last
    pose (10 Nov, phantom11: "show progress for each take")"""
    take = g.get("placement")
    label = f"{take:g} m" if isinstance(take, (int, float)) else str(take if take is not None else "?")
    pose, of = g.get("pose"), g.get("of")
    try:
        pose, of = int(pose), int(of)
        line = f"take {label}: pose {pose}/{of} {_bar(pose, of)} {round(100.0 * pose / of) if of else 0:.0f} %"
    except (TypeError, ValueError):
        line = f"take {label}: pose {pose} of {of}"
    if g.get("eta_s") is not None:
        try:
            if float(g["eta_s"]) > 0:
                line += f"  {_eta(g['eta_s'])}"
        except (TypeError, ValueError):
            pass
    L = [line[:width]]
    last = g.get("last") or {}
    cmd, meas = last.get("command") or [], last.get("measured") or []
    if len(cmd) >= 2 and len(meas) >= 2:
        L.append(f"  last pose {cmd[0]:+.0f}/{cmd[1]:+.0f} deg, measured {meas[0]:+.1f}/{meas[1]:+.1f}, "
                 f"{last.get('tags_l', 0)} tags, {'board found' if last.get('found') else 'NO BOARD'}"[:width])
    return L


QUALITY_LEFT_W = 44                  # the image-section table; the held-out table to its right
_ROW_NAMES = ("top", "middle", "bottom")
_HELD_NAMES = (("TL", "T", "TR"), ("L", "C", "R"), ("BL", "B", "BR"))


def _col(text, colour, ansi):
    """(plain, shown): padding is counted on the plain text"""
    return text, (f"{colour}{text}{PLAIN}" if ansi and colour else text)


def _pct_cell(v, ansi):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return _col(f"{'?':>4}", None, ansi)
    if v >= 100:
        return _col(f"{'ok':>4}", GREEN, ansi)
    return _col(f"{v:>4}", YELLOW if v >= 50 else RED, ansi)


def _join(cells):
    return "".join(c[0] for c in cells), "".join(c[1] for c in cells)


def _quality_left(q, ansi) -> list:
    cams = q.get("cams") or {}
    if not isinstance(cams, dict) or not any(isinstance(cams.get(k), dict) for k in ("0", "1")):
        return []
    rows = [_col("image sections: % of target (ok = reached)", None, ansi),
            _col(f"{'':8}{'left camera':<16}{'right camera':<12}", None, ansi)]
    for i, name in enumerate(_ROW_NAMES):
        cells = [_col(f"{name:<8}", None, ansi)]
        for k in ("0", "1"):
            pct = (cams.get(k) or {}).get("pct") or []
            row = pct[i] if i < len(pct) and isinstance(pct[i], (list, tuple)) else []
            cells += [_pct_cell(row[j], ansi) if j < len(row) else _col("    ", None, ansi) for j in range(3)]
            cells.append(_col("    ", None, ansi))
        rows.append(_join(cells))
    return rows


def _held_cell(ns, want):
    vals = []
    for n in ns or ():
        try:
            n = int(n)
        except (TypeError, ValueError):
            vals.append("?")
            continue
        vals.append("ok" if want and n >= want else "-" if n == 0 else str(n))
    return "/".join(vals) or "?"


def _quality_right(q, ansi) -> list:
    h = q.get("heldout") or {}
    sec = h.get("sections")
    if not isinstance(sec, (list, tuple)) or not sec:
        return []
    want = h.get("want")
    head = "held-out check"
    if h.get("judged") is not None and h.get("judged_min") is not None:
        head += f": judged {h['judged']} of {h['judged_min']} needed"
    bands = "/".join(str(b) for b in h.get("bands") or ())
    rows = [_col(head, None, ansi), _col(f"holds per band {bands} m" if bands else "holds per distance band", None,
                                         ansi)]
    for i, names in enumerate(_HELD_NAMES):
        row = sec[i] if i < len(sec) and isinstance(sec[i], (list, tuple)) else []
        cells = []
        for j, nm in enumerate(names):
            if j >= len(row):
                continue
            v = _held_cell(row[j], want)
            done = all(x == "ok" for x in v.split("/"))
            cells.append(_col(f"{nm:<2} ", None, ansi))
            cells.append(_col(f"{v:<9}", GREEN if done else None, ansi))
        rows.append(_join(cells))
    return rows


def _quality_lines(s, width=100, ansi=False) -> list:
    """the per-section data status (state["quality"], quality.snapshot), at most 7 lines within width: the image
    sections of both cameras (% of the target, ok = reached; on a terminal green ok, yellow 50-99, red under 50)
    beside the held-out check per section and distance band, then the far bands and what still blocks the stop"""
    q = s.get("quality")
    if not isinstance(q, dict):
        return []
    if s.get("phase") not in QUALITY_PHASES and not s.get("trial"):
        return []
    left, right = _quality_left(q, ansi), _quality_right(q, ansi)
    L = []
    for k in range(max(len(left), len(right))):
        lp, ls = left[k] if k < len(left) else ("", "")
        rp, rs = right[k] if k < len(right) else ("", "")
        if left and right:
            pad = " " * max(1, QUALITY_LEFT_W - len(lp))
            plain, shown = lp + pad + rp, ls + pad + rs
        else:
            plain, shown = lp + rp, ls + rs
        if plain.strip():
            L.append(shown if len(plain) <= width else plain[:width])
    h = q.get("heldout") or {}
    far = h.get("far")
    if isinstance(far, dict) and far:
        fm = h.get("far_min")
        parts = []
        for band, n in far.items():
            ok = fm is not None and n is not None and float(n) >= float(fm)
            parts.append(f"{band} m {n}" + (f"/{fm}" if fm is not None else "") + (" ok" if ok else ""))
        L.append(textwrap.shorten("far bands: " + ", ".join(parts), width, placeholder=" ..."))
    if q.get("stop_met"):
        L.append("coverage: enough to stop")
    elif q.get("blocking"):
        L.append(textwrap.shorten("still needed: " + "; ".join(str(b) for b in q["blocking"]), width,
                                  placeholder=" ..."))
    return L


def _detail_lines(s, width=100) -> list:
    """this phase's details: the full session's phase lines first, then the board, the preflight, the cameras"""
    L = _full_lines(s, width) if s.get("mode") == "full" else []
    b = s.get("board")
    cap = s.get("capture") or {}
    if s.get("mode") != "full" or s.get("phase") in BOARD_PHASES:
        if b:
            cut = f", cut by the {'/'.join(b['cut'])} edge" if b.get("cut") else ""
            L.append(f"board: {b.get('z_m', 0):.2f} m, {b.get('tags', 0)} tags{cut}")
        elif cap:
            L.append("board: not seen")
    L += _preflight_lines(s, width)
    if cap:
        cams = cap.get("cams") or {}
        per = ", ".join(f"queue {q}: {c.get('fps', 0):.1f} fps, {c.get('none_reads', 0)} empty reads"
                        for q, c in sorted(cams.items()))
        sync = cap.get("sync_median_ms")
        L.append(f"cameras: {cap.get('pairs', 0)} pairs; {per}; L/R sync "
                 f"{'?' if sync is None else f'{sync:.3f}'} ms")
    return L


def _reasons(rs, n=4, width=100):
    return [f"    - {str(r)[:width - 6]}" for r in list(rs or [])[:n]]


def _full_lines(s, width=100) -> list:
    """the full session's screen: what the current phase needs the operator to see (the motor check list whole only
    in MOTORS; in HOLD only its problems)"""
    L = []
    mo = s.get("motors")
    if mo and s.get("phase") in ("MOTORS", "HOLD"):
        L.append(f"motors: {mo.get('detail', '')}"[:width])
        for c in mo.get("checks") or ():
            if s.get("phase") == "MOTORS" or c.get("status") not in ("ok", "info"):
                L.append(_check_line(c, width + 10, gate=False))
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
    """draws a screen only when its text changed, at most once per min_draw_s (a change waiting longer is drawn at
    the next state; flush() draws it when leaving); a key is acknowledged on screen at once, then calibd's answer
    (state["last_key"]) replaces it. Speech does not depend on drawing."""

    def __init__(self, send, speaker, clock=time.monotonic, out=print, speak_max_age_s=SPEAK_MAX_AGE_S, ansi=False,
                 min_draw_s=MIN_DRAW_S):
        self.send_line, self.speaker, self.clock, self.out = send, speaker or Silent(), clock, out
        self.ansi = ansi
        self.min_draw_s = min_draw_s
        self.screen = None              # the newest rendered text; _drawn: the one on screen, drawn at _drawn_t
        self._drawn = self._drawn_t = None
        self._notes = []
        self._pressed = None            # the operator's last key: {key, at (laptop clock), before, answer, answer_at}
        self._lock = threading.RLock()  # the key thread draws too
        self.max_age = speak_max_age_s
        self.spoken = None
        self._fresh = True              # nothing seen yet: old prompts (attaching late) are not spoken
        self._hb = None
        self.state = None
        self._step_n = None             # calibd's step counter and when (laptop clock) it last changed
        self._step_seen = None
        self.connected = True           # the attach link: False between link_lost() and link_back()
        self._lost = None               # (robot, attempt) while it is down
        self._back_until = None         # "Connection back" shown until then (laptop clock)

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
        with self._lock:
            self._notes = self.notes(st)
            self._answer(st)
            self._draw()
        return st

    def _draw(self, force=False) -> bool:
        with self._lock:
            now = self.clock()
            st = self.state or {}
            if not self.connected:
                st = dict(st, attached=False)       # the last state says "link up": it is not
            pf = getattr(self.speaker, "prompts", None)
            self.screen = render(st, notes=self.link_notes() + self._notes, ansi=self.ansi,
                                 ack=self.link_ack(now) + self.ack_lines(now),
                                 prompts=pf(float(st.get("updated") or 0)) if pf else None)
            if self.screen == self._drawn:
                return False
            if not force and self._drawn_t is not None and now - self._drawn_t < self.min_draw_s:
                return False
            self.out(self.screen)
            self._drawn, self._drawn_t = self.screen, now
            return True

    def flush(self) -> bool:
        """draw the newest screen if it is not the one shown (leaving: nothing waits for the next state)"""
        with self._lock:
            if self.screen is None or self.screen == self._drawn:
                return False
            self.out(self.screen)
            self._drawn, self._drawn_t = self.screen, self.clock()
            return True

    def link_lost(self, robot, attempt):
        """the attach link dropped; attempt N of re-attaching is next (drawn at once: no state line comes now)"""
        with self._lock:
            self.connected, self._lost, self._back_until = False, (str(robot), int(attempt)), None
            self._draw(force=True)

    def link_back(self):
        """state lines come again: "Connection back" for LINK_BACK_S; calibd's step counter is watched afresh"""
        with self._lock:
            self.connected, self._lost = True, None
            self._back_until = self.clock() + LINK_BACK_S
            self._step_n = self._step_seen = None
            self._hb = None                     # the first heartbeat goes at once

    def link_notes(self) -> list:
        if self._lost is None:
            return []
        robot, n = self._lost
        return [f"Connection lost to {robot}: reconnecting (attempt {n}). Q = leave",
                "calibd holds the robot by itself 5 s after it stops hearing this screen: once the connection is "
                "back, the robot is holding; press G when ready",
                "keys pressed now are not sent (they are not queued): press them again once the connection is back"]

    def link_ack(self, now) -> list:
        if self._back_until is None or now >= self._back_until:
            return []
        if (self.state or {}).get("phase") == "HOLD":
            return ["Connection back: the robot is holding; press G when ready"]
        return ["Connection back"]

    def _answer(self, st):
        """calibd's answer to the key pressed here: a last_key for that key that was not there when it was pressed"""
        p, lk = self._pressed, st.get("last_key")
        if p and p["answer"] is None and isinstance(lk, dict) and lk != p["before"] and \
                str(lk.get("key", "")).upper() == p["key"]:
            p["answer"], p["answer_at"] = lk, self.clock()

    def ack_lines(self, now) -> list:
        """10 Nov: "i press Y and nothing is shown": the key at once, then calibd's answer for KEY_SHOW_S"""
        p = self._pressed
        if not p:
            return []
        k, a = p["key"], p["answer"]
        if p.get("unsent"):
            if now - p["at"] > KEY_SHOW_S:
                return []
            return [f"{k} NOT sent: no connection to the robot (keys are not queued); press it again once the "
                    "connection is back"]
        if a is not None:
            if now - p["answer_at"] > KEY_SHOW_S:
                return []
            why = str(a.get("why") or "").strip()
            if a.get("accepted"):
                return [f"{k}: {why or 'accepted'}"]
            return [f"{k} ignored: {why}" if why else f"{k} ignored"]
        age = now - p["at"]
        if age < KEY_NO_ANSWER_S:
            return [f"you pressed {k}; waiting for calibd"]
        if age < KEY_GIVE_UP_S:
            return [f"{k} sent, no answer from calibd yet"]
        return []

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
        if not self.connected:
            return
        t = self.clock()
        if self._hb is None or t - self._hb >= HB_S:
            self._hb = t
            self.send_line(json.dumps({"hb": t}))

    def key(self, k):
        k = str(k).upper()
        with self._lock:
            self._pressed = {"key": k, "at": self.clock(), "before": (self.state or {}).get("last_key"),
                             "answer": None, "answer_at": None, "unsent": not self.connected}
        if self.connected:
            self.send_line(json.dumps({"key": k}))
        self._draw(force=True)          # the acknowledgement now, not at the next state
