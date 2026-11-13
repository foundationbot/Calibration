"""Camera recovery (design plan section 8): calibd acts on a stalled head camera instead of only showing it.

12 Nov, phantom10 (session 20261112-182519-P10): the head camera dropped off USB at 18:46:31, flapped between
SuperSpeed (usb 2-3.1) and USB 2 (usb 1-4.1) and settled on SuperSpeed 19 s later, but the producer never reopened it:
the frames stayed frozen for 287 s, the screen said "camera stalled", and nothing acted on the health watch's
producer_restart (nothing executed health actions at all).

Recovery.tick(t, phase, stalled_s, events, sample, own_until, busy) runs once per calibd step, in the loop thread.
stalled_s: the newest left/right pair's age while calibd flags the camera stalled (daemon.STALE_S), else None. The
restart itself (the no-overlap cycle: seconds) runs in a worker thread and a later tick picks up how it went.

One episode = from the newest pair going stale until fresh pairs come again. It acts when the pair is RESTART_AFTER_S
old (daemon.STALE_S = 1 s only flags a stall), or when health asks producer_restart for a reason of its own other than
a freeze (sync, desync_persists; a freeze is the stale pair above). Only in FRAME_PHASES (a HOLD counts as the phase it
came from), never inside the tool's own restart window (own_until: health.OWN_WINDOW_S after producer_own_t), never
while the flow restarts the producer itself (busy). Acting:
  a restart of ours within SUSPECT_WINDOW_S (10 min) before   -> hardware_suspect: spoken, banner, report, quoting the
                                                                 dmesg USB lines on the head camera's ports
  MAX_RESTARTS of ours within CAP_WINDOW_S already            -> no restart (never a restart loop), said once
  a head camera board off the USB bus or below SuperSpeed     -> wait up to USB_WAIT_S for it on SuperSpeed first (a
                                                                 producer opened mid re-enumeration sees no camera)
  -> ONE restart (restart(reason): the flow's own path, FullFlow.restart_cameras), at most one per episode
  -> fresh pairs again: "Camera back."; none FRAMES_AFTER_S after the restart: camera_restart_failed, no second try.
Events (events.jsonl): camera_restart_requested, camera_usb_wait / camera_usb_back / camera_usb_timeout,
camera_restart_done, camera_restart_failed, camera_restart_skipped, hardware_suspect, camera_recovered.
report_lines(as_dict()) is the session report's camera section."""
import re
import threading
import time
from dataclasses import dataclass, field

from calibtool import health

FRAME_PHASES = frozenset({"TRIAL", "PLACE", "GRID", "IMU_PLACE", "IMU_SWEEP", "CHECK_PLACE", "CHECK_GRID",
                          "POST_PLACE", "POSTCHECK"})
RESTART_AFTER_S = 6.0           # the newest pair this old -> act (a pose grab gives up after 6 s too)
USB_WAIT_S = 30.0               # 12 Nov: the board flapped SS <-> USB 2 for 19 s before it settled on SuperSpeed
FRAMES_AFTER_S = 45.0           # after the restart's pod is Running (phantom11: the camera re-enumerates in ~13 s)
SUSPECT_WINDOW_S = health.SUSPECT_WINDOW_S
CAP_WINDOW_S = health.SENTENCE_WINDOW_S
MAX_RESTARTS = 3                # restarts of ours within CAP_WINDOW_S; beyond them the camera is left alone
SUPERSPEED = health.SUPERSPEED
LINES_BEFORE_S = 30.0           # dmesg lines first seen this long before an episode began belong to it ...
LINES_AFTER_S = 20.0            # ... and this long after it ended (the sampler reads dmesg every 10 s)
LINES_KEEP = 8
EPISODES_KEEP = 20

STOPPED_TEXT = "The camera stopped. Restarting it."
STEP_TEXT = "The cameras are out of step. Restarting them."
USB_TEXT = "The camera dropped off USB. Waiting for it to come back."
USB_BACK_TEXT = "Camera on USB again. Restarting it."
USB_TIMEOUT_TEXT = "The camera is still not back on USB. Restarting it anyway."
BACK_TEXT = "Camera back."
FAILED_TEXT = "The camera did not come back after the restart. Check the screen."
CAPPED_TEXT = "The camera keeps stopping. No more automatic restarts. Check the head camera cable."

_USB = re.compile(r"usb (\d+-[\d.]+):", re.I)
_TIMES = {2: "twice", 3: "three times"}


@dataclass
class Episode:
    start: float
    reason: str                     # stall | sync | desync_persists
    state: str = "acting"           # usb_wait | restarting | frames_wait | given_up
    usb_wait_t: object = None
    restart_t: object = None
    restart_done_t: object = None
    end: object = None
    outcome: object = None          # recovered | recovered_by_itself | restart_failed | no_frames | capped | left_phase
    error: object = None
    usb_missing: bool = False       # a head camera board absent from the USB bus during the episode
    min_speed: object = None        # the lowest link speed seen (Mbit/s)
    suspect: object = None          # the hardware_suspect this episode raised (dict, see Recovery._suspect)

    def as_dict(self, lines=()) -> dict:
        return {"start": self.start, "end": self.end, "reason": self.reason, "state": self.state,
                "outcome": self.outcome, "restart_t": self.restart_t, "restart_done_t": self.restart_done_t,
                "usb_missing": self.usb_missing, "min_speed": self.min_speed, "error": self.error,
                "lines": list(lines)}


class _Job:
    """the restart in a daemon thread: done(), error() (None or "Type: message")"""

    def __init__(self, fn):
        self._done, self._err = threading.Event(), None

        def run():
            try:
                fn()
            except Exception as e:  # noqa: BLE001  (said and recorded; the episode does not retry)
                self._err = f"{type(e).__name__}: {e}"
            finally:
                self._done.set()
        threading.Thread(target=run, name="camera-restart", daemon=True).start()

    def done(self) -> bool:
        return self._done.is_set()

    def error(self):
        return self._err


class Recovery:
    """restart(reason): restarts the camera producer (blocking; raises on failure); event(kind, **kw): events.jsonl;
    say(text): a spoken prompt; head_serials / head_ports: the preflight baseline's head camera boards and their USB
    ports; on_restart(t, reason): told about each restart (health.Watch.note_restart); run(fn) -> a job with done()
    and error() (default: a thread)"""

    def __init__(self, restart, event, say, head_serials=(), head_ports=(), on_restart=None, run=_Job):
        self.restart, self.event, self.say = restart, event, say
        self.serials = tuple(head_serials or ())
        self.home_ports = tuple(head_ports or ())
        self.ports = set(self.home_ports)
        self.on_restart, self.run = on_restart, run
        self.ep = None
        self.episodes = []              # closed episodes
        self.restarts = []              # when ours ran
        self.suspects = []              # [{"t", "sentence", "spoken", "lines"}]
        self._job = None
        self._usb = None                # the newest {serial: Mbit/s}
        self._seen = None               # dmesg lines already seen (None: the first read is history, not news)
        self._recent = []               # [(t first seen, line)] USB lines

    # -------------------------------------------------------------- what the sampler says

    def _observe(self, t, sample):
        if not sample:
            return
        usb = sample.get("usb")
        if usb is not None:
            self._usb = dict(usb)
            for s, p in (sample.get("usb_ports") or {}).items():
                if s in self.serials and p:
                    self.ports.add(str(p))
            if self.ep is not None:
                self._note_usb(self.ep)
        lines = sample.get("dmesg")
        if lines is None:
            return
        if self._seen is None:
            self._seen = set(lines)
            return
        for ln in lines:
            if ln in self._seen:
                continue
            self._seen.add(ln)
            m = _USB.search(ln)
            if not m:
                continue
            if any(s and s in ln for s in self.serials):        # "usb 1-4.1: SerialNumber: <board>": its port now
                self.ports.add(m.group(1))
            self._recent.append((t, ln))
        self._recent = [x for x in self._recent if t - x[0] <= CAP_WINDOW_S + LINES_BEFORE_S]

    def _note_usb(self, ep):
        if self._usb is None:
            return
        for s in self.serials:
            sp = self._usb.get(s)
            if sp is None:
                ep.usb_missing = True
            else:
                ep.min_speed = int(sp) if ep.min_speed is None else min(ep.min_speed, int(sp))

    def _usb_ok(self) -> bool:
        """every head camera board on the bus at SuperSpeed (unknown boards or no reading: True, nothing to wait for)"""
        if not self.serials or self._usb is None:
            return True
        return all(self._usb.get(s) is not None and self._usb[s] >= SUPERSPEED for s in self.serials)

    def _usb_words(self) -> str:
        out = []
        for s in self.serials:
            sp = (self._usb or {}).get(s)
            out.append(f"{s} not on the USB bus" if sp is None else f"{s} at {sp} Mbit/s")
        return ", ".join(out)

    def _on_head_port(self, port) -> bool:
        return any(port == p or port.startswith(p + ".") for p in self.ports)

    def _lines(self, ep) -> list:
        lo = ep.start - LINES_BEFORE_S
        hi = None if ep.end is None else ep.end + LINES_AFTER_S
        out = [ln for tl, ln in self._recent if tl >= lo and (hi is None or tl <= hi)
               and self._on_head_port(_USB.search(ln).group(1))]
        return out[-LINES_KEEP:]

    def _evidence(self, ep) -> dict:
        lines = self._lines(ep)
        low = [ln.lower() for ln in lines]
        dropped = ep.usb_missing or any("disconnect" in ln for ln in low)
        usb2 = (ep.min_speed is not None and ep.min_speed < SUPERSPEED) or any(
            "high-speed" in ln or "full-speed" in ln for ln in low)
        return {"dropped": dropped, "usb2": usb2, "lines": lines}

    # -------------------------------------------------------------- the step

    def tick(self, t, phase, stalled_s, events=(), sample=None, own_until=None, busy=False):
        self._observe(t, sample)
        fresh = stalled_s is None
        if self.ep is not None:
            self._advance(t, phase, fresh)
            return
        if phase not in FRAME_PHASES or busy or (own_until is not None and t <= own_until):
            return
        reason = None
        if not fresh and float(stalled_s) >= RESTART_AFTER_S:
            reason = "stall"
        else:
            for e in events or ():
                if getattr(e, "action", None) == "producer_restart" and getattr(e, "code", None) != "freeze":
                    reason = e.code
                    break
        if reason is not None:
            self._begin(t, reason, stalled_s)

    def _begin(self, t, reason, stalled_s):
        ep = Episode(start=t - float(stalled_s or 0.0), reason=reason)
        self.ep = ep
        self._note_usb(ep)
        self.event("camera_restart_requested", reason=reason, stalled_s=stalled_s, usb=self._usb_words() or None)
        if any(0 <= t - r <= SUSPECT_WINDOW_S for r in self.restarts):
            self._suspect(t, ep)
        n = sum(1 for r in self.restarts if 0 <= t - r <= CAP_WINDOW_S)
        if n >= MAX_RESTARTS:
            ep.state, ep.outcome = "given_up", "capped"
            self.event("camera_restart_skipped", why=f"{n} automatic restarts in {CAP_WINDOW_S / 60:.0f} min "
                       "already: no more (never a restart loop)")
            self.say(CAPPED_TEXT)
            return
        if reason == "stall" and not self._usb_ok():
            ep.state, ep.usb_wait_t = "usb_wait", t
            self.event("camera_usb_wait", usb=self._usb_words(), wait_s=USB_WAIT_S)
            self.say(USB_TEXT)
            return
        self._restart(t, STOPPED_TEXT if reason == "stall" else STEP_TEXT)

    def _restart(self, t, text):
        ep = self.ep
        ep.state, ep.restart_t = "restarting", t
        self.restarts.append(t)
        if self.on_restart is not None:
            self.on_restart(t, ep.reason)
        self.say(text)
        self._job = self.run(lambda: self.restart(ep.reason))

    def _advance(self, t, phase, fresh):
        ep = self.ep
        if ep.state == "restarting":
            if not self._job.done():
                return
            err = self._job.error()
            self._job = None
            if err:
                ep.state, ep.outcome, ep.error = "given_up", "restart_failed", err
                self.event("camera_restart_failed", why="the producer restart failed", error=err)
                self.say(FAILED_TEXT)
            else:
                ep.state, ep.restart_done_t = "frames_wait", t
                self.event("camera_restart_done", seconds=round(t - ep.restart_t, 1))
            return
        if fresh:
            self._end(t)
            return
        if ep.state == "usb_wait":
            if phase not in FRAME_PHASES:
                ep.outcome = "left_phase"
                self.event("camera_usb_wait_ended", why=f"the session left the camera phases ({phase}): no restart")
                self._end(t, say=False)
            elif self._usb_ok():
                self.event("camera_usb_back", waited_s=round(t - ep.usb_wait_t, 1), usb=self._usb_words())
                self._restart(t, USB_BACK_TEXT)
            elif t - ep.usb_wait_t >= USB_WAIT_S:
                self.event("camera_usb_timeout", waited_s=round(t - ep.usb_wait_t, 1), usb=self._usb_words())
                self._restart(t, USB_TIMEOUT_TEXT)
        elif ep.state == "frames_wait" and t - ep.restart_done_t >= FRAMES_AFTER_S:
            ep.state, ep.outcome = "given_up", "no_frames"
            self.event("camera_restart_failed", why=f"no frames {FRAMES_AFTER_S:.0f} s after the restart",
                       usb=self._usb_words() or None)
            self.say(FAILED_TEXT)

    def _end(self, t, say=True):
        ep = self.ep
        ep.end = t
        if ep.outcome is None:
            ep.outcome = "recovered" if ep.restart_t is not None else "recovered_by_itself"
        lines = self._lines(ep)
        self.event("camera_recovered", outcome=ep.outcome, stalled_s=round(t - ep.start, 1),
                   restarted=ep.restart_t is not None, lines=lines)
        if say and ep.outcome != "left_phase":
            self.say(BACK_TEXT)
        self.episodes = (self.episodes + [ep])[-EPISODES_KEEP:]
        self.ep = None
        if ep.suspect is not None:
            self._refresh_suspect(ep)

    # -------------------------------------------------------------- hardware suspicion

    def _window(self, t, ep) -> list:
        """the episodes of ours within SUSPECT_WINDOW_S that restarted, then ep"""
        return [e for e in self.episodes if e.restart_t is not None and 0 <= t - e.restart_t <= SUSPECT_WINDOW_S] \
            + [ep]

    def _suspect(self, t, ep):
        eps = self._window(t, ep)
        sentence, spoken, lines = self.sentence(eps)
        s = {"t": t, "starts": [e.start for e in eps], "sentence": sentence, "spoken": spoken, "lines": lines}
        ep.suspect = s
        self.suspects.append(s)
        self.event("hardware_suspect", sentence=sentence, lines=lines)
        self.say(spoken)

    def _refresh_suspect(self, ep):
        """at the episode's end, with the dmesg lines that came in later: the sentence and lines again (unspoken)"""
        s = ep.suspect
        eps = [e for e in self.episodes if e.start in s["starts"]]
        sentence, _, lines = self.sentence(eps)
        if (sentence, lines) != (s["sentence"], s["lines"]):
            s["sentence"], s["lines"] = sentence, lines
            self.event("hardware_suspect", sentence=sentence, lines=lines, update=True)

    def sentence(self, eps):
        """-> (the banner / report sentence, the spoken one, the quoted lines) for these episodes (oldest first)"""
        n = len(eps)
        times = _TIMES.get(n, f"{n} times")
        span = (eps[-1].start - eps[0].start) / 60.0
        mins = max(1, round(span))
        within = "within a minute" if span < 1 else f"in {mins} minute{'s' if mins > 1 else ''}"
        ev = [self._evidence(e) for e in eps]
        lines = []
        for e in ev:
            lines += [ln for ln in e["lines"] if ln not in lines]
        lines = lines[-LINES_KEEP:]
        ports = list(self.home_ports) or sorted(self.ports)
        dropped = [e["dropped"] for e in ev]
        usb2 = any(e["usb2"] for e in ev)
        if any(dropped) or usb2:
            bits = ([f"port {', '.join(ports)}"] if ports else []) + (["fell back to USB 2"] if usb2 else [])
            detail = f" ({', '.join(bits)})" if bits else ""
            if all(dropped):
                text = f"Camera dropped off USB {times} {within}{detail}: suspect the head camera cable or hub."
            elif any(dropped):
                text = (f"The camera stopped {times} {within}; it dropped off USB{detail}: suspect the head camera "
                        "cable or hub.")
            else:
                text = (f"The camera stopped {times} {within}; its USB link fell back{detail}: suspect the head "
                        "camera cable or hub.")
            spoken = f"The camera stopped {times} {within}. Suspect the head camera cable or hub."
        else:
            text = (f"The camera stopped {times} {within} with no USB drop seen: suspect the camera board or the "
                    "producer.")
            spoken = f"The camera stopped {times} {within}. Suspect the camera board."
        return text, spoken, lines

    # -------------------------------------------------------------- the screen and the report

    def banner(self, t) -> list:
        out = []
        ep = self.ep
        if ep is not None:
            if ep.state == "usb_wait":
                out.append(f"CAMERA: head camera board {self._usb_words()}; waiting up to {USB_WAIT_S:.0f} s for it "
                           f"on SuperSpeed before restarting the producer ({t - ep.usb_wait_t:.0f} s so far)")
            elif ep.state == "restarting":
                out.append("CAMERA: restarting the camera producer (automatic: one restart per stall)")
            elif ep.state == "frames_wait":
                out.append(f"CAMERA: producer restarted; waiting for frames ({t - ep.restart_done_t:.0f} s of "
                           f"{FRAMES_AFTER_S:.0f})")
            elif ep.outcome == "capped":
                out.append(f"CAMERA: {MAX_RESTARTS} automatic restarts in {CAP_WINDOW_S / 60:.0f} min already: no "
                           "more (never a restart loop). The session waits for frames; check the head camera cable; "
                           "A aborts")
            elif ep.outcome == "restart_failed":
                out.append(f"CAMERA: the automatic producer restart failed: {ep.error}. The session waits for frames; "
                           "A aborts")
            elif ep.outcome == "no_frames":
                out.append(f"CAMERA: no frames {FRAMES_AFTER_S:.0f} s after the automatic restart (one per stall). "
                           "Suspect the head camera, its cable or hub; the session waits for frames; A aborts")
        if self.suspects:
            out.append("SUSPECT: " + self.suspects[-1]["sentence"])
        return out

    def as_dict(self, t) -> dict:
        ep = self.ep
        eps = self.episodes + ([ep] if ep is not None else [])
        return {"active": ep is not None, "state": None if ep is None else ep.state,
                "stalled_since": None if ep is None else ep.start, "restarts": list(self.restarts),
                "episodes": [e.as_dict(self._lines(e)) for e in eps],
                "suspects": [dict(s) for s in self.suspects], "banner": self.banner(t)}


def _hms(t) -> str:
    return time.strftime("%H:%M:%S", time.localtime(float(t)))


def report_lines(d) -> list:
    """the session report's camera section (markdown lines) from Recovery.as_dict(); [] when nothing happened"""
    if not d or not (d.get("episodes") or d.get("suspects")):
        return []
    eps = d.get("episodes") or []
    n_r = len(d.get("restarts") or [])
    out = [f"- camera: {len(eps)} stall(s) acted on, {n_r} automatic producer restart(s)"]
    for e in eps:
        dur = "" if e.get("end") is None else f", {e['end'] - e['start']:.0f} s"
        what = {"recovered": "back after a restart", "recovered_by_itself": "back by itself",
                "restart_failed": f"restart failed ({e.get('error')})", "no_frames": "no frames after the restart",
                "capped": "not restarted (too many restarts)", "left_phase": "not restarted (session moved on)"}.get(
            e.get("outcome"), f"still {e.get('state')}")
        usb = []
        if e.get("usb_missing"):
            usb.append("off the USB bus")
        if e.get("min_speed") is not None and e["min_speed"] < SUPERSPEED:
            usb.append(f"down to {e['min_speed']} Mbit/s")
        out.append(f"  - {_hms(e['start'])} {e.get('reason')}: {what}{dur}" + (f"; {', '.join(usb)}" if usb else ""))
    for s in d.get("suspects") or []:
        out.append(f"  - SUSPECT ({_hms(s['t'])}): {s['sentence']}")
        out += [f"    - `{ln}`" for ln in s.get("lines") or []]
    return out
