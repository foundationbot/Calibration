"""The health watch and its hardware suspicions (design plan section 8, v1 short list; baseline-relative).

Watch(baseline).observe(t, sample) -> [Event]. The sample is what the daemon gathers about once a second: the capture
counters per camera queue (capture.Capture.snapshot()["cams"]), the left/right sync, and the system probes (boot id,
wall clock, k0s answering, the IMU ring's head counter, USB speed per head-camera serial, new dmesg lines, new producer
log lines, the producer deployment's generation / pod uid / args, the 1-min load). Each rule fires once per episode:
again only after its condition has cleared. note_restart() records a producer restart (by whatever did it); own_change()
marks a restart the tool itself runs, so the producer changing under it is not "someone else".

Rules (condition -> event code, level, action):
  reads None on a queue REATTACH_AFTER times in a row        desync, warn, reattach
  ... still None after a re-attach later in that episode      desync_persists, warn, producer_restart
  no new frame on a queue for FREEZE_S                         freeze, warn, producer_restart
  ... a freeze within 10 min of a producer restart             + hardware, suspect
  fps < 0.7 x baseline while load < 1.5 x baseline             fps_drop, warn
  median |left - right stamp| > 0.5 ms                         sync, warn, producer_restart (once)
  ... still > 0.5 ms after a restart                           sync_persists, stop, refuse_collect
  boot id changed / clock before 2020                          reboot / clock, stop, preflight
  k0s not answering                                            k0s, warn
  IMU ring head not advancing for IMU_SILENT_S (reading time)  imu_silent, warn ("check the head power switch first")
  head-camera USB link below SuperSpeed or missing             usb_link, suspect (no restart: a restart cannot fix it)
  dmesg reset/disconnect on the head camera's USB port         usb_reset, suspect, quoting the line
  producer log "Device crashed"                                device_crashed, suspect, quoting the line
  producer generation / pod / args changed, not by the tool    producer_changed, stop, stop_collect
suspicion_sentence(watch, now) puts the restart history in one sentence for the operator and the report."""
import contextlib
import re
import time
from dataclasses import dataclass, field

REATTACH_AFTER = 25
FREEZE_S = 2.0
FPS_DROP = 0.7
LOAD_EXPLAINS = 1.5
SYNC_MS = 0.5
IMU_SILENT_S = 1.0
SUPERSPEED = 5000
SUSPECT_WINDOW_S = 600.0        # a second freeze within this of a restart -> hardware suspicion
SENTENCE_WINDOW_S = 1200.0      # restarts counted in suspicion_sentence
USB_BEFORE_S = 60.0             # a USB reset this long before a restart counts as its cause
CLOCK_MIN_S = 1.577e9           # 2020-01-01
_USB_LINE = re.compile(r"usb (\d+-[\d.]+):.*(reset|disconnect)", re.I)


@dataclass
class Event:
    t: float
    level: str                  # info | warn | stop | suspect
    code: str
    message: str
    action: object = None       # reattach | producer_restart | refuse_collect | preflight | stop_collect | None
    lines: tuple = ()           # evidence quoted (dmesg / producer log)

    def as_dict(self) -> dict:
        return {"t": self.t, "level": self.level, "code": self.code, "message": self.message, "action": self.action,
                "lines": list(self.lines)}


@dataclass
class _Restart:
    t: float
    reason: str


@dataclass
class Watch:
    baseline: dict
    active: set = field(default_factory=set)
    restarts: list = field(default_factory=list)
    usb_resets: list = field(default_factory=list)          # (t, line)
    events: list = field(default_factory=list)

    def __post_init__(self):
        self._seen = set()
        self._imu_last = None                               # (head, t of its last change)
        self._own = False
        self._producer = dict(self.baseline.get("producer") or {})
        self._sync_restart_t = None
        self._desync_ra = {}                                # queue -> its re-attach count when the episode began

    # -------------------------------------------------------------- bookkeeping

    def note_restart(self, t, reason):
        self.restarts.append(_Restart(float(t), str(reason)))
        if reason == "sync":
            self._sync_restart_t = float(t)

    @contextlib.contextmanager
    def own_change(self):
        self._own = True
        try:
            yield
        finally:
            self._own = False

    def adopt_producer(self, prod):
        """the producer as the tool's own restart left it: the new reference"""
        self._producer = dict(prod)

    def _episode(self, key, cond, make):
        """make() -> [Event] once when cond turns true; clears when it turns false"""
        if cond:
            if key in self.active:
                return []
            self.active.add(key)
            return make()
        self.active.discard(key)
        return []

    # -------------------------------------------------------------- the rules

    def observe(self, t, s) -> list:
        out = []
        b = self.baseline
        for q, c in sorted((s.get("cams") or {}).items()):
            nr, ra = int(c.get("none_run") or 0), int(c.get("reattaches") or 0)
            desync = nr >= REATTACH_AFTER
            # the reader re-attaches at the 25th empty read, so the episode's first sample may already count one
            # re-attach (review 5): "persists" = a re-attach later in this episode did not help
            if desync:
                ra0 = self._desync_ra.setdefault(q, ra)
            else:
                ra0 = self._desync_ra.pop(q, ra)
            out += self._episode(("desync", q), desync, lambda q=q, nr=nr: [Event(
                t, "warn", "desync", f"Camera queue {q}: the last {nr} frame reads came back empty (frame pool out "
                f"of step); re-attaching the tool's own reader.", "reattach")])
            out += self._episode(("desync_persists", q), desync and ra > ra0, lambda q=q: [Event(
                t, "warn", "desync_persists", f"Camera queue {q}: reads still empty after re-attaching; the producer "
                "needs a restart.", "producer_restart")])
            since = c.get("since_frame_s")
            frozen = since is not None and since >= FREEZE_S and not desync

            def freeze(q=q, since=since):
                ev = [Event(t, "warn", "freeze", f"Camera queue {q}: no new frame for {since:.1f} s (frozen); the "
                            "producer needs a restart.", "producer_restart")]
                if any(0 <= t - r.t <= SUSPECT_WINDOW_S for r in self.restarts):
                    ev.append(Event(t, "suspect", "hardware", f"Camera queue {q} froze again within 10 min of a "
                                    "producer restart: suspect the head camera, its cable or hub, not software."))
                return ev
            out += self._episode(("freeze", q), frozen, freeze)
            fps, load = c.get("fps"), s.get("load")
            drop = (fps is not None and b.get("fps") and c.get("frames", 0) > 0 and fps < FPS_DROP * b["fps"]
                    and load is not None and load < LOAD_EXPLAINS * b.get("load", load))
            out += self._episode(("fps", q), bool(drop), lambda q=q, fps=fps, load=load: [Event(
                t, "warn", "fps_drop", f"Camera queue {q}: {fps:.1f} fps against {b['fps']:.0f} at the start, and "
                f"the CPU load ({load:.1f}) does not explain it.")])
        sync = s.get("sync_median_ms")
        bad_sync = sync is not None and sync > SYNC_MS
        restarted = self._sync_restart_t is not None
        out += self._episode("sync", bad_sync and not restarted, lambda: [Event(
            t, "warn", "sync", f"Left/right capture stamps {sync:.3f} ms apart (limit {SYNC_MS} ms); restarting the "
            "producer once.", "producer_restart")])
        out += self._episode("sync_persists", bad_sync and restarted, lambda: [Event(
            t, "stop", "sync_persists", f"Left/right capture stamps still {sync:.3f} ms apart after a producer "
            "restart: not collecting.", "refuse_collect")])
        boot = s.get("boot_id")
        out += self._episode("reboot", bool(boot) and boot != b.get("boot_id"), lambda: [Event(
            t, "stop", "reboot", "The robot rebooted (boot id changed): motors are off; preflight runs again, and the "
            "motor check and Y are needed again.", "preflight")])
        clk = s.get("clock_s")
        out += self._episode("clock", clk is not None and clk < CLOCK_MIN_S, lambda: [Event(
            t, "stop", "clock", f"The robot's clock reads {time.strftime('%Y-%m-%d', time.gmtime(clk))}: not set; "
            "preflight again once it is.", "preflight")])
        out += self._episode("k0s", s.get("k0s_ok") is False, lambda: [Event(
            t, "warn", "k0s", "k0s (kubectl) is not answering: the producer cannot be checked or restarted now.")])
        out += self._imu_rule(t, s.get("imu_head"), s.get("imu_t"))
        usb = s.get("usb")
        if usb is not None:
            for ser in b.get("head_serials") or ():
                sp = usb.get(ser)
                slow = sp is None or sp < SUPERSPEED
                words = "is not on the USB bus" if sp is None else f"runs at {sp} Mbit/s (below SuperSpeed)"
                out += self._episode(("usb", ser), slow, lambda ser=ser, words=words: [Event(
                    t, "suspect", "usb_link", f"The head camera board {ser} {words}: a cable, hub or port fault a "
                    "producer restart cannot fix.")])
        ports = tuple(b.get("head_ports") or ())
        for ln in s.get("dmesg") or ():
            m = _USB_LINE.search(ln)
            if not m or ln in self._seen:
                continue
            self._seen.add(ln)
            if any(m.group(1) == p or m.group(1).startswith(p + ".") for p in ports):
                self.usb_resets.append((t, ln))
                out.append(Event(t, "suspect", "usb_reset", f"USB {m.group(2).lower()} on the head camera's port "
                                 f"{m.group(1)}.", None, (ln,)))
        for ln in s.get("producer_log") or ():
            if "Device crashed" in ln and ln not in self._seen:
                self._seen.add(ln)
                out.append(Event(t, "suspect", "device_crashed", "The camera producer logged 'Device crashed': "
                                 "suspect the camera board.", None, (ln,)))
        prod = s.get("producer")
        if prod is not None and not self._own:
            keys = ("generation", "pod_uid", "args")
            changed = [k for k in keys if k in self._producer and prod.get(k) != self._producer.get(k)]
            out += self._episode("producer", bool(changed), lambda: [Event(
                t, "stop", "producer_changed", "Someone else changed the camera producer (" + ", ".join(
                    f"{k} {self._producer.get(k)} -> {prod.get(k)}" for k in changed) + "): collection stopped; "
                "tell the operator.", "stop_collect")])
        self.events += out
        return out

    def _imu_rule(self, t, head, t_read=None):
        """silent: the head counter unchanged for IMU_SILENT_S of the readings' own time (t_read: when the sampler
        read it; a stale reading repeated by the sampler says nothing)"""
        if head is None:
            return []
        tr = t if t_read is None else t_read
        if self._imu_last is None or head != self._imu_last[0]:
            self._imu_last = (head, tr)
        silent = tr - self._imu_last[1] >= IMU_SILENT_S
        return self._episode("imu", silent, lambda: [Event(
            t, "warn", "imu_silent", "The head IMU has stopped publishing: check the head power switch first.")])


def _ordinal(n) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def suspicion_sentence(w: Watch, now, window_s=SENTENCE_WINDOW_S):
    """one sentence on the producer restarts within window_s before now (None for fewer than 2)"""
    rs = [r for r in w.restarts if 0 <= now - r.t <= window_s]
    if len(rs) < 2:
        return None
    head = f"{_ordinal(len(rs))} producer restart in {round(window_s / 60)} min"
    hhmm = [time.strftime("%H:%M", time.localtime(r.t)) for r in rs]
    with_usb = [any(0 <= r.t - tu <= USB_BEFORE_S for tu, _ in w.usb_resets) for r in rs]
    if all(with_usb):
        return (f"{head}; USB reset on the head camera's port before each ({', '.join(hhmm)}) -> suspect the head "
                "camera cable or hub, not software.")
    if any(with_usb):
        at = ", ".join(h for h, u in zip(hhmm, with_usb) if u)
        return (f"{head}; USB reset on the head camera's port before some ({at}) -> suspect the head camera cable or "
                "hub.")
    return (f"{head}; no USB reset logged ({', '.join(hhmm)}) -> suspect the camera board or the producer; the session "
            "report has the log lines.")
