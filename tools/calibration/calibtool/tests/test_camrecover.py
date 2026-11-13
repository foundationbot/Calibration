"""calibtool.camrecover: what calibd does about a stalled head camera (12 Nov, phantom10: the camera dropped off USB,
flapped SuperSpeed <-> USB 2 for 19 s, came back on SuperSpeed, and the frames stayed frozen 287 s because nothing
restarted the producer).

Fakes only: a producer that records restarts (and can fail), a job runner the test finishes by hand (the restart's
worker thread), a USB sampler that flaps SS -> gone -> USB 2 -> SS with the matching dmesg lines. Asserted: one restart
per stall episode, the wait for SuperSpeed before it, hardware suspicion on a second stall within 10 min quoting the
USB lines, never a restart inside the tool's own restart window or outside the camera phases, at most MAX_RESTARTS in
20 min, every step spoken and in events, the report's camera section."""
import time
from types import SimpleNamespace

import pytest

from calibtool import camrecover as cr, health

SER = "1844301011B5F0F500"
PORT = "2-3.1"
SS, HS = 5000, 480


class Producer:
    def __init__(self, fail=None):
        self.restarts, self.fail = [], fail

    def __call__(self, reason):
        self.restarts.append(reason)
        if self.fail:
            raise RuntimeError(self.fail)


class Jobs:
    """the restart's worker thread, finished by the test: finish() runs the restart and marks it done"""

    def __init__(self, auto=True):
        self.auto, self.jobs = auto, []

    def __call__(self, fn):
        j = SimpleNamespace(fn=fn, ok=False, err=None)
        j.done = lambda: j.ok
        j.error = lambda: j.err
        self.jobs.append(j)
        if self.auto:
            self.finish(j)
        return j

    def finish(self, j=None):
        j = j or self.jobs[-1]
        try:
            j.fn()
        except Exception as e:  # noqa: BLE001
            j.err = f"{type(e).__name__}: {e}"
        j.ok = True


class Usb:
    """the sampler's facts: the head board's link speed (None = off the bus), its port, dmesg's last lines"""

    def __init__(self):
        self.speed, self.port, self.dmesg = SS, PORT, ["[Thu Nov 12 18:00:00 2026] usb 2-3.1: old line"]

    def sample(self):
        usb = {} if self.speed is None else {SER: self.speed}
        ports = {} if self.speed is None else {SER: self.port}
        return {"usb": usb, "usb_ports": ports, "dmesg": list(self.dmesg)}

    def log(self, line):
        self.dmesg = (self.dmesg + [line])[-80:]

    def drop(self, hhmmss="18:46:31"):
        self.speed = None
        self.log(f"[Thu Nov 12 {hhmmss} 2026] usb 2-3.1: USB disconnect, device number 4")

    def usb2(self, hhmmss="18:46:35"):
        self.speed, self.port = HS, "1-4.1"
        self.log(f"[Thu Nov 12 {hhmmss} 2026] usb 1-4.1: new high-speed USB device number 7 using tegra-xusb")
        self.log(f"[Thu Nov 12 {hhmmss} 2026] usb 1-4.1: SerialNumber: {SER}")

    def superspeed(self, hhmmss="18:46:50"):
        self.speed, self.port = SS, PORT
        self.log(f"[Thu Nov 12 {hhmmss} 2026] usb 2-3.1: new SuperSpeed USB device number 8 using tegra-xusb")


class Rig:
    """a Recovery, its fakes and a clock; step(seconds, stalled) runs ticks at the loop's pace"""

    def __init__(self, producer=None, jobs=None, phase="PLACE"):
        self.t = 1.79e9
        self.producer = producer or Producer()
        self.jobs = jobs or Jobs()
        self.usb = Usb()
        self.events, self.said, self.noted = [], [], []
        self.phase = phase
        self.own_until, self.busy = None, False
        self.stall_t = None             # when the newest pair went stale (None: fresh)
        self.r = cr.Recovery(self.producer, event=lambda kind, **kw: self.events.append((kind, kw)),
                             say=self.said.append, head_serials=[SER], head_ports=[PORT],
                             on_restart=lambda t, why: self.noted.append((t, why)), run=self.jobs)
        self.tick()

    def stall(self):
        self.stall_t = self.t

    def back(self):
        self.stall_t = None

    def tick(self, events=()):
        stalled = None if self.stall_t is None else self.t - self.stall_t
        self.r.tick(self.t, self.phase, stalled, events, self.usb.sample(), own_until=self.own_until,
                    busy=self.busy)

    def run(self, seconds, dt=0.15):
        for _ in range(int(round(seconds / dt))):
            self.t += dt
            self.tick()

    def kinds(self):
        return [k for k, _ in self.events]

    def ev(self, kind):
        return [kw for k, kw in self.events if k == kind]


def test_a_stall_gets_one_restart_after_6_s_and_camera_back_is_said():
    g = Rig()
    g.stall()
    g.run(5.5)
    assert g.producer.restarts == [] and g.said == []           # 1 s only flags it; 6 s acts
    g.run(1.0)
    assert g.producer.restarts == ["stall"]
    assert g.said == [cr.STOPPED_TEXT]
    assert g.noted and g.noted[0][1] == "stall"                 # health.Watch.note_restart
    assert g.kinds()[:2] == ["camera_restart_requested", "camera_restart_done"]
    g.run(3.0)
    g.back()
    g.run(0.3)
    assert g.said == [cr.STOPPED_TEXT, cr.BACK_TEXT]
    assert g.ev("camera_recovered")[0]["outcome"] == "recovered"
    assert g.r.ep is None and g.r.as_dict(g.t)["active"] is False


def test_at_most_one_restart_per_episode_then_failed_and_waiting():
    g = Rig()
    g.stall()
    g.run(7.0)
    assert g.producer.restarts == ["stall"]
    g.run(cr.FRAMES_AFTER_S + 5)                                # still frozen after the restart
    assert g.producer.restarts == ["stall"]                     # never a second one in the episode
    assert g.ev("camera_restart_failed")[0]["why"].startswith("no frames")
    assert g.said[-1] == cr.FAILED_TEXT
    assert any("no frames" in b for b in g.r.banner(g.t))
    g.run(120)
    assert g.producer.restarts == ["stall"]
    g.back()
    g.run(0.3)
    assert g.said[-1] == cr.BACK_TEXT


def test_a_failed_restart_is_said_and_not_retried():
    g = Rig(producer=Producer(fail="the producer pod did not go away within 60 s"))
    g.stall()
    g.run(7.0)
    assert g.ev("camera_restart_failed")[0]["error"].endswith("within 60 s")
    assert g.said == [cr.STOPPED_TEXT, cr.FAILED_TEXT]
    g.run(60)
    assert len(g.producer.restarts) == 1
    assert any("restart failed" in b for b in g.r.banner(g.t))


def test_the_restart_waits_for_the_camera_back_on_superspeed():
    """12 Nov: off the bus at 18:46:31, USB 2 on another port, SuperSpeed again at 18:46:50"""
    g = Rig()
    g.stall()
    g.run(2.0)
    g.usb.drop()
    g.run(5.0)
    assert g.producer.restarts == [] and g.r.ep.state == "usb_wait"
    assert g.said == [cr.USB_TEXT]
    assert "not on the USB bus" in g.ev("camera_usb_wait")[0]["usb"]
    assert any("waiting up to 30 s" in b for b in g.r.banner(g.t))
    g.usb.usb2()
    g.run(8.0)
    assert g.producer.restarts == []                            # USB 2 is not back
    g.usb.superspeed()
    g.run(0.3)
    assert g.producer.restarts == ["stall"]
    assert g.said == [cr.USB_TEXT, cr.USB_BACK_TEXT]
    assert "camera_usb_back" in g.kinds()
    ep = g.r.ep
    assert ep.usb_missing and ep.min_speed == HS
    g.back()
    g.run(0.3)
    d = g.r.as_dict(g.t)["episodes"][0]
    assert any("USB disconnect" in ln for ln in d["lines"]) and any("1-4.1" in ln for ln in d["lines"])


def test_a_camera_that_stays_off_usb_gets_its_one_restart_after_30_s():
    g = Rig()
    g.usb.drop()
    g.stall()
    g.run(6.5)
    assert g.r.ep.state == "usb_wait"
    g.run(cr.USB_WAIT_S - 1)
    assert g.producer.restarts == []
    g.run(1.5)
    assert g.producer.restarts == ["stall"]
    assert g.said == [cr.USB_TEXT, cr.USB_TIMEOUT_TEXT]
    assert "camera_usb_timeout" in g.kinds()


def test_frames_back_while_waiting_for_usb_needs_no_restart():
    g = Rig()
    g.usb.drop()
    g.stall()
    g.run(7.0)
    g.usb.superspeed()
    g.back()
    g.run(0.3)
    assert g.producer.restarts == []
    assert g.ev("camera_recovered")[0]["outcome"] == "recovered_by_itself"
    assert g.said == [cr.USB_TEXT, cr.BACK_TEXT]


def test_a_second_stall_within_10_min_is_a_hardware_suspicion_quoting_the_usb_lines():
    g = Rig()
    g.stall()
    g.usb.drop("18:40:31")
    g.run(3.0)
    g.usb.superspeed("18:40:40")
    g.run(4.0)
    g.back()
    g.run(1.0)
    assert g.producer.restarts == ["stall"] and g.ev("hardware_suspect") == []
    g.run(6 * 60 - 8)                                           # 6 minutes after the first stall
    g.stall()
    g.run(2.0)
    g.usb.drop()
    g.run(1.0)
    g.usb.usb2()
    g.run(3.5)
    s = g.ev("hardware_suspect")
    assert len(s) == 1
    assert s[0]["sentence"] == ("Camera dropped off USB twice in 6 minutes (port 2-3.1, fell back to USB 2): suspect "
                                "the head camera cable or hub.")
    assert any("USB disconnect" in ln for ln in s[0]["lines"])
    assert "The camera stopped twice in 6 minutes. Suspect the head camera cable or hub." in g.said
    assert any(b.startswith("SUSPECT: Camera dropped off USB twice") for b in g.r.banner(g.t))
    g.usb.superspeed()
    g.run(0.3)
    assert g.producer.restarts == ["stall", "stall"]            # one restart in this episode too, no loop


def test_a_stall_more_than_10_min_after_the_last_restart_is_not_suspected():
    g = Rig()
    g.stall()
    g.run(7.0)
    g.back()
    g.run(health.SUSPECT_WINDOW_S + 5)
    g.stall()
    g.run(7.0)
    assert g.ev("hardware_suspect") == [] and g.producer.restarts == ["stall", "stall"]


def test_no_usb_drop_seen_points_at_the_board_not_the_cable():
    g = Rig()
    for _ in range(2):
        g.stall()
        g.run(7.0)
        g.back()
        g.run(60)
    assert g.ev("hardware_suspect")[0]["sentence"] == (
        "The camera stopped twice in 1 minute with no USB drop seen: suspect the camera board or the producer.")
    assert "The camera stopped twice in 1 minute. Suspect the camera board." in g.said


def test_never_a_restart_loop_at_most_three_in_20_min():
    g = Rig()
    for _ in range(cr.MAX_RESTARTS):
        g.stall()
        g.run(7.0)
        g.back()
        g.run(30)
    assert len(g.producer.restarts) == cr.MAX_RESTARTS
    g.stall()
    g.run(60)
    assert len(g.producer.restarts) == cr.MAX_RESTARTS
    assert g.said.count(cr.CAPPED_TEXT) == 1
    assert "camera_restart_skipped" in g.kinds()
    assert any("no more" in b for b in g.r.banner(g.t))


def test_no_restart_in_the_tools_own_restart_window_or_while_the_flow_restarts():
    g = Rig()
    g.own_until = g.t + health.OWN_WINDOW_S
    g.stall()
    g.run(30)
    assert g.producer.restarts == []
    g.busy = True
    g.run(40)                                                   # past the own window, the flow is restarting
    assert g.producer.restarts == []
    g.busy = False
    g.run(0.3)
    assert g.producer.restarts == ["stall"]


def test_only_in_the_phases_that_need_frames():
    g = Rig(phase="FIT")
    g.stall()
    g.run(20)
    assert g.producer.restarts == []
    g.phase = "GRID"
    g.run(0.3)
    assert g.producer.restarts == ["stall"]


def test_waiting_for_usb_ends_without_a_restart_when_the_session_leaves_the_camera_phases():
    g = Rig()
    g.usb.drop()
    g.stall()
    g.run(7.0)
    g.phase = "REPORT"
    g.run(0.3)
    assert g.producer.restarts == [] and g.r.ep is None
    assert g.ev("camera_recovered")[0]["outcome"] == "left_phase"


def test_health_asking_a_restart_for_sync_acts_at_once_a_freeze_waits_for_6_s():
    g = Rig()
    freeze = health.Event(g.t, "warn", "freeze", "frozen", "producer_restart")
    g.stall()
    g.t += 2.0
    g.tick([freeze])
    assert g.producer.restarts == []
    g.back()
    g.run(1.0)
    sync = health.Event(g.t, "warn", "sync", "0.9 ms apart", "producer_restart")
    g.tick([sync])
    assert g.producer.restarts == ["sync"]
    assert g.said == [cr.STEP_TEXT]
    assert g.noted[-1][1] == "sync"                             # health's sync_persists rule needs it
    g.run(0.3)
    assert g.said[-1] == cr.BACK_TEXT


def test_the_restart_runs_in_a_worker_thread_and_the_loop_goes_on():
    g = Rig(jobs=Jobs(auto=False))
    g.stall()
    g.run(7.0)
    assert g.r.ep.state == "restarting" and "restarting the camera producer" in g.r.banner(g.t)[0]
    g.run(5.0)                                                  # the loop keeps ticking, nothing more asked
    assert len(g.jobs.jobs) == 1
    g.jobs.finish()
    g.run(0.3)
    assert g.r.ep.state == "frames_wait"


def test_the_default_runner_is_a_thread():
    p = Producer(fail="boom")
    j = cr._Job(lambda: p("stall"))
    for _ in range(100):
        if j.done():
            break
        time.sleep(0.01)
    assert j.done() and j.error() == "RuntimeError: boom"


def test_the_report_names_each_stall_and_the_suspicion_with_its_lines():
    g = Rig()
    g.stall()
    g.usb.drop("18:40:31")
    g.run(7.0)
    g.usb.superspeed("18:40:40")
    g.run(0.3)
    g.back()
    g.run(1.0)
    g.run(300)
    g.stall()
    g.usb.drop()
    g.run(7.0)
    lines = cr.report_lines(g.r.as_dict(g.t))
    assert lines[0] == "- camera: 2 stall(s) acted on, 1 automatic producer restart(s)"
    assert "stall: back after a restart" in lines[1] and "off the USB bus" in lines[1]
    assert any(ln.lstrip().startswith("- SUSPECT") and "dropped off USB twice in 5 minutes" in ln for ln in lines)
    assert any("`[Thu Nov 12 18:46:31 2026] usb 2-3.1: USB disconnect" in ln for ln in lines)
    assert cr.report_lines({}) == [] and cr.report_lines(None) == []
