"""calibtool.health: the section-8 v1 watch. Each rule fires once per episode (and again only after its condition
cleared), says what it saw in plain words, and names the action (re-attach, producer restart, preflight again, stop
collecting) or a hardware suspicion quoting the evidence."""
import time

import pytest

pytest.importorskip("numpy")

from calibtool import health as hw  # noqa: E402

BASE = {"fps": 25.0, "load": 9.0, "boot_id": "b1", "head_serials": ["19443010A1F94D2E00"], "head_ports": ["2-3.1"],
        "producer": {"generation": 1276, "pod_uid": "u1", "args": ["--width", "1280"]}}


def cam(frames=100, none_run=0, reattaches=0, since=0.04, fps=25.0):
    return {"frames": frames, "none_run": none_run, "reattaches": reattaches, "since_frame_s": since, "fps": fps}


def good(**over):
    s = {"cams": {"3": cam(), "0": cam()}, "sync_median_ms": 0.014, "boot_id": "b1", "clock_s": 1.79e9, "k0s_ok": True,
         "imu_head": None, "usb": {"19443010A1F94D2E00": 10000}, "dmesg": [], "producer_log": [],
         "producer": dict(BASE["producer"]), "load": 9.0}
    s.update(over)
    return s


def codes(evs):
    return [e.code for e in evs]


def watch():
    return hw.Watch(BASE)


def test_all_good_is_quiet():
    w = watch()
    assert w.observe(0.0, good(imu_head=100)) == []
    assert w.observe(1.0, good(imu_head=600)) == []


def test_desync_reattach_then_restart():
    w = watch()
    e = w.observe(0.0, good(cams={"3": cam(none_run=25), "0": cam()}))
    assert codes(e) == ["desync"] and e[0].action == "reattach" and e[0].level == "warn"
    assert "queue 3" in e[0].message
    assert w.observe(0.5, good(cams={"3": cam(none_run=30), "0": cam()})) == []        # same episode
    e = w.observe(1.0, good(cams={"3": cam(none_run=50, reattaches=1), "0": cam()}))
    assert codes(e) == ["desync_persists"] and e[0].action == "producer_restart"


def test_desync_rearms_after_clearing():
    w = watch()
    w.observe(0.0, good(cams={"3": cam(none_run=25), "0": cam()}))
    w.observe(1.0, good())
    assert codes(w.observe(2.0, good(cams={"3": cam(none_run=25), "0": cam()}))) == ["desync"]


def test_freeze_then_second_freeze_in_ten_minutes_is_suspected():
    w = watch()
    e = w.observe(0.0, good(cams={"3": cam(since=2.5), "0": cam()}))
    assert codes(e) == ["freeze"] and e[0].action == "producer_restart"
    w.note_restart(5.0, "freeze")
    w.observe(10.0, good())
    e = w.observe(300.0, good(cams={"3": cam(since=3.0), "0": cam()}))
    assert codes(e) == ["freeze", "hardware"]
    assert e[1].level == "suspect"


def test_fps_drop_only_when_cpu_does_not_explain_it():
    w = watch()
    assert codes(w.observe(0.0, good(cams={"3": cam(fps=12.0), "0": cam()}, load=20.0))) == []
    assert codes(w.observe(1.0, good(cams={"3": cam(fps=12.0), "0": cam()}, load=9.5))) == ["fps_drop"]


def test_sync_restart_once_then_refuse():
    w = watch()
    e = w.observe(0.0, good(sync_median_ms=0.6))
    assert codes(e) == ["sync"] and e[0].action == "producer_restart" and e[0].level == "warn"
    w.note_restart(1.0, "sync")
    e = w.observe(30.0, good(sync_median_ms=0.7))
    assert codes(e) == ["sync_persists"] and e[0].level == "stop" and e[0].action == "refuse_collect"


def test_boot_id_change_means_preflight_again():
    e = watch().observe(0.0, good(boot_id="b2"))
    assert codes(e) == ["reboot"] and e[0].level == "stop" and e[0].action == "preflight"


def test_clock_1970():
    e = watch().observe(0.0, good(clock_s=100.0))
    assert codes(e) == ["clock"] and e[0].level == "stop"


def test_k0s_refused():
    e = watch().observe(0.0, good(k0s_ok=False))
    assert codes(e) == ["k0s"] and e[0].level == "warn"


def test_imu_silent_points_at_the_head_power_switch():
    w = watch()
    w.observe(0.0, good(imu_head=100))
    assert w.observe(0.5, good(imu_head=100)) == []
    e = w.observe(1.2, good(imu_head=100))
    assert codes(e) == ["imu_silent"] and "head power switch" in e[0].message
    assert w.observe(2.0, good(imu_head=100)) == []
    assert w.observe(2.5, good(imu_head=300)) == []
    w.observe(3.0, good(imu_head=300))
    assert codes(w.observe(4.5, good(imu_head=300))) == ["imu_silent"]


def test_usb_link_below_superspeed_is_hardware_without_restart():
    e = watch().observe(0.0, good(usb={"19443010A1F94D2E00": 480}))
    assert codes(e) == ["usb_link"] and e[0].level == "suspect" and e[0].action is None
    assert "480" in e[0].message


def test_head_camera_missing_from_usb():
    e = watch().observe(0.0, good(usb={}))
    assert codes(e) == ["usb_link"]


def test_dmesg_reset_on_the_head_port_only():
    w = watch()
    line_dongle = "[Mon Nov  9 19:12:33 2026] usb 2-1.4: USB disconnect, device number 7"
    assert w.observe(0.0, good(dmesg=[line_dongle])) == []
    line = "[Mon Nov  9 19:12:33 2026] usb 2-3.1: reset SuperSpeed USB device number 12 using tegra-xusb"
    e = w.observe(1.0, good(dmesg=[line]))
    assert codes(e) == ["usb_reset"] and e[0].level == "suspect" and line in e[0].lines
    assert w.observe(2.0, good(dmesg=[line])) == []                 # a line already seen does not fire again


def test_producer_device_crashed():
    line = ("[19443010A1F94D2E00] [1.4.1] [743.9] [system] [critical] Device crashed, but no crash dump could be "
            "extracted.")
    e = watch().observe(0.0, good(producer_log=[line]))
    assert codes(e) == ["device_crashed"] and e[0].level == "suspect" and line in e[0].lines


def test_someone_else_changes_the_producer():
    w = watch()
    e = w.observe(0.0, good(producer={"generation": 1277, "pod_uid": "u2", "args": ["--width", "640"]}))
    assert codes(e) == ["producer_changed"] and e[0].level == "stop"


def test_own_restart_is_not_someone_else():
    w = watch()
    with w.own_change():
        assert w.observe(0.0, good(producer={"generation": 1276, "pod_uid": "u2", "args": ["--width", "1280"]})) == []
    w.adopt_producer({"generation": 1276, "pod_uid": "u2", "args": ["--width", "1280"]})
    assert w.observe(1.0, good(producer={"generation": 1276, "pod_uid": "u2", "args": ["--width", "1280"]})) == []


def test_events_are_json_ready():
    import json
    e = watch().observe(0.0, good(boot_id="b2"))
    json.dumps([x.as_dict() for x in e])


def _t(hh, mm):
    lt = list(time.localtime(1.79e9))
    lt[3], lt[4], lt[5] = hh, mm, 0
    return time.mktime(time.struct_time(lt))


def test_suspicion_sentence_restarts_with_usb_resets():
    w = watch()
    for hh, mm in ((15, 21), (15, 31), (15, 41)):
        w.observe(_t(hh, mm) - 5, good(dmesg=[f"[x] usb 2-3.1: reset SuperSpeed USB device {hh}{mm}"]))
        w.note_restart(_t(hh, mm), "freeze")
    s = hw.suspicion_sentence(w, _t(15, 41))
    assert s == ("3rd producer restart in 20 min; USB reset on the head camera's port before each (15:21, 15:31, "
                 "15:41) -> suspect the head camera cable or hub, not software.")


def test_suspicion_sentence_without_usb_resets():
    w = watch()
    for mm in (21, 41):
        w.note_restart(_t(15, mm), "freeze")
    s = hw.suspicion_sentence(w, _t(15, 41))
    assert s.startswith("2nd producer restart in 20 min; no USB reset logged")


def test_no_sentence_for_a_single_restart():
    w = watch()
    w.note_restart(_t(15, 0), "freeze")
    assert hw.suspicion_sentence(w, _t(15, 5)) is None


def test_a_stale_imu_sample_is_not_silence():
    """found on phantom10 9 Nov: the sampler's IMU reading was not refreshed while its kubectl calls ran, and the rule
    called that silence. The rule judges the reading's own time (imu_t), not when the watch looked"""
    w = watch()
    w.observe(0.0, good(imu_head=100, imu_t=0.0))
    for t in (1.0, 2.0, 3.0, 4.0):
        assert w.observe(t, good(imu_head=100, imu_t=0.0)) == []        # the same old reading: no judgement
    assert w.observe(5.0, good(imu_head=600, imu_t=5.0)) == []
    assert codes(w.observe(6.5, good(imu_head=600, imu_t=6.5))) == ["imu_silent"]


def test_desync_comes_first_even_when_the_reader_already_reattached():
    """review 5: capture re-attaches at the 25th empty read, so the first sample the watch sees already has
    reattaches == 1, and the old rule skipped 'desync' and asked for a producer restart at once"""
    w = watch()
    e = w.observe(0.0, good(cams={"3": cam(none_run=25, reattaches=1), "0": cam()}))
    assert codes(e) == ["desync"]
    assert w.observe(0.5, good(cams={"3": cam(none_run=40, reattaches=1), "0": cam()})) == []
    e = w.observe(1.5, good(cams={"3": cam(none_run=60, reattaches=2), "0": cam()}))
    assert codes(e) == ["desync_persists"]
    w.observe(2.0, good())
    e = w.observe(3.0, good(cams={"3": cam(none_run=25, reattaches=3), "0": cam()}))
    assert codes(e) == ["desync"]               # a new episode starts over


def test_the_tools_own_producer_restart_raises_nothing_and_relearns_the_fps():
    """10 Nov, phantom10/phantom11: every own restart gave "someone else changed the camera producer", "head camera
    not on the USB bus", "USB disconnect on the head camera's port" and then "15 fps against 25" for the whole session"""
    w = watch()
    line = "[Tue Nov 10 14:35:15 2026] usb 2-3.1: USB disconnect, device number 4"
    new = {"generation": 1285, "pod_uid": "u9", "args": ["--width", "1280", "--fps", "15"]}
    during = good(usb={}, dmesg=[line], producer=new, cams={"3": cam(fps=0.0), "0": cam(fps=0.0)}, own_until=60.0)
    assert w.observe(10.0, during) == []
    after = good(producer=new, dmesg=[line], cams={"3": cam(fps=15.0), "0": cam(fps=15.0)}, own_until=60.0)
    assert w.observe(61.0, after) == []                    # same producer, same line, 15 fps is the new normal
    assert w.observe(62.0, good(producer=new, cams={"3": cam(fps=8.0), "0": cam(fps=8.0)}))[0].code == "fps_drop"


def test_a_change_after_the_own_window_is_someone_else_again():
    w = watch()
    w.observe(10.0, good(own_until=60.0))
    e = w.observe(100.0, good(producer={"generation": 1290, "pod_uid": "u3", "args": ["--width", "640"]},
                              own_until=60.0))
    assert "producer_changed" in codes(e)
