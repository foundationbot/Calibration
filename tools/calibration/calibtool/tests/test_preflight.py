"""calibtool.preflight (design section 4 step 0; nothing moves). Every check says ok / warn / fail / info with a plain
detail and belongs to a gate: "collect" (needed to read the cameras and collect: M2's acceptance is all of them ok)
or "motion" (needed before the motors are enabled, M5: reported now, not blocking collection)."""
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

from calibtool import motors, preflight as pf, profile  # noqa: E402

HEAD_JSON = json.dumps({"boards": {"0": {"mxid": "19443010A1F94D2E00"}, "1": {"mxid": "19443010D195852600"}},
                        "cameras": {"ego_right": {"queue_id": 0, "board": 0}, "ego_left": {"queue_id": 3, "board": 0},
                                    "right_wrist": {"queue_id": 1, "board": 1}}})
SPEC = """
metadata: {name: producer, generation: 1276}
spec:
  replicas: 1
  strategy: {type: Recreate}
  template:
    spec:
      containers: [{name: producer, image: img, args: ["--width", "1280", "--height", "800", "--fps", "25"]}]
"""


def texture(h=200, w=320, seed=0):
    rng = np.random.default_rng(seed)
    import cv2
    return cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (5, 5), 1.5)


def frames(sync_ms=0.014, n=3, d=12, none_reads=0, size=(1280, 800)):
    yl = texture(size[1], size[0])
    yr = np.roll(yl, -d, axis=1)                    # the right camera sees the scene d px further left
    return {"pairs": n, "wanted": 3, "sync_median_ms": sync_ms, "size": size, "none_reads": none_reads,
            "fps": {"3": 25.0, "0": 25.0}, "left": yl, "right": yr}


def probes(**over):
    p = {
        "identity": lambda: ("phantom10", "60776b088362aaaaaaaaaaaaaaaaaaaa"),
        "lock": lambda: (True, "/tmp/calibtool-m2/calibd.lock"),
        "clock": lambda: 1.794e9,
        "boot_id": lambda: "fb744b57",
        "k0s": lambda: (True, "pods listed"),
        "autonomy": lambda: None,
        "pods": lambda: ["dma-video/producer-x", "phantom/dma-recorder-x", "positronic/vla-cli-x"],
        "producer_spec": lambda: SPEC,
        "head_camera_json": lambda: HEAD_JSON,
        "robot_env": lambda: None,
        "okvis": lambda: {"running": False, "detail": "okvis2x DaemonSet desired 0"},
        "usb": lambda: {"19443010A1F94D2E00": (10000, "2-3.1"), "19443010D195852600": (10000, "2-2")},
        "frames": lambda: frames(),
        "imu": lambda: (100, 350),
        "master": lambda: motors.MasterStatus(True, "2.1.1", 6065.3, 0, False, None),
        "motor_tool": lambda: {"path": "/root/p11-cal/dma_motor_motion", "md5": "208231a2", "has_targets": True,
                               "error": None},
        "desired_still": lambda: True,
        "shm_count": lambda: 140,
        "disk": lambda: (True, 225.0),
        "load": lambda: (9.4, 9),
    }
    p.update(over)
    return p


def run(**over):
    return pf.run(probes(**over), profile.load("P10"))


def by(res):
    return {c.name: c for c in res.checks}


def test_all_good_is_green_on_the_collect_gate():
    r = run()
    bad = [c for c in r.checks if c.gate == "collect" and c.status == "fail"]
    assert not bad, bad
    assert pf.green(r.checks)
    assert pf.green(r.checks, gate="motion")
    assert all(c.status in ("ok", "info") for c in r.checks), [c for c in r.checks if c.status not in ("ok", "info")]


def test_the_checks_are_all_there():
    names = set(by(run()))
    assert {"identity", "lock", "clock", "k0s", "autonomy", "producer", "head_camera_json", "okvis", "camera_users",
            "usb_link", "frames", "sync", "left_right", "imu", "disk", "shm", "policy_pods", "master", "motors_off",
            "desired_still", "motor_tool", "estop", "gantry"} <= names


@pytest.mark.parametrize("name,over,word", [
    ("identity", {"identity": lambda: ("phantom10", "deadbeef0000ffff")}, "machine-id"),
    ("lock", {"lock": lambda: (False, "held by pid 12")}, "held"),
    ("clock", {"clock": lambda: 1000.0}, "1970"),
    ("k0s", {"k0s": lambda: (False, "connection refused")}, "refused"),
    ("autonomy", {"autonomy": lambda: "autonomy: ARMED"}, "armed"),
    ("usb_link", {"usb": lambda: {"19443010A1F94D2E00": (480, "2-3.1")}}, "480"),
    ("frames", {"frames": lambda: frames(n=1)}, "1 of 3"),
    ("frames", {"frames": lambda: frames(none_reads=4)}, "empty"),
    ("frames", {"frames": lambda: frames(size=(640, 400))}, "640x400"),
    ("sync", {"frames": lambda: frames(sync_ms=0.6)}, "0.600"),
    ("left_right", {"frames": lambda: frames(d=-12)}, "swapped"),
    ("imu", {"imu": lambda: (100, 100)}, "head power switch"),
    ("disk", {"disk": lambda: (False, 3.0)}, "3.0 GB"),
])
def test_each_failing_probe_fails_its_check(name, over, word):
    c = by(run(**over))[name]
    assert c.status == "fail", c
    assert word in c.detail
    assert c.gate == "collect"
    assert not pf.green(run(**over).checks)


def test_policy_pods_block_motion_not_collection():
    r = run(pods=lambda: ["positronic/ghost-wbc-teleop-bsjrv", "positronic/foundation-vla-mbf94",
                          "dma-video/producer-x"])
    c = by(r)["policy_pods"]
    assert c.gate == "motion" and c.status == "fail"
    assert "ghost-wbc-teleop-bsjrv" in c.detail and "foundation-vla-mbf94" in c.detail
    assert pf.green(r.checks) and not pf.green(r.checks, gate="motion")


def test_motion_gate_checks():
    r = run(master=lambda: motors.MasterStatus(False, None, None, None, None, "dma-ethercat is inactive"),
            motor_tool=lambda: {"path": "/usr/bin/dma_motor_motion", "md5": "3808", "has_targets": False,
                                "error": None},
            desired_still=lambda: False)
    b = by(r)
    for n in ("master", "motor_tool", "desired_still"):
        assert b[n].gate == "motion" and b[n].status == "fail", n
    assert "targets" in b["motor_tool"].detail
    assert pf.green(r.checks)


def test_motors_already_on_is_reported():
    r = run(master=lambda: motors.MasterStatus(True, "2.1.1", 1.0, 1, False, None))
    assert by(r)["motors_off"].status == "warn"


def test_left_right_without_texture_is_inconclusive_not_a_failure():
    f = frames()
    f["left"] = np.full((800, 1280), 90, np.uint8)
    f["right"] = f["left"].copy()
    c = by(run(frames=lambda: f))["left_right"]
    assert c.status == "warn" and "could not tell" in c.detail


def test_okvis_running_is_listed_as_a_camera_user():
    r = run(okvis=lambda: {"running": True, "detail": "okvis2x pod Running"})
    assert by(r)["okvis"].status == "info" and "Running" in by(r)["okvis"].detail


def test_snapshot_and_health_baseline():
    r = run()
    assert r.snapshot["producer_yaml"] == SPEC and r.snapshot["head_camera_json"] == HEAD_JSON
    b = r.baseline
    assert b["boot_id"] == "fb744b57" and b["fps"] == 25.0 and b["load"] == 9.4
    assert b["head_serials"] == ["19443010A1F94D2E00"] and b["head_ports"] == ["2-3.1"]
    assert b["producer"]["generation"] == 1276 and b["producer"]["args"][:2] == ["--width", "1280"]


def test_a_probe_that_raises_is_a_failed_check_not_a_crash():
    def boom():
        raise OSError("no such file")
    c = by(run(head_camera_json=boom))["head_camera_json"]
    assert c.status == "fail" and "OSError" in c.detail


def test_checks_as_dicts_are_json_ready():
    r = run()
    json.dumps([c.as_dict() for c in r.checks])


# ------------------------------------------------------------------ pure helpers

def test_disparity_sign():
    f = frames(d=9)
    s, d = pf.disparity_sign(f["left"][:100, :320], f["right"][:100, :320])
    assert s == 1 and d == 9
    s, d = pf.disparity_sign(f["right"][:100, :320], f["left"][:100, :320])
    assert s == -1 and d == -9


def test_head_boards_from_head_camera_json():
    assert pf.head_serials(HEAD_JSON, (3, 0)) == ["19443010A1F94D2E00"]


def test_usb_speeds_from_sysfs(tmp_path):
    for name, vid, serial, speed in (("2-3.1", "03e7", "19443010A1F94D2E00", "10000"),
                                     ("2-1.4", "0bda", "x", "5000"), ("2-2", "03e7", "19443010D195852600", "480")):
        d = tmp_path / name
        d.mkdir()
        (d / "idVendor").write_text(vid + "\n")
        (d / "serial").write_text(serial + "\n")
        (d / "speed").write_text(speed + "\n")
    assert pf.usb_speeds(tmp_path) == {"19443010A1F94D2E00": (10000, "2-3.1"), "19443010D195852600": (480, "2-2")}


def test_policy_pod_names():
    names = ["positronic/ghost-wbc-teleop-bsjrv", "positronic/foundation-vla-mbf94", "phantom/dma-recorder-gdvq4",
             "okvis/nav-agent-gfpf8", "positronic/vla-cli-42xrc", "x/phantom-locomotion-abc", "x/omni-wbc-1"]
    assert pf.policy_pods(names) == ["positronic/ghost-wbc-teleop-bsjrv", "positronic/foundation-vla-mbf94",
                                     "x/phantom-locomotion-abc", "x/omni-wbc-1"]
