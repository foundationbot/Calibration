"""calibtool.gravity: 'up' in camera-0 coordinates for the placement guidance (9 Nov trial: the head camera looks
steeply down from ~1.5 m, so a board's distance from the camera says little about where on the floor it stands).

Read-only: the head IMU's accelerometer (slot 0 of /dev/shm/raw_imu_actuals, DMA.common's ShmRingBuffer<RawImuFrame>)
averaged over the newest ~0.5 s of the ring, rotated into camera 0 by R_SC of the live OKVIS config (robot.env's
OKVIS_CALIBRATION, /etc/okvis in the container = /etc/phantomos/okvis on the host). Fakeable: Gravity takes a reader
callable and R_SC."""
import json
import struct

import pytest

np = pytest.importorskip("numpy")

from calibtool import gravity as gv  # noqa: E402

SLOT = 832                  # sizeof(Slot<RawImuFrame>) on the robot (852160 = 192 + 1024 x 832, measured 9 Nov)


def write_ring(path, accs, slot_imu=0, capacity=1024, start=0, t0_ns=5_000_000_000, dt_ns=2_000_000, other=(1, 2, 3),
               magic=gv.MAGIC, torn=()):
    """a ring in DMA.common's layout: 192-byte header (magic, version, slot_size, capacity; head at 64), slots of
    [u64 sequence | i64 timestamp_ns, u32 cycle, u8 num_imus, 32 x (accel xyz, gyro xyz) float32]"""
    b = bytearray(gv.HEADER_BYTES + capacity * SLOT)
    struct.pack_into("<QQQQ", b, 0, magic, 1, SLOT, capacity)
    for k, a in enumerate(accs):
        pos = start + k
        off = gv.HEADER_BYTES + (pos % capacity) * SLOT
        struct.pack_into("<Q", b, off, 0 if pos in torn else pos + 1)
        struct.pack_into("<qIB", b, off + 8, t0_ns + k * dt_ns, pos, 2)
        for i in range(2):
            acc = a if i == slot_imu else other
            struct.pack_into("<6f", b, off + 8 + 16 + 24 * i, *acc, 0.0, 0.0, 0.0)
    struct.pack_into("<Q", b, gv.HEAD_OFFSET, start + len(accs))
    path.write_bytes(bytes(b))
    return path


UP = (0.3, -9.7, 1.2)


def test_ring_accel_averages_the_newest_half_second(tmp_path):
    accs = [(0.0, 0.0, 0.0)] * 250 + [UP] * 250          # 1 s at 500 Hz: only the newest 0.5 s counts
    acc, n = gv.ring_accel(write_ring(tmp_path / "r", accs), slot=0, window_s=0.5)
    assert np.allclose(acc, UP, atol=1e-5)
    assert 245 <= n <= 251


def test_ring_accel_reads_across_the_wrap_and_the_right_slot(tmp_path):
    p = write_ring(tmp_path / "r", [UP] * 40, slot_imu=1, capacity=16, start=30)
    acc, n = gv.ring_accel(p, slot=1, window_s=0.5)
    assert np.allclose(acc, UP, atol=1e-5) and n == 16
    acc0, _ = gv.ring_accel(p, slot=0, window_s=0.5)
    assert np.allclose(acc0, (1, 2, 3))


def test_ring_accel_skips_torn_slots(tmp_path):
    accs = [UP] * 10 + [(50.0, 50.0, 50.0)] + [UP] * 10
    acc, n = gv.ring_accel(write_ring(tmp_path / "r", accs, torn={10}), window_s=0.5)
    assert np.allclose(acc, UP, atol=1e-5) and n == 20


def test_ring_accel_refuses_a_foreign_or_empty_ring(tmp_path):
    with pytest.raises(ValueError, match="magic"):
        gv.ring_accel(write_ring(tmp_path / "a", [UP], magic=0x1234))
    with pytest.raises(ValueError, match="no samples"):
        gv.ring_accel(write_ring(tmp_path / "b", []))


def cam_pitched_down(deg):
    """'up' in camera coordinates (x right, y down, z forward) for a camera pitched deg below the horizon"""
    a = np.radians(deg)
    return np.array([0.0, -np.cos(a), -np.sin(a)])


def test_up_in_camera_and_the_pitch_it_implies():
    R_SC = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # camera -> IMU (some fixed mount)
    up_c = cam_pitched_down(50.0)
    acc_s = 9.81 * (R_SC @ up_c)
    up = gv.up_in_camera(acc_s, R_SC)
    assert np.allclose(up, up_c, atol=1e-9)
    assert gv.pitch_down_deg(up) == pytest.approx(50.0)
    assert gv.pitch_down_deg(cam_pitched_down(0.0)) == pytest.approx(0.0, abs=1e-9)


def test_live_config_path_from_robot_env():
    env = "# okvis\nexport OKVIS_IMAGE=x\nOKVIS_CALIBRATION=/etc/okvis/p11_kb4t_1280_imu.yaml\nOTHER=1\n"
    assert str(gv.okvis_config_path(env)) == "/etc/phantomos/okvis/p11_kb4t_1280_imu.yaml"
    assert str(gv.okvis_config_path('OKVIS_CALIBRATION="/etc/okvis/a.yaml"  # live\n')) == \
        "/etc/phantomos/okvis/a.yaml"
    assert str(gv.okvis_config_path("OKVIS_CALIBRATION=/data/b.yaml\n")) == "/data/b.yaml"
    assert gv.okvis_config_path("# OKVIS_CALIBRATION=/etc/okvis/c.yaml\n") is None


def _okvis_yaml(R):
    rows = []
    for i in range(2):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (0.03 * i, 0.0, 0.15)
        rows.append("    - {T_SC:\n        [" + ", ".join(repr(float(x)) for x in T.ravel()) + "],\n"
                    "        image_dimension: [1280, 800],\n        distortion_coefficients: [0.0, 0.0, 0.0, 0.0],\n"
                    "        distortion_type: equidistant,\n        focal_length: [568.0, 568.0],\n"
                    "        principal_point: [640.0, 400.0]}\n")
    return ("%YAML:1.0\ncameras:\n" + "".join(rows) + "imu_parameters:\n    T_BS:\n"
            "        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]\n")


def test_r_sc_of_camera_0_from_the_live_config(tmp_path):
    pytest.importorskip("cv2")
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    (tmp_path / "okvis").mkdir()
    (tmp_path / "okvis" / "live.yaml").write_text(_okvis_yaml(R))
    env = tmp_path / "robot.env"
    env.write_text("OKVIS_CALIBRATION=/etc/okvis/live.yaml\n")
    R_SC, cfg = gv.live_r_sc([tmp_path / "missing.env", env], mounts=(("/etc/okvis", str(tmp_path / "okvis")),))
    assert np.allclose(R_SC, R) and cfg == tmp_path / "okvis" / "live.yaml"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_gravity_reads_once_then_refreshes_occasionally():
    c, calls = Clock(), []
    up_c = cam_pitched_down(50.0)

    def read():
        calls.append(c.t)
        return 9.81 * up_c, 250
    g = gv.Gravity(read, np.eye(3), clock=c, refresh_s=30.0)
    assert np.allclose(g.up(), up_c)
    c.t = 10.0
    g.up()
    assert calls == [0.0]
    c.t = 31.0
    g.up()
    assert calls == [0.0, 31.0]
    st = g.status()
    json.dumps(st)
    assert st["ok"] and st["pitch_down_deg"] == pytest.approx(50.0, abs=0.1)


def test_gravity_unavailable_says_why_and_gives_none():
    def broken():
        raise FileNotFoundError("/dev/shm/raw_imu_actuals")
    g = gv.Gravity(broken, np.eye(3), clock=Clock())
    assert g.up() is None
    st = g.status()
    assert not st["ok"] and "raw_imu_actuals" in st["note"] and "distance only" in st["note"]
    g = gv.Gravity(lambda: (np.zeros(3), 1), None, clock=Clock(), note="no OKVIS config: robot.env not found")
    assert g.up() is None and "robot.env" in g.status()["note"]


def test_gravity_refuses_a_reading_that_is_not_gravity():
    """a robot being moved or a broken slot: |acc| far from 9.81 is not used"""
    g = gv.Gravity(lambda: ((0.0, -3.0, 0.0), 250), np.eye(3), clock=Clock())
    assert g.up() is None and "9.81" in g.status()["note"]


def test_a_failed_refresh_keeps_the_last_good_reading():
    c = Clock()
    vals = iter([(9.81 * cam_pitched_down(40.0), 250)])

    def read():
        return next(vals)
    g = gv.Gravity(read, np.eye(3), clock=c, refresh_s=30.0)
    first = g.up()
    c.t = 40.0
    assert np.allclose(g.up(), first)
    assert g.status()["ok"] and "kept" in g.status()["note"]
