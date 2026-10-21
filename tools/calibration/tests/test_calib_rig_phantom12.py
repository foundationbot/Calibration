"""The phantom12 rig file and what every step derives from it."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "scripts"))
import rigcfg  # noqa: E402

RIG = PKG / "config" / "rig-phantom12.yaml"


@pytest.fixture
def rig():
    return rigcfg.load(RIG)


def test_phantom12_uses_the_aprilgrid(rig):
    assert rig.board.kind == "aprilgrid"


def test_phantom12_left_eye_is_dma_queue_1_and_okvis_cam0(rig):
    # /etc/phantom/head_camera.json on phantom12: queue 1 'left' socket B,
    # queue 2 'right' socket D. Step 60 measures the side from the disparity sign.
    assert rig.cam_map() == [("cam1", "cam0"), ("cam2", "cam1")]
    assert rig.okvis_queues() == [1, 2]
    assert rig.okvis_sides() == ["left", "right"]
    assert rig.expected_dx_sign() == +1


def test_phantom12_imu_is_neckpitch_at_index_2(rig):
    # read from /dev/shm/config on phantom12, 2026-10-14
    assert (rig.imu["name"], rig.imu["index"]) == ("NeckPitch", 2)


def test_phantom12_borrows_no_imu_in_head_transform(rig):
    # phantom13's T_BS is that robot's IMU mount; step 70 must take phantom12's
    # from its own template instead of silently injecting another robot's.
    with pytest.raises(SystemExit) as e:
        rig.t_bs_default_tokens()
    assert "template" in str(e.value)


def test_the_board_is_measured_at_a_0120_m_pitch(rig, capsys):
    # Steel rule, 2026-10-14: 720 mm over 6 pitches along a row, 480 mm over 4 down
    # a column. Step 65 compares the triangulated board against this pitch.
    rigcfg.load(RIG)
    assert rig.target_measured
    assert "MEASURE" not in capsys.readouterr().out
    assert rig.board.pitch == pytest.approx(0.120)


def test_phantom12_imu_noise_is_the_measured_model(rig):
    imu = yaml.safe_load(rig.resolve(rig.imu["kalibr_yaml"]).read_text())
    # Allan analysis 2026-10-14 (x10 inflation): far below phantom13's inherited
    # 1.6e-3 gyro / 1.1e-2 accel / 5.2e-3 accel random walk.
    assert imu["gyroscope_noise_density"] == pytest.approx(5.0e-4)
    assert imu["accelerometer_noise_density"] == pytest.approx(8.0e-3)
    assert imu["accelerometer_random_walk"] == pytest.approx(2.0e-4)
    assert imu["update_rate"] == 500
