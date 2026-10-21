"""T_BS steps: reading transforms and intrinsics from an OKVIS config."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import okviscfg  # noqa: E402

REAL = HERE.parent / "results/phantom12-2026-10-15/imucam/phantom12_stereo.yaml"

TINY = """%YAML:1.0
cameras:
    - {T_SC: [1, 0, 0, 0.1,  0, 1, 0, 0.2,  0, 0, 1, 0.3,  0, 0, 0, 1],
       image_dimension: [1280, 800], distortion_coefficients: [0.1, 0.2, 0.3, 0.4],
       distortion_type: DTYPE, focal_length: [500, 501], principal_point: [640, 400]}
imu_parameters:
    T_BS: [0, 1, 0, 0.01,  -1, 0, 0, 0.02,  0, 0, 1.000004, 0.03,  0, 0, 0, 1]
"""


def _write(tmp_path, text):
    p = tmp_path / "rig.yaml"
    p.write_text(text)
    return p


def test_reads_a_minimal_config_and_cleans_its_rotations(tmp_path):
    r = okviscfg.load(_write(tmp_path, TINY.replace("DTYPE", "equidistant")))
    assert r.lens == "pinhole-equi" and len(r.T_SC) == 1
    assert np.allclose(r.T_SC[0][:3, 3], [0.1, 0.2, 0.3])
    assert np.allclose(r.K[0], [[500, 0, 640], [0, 501, 400], [0, 0, 1]])
    assert np.allclose(r.D[0].ravel(), [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(r.T_BS[:3, 3], [0.01, 0.02, 0.03])
    R = r.T_BS[:3, :3]
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-12 and r.max_nonortho > 5e-6
    assert np.allclose(R, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-5)
    assert okviscfg.load(_write(tmp_path, TINY.replace("DTYPE", "radialtangential"))).lens == "pinhole-radtan"
    five = TINY.replace("DTYPE", "equidistant").replace("0.3, 0.4]", "0.3, 0.4, 0.5]")
    assert np.allclose(okviscfg.load(_write(tmp_path, five)).D[0].ravel(), [0.1, 0.2, 0.3, 0.4])   # OKVIS's four


@pytest.mark.parametrize("change, match", [
    (("0, 1, 0, 0.01", "0, 0.1, 0, 0.01"), "not a rotation"),            # a digit typo
    (("1.000004, 0.03", "-1.000004, 0.03"), "not a rotation"),           # a mirror image
    (("0, 0, 0, 1]\n", "0.01, 0.02, 0.03, 1]\n"), "bottom row"),          # column-major
    (("0.3, 0.4]", "0.3]"), "want at least 4"),                            # too few coefficients
    (("0, 0, 1.000004, 0.03", "0, 0, 1.002, 0.03"), "not a rotation"),    # cleaning would move it 2e-3
    (("focal_length: [500, 501]", "focal_length: [500, 501, 502]"), "focal_length"),
    (("distortion_type: DTYPE, ", ""), "distortion_type"),
    (("focal_length: [500, 501]", "focal_lenght: [500, 501]"), "focal_length"),   # a missing key
    (("distortion_type: DTYPE", "cam_model: eucm, distortion_type: DTYPE"), "eucm"),
    (("DTYPE", "plumb_bob"), "distortion types"),
])
def test_a_config_that_is_not_what_it_seems_is_refused(tmp_path, change, match):
    text = TINY.replace(*change).replace("DTYPE", "equidistant")
    assert text != TINY.replace("DTYPE", "equidistant")
    with pytest.raises(ValueError, match=match):
        okviscfg.load(_write(tmp_path, text))


KB4T = "[0.1, 0.2, 0.3, 0.4, 0.001, -0.002]"          # k1..k4, p1, p2


def test_a_kb4t_config_keeps_all_six_numbers(tmp_path):
    """distortion_type kb4t (OKVIS's KB4T lens: equidistant k1..k4 plus tangential p1, p2) loads as pinhole-kb4t with D
    6x1; every other type keeps the first four numbers, as OKVIS reads them"""
    text = TINY.replace("DTYPE", "kb4t").replace("[0.1, 0.2, 0.3, 0.4]", KB4T)
    r = okviscfg.load(_write(tmp_path, text))
    assert r.lens == "pinhole-kb4t" and r.D[0].shape == (6, 1)
    assert np.array_equal(r.D[0].ravel(), [0.1, 0.2, 0.3, 0.4, 0.001, -0.002])
    assert np.allclose(r.K[0], [[500, 0, 640], [0, 501, 400], [0, 0, 1]])
    equi = okviscfg.load(_write(tmp_path, TINY.replace("DTYPE", "equidistant").replace("[0.1, 0.2, 0.3, 0.4]", KB4T)))
    assert equi.lens == "pinhole-equi" and equi.D[0].shape == (4, 1)


@pytest.mark.parametrize("coefficients", ["[0.1, 0.2, 0.3, 0.4]", "[0.1, 0.2, 0.3, 0.4, 0.001]",
                                          "[0.1, 0.2, 0.3, 0.4, 0.001, -0.002, 0.5]"])
def test_a_kb4t_block_without_exactly_six_numbers_is_refused(tmp_path, coefficients):
    """OKVIS stops on such a block (LOG(FATAL)): this package refuses it too"""
    text = TINY.replace("DTYPE", "kb4t").replace("[0.1, 0.2, 0.3, 0.4]", coefficients)
    with pytest.raises(ValueError, match="kb4t takes exactly 6"):
        okviscfg.load(_write(tmp_path, text))


def test_mixed_lens_models_are_refused(tmp_path):
    cams = TINY.split("imu_parameters:")[0].split("cameras:\n")[1]
    text = ("%YAML:1.0\ncameras:\n" + cams.replace("DTYPE", "equidistant") + cams.replace("DTYPE", "radialtangential")
            + "imu_parameters:" + TINY.split("imu_parameters:")[1])
    with pytest.raises(ValueError, match="distortion types"):
        okviscfg.load(_write(tmp_path, text))
    kb4t = cams.replace("DTYPE", "kb4t").replace("[0.1, 0.2, 0.3, 0.4]", KB4T)
    text = ("%YAML:1.0\ncameras:\n" + cams.replace("DTYPE", "equidistant") + kb4t
            + "imu_parameters:" + TINY.split("imu_parameters:")[1])
    with pytest.raises(ValueError, match="distortion types"):                 # OKVIS stops on this pair too
        okviscfg.load(_write(tmp_path, text))


@pytest.mark.skipif(not REAL.exists(), reason="phantom12 result not in this checkout")
def test_phantom12s_config_has_the_known_geometry():
    """Pins the conventions the T_BS steps rely on (T_SC = camera in IMU, T_BS = IMU in Head):
    read the other way round, none of these numbers come out."""
    r = okviscfg.load(REAL)
    assert r.lens == "pinhole-equi" and len(r.T_SC) == 2 and r.max_nonortho < 1e-5
    base = np.linalg.inv(r.T_SC[0]) @ r.T_SC[1]
    assert abs(np.linalg.norm(base[:3, 3]) - 0.04483) < 5e-5            # 44.83 mm (stereo fit)
    assert base[0, 3] > 0.04                                            # camera 1 to camera 0's right
    for T in r.T_SC:
        T_BC = r.T_BS @ T
        z, x = T_BC[:3, 2], T_BC[:3, 0]
        assert abs(np.degrees(np.arcsin(z[2])) + 30.0) < 0.5          # optical axis 30 deg down
        assert z[0] > 0.85 and x[1] < -0.99                            # forward; image x = robot right
        assert 0.13 < T_BC[0, 3] < 0.15                                # 142 mm ahead of the Head origin
    assert abs(r.K[0][0, 0] - 562.914) < 1e-3 and abs(r.K[1][0, 0] - 565.364) < 1e-3
    assert abs(r.K[1][0, 2] - 641.347) < 1e-3 and abs(r.K[1][1, 2] - 383.572) < 1e-3
    assert abs(r.D[0][0, 0] + 0.012380) < 1e-5 and abs(r.D[1][0, 0] + 0.012067) < 1e-5
