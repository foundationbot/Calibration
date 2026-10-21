"""Step 70's image_delay derivation and its stamping-bug guard."""
import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
make_yaml = importlib.import_module("70_make_mk13_yaml")


def test_image_delay_is_minus_the_mean_kalibr_timeshift():
    # Kalibr: t_imu = t_cam + shift.  OKVIS: t_cam_correct = t_cam - image_delay.
    delay, _ = make_yaml.image_delay_from_timeshifts([0.0012, 0.0011])
    assert delay == pytest.approx(-0.00115)


def test_a_few_ms_of_timeshift_raises_no_warning():
    _, warning = make_yaml.image_delay_from_timeshifts([0.0012, 0.0011])
    assert warning is None


def test_a_rebase_sized_timeshift_is_flagged():
    # phantom13 2026-08-22 on rebased stamps: -23.6 / -24.6 ms -> image_delay 0.0241,
    # which current OKVIS (no live rebase) would apply as a real 24 ms error.
    delay, warning = make_yaml.image_delay_from_timeshifts([-0.0236, -0.0246])
    assert delay == pytest.approx(0.0241)
    assert warning is not None and "rebase" in warning


# ---- step 70 end to end on the phantom12 rig ------------------------------------
PKG = Path(__file__).resolve().parent.parent
RIG12 = PKG / "config" / "rig-phantom12.yaml"

TEMPLATE = """%YAML:1.0
# old header line
cameras:
    - {T_SC:
        [1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1],
        image_dimension: [1280, 800],
        distortion_coefficients: [0, 0, 0, 0],
        distortion_type: radialtangential,
        focal_length: [500, 500],
        principal_point: [640, 400],
        dma_queue: 1}
    - {T_SC:
        [1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1],
        image_dimension: [1280, 800],
        distortion_coefficients: [0, 0, 0, 0],
        distortion_type: radialtangential,
        focal_length: [500, 500],
        principal_point: [640, 400],
        dma_queue: 2}
camera_parameters:
    image_delay: 0.0241
imu_parameters:
    s_a: [1.0, 1.0, 1.0]
{tbs}"""

TBS = """    T_BS:
        [1, 0, 0, 0.01,
         0, 1, 0, 0.02,
         0, 0, 1, 0.03,
         0, 0, 0, 1]
"""


def _camchain(tmp_path, shift=0.0012):
    T_ci = [[1, 0, 0, 0.05], [0, 1, 0, -0.15], [0, 0, 1, -0.10], [0, 0, 0, 1]]
    cam = lambda f: {"camera_model": "pinhole", "distortion_model": "equidistant",
                     "distortion_coeffs": [-0.02, 0.05, -0.05, 0.02],
                     "intrinsics": [f, f, 640.0, 400.0], "resolution": [1280, 800],
                     "T_cam_imu": T_ci, "timeshift_cam_imu": shift}
    cc = {"cam0": cam(568.0), "cam1": {**cam(569.0),
          "T_cn_cnm1": [[1, 0, 0, -0.08], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}}
    p = tmp_path / "camchain-imucam.yaml"
    p.write_text(yaml.safe_dump(cc))
    return p


def _run_step70(monkeypatch, tmp_path, template_text, camchain):
    tpl = tmp_path / "template.yaml"
    tpl.write_text(template_text)
    out = tmp_path / "phantom12_stereo.new.yaml"
    monkeypatch.setattr(sys, "argv", ["70", str(tpl), str(camchain), str(out),
                                      "--rig", str(RIG12)])
    make_yaml.main()
    return out


def test_phantom12_template_without_T_BS_is_refused(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as e:
        _run_step70(monkeypatch, tmp_path, TEMPLATE.replace("{tbs}", ""), _camchain(tmp_path))
    assert "T_BS" in str(e.value)
    assert not (tmp_path / "phantom12_stereo.new.yaml").exists()


def test_phantom12_yaml_is_written_from_the_camchain(monkeypatch, tmp_path):
    out = _run_step70(monkeypatch, tmp_path, TEMPLATE.replace("{tbs}", TBS), _camchain(tmp_path))
    text = out.read_text()
    assert "image_delay: -0.0012" in text                 # raw host-clock stamps
    assert text.count("distortion_type: equidistant") == 2
    assert "focal_length: [568, 568]" in text and "focal_length: [569, 569]" in text
    assert "0, 1, 0, 0.02" in text                        # template T_BS preserved
    # T_SC = inv(T_cam_imu): translation of the inverse of [I | (0.05,-0.15,-0.10)]
    assert "-0.05" in text and "0.15" in text


def test_phantom12_header_describes_an_aprilgrid_fit(monkeypatch, tmp_path):
    out = _run_step70(monkeypatch, tmp_path, TEMPLATE.replace("{tbs}", TBS), _camchain(tmp_path))
    header = "".join(l for l in out.read_text().splitlines(True) if l.startswith("#"))
    assert "aprilgrid" in header
    assert "checkerboard" not in header
    assert "step 30" not in header        # no flip filtering exists for an aprilgrid
