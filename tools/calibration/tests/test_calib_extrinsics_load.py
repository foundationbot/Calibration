"""Step 80's config reader (it parses the bytes OKVIS reads itself): a kb4t block (step 71 --okvis-kb4t) keeps its six
distortion numbers, the others (the template's radialtangential) their four, as okviscfg reads them."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
m80 = importlib.import_module("80_validate_extrinsics")
okviscfg = importlib.import_module("okviscfg")
PKG = Path(__file__).resolve().parents[1]
FITA = PKG / "results" / "phantom12-2026-10-21-okvis-kb4t" / "phantom12_stereo_dense_thor-fita.yaml"
TEMPLATE = PKG / "config" / "okvis-template-phantom12.yaml"      # radialtangential


@pytest.mark.parametrize("path,kind,n", [(FITA, "kb4t", 6), (TEMPLATE, "radtan", 4)])
def test_load_calib_keeps_each_lens_its_own_distortion_numbers(path, kind, n):
    cal, ref = m80.load_calib(path), okviscfg.load(path)
    assert [lens.kind for lens in cal["lens"]] == [kind, kind]
    for c in (0, 1):
        assert cal["d"][c].size == n and np.allclose(cal["d"][c], np.ravel(ref.D[c])[:n]), c


def test_a_kb4t_config_undistorts_through_step_80s_lens():
    """before the fix the kb4t lens got four numbers and refused them (ValueError) in check_gravity"""
    cal = m80.load_calib(FITA)
    K = np.array([[cal["f"][0][0], 0, cal["c"][0][0]], [0, cal["f"][0][1], cal["c"][0][1]], [0, 0, 1.0]])
    pts = np.array([[[cal["c"][0][0], cal["c"][0][1]]]])
    assert np.allclose(cal["lens"][0].undistort(pts, K, cal["d"][0]).reshape(-1, 2), 0, atol=1e-2)
