"""calibtool.scaling: an OKVIS config written for another producer size (design 6.3).

The rule pinned here: focal x s; principal point (c + 0.5) x s - 0.5 (OKVIS's own resize, ViParametersReader.cpp
"fx_new = fx * s, cx_new = (cx + 0.5) * s - 0.5"); image_dimension exact integers; resize_factor written so that
OKVIS's int(dim x resize_factor) (ThreadedSlam.cpp, a double; float32 checked too) gives exactly 640x400 for VIO.
Nothing else in the file changes."""
import difflib
import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from calibtool import scaling  # noqa: E402

VIO = (640, 400)


def _okvis_yaml(**kw):
    spec = importlib.util.spec_from_file_location("_t_livecfg", Path(__file__).with_name("test_livecfg.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.okvis_yaml(**kw)


def _cams(text):
    import cv2
    fs = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    cams = fs.getNode("cameras")
    out = []
    for i in range(cams.size()):
        c = cams.at(i)
        get = lambda k: [c.getNode(k).at(j).real() for j in range(c.getNode(k).size())]   # noqa: E731
        out.append({k: get(k) for k in ("image_dimension", "focal_length", "principal_point", "T_SC",
                                         "distortion_coefficients")})
    rf = fs.getNode("frontend_parameters").getNode("resize_factor")
    return out, (None if rf.empty() else rf.real())


def _rf_text(text):
    lines = [ln for ln in text.splitlines() if ln.split("#")[0].strip().startswith("resize_factor:")]
    assert len(lines) == 1
    return lines[0].split("#")[0].split(":", 1)[1].strip()


def _changed(a, b):
    """the lines that differ between two texts (removed and added)"""
    return [ln[1:] for ln in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
            if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]


KEYS = ("image_dimension:", "focal_length:", "principal_point:", "resize_factor:")


@pytest.mark.parametrize("s,size", [(0.75, (960, 600)), (0.5, (640, 400)), (1.0, (1280, 800))])
def test_scale_config(s, size):
    src = _okvis_yaml()
    out = scaling.scale_config(src, s)
    a, rf_a = _cams(src)
    b, rf_b = _cams(out)
    for ca, cb in zip(a, b):
        assert cb["image_dimension"] == list(size)
        assert np.allclose(cb["focal_length"], [f * s for f in ca["focal_length"]], rtol=0, atol=1e-12)
        assert np.allclose(cb["principal_point"], [(c + 0.5) * s - 0.5 for c in ca["principal_point"]], rtol=0,
                           atol=1e-12)
        for k in ("T_SC", "distortion_coefficients"):
            assert cb[k] == ca[k]
    rf = _rf_text(out)
    for conv in (float, np.float32):
        f = conv(rf)
        assert (int(size[0] * f), int(size[1] * f)) == VIO, (rf, conv)
        assert (int(np.float64(size[0]) * np.float64(f)), int(np.float64(size[1]) * np.float64(f))) == VIO
    assert scaling.config_size(out) == size
    changed = _changed(src, out)
    assert all(ln.split("#")[0].strip().startswith(KEYS) for ln in changed), changed
    if s == 1.0:
        assert out == src                                     # 0.5 already gives 640x400: nothing to change


def test_resize_factor_values():
    assert scaling.resize_factor_text((1280, 800), VIO) == "0.5"
    assert scaling.resize_factor_text((640, 400), VIO) == "1.0"
    t = scaling.resize_factor_text((960, 600), VIO)
    assert t == "0.6666667"
    assert float(t) > 2 / 3                               # 2/3 as a double: int(960 * 0.666...) = 639 risk
    with pytest.raises(ValueError):
        scaling.resize_factor_text((600, 400), VIO)      # cannot hit both sides


def test_comment_lines_are_left_alone():
    src = _okvis_yaml()
    out = scaling.scale_config(src, 0.75)
    assert "# a test config; focal_length: [1, 2] in a comment is not a key" in out
    assert "# -> 640x400 for the whole VIO" in out           # the trailing comment of resize_factor is kept


@pytest.mark.parametrize("s", [0.75, 0.5, 1.0])
def test_unscale_inverts_scale(s):
    src = _okvis_yaml()
    back = scaling.unscale_config(scaling.scale_config(src, s), s)
    a, rf_a = _cams(src)
    b, rf_b = _cams(back)
    for ca, cb in zip(a, b):
        for k in ca:
            assert np.allclose(ca[k], cb[k], rtol=0, atol=1e-9), k
    assert rf_b == rf_a == 0.5
    assert all(ln.split("#")[0].strip().startswith(KEYS) for ln in _changed(src, back))


def test_scale_refuses_fractional_sizes_and_missing_resize_factor():
    with pytest.raises(ValueError, match="image_dimension"):
        scaling.scale_config(_okvis_yaml(), 0.33)
    no_rf = _okvis_yaml(resize=None)
    with pytest.raises(ValueError, match="resize_factor"):
        scaling.scale_config(no_rf, 0.75)
    out = scaling.scale_config(no_rf, 0.5)                   # VIO at 640x400 without resizing: no key needed
    assert scaling.config_size(out) == (640, 400) and "resize_factor" not in out.split("frontend_parameters")[1]


def test_factor():
    assert scaling.factor((960, 600), (1280, 800)) == 0.75
    assert scaling.factor((640, 400), (1280, 800)) == 0.5
    with pytest.raises(ValueError):
        scaling.factor((960, 540), (1280, 800))             # not one factor for both sides


def test_scaled_pp():
    assert scaling.scaled_pp(645.3093383991189, 0.75) == (645.3093383991189 + 0.5) * 0.75 - 0.5
    assert abs(scaling.scaled_pp(645.3093383991189, 0.75) - 645.3093383991189 * 0.75 - (-0.125)) < 1e-12
    assert abs(scaling.unscaled_pp(scaling.scaled_pp(377.83, 0.75), 0.75) - 377.83) < 1e-12


# ---------------------------------------------------------------- the stored phantom10 configs (fast, need data)


P10_PAIRS = [("phantom11_stereo_kb4t_20261105_1280_rscfix.yaml", "phantom11_stereo_kb4t_20261105_960_rscfix.yaml"),
             ("phantom11_stereo_kb4t_20261106_1280_imu.yaml", "phantom11_stereo_kb4t_20261106_960_imu.yaml")]


@pytest.mark.parametrize("native,deployed", P10_PAIRS)
def test_p10_960_configs_used_c_times_s(data_root, native, deployed):
    """phantom10's 960 configs were made by cal960.py (c x s): ours differ from them by exactly -0.125 px in the
    principal point (0.5 s - 0.5 at s = 0.75), the focal length is the same number, and resize_factor 0.6666667 vs
    their 0.66666669 (both give 640x400)"""
    d = data_root / "p10" / "etc-okvis"
    src, theirs = (d / native).read_text(), (d / deployed).read_text()
    ours = scaling.scale_config(src, 0.75)
    a, _ = _cams(ours)
    b, rf_b = _cams(theirs)
    for ca, cb in zip(a, b):
        assert ca["image_dimension"] == cb["image_dimension"] == [960, 600]
        assert np.allclose(ca["focal_length"], cb["focal_length"], rtol=0, atol=1e-12)
        assert np.allclose(np.subtract(ca["principal_point"], cb["principal_point"]), -0.125, atol=1e-9)
        assert ca["T_SC"] == cb["T_SC"] and ca["distortion_coefficients"] == cb["distortion_coefficients"]
    assert _rf_text(ours) == "0.6666667" and _rf_text(theirs) == "0.66666669"
    changed = _changed(src, ours)
    assert all(ln.split("#")[0].strip().startswith(KEYS) for ln in changed), changed
    assert len(changed) == 2 * (2 * 3 + 1)                   # per camera 3 keys, plus resize_factor; - and + lines


@pytest.mark.replay
def test_960_config_unscaled_scores_on_1280_data(data_root, tmp_path):
    """phantom10's live 960 config (made from the 1280 rscfix config by c x s) brought back to 1280 by the inverse
    rule scores on the 1280 TEST pairs like the 1280 config it came from (their principal points differ by 0.5/0.75 -
    0.5 = 0.17 px)"""
    from calibtool import livecfg, sessions
    st = sessions.get("p10_1106")
    d = data_root / "p10" / "etc-okvis"
    test_npz = data_root / st.recorded.files["test_npz"]
    l960 = livecfg.load(d / "phantom11_stereo_kb4t_20261105_960_rscfix.yaml", at_size=(1280, 800))
    l1280 = livecfg.load(d / "phantom11_stereo_kb4t_20261105_1280_rscfix.yaml")
    assert l960.size == (1280, 800) and l960.scaled_from == (960, 600)
    assert np.allclose(l960.intr[0][:2], l1280.intr[0][:2], rtol=1e-12)
    assert np.allclose(np.subtract(l960.intr[0][2:4], l1280.intr[0][2:4]), 0.5 / 0.75 - 0.5, atol=1e-9)
    a = livecfg.summary(l960, test_npz, st.rig_path, tmp_path / "a")
    b = livecfg.summary(l1280, test_npz, st.rig_path, tmp_path / "b")
    print(f"P10 960->1280: median {a.median_pct:+.4f} IQR {a.iqr_pct} rows {a.rows_px:.4f}; 1280: median "
          f"{b.median_pct:+.4f} IQR {b.iqr_pct} rows {b.rows_px:.4f}")
    assert a.n_pairs == b.n_pairs == st.recorded.test_pairs_16
    assert abs(a.median_pct - b.median_pct) < 0.05 and abs(a.rows_px - b.rows_px) < 0.1
    assert abs(b.median_pct - st.recorded.summary_live.median_pct) < 0.02
