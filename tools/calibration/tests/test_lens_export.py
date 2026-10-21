"""Step 71: the canonical JSON, a Kalibr camchain the package's own checks accept, an OKVIS config OpenCV loads (only
on request: the 4-term lens is unusable for depth; --okvis-kb4t writes the whole KB4T lens for an OKVIS image with the
kb4t change)."""
import importlib
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
yaml = pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = importlib.import_module("lens_synth")
m71 = importlib.import_module("71_export_lens")
m66 = importlib.import_module("66_validate_lens")
okviscfg = importlib.import_module("okviscfg")
rm = importlib.import_module("rigmath")
TPL = importlib.import_module("test_calib_make_yaml")       # step 70's test template (two blocks, dma_queue 1, 2)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("x")
    P = S.pipeline()
    assert m66.main([str(P / "fit.json"), str(P / "test.npz"), str(out / "v"), "--uncertainty", str(P / "unc.json"),
                     "--rig", str(S.RIG12)]) == 0
    tpl = out / "template.yaml"
    tpl.write_text(TPL.TEMPLATE.replace("{tbs}", TPL.TBS))
    assert m71.main([str(P / "fit.json"), str(out / "v" / "report.json"), str(out / "e"), "--okvis-template", str(tpl),
                     "--allow-4term-okvis", "--rig", str(S.RIG12)]) == 0
    return out


def test_the_canonical_json_keeps_all_ten_numbers_and_the_masks(exported):
    fit = json.loads((S.pipeline() / "fit.json").read_text())
    c = json.loads((exported / "e" / "phantom12-kb4t.json").read_text())
    assert c["cameras"][0]["intr"] == fit["intr"][0] and len(c["intr_names"]) == 10
    assert c["validation"]["verdict"] == "PASS" and "top-left" in c["validation"]["not_validated_cells"]
    assert c["cameras"][0]["side"] == "left" and c["cameras"][1]["dma_queue"] == 2
    assert c["p_drop_depth_pct"]["within_450px"] == pytest.approx(9.3, abs=0.5)      # the fit is close to Fit A
    assert c["fit_verdict"] == fit["verdict"] and c["fit_verdict"].startswith("converged")
    assert [x["points_without_ray"] for x in c["p_drop_cost_px"]] == [0, 0]


def test_the_camchain_is_kb4_with_the_inverse_stereo_pose(exported):
    rig = S.rig12()
    cc = yaml.safe_load((exported / "e" / "camchain-kb4t-equi4.yaml").read_text())
    fit = json.loads((S.pipeline() / "fit.json").read_text())
    assert np.allclose(np.array(cc["cam1"]["T_cn_cnm1"]) @ np.array(fit["C"]), np.eye(4), atol=1e-12)
    assert cc["cam0"]["distortion_coeffs"] == fit["intr"][0][4:8]
    rig.camchain_lens(cc, "cam0"), rig.camchain_lens(cc, "cam1")    # model and resolution as the rig says
    text = (exported / "e" / "camchain-kb4t-equi4.yaml").read_text()
    assert "DROPPED" in text and "% within 450 px" in text                           # the depth cost, as a comment


def test_the_okvis_config_loads_and_composes_t_sc1(exported):
    fit = json.loads((S.pipeline() / "fit.json").read_text())
    o = okviscfg.load(exported / "e" / "phantom12_stereo.yaml")
    assert o.lens == "pinhole-equi"
    assert np.allclose(o.T_SC[0], np.eye(4)) and np.allclose(o.T_SC[1], np.array(fit["C"]), atol=1e-9)
    assert np.allclose(o.K[0][0, 0], fit["intr"][0][0]) and np.allclose(o.D[1].ravel(), fit["intr"][1][4:8])
    text = (exported / "e" / "phantom12_stereo.yaml").read_text()
    assert "APPROXIMATION" in text and "0, 1, 0, 0.02" in text                     # T_BS kept
    assert "% within 450 px" in text


def test_okvis_kb4t_writes_the_whole_lens(exported, tmp_path, capsys):
    """--okvis-kb4t: distortion_type kb4t with all six coefficients per block, in OKVIS block order; the file loads
    through okviscfg as kb4t with the fit's ten numbers per camera, and its header says who reads what and that OKVIS
    images without the kb4t change refuse it"""
    P = S.pipeline()
    fit = json.loads((P / "fit.json").read_text())
    assert m71.main([str(P / "fit.json"), str(exported / "v" / "report.json"), str(tmp_path / "e"),
                     "--okvis-template", str(exported / "template.yaml"), "--okvis-kb4t", "--rig", str(S.RIG12)]) == 0
    o = okviscfg.load(tmp_path / "e" / "phantom12_stereo.yaml")
    assert o.lens == "pinhole-kb4t"
    for block, cam in enumerate(S.rig12().okvis_order):
        K, D = o.K[block], o.D[block].ravel()
        got = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D]
        assert len(got) == 10 and np.allclose(got, fit["intr"][cam], rtol=1e-14, atol=0)
    assert np.allclose(o.T_SC[0], np.eye(4)) and np.allclose(o.T_SC[1], np.array(fit["C"]), atol=1e-9)
    text = (tmp_path / "e" / "phantom12_stereo.yaml").read_text()
    assert text.count("distortion_type: kb4t") == 2 and "APPROXIMATION" not in text
    assert "depth: full KB4T" in text and "VIO: k1..k4" in text and "older images refuse this file" in text
    assert "KB4T rectification" in text and "0, 1, 0, 0.02" in text                 # deploy check; T_BS kept
    assert "kb4t" in capsys.readouterr().out


@pytest.mark.parametrize("flags, msg", [
    (["--okvis-template", "TPL", "--okvis-kb4t", "--allow-4term-okvis"], "not both"),
    (["--okvis-kb4t"], "needs --okvis-template"),
])
def test_okvis_kb4t_is_refused_with_the_4term_flag_or_without_a_template(exported, tmp_path, flags, msg):
    flags = [str(exported / "template.yaml") if f == "TPL" else f for f in flags]
    with pytest.raises(SystemExit) as e:
        m71.main([str(S.pipeline() / "fit.json"), str(exported / "v" / "report.json"), str(tmp_path / "e"), *flags,
                  "--rig", str(S.RIG12)])
    assert msg in str(e.value)
    assert not (tmp_path / "e").exists()                                           # refused before anything is written


def test_the_drop_cost_is_measured_not_quoted():
    cost = m71.p_drop_cost(S.I0, (1280, 800))                          # Fit A camera 0 (scratch run, 20 Oct)
    assert cost["within_350px"] == pytest.approx(0.90, abs=0.05) and cost["whole_image"] == pytest.approx(4.16, abs=0.1)
    assert cost["points_without_ray"] == 0


def test_the_drop_cost_leaves_out_pixels_where_the_lens_has_no_ray(monkeypatch):
    """k1 = -0.2: the KB4 polynomial turns over at theta_d 0.86 rad, ~490 px from the principal point, inside the image.
    Beyond it no ray exists; unproject's Newton steps return a finite wrong ray there (a 25000 px "shift" before this
    test), or NaN. Such grid points are left out and counted"""
    lens = S.I0.copy()
    lens[4] = -0.2
    cost = m71.p_drop_cost(lens, (1280, 800))
    assert cost["whole_image"] < 10 and cost["within_350px"] == pytest.approx(0.90, abs=0.05)
    assert 100 < cost["points_without_ray"] < 2560
    real = m71.lensmodel.unproject

    def nan_rays(model, intr, uv):
        ray = real(model, intr, uv)
        ray[:5] = np.nan
        return ray
    monkeypatch.setattr(m71.lensmodel, "unproject", nan_rays)
    cost = m71.p_drop_cost(S.I0, (1280, 800))
    assert cost["points_without_ray"] == 5 and np.isfinite(cost["whole_image"])


def test_the_drop_depth_cost_is_measured():
    """Fit A against its p1 = p2 = 0 truncation at 1 m (amendment of 21 Oct, measured on the real Fit A file: centre
    0.58 %, 4.54 % within 350 px, 9.30 % within 450 px, 25.2 % within 560 px); lens_synth's C_TRUE is Fit A's stereo
    pose turned 0.9 deg about z, which changes these by under 0.01"""
    cost = m71.p_drop_depth_cost([S.I0, S.I1], S.C_TRUE, (1280, 800))
    assert cost["within_450px"] == pytest.approx(9.30, abs=0.3) and cost["centre_cell"] < 1.0
    swapped = [np.r_[S.I0[:8], 0.0, 0.0], np.r_[S.I1[:8], S.I0[8:]]]   # camera 0's p1, p2 on camera 1: 8.7 %
    assert m71.p_drop_depth_cost(swapped, S.C_TRUE, (1280, 800))["within_450px"] > 5.0     # both cameras truncated


def test_no_okvis_config_without_allow_4term_okvis(exported, tmp_path, capsys):
    P = S.pipeline()
    assert m71.main([str(P / "fit.json"), str(exported / "v" / "report.json"), str(tmp_path / "e"),
                     "--okvis-template", str(exported / "template.yaml"), "--rig", str(S.RIG12)]) == 0
    out = capsys.readouterr().out
    assert not (tmp_path / "e" / "phantom12_stereo.yaml").exists()
    assert (tmp_path / "e" / "phantom12-kb4t.json").exists() and (tmp_path / "e" / "camchain-kb4t-equi4.yaml").exists()
    assert "--allow-4term-okvis" in out and "depth" in out and "% within 450 px" in out
    assert "--okvis-kb4t" in out                                                    # the export for depth


def test_only_a_pass_for_this_calibration_is_exported(exported, tmp_path):
    rep = json.loads((exported / "v" / "report.json").read_text())
    for change, msg in (({"verdict": "FAIL"}, "only a PASS"), ({"calibration": {**rep["calibration"], "sha256": "0"}},
                                                              "another calibration")):
        bad = tmp_path / "r.json"
        bad.write_text(json.dumps({**rep, **change}))
        with pytest.raises(SystemExit) as e:
            m71.main([str(S.pipeline() / "fit.json"), str(bad), str(tmp_path / "e"), "--rig", str(S.RIG12)])
        assert msg in str(e.value)


def test_only_a_converged_fit_is_exported(exported, tmp_path):
    """an unconverged step-45 file has the same numbers as the report judged (same sha256), so only its verdict can
    stop it; a file with no verdict is not step 45's output"""
    fit = json.loads((S.pipeline() / "fit.json").read_text())
    for verdict in ("unconverged: 1 attempt of at most 3 did not reach scipy's convergence", None):
        bad = tmp_path / "fit.json"
        bad.write_text(json.dumps({k: v for k, v in fit.items() if k != "verdict"} | ({"verdict": verdict} if verdict
                                                                                        else {})))
        with pytest.raises(SystemExit) as e:
            m71.main([str(bad), str(exported / "v" / "report.json"), str(tmp_path / "e"), "--rig", str(S.RIG12)])
        assert "only a converged fit" in str(e.value) and repr(verdict) in str(e.value)
    assert not (tmp_path / "e").exists()
