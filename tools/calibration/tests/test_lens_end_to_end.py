"""The KB4T steps chained on a synthetic recording: corners -> 45 -> 46 -> 66 -> 71 gives a PASS and files the next
consumers load (OKVIS via OpenCV, Kalibr's camchain via the package's checks). The OKVIS config is written only with
--allow-4term-okvis (Task 10's amendment, 21 Oct: its 4-term lens is unusable for stereo depth), so the chain passes
that flag and checks the file says so."""
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
TPL = importlib.import_module("test_calib_make_yaml")


def test_corners_to_export(tmp_path):
    P = S.pipeline()                                          # steps 45 and 46 on the fit sweep
    rig = ["--rig", str(S.RIG12)]
    assert importlib.import_module("66_validate_lens").main(
        [str(P / "fit.json"), str(P / "test.npz"), str(tmp_path / "v"), "--fit-report", str(P / "fit.json"),
         "--uncertainty", str(P / "unc.json")] + rig) == 0
    tpl = tmp_path / "template.yaml"
    tpl.write_text(TPL.TEMPLATE.replace("{tbs}", TPL.TBS))
    assert importlib.import_module("71_export_lens").main(
        [str(P / "fit.json"), str(tmp_path / "v" / "report.json"), str(tmp_path / "e"), "--okvis-template", str(tpl),
         "--allow-4term-okvis"] + rig) == 0
    fit = json.loads((P / "fit.json").read_text())
    rep = json.loads((tmp_path / "v" / "report.json").read_text())
    assert fit["verdict"].startswith("converged") and rep["verdict"] == "PASS"
    assert importlib.import_module("okviscfg").load(tmp_path / "e" / "phantom12_stereo.yaml").lens == "pinhole-equi"
    assert "NOT FOR STEREO DEPTH" in (tmp_path / "e" / "phantom12_stereo.yaml").read_text()
    cc = yaml.safe_load((tmp_path / "e" / "camchain-kb4t-equi4.yaml").read_text())
    S.rig12().camchain_lens(cc, "cam1")
    assert abs(fit["baseline_mm"] - 1e3 * np.linalg.norm(S.C_TRUE[:3, 3])) < 0.1
