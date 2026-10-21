"""Step 30 only applies to checkerboards."""
import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("yaml")
pytest.importorskip("cv2")

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "scripts"))
label = importlib.import_module("30_label_canonical")


def test_step30_is_a_no_op_for_an_aprilgrid(monkeypatch, tmp_path, capsys):
    # tag ids anchor every corner, so there is no 180 deg ambiguity to relabel;
    # the camera-IMU fit takes step 10's imucam tree directly.
    monkeypatch.setattr(sys, "argv", ["30", str(tmp_path / "tree"), str(tmp_path / "cc.yaml"),
                                      str(tmp_path / "out"),
                                      "--rig", str(PKG / "config" / "rig-phantom12.yaml")])
    with pytest.raises(SystemExit) as e:
        label.main()
    assert e.value.code == 0
    assert "not needed" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
