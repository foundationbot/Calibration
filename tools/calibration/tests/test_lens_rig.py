"""The rig-file values the KB4T lens steps add: the nominal baseline and the lens gates."""
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "scripts"))
import rigcfg  # noqa: E402


def test_nominal_baselines_of_both_rigs():
    assert rigcfg.load(PKG / "config" / "rig-phantom12.yaml").nominal_baseline_m == pytest.approx(0.0445)
    assert rigcfg.load(PKG / "config" / "rig.yaml").nominal_baseline_m == pytest.approx(0.079)


def test_lens_gates_are_the_22_oct_values_in_both_rigs():
    """hold-gate plan (22 Oct) rule Z: far boards by distance (1.3 m), gross misfits by rms (1.80 px); the rest are the
    20 Oct values"""
    want = {"lens_hold_rms_max_px": 1.8, "lens_hold_max_z_m": 1.3, "lens_bad_hold_max_frac": 0.20,
            "lens_repeat_max_pct": 0.5, "lens_pass_pct": 0.5, "lens_pass_corner_pct": 1.0, "lens_min_holds": 8}
    for name in ("rig-phantom12.yaml", "rig.yaml"):
        assert rigcfg.load(PKG / "config" / name).lens_gates() == want


def _rig_copy(tmp_path, edit):
    cfg = tmp_path / "config"
    shutil.copytree(PKG / "config", cfg)
    p = cfg / "rig-phantom12.yaml"
    d = yaml.safe_load(p.read_text())
    edit(d)
    p.write_text(yaml.safe_dump(d))
    return p


def test_a_rig_file_without_the_new_keys_still_loads(tmp_path):
    def edit(d):
        del d["cameras"]["nominal_baseline_m"]
        for k in list(d["gates"]):
            if k.startswith("lens_"):
                del d["gates"][k]
    rig = rigcfg.load(_rig_copy(tmp_path, edit))
    assert rig.nominal_baseline_m is None
    assert rig.lens_gates()["lens_min_holds"] == 8 and isinstance(rig.lens_gates()["lens_min_holds"], int)


def test_a_rig_file_gate_overrides_the_default(tmp_path):
    rig = rigcfg.load(_rig_copy(tmp_path, lambda d: d["gates"].update(lens_hold_rms_max_px=0.5)))
    assert rig.lens_gates()["lens_hold_rms_max_px"] == 0.5
