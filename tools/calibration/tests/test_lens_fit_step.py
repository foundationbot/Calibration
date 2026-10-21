"""Step 45 end to end on synthetic corner files: the quality gate, the rejection rule, the camera-sides check and
the refusal of an unconverged fit."""
import functools
import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
yaml = pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = importlib.import_module("lens_synth")
m45 = importlib.import_module("45_fit_lens")


def _npz(tmp_path, d, name="fit.npz"):
    p = tmp_path / name
    np.savez_compressed(p, **d)
    return p


def test_noisy_holds_are_dropped_listed_and_the_fit_repeated(tmp_path):
    """2 of 18 holds at 0.6 px per axis (~0.85 px rms: the far-board kind) among 0.3 px ones (~0.42 px)"""
    d = S.views([S.I0, S.I1], S.C_TRUE, 18, seed=24, noisy_holds=(5, 12))
    out = tmp_path / "fit.json"
    assert m45.main([str(_npz(tmp_path, d)), str(out), "--rig", str(S.RIG12)]) == 0
    r = json.loads(out.read_text())
    assert [h for h, _ in r["dropped_holds"]] == [5, 12]
    assert all(v > 0.6 for _, v in r["dropped_holds"])
    assert r["fit_holds"] == 16 and not {5, 12} & set(r["kept_holds"])
    assert r["success"] and r["verdict"].startswith("converged") and r["model"] == "KB4T"
    assert r["inputs"]["sha256"] and r["git_sha"] and len(r["intr"][0]) == 10


def test_a_recording_with_too_many_bad_holds_is_rejected(tmp_path):
    d = S.views([S.I0, S.I1], S.C_TRUE, 18, seed=24, noisy_holds=(1, 5, 9, 12))    # 22 % > 20 %
    out = tmp_path / "fit.json"
    assert m45.main([str(_npz(tmp_path, d)), str(out), "--rig", str(S.RIG12)]) == 2
    assert json.loads(out.read_text())["verdict"].startswith("rejected")


def test_swapped_sides_in_the_rig_file_are_refused_after_the_fit_finds_the_true_side(tmp_path):
    """the rig file says kalibr cam0 is the RIGHT eye, so the nominal start puts camera 1 at -x; the data put it at
    +x. The fit must still land on +x (it does from the wrong-side start: 20 Oct scratch run) and step 45 exit 3"""
    cfg = tmp_path / "config"
    shutil.copytree(S.PKG / "config", cfg)
    p = cfg / "rig-phantom12.yaml"
    rig = yaml.safe_load(p.read_text())
    rig["cameras"]["kalibr"][0]["side"], rig["cameras"]["kalibr"][1]["side"] = "right", "left"
    p.write_text(yaml.safe_dump(rig))
    out = tmp_path / "fit.json"
    assert m45.main([str(_npz(tmp_path, S.fit_sweep())), str(out), "--rig", str(p)]) == 3
    r = json.loads(out.read_text())
    assert r["verdict"].startswith("camera sides")
    assert r["C"][0][3] == pytest.approx(S.C_TRUE[0, 3], abs=1e-4)      # converged to the true side


def test_a_rig_without_a_nominal_baseline_is_refused(tmp_path):
    cfg = tmp_path / "config"
    shutil.copytree(S.PKG / "config", cfg)
    p = cfg / "rig-phantom12.yaml"
    rig = yaml.safe_load(p.read_text())
    del rig["cameras"]["nominal_baseline_m"]
    p.write_text(yaml.safe_dump(rig))
    with pytest.raises(SystemExit) as e:
        m45.main([str(_npz(tmp_path, S.fit_sweep())), str(tmp_path / "fit.json"), "--rig", str(p)])
    assert "nominal_baseline_m" in str(e.value)


def test_a_fit_stopped_at_its_cap_is_unconverged(tmp_path, monkeypatch):
    """converged or nothing: with the cap at 3 evaluations and one attempt scipy cannot report success, and step 45
    must say the fit is unconverged (exit 2, success false) instead of judging its numbers. fit_from_nominal looks
    fit_converged up in stereofit's namespace, so patching it there reaches step 45"""
    monkeypatch.setattr(m45.sf, "fit_converged", functools.partial(m45.sf.fit_converged, max_nfev=3, attempts=1))
    out = tmp_path / "fit.json"
    assert m45.main([str(_npz(tmp_path, S.fit_sweep())), str(out), "--rig", str(S.RIG12)]) == 2
    r = json.loads(out.read_text())
    assert r["verdict"].startswith("unconverged") and r["success"] is False
    assert len(r["attempts"]) == 1 and r["attempts"][0]["success"] is False
    assert "1 attempt of at most 1" in r["verdict"] and "cap 3 evaluations per solve round" in r["verdict"]


def test_pairs_whose_pnp_start_failed_are_counted(tmp_path, monkeypatch):
    """a pair whose PnP start failed is left out of the fit; how many is in fit.json, not silent"""
    real = m45.sf.pnp_poses

    def failing(data, *a, **k):
        poses, ok = real(data, *a, **k)
        return poses, ok & ~np.isin(data.src_pairs, [0, 1])
    monkeypatch.setattr(m45.sf, "pnp_poses", failing)
    out = tmp_path / "fit.json"
    assert m45.main([str(_npz(tmp_path, S.fit_sweep())), str(out), "--rig", str(S.RIG12)]) == 0
    r = json.loads(out.read_text())
    assert r["pnp_failed_pairs"] == 2 and r["fit_pairs"] == 46


def _strict(text):
    """json.loads that refuses Infinity and NaN (not JSON; most readers outside Python reject them)"""
    def refuse(c):
        raise ValueError(f"non-finite number {c} in the JSON")
    return json.loads(text, parse_constant=refuse)


def test_a_hold_with_no_corner_within_the_gross_limit_is_written_as_null(tmp_path, monkeypatch):
    """stereofit.hold_rms gives inf to a hold none of whose corners fits within GROSS_PX; fit.json stays strict JSON
    (null there). 10 of 48 holds so: rejected (21 %), with no warm refit"""
    real = m45.sf.hold_rms
    monkeypatch.setattr(m45.sf, "hold_rms", lambda *a: {h: (np.inf if h < 10 else v) for h, v in real(*a).items()})
    out = tmp_path / "fit.json"
    assert m45.main([str(_npz(tmp_path, S.fit_sweep())), str(out), "--rig", str(S.RIG12)]) == 2
    r = _strict(out.read_text())
    assert r["verdict"].startswith("rejected") and r["dropped_holds"][0] == [0, None]
