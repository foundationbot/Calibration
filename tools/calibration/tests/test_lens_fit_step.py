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
    """2 of 18 holds at 1.4 px per axis (~2 px rms: a gross misfit, over the 1.80 px gate) among 0.3 px ones (~0.42 px)"""
    d = S.views([S.I0, S.I1], S.C_TRUE, 18, seed=24, noisy_holds=(5, 12), noisy=1.4)
    out = tmp_path / "fit.json"
    assert m45.main([str(_npz(tmp_path, d)), str(out), "--rig", str(S.RIG12)]) == 0
    r = json.loads(out.read_text())
    assert [h for h, _ in r["dropped_holds"]] == [5, 12]
    assert all(v > 1.8 for _, v in r["dropped_holds"]) and r["dropped_far_holds"] == []
    assert r["fit_holds"] == 16 and not {5, 12} & set(r["kept_holds"])
    assert r["success"] and r["verdict"].startswith("converged") and r["model"] == "KB4T"
    assert r["inputs"]["sha256"] and r["git_sha"] and len(r["intr"][0]) == 10


def test_a_recording_with_too_many_bad_holds_is_rejected(tmp_path):
    d = S.views([S.I0, S.I1], S.C_TRUE, 18, seed=24, noisy_holds=(1, 5, 9, 12), noisy=1.4)    # 22 % > 20 %
    out = tmp_path / "fit.json"
    assert m45.main([str(_npz(tmp_path, d)), str(out), "--rig", str(S.RIG12)]) == 2
    assert json.loads(out.read_text())["verdict"].startswith("rejected")


def _joined(a, b):
    """two step-15 npz dicts as one recording: b's images and holds numbered after a's"""
    n, h = len(a["img_hold"]), int(a["img_hold"].max()) + 1
    out = {k: np.r_[a[k], b[k] + (h if k == "img_hold" else n if k == "det_img" else 0)] for k in a}
    return out


def test_far_boards_are_dropped_by_distance_however_well_they_fit(tmp_path):
    """hold-gate plan rule Z: a hold whose board sits beyond lens_hold_max_z_m (1.3 m) is dropped even at 0.3 px noise
    (synthetic far boards fit well; the real far boards' fault is not their rms), listed with its depth, and counts
    toward the rejection fraction like a misfit"""
    near = S.views([S.I0, S.I1], S.C_TRUE, 18, seed=24)
    far = S.views([S.I0, S.I1], S.C_TRUE, 2, seed=25, z_range=(1.45, 1.6), targets=[(643, 385)])
    out = tmp_path / "fit.json"
    assert m45.main([str(_npz(tmp_path, _joined(near, far))), str(out), "--rig", str(S.RIG12)]) == 0
    r = json.loads(out.read_text())
    assert [h for h, _ in r["dropped_far_holds"]] == [18, 19] and all(z > 1.3 for _, z in r["dropped_far_holds"])
    assert r["dropped_holds"] == [] and r["fit_holds"] == 18 and not {18, 19} & set(r["kept_holds"])
    assert len(r["hold_z_m"]) == 20 and all(z < 1.3 for h, z in r["hold_z_m"] if h < 18)


def test_misfits_and_far_boards_together_reject_the_recording(tmp_path):
    """2 misfits + 2 far boards of 18 holds = 22 % > 20 %: rejected, although each kind alone is under the limit"""
    near = S.views([S.I0, S.I1], S.C_TRUE, 16, seed=24, noisy_holds=(5, 12), noisy=1.4)
    far = S.views([S.I0, S.I1], S.C_TRUE, 2, seed=25, z_range=(1.45, 1.6), targets=[(643, 385)])
    out = tmp_path / "fit.json"
    assert m45.main([str(_npz(tmp_path, _joined(near, far))), str(out), "--rig", str(S.RIG12)]) == 2
    r = json.loads(out.read_text())
    assert r["verdict"].startswith("rejected") and "1.3 m" in r["verdict"]


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


def _saved_fit(tmp_path, npz):
    """a converged step-45 fit of npz, as --start reads it"""
    out = tmp_path / "first.json"
    assert m45.main([str(npz), str(out), "--rig", str(S.RIG12)]) == 0
    return out


def test_start_from_a_saved_fit_skips_the_fit_from_nominal(tmp_path, monkeypatch):
    """--start: a warm fit from a saved fit's lenses and C on the SAME corners, then the usual gate and checks (hold-gate
    plan D5); the fit from nominal is not run, and fit.json names the start and its sha256"""
    npz = _npz(tmp_path, S.fit_sweep())
    first = _saved_fit(tmp_path, npz)

    def no_nominal(*a, **k):
        raise AssertionError("--start must not fit from nominal")
    monkeypatch.setattr(m45.sf, "fit_from_nominal", no_nominal)
    out = tmp_path / "fit.json"
    assert m45.main([str(npz), str(out), "--rig", str(S.RIG12), "--start", str(first)]) == 0
    r, f = json.loads(out.read_text()), json.loads(first.read_text())
    assert r["verdict"].startswith("converged") and r["success"]
    assert r["start"]["path"] == str(first) and r["start"]["sha256"] == m45.sf.file_sha256(first)
    assert np.allclose(r["C"], f["C"], atol=1e-6) and np.allclose(r["intr"], f["intr"], atol=1e-4)


def test_start_from_a_fit_of_other_corners_is_refused(tmp_path):
    """a saved fit of another corner file is not a start for this one: refused before any fitting, nothing written"""
    first = _saved_fit(tmp_path, _npz(tmp_path, S.fit_sweep()))
    other = _npz(tmp_path, S.views([S.I0, S.I1], S.C_TRUE, 18, seed=99), "other.npz")
    out = tmp_path / "fit.json"
    with pytest.raises(SystemExit) as e:
        m45.main([str(other), str(out), "--rig", str(S.RIG12), "--start", str(first)])
    assert "sha256" in str(e.value) and not out.exists()


def test_start_from_an_unconverged_fit_is_refused(tmp_path):
    npz = _npz(tmp_path, S.fit_sweep())
    first = _saved_fit(tmp_path, npz)
    f = json.loads(first.read_text())
    f["success"] = False
    first.write_text(json.dumps(f))
    with pytest.raises(SystemExit) as e:
        m45.main([str(npz), str(tmp_path / "fit.json"), "--rig", str(S.RIG12), "--start", str(first)])
    assert "converged" in str(e.value)
