"""calibtool.livecfg: a robot's live OKVIS config as a calibration the scorer can judge, and the RADTAN lens model
registered in lensmodel only inside radtan_scope().

Fast tests use synthetic configs; `replay` tests score phantom21's live radtan config on the stored TEST split and
compare with phase A's reference numbers (test_reproduce.P21_LIVE_RADTAN, made with the kit's shim)."""
import importlib.util
import json
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from calibtool import livecfg, pipeline, sessions  # noqa: E402
import lensmodel  # noqa: E402  (calibtool.paths put the scripts folder on sys.path)

ENTRY_TOL_PCT = 0.02


def _row(v):
    return "[" + ", ".join(repr(float(x)) for x in np.ravel(v)) + "]"


def okvis_yaml(dist_type="radialtangential", dist=(-0.25, 0.046, -2.8e-4, -3.1e-4), size=(1280, 800),
               f=((583.5, 582.9), (582.7, 582.1)), c=((643.37, 388.85), (637.23, 384.70)), resize=0.5,
               T1_t=(0.0116, -0.0017, 0.1494)):
    """a minimal OKVIS config in the robot files' layout (flow-mapping camera blocks, %YAML:1.0)"""
    T0 = np.eye(4)
    T0[:3, 3] = (-0.0316, -0.0026, 0.1496)
    a = np.deg2rad(1.0)
    T1 = np.eye(4)
    T1[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    T1[:3, 3] = T1_t
    cams = []
    for i, T in enumerate((T0, T1)):
        cams.append(f"""\
    - {{T_SC:
        {_row(T)},
        image_dimension: [{size[0]}, {size[1]}],
        distortion_coefficients: {_row(dist)},
        distortion_type: {dist_type},
        focal_length: {_row(f[i])},
        principal_point: {_row(c[i])},
        cam_model: pinhole,
        dma_queue: {2 - 2 * i}}} # OKVIS cam{i}
""")
    rf = "" if resize is None else f"    resize_factor: {resize}  # -> 640x400 for the whole VIO\n"
    return ("%YAML:1.0\n# a test config; focal_length: [1, 2] in a comment is not a key\ncameras:\n" + "\n".join(cams)
            + "\nimu_parameters:\n    use: true\n    T_BS:\n"
            "        [1.0, 0.0, 0.0, 0.0,\n         0.0, 1.0, 0.0, 0.0,\n"
            "         0.0, 0.0, 1.0, 0.0,\n         0.0, 0.0, 0.0, 1.0]\n\nfrontend_parameters:\n" + rf
            + "    detection_threshold: 42.0\n")


def _write(tmp_path, text, name="cfg.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


# ---------------------------------------------------------------- load: the lens model read, never inferred


@pytest.mark.parametrize("dtype,dist,model", [
    ("radialtangential", (-0.25, 0.046, -2.8e-4, -3.1e-4), "RADTAN"),
    ("equidistant", (0.01, -0.02, 0.003, -0.001), "KB4"),
    ("kb4t", (0.01, -0.02, 0.003, -0.001, 1e-4, -2e-4), "KB4T")])
def test_load_reads_distortion_type(tmp_path, dtype, dist, model):
    live = livecfg.load(_write(tmp_path, okvis_yaml(dtype, dist)))
    assert live.model == model and live.distortion_type == dtype
    assert live.size == (1280, 800)
    assert len(live.intr) == 2 and all(len(v) == 4 + len(dist) for v in live.intr)
    assert np.allclose(live.intr[0][:4], [583.5, 582.9, 643.37, 388.85])
    assert np.allclose(live.intr[1][4:], dist)


def test_radtan_with_four_numbers_is_not_kb4(tmp_path):
    """run_lens_kb4t.okvis_calibration reads 4 coefficients as KB4: a radtan config scored as a fisheye"""
    live = livecfg.load(_write(tmp_path, okvis_yaml("radialtangential")))
    assert live.model == "RADTAN"


def test_unknown_or_missing_distortion_type_is_refused(tmp_path):
    with pytest.raises(ValueError):
        livecfg.load(_write(tmp_path, okvis_yaml("omni")))
    t = okvis_yaml().replace("        distortion_type: radialtangential,\n", "")
    with pytest.raises(ValueError, match="distortion_type"):
        livecfg.load(_write(tmp_path, t, "nodist.yaml"))


def test_stereo_pose_is_cam1_in_cam0(tmp_path):
    live = livecfg.load(_write(tmp_path, okvis_yaml()))
    import okviscfg
    c = okviscfg.load(tmp_path / "cfg.yaml")
    assert np.allclose(live.C, np.linalg.inv(c.T_SC[0]) @ c.T_SC[1], atol=1e-15)
    assert np.isclose(np.linalg.norm(live.C[:3, 3]), np.linalg.norm(np.subtract((0.0116, -0.0017, 0.1494),
                                                                                 (-0.0316, -0.0026, 0.1496))))


def test_write_json_is_a_step66_calibration(tmp_path):
    live = livecfg.load(_write(tmp_path, okvis_yaml("kb4t", (0.01, -0.02, 0.003, -0.001, 1e-4, -2e-4))))
    p = live.write_json(tmp_path / "live.json")
    import stereofit as sf
    model, intr, C = sf.load_calibration(str(p))
    assert model == "KB4T" and np.allclose(intr[0], live.intr[0]) and np.allclose(C, live.C)
    assert json.loads(p.read_text())["source"] == str(tmp_path / "cfg.yaml")


def test_load_at_another_size_unscales(tmp_path):
    """a config written for 960x600 loads as the 1280x800 calibration it was scaled from (scaling.unscale_config)"""
    f, c = ((450.0, 449.5), (449.0, 448.0)), ((480.2, 300.4), (478.9, 299.1))
    live = livecfg.load(_write(tmp_path, okvis_yaml("equidistant", (0.01, 0, 0, 0), size=(960, 600), f=f, c=c,
                                                    resize=0.66666669)), at_size=(1280, 800))
    assert live.size == (1280, 800)
    assert np.allclose(live.intr[0][:4], [450.0 / 0.75, 449.5 / 0.75, (480.2 + 0.5) / 0.75 - 0.5,
                                          (300.4 + 0.5) / 0.75 - 0.5], atol=1e-9)
    assert livecfg.load(tmp_path / "cfg.yaml").size == (960, 600)


def test_load_at_another_size_needs_no_resize_factor(tmp_path):
    """a 640x400 config has no resize_factor (VIO runs at the image size; scaling.scale_config writes it so): it still
    loads at 1280x800, since scoring needs only the lens, never the VIO resize"""
    from calibtool import scaling
    native = okvis_yaml("equidistant", (0.01, 0, 0, 0), resize=None)
    small = scaling.scale_config(native, 0.5)
    assert "resize_factor" not in small and scaling.config_size(small) == (640, 400)
    live = livecfg.load(_write(tmp_path, small, "640.yaml"), at_size=(1280, 800))
    ref = livecfg.load(_write(tmp_path, native, "1280.yaml"))
    assert live.size == (1280, 800) and live.scaled_from == (640, 400)
    for a, b in zip(live.intr, ref.intr):
        assert np.allclose(a, b, rtol=0, atol=1e-9)


# ---------------------------------------------------------------- the RADTAN model


def _radtan_intr():
    return np.array([583.4986, 582.9690, 643.3657, 388.8541, -0.2489, 0.04583, -2.805e-4, -3.080e-4])


def _points(n=400, seed=1):
    rng = np.random.default_rng(seed)
    ang = np.deg2rad(rng.uniform(-45, 45, (n, 2)))       # within the lens's field (radtan rises to ~55 deg here)
    z = rng.uniform(0.3, 3.0, n)
    return np.c_[np.tan(ang[:, 0]) * z, np.tan(ang[:, 1]) * z, z]


def test_radtan_projection_equals_opencv():
    intr, P = _radtan_intr(), _points()
    K = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]])
    want, _ = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, intr[4:])
    got = livecfg.radtan_project(intr, P)
    assert np.abs(got - want.reshape(-1, 2)).max() < 1e-9


def test_radtan_unproject_inverts_project():
    intr, P = _radtan_intr(), _points()
    rays = livecfg.radtan_unproject(intr, livecfg.radtan_project(intr, P))
    assert np.allclose(np.linalg.norm(rays, axis=1), 1)
    want = P / np.linalg.norm(P, axis=1, keepdims=True)
    assert np.abs(rays - want).max() < 1e-10


def test_radtan_scope_registers_and_restores():
    before_models, before = dict(lensmodel.MODELS), (lensmodel.project, lensmodel.unproject)
    models_obj = lensmodel.MODELS
    assert "RADTAN" not in lensmodel.MODELS
    P = _points(20)
    kb = np.array([580.0, 580.0, 640.0, 400.0, 0.01, -0.02, 0.003, -0.001, 1e-4, -2e-4])
    kb_px = lensmodel.project("KB4T", kb, P)
    with livecfg.radtan_scope():
        assert lensmodel.MODELS["RADTAN"] == 4
        assert np.abs(lensmodel.project("RADTAN", _radtan_intr(), P)
                      - livecfg.radtan_project(_radtan_intr(), P)).max() == 0
        r = lensmodel.unproject("RADTAN", _radtan_intr(), lensmodel.project("RADTAN", _radtan_intr(), P))
        assert np.allclose(r, P / np.linalg.norm(P, axis=1, keepdims=True), atol=1e-10)
        assert np.array_equal(lensmodel.project("KB4T", kb, P), kb_px)          # other models untouched
    assert lensmodel.MODELS == before_models and lensmodel.MODELS is models_obj
    assert (lensmodel.project, lensmodel.unproject) == before
    with pytest.raises(ValueError):
        lensmodel.project("RADTAN", _radtan_intr(), P)


def test_radtan_scope_restores_on_exception():
    before_models, before = dict(lensmodel.MODELS), (lensmodel.project, lensmodel.unproject)
    with pytest.raises(RuntimeError, match="inside"):
        with livecfg.radtan_scope():
            raise RuntimeError("inside")
    assert lensmodel.MODELS == before_models
    assert (lensmodel.project, lensmodel.unproject) == before


def test_radtan_scope_nests():
    before = (lensmodel.project, lensmodel.unproject)
    with livecfg.radtan_scope():
        with livecfg.radtan_scope():
            pass
        assert "RADTAN" in lensmodel.MODELS                  # the outer scope is still active
    assert "RADTAN" not in lensmodel.MODELS and (lensmodel.project, lensmodel.unproject) == before


def test_radtan_scope_two_threads_closing_out_of_order():
    """thread A enters, thread B enters, A leaves first: RADTAN stays registered while B is inside, and lensmodel is
    exactly as before once B has left too (no scope may restore another one's saved state)"""
    import threading
    before_models, before = dict(lensmodel.MODELS), (lensmodel.project, lensmodel.unproject)
    a_in, b_in, a_out, b_check = (threading.Event() for _ in range(4))
    seen = {}

    def a():
        with livecfg.radtan_scope():
            a_in.set()
            b_in.wait(5)
        a_out.set()

    def b():
        a_in.wait(5)
        with livecfg.radtan_scope():
            b_in.set()
            a_out.wait(5)
            seen["models"] = "RADTAN" in lensmodel.MODELS
            seen["px"] = lensmodel.project("RADTAN", _radtan_intr(), _points(5))
        b_check.set()

    ts = [threading.Thread(target=f) for f in (a, b)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert b_check.is_set()
    assert seen["models"] and np.array_equal(seen["px"], livecfg.radtan_project(_radtan_intr(), _points(5)))
    assert lensmodel.MODELS == before_models and "RADTAN" not in lensmodel.MODELS
    assert (lensmodel.project, lensmodel.unproject) == before


def test_model_scope_only_for_radtan():
    with livecfg.model_scope("KB4T"):
        assert "RADTAN" not in lensmodel.MODELS
    with livecfg.model_scope("RADTAN"):
        assert "RADTAN" in lensmodel.MODELS
    assert "RADTAN" not in lensmodel.MODELS


# ---------------------------------------------------------------- replay: phantom21's live radtan config


def _reference():
    spec = importlib.util.spec_from_file_location("_t_reproduce", Path(__file__).with_name("test_reproduce.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.P21_LIVE_RADTAN


@pytest.mark.replay
@pytest.mark.usefixtures("recorded_trim")  # recorded numbers: at most 4 trim rounds
def test_p21_live_radtan_scored_in_scope_reproduces_reference(data_root, tmp_path):
    """the live radtan config read by livecfg.load, scored in this process inside radtan_scope (summary) and by step 66
    through livecfg's runner: phase A's reference numbers (made with the kit's shim, scipy: so scored with scipy
    here), per entry within 0.02 %"""
    ref = _reference()
    st = sessions.get("p21_1105")
    test_npz = data_root / st.recorded.files["test_npz"]         # = split16(p21_1105)'s TEST file, array for array
    live = livecfg.load(st.live_config)
    assert live.model == "RADTAN" and live.size == (1280, 800)
    rec = json.loads((data_root / st.recorded.files["live_json"]).read_text())
    assert np.allclose(np.array(live.intr), rec["intr"], atol=1e-9) and np.allclose(live.C, rec["C"], atol=1e-9)
    before = dict(lensmodel.MODELS)
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(livecfg.validate66, live, test_npz, tmp_path / "validate-live", st.rig_path, solver="scipy")
        sm = livecfg.summary(live, test_npz, st.rig_path, tmp_path, solver="scipy")
        v = fut.result()
    assert lensmodel.MODELS == before
    print("P21 LIVE via livecfg: " + json.dumps({"validate": v.as_dict(), "summary": sm.as_dict()}, default=str))
    off = []
    if v.verdict != ref["verdict"]:
        off.append(f"verdict {v.verdict} vs {ref['verdict']}")
    if set(v.entries) != set(ref["entries"]):
        off.append(f"entries {sorted(v.entries)} vs {sorted(ref['entries'])}")
    off += [f"entry {k}: {v.entries[k]:+.4f} vs {x:+.4f}" for k, x in ref["entries"].items()
            if k in v.entries and abs(v.entries[k] - x) > ENTRY_TOL_PCT]
    rs = ref["summary"]
    for name, g, w in (("median", sm.median_pct, rs["median_pct"]), ("iqr low", sm.iqr_pct[0], rs["iqr_pct"][0]),
                       ("iqr high", sm.iqr_pct[1], rs["iqr_pct"][1])):
        if abs(g - w) > ENTRY_TOL_PCT:
            off.append(f"summary {name} {g:+.4f} vs {w:+.4f}")
    if abs(sm.rows_px - rs["rows_px"]) > 1e-3 or abs(sm.baseline_mm - rs["baseline_mm"]) > 1e-3:
        off.append(f"rows {sm.rows_px:.4f} / baseline {sm.baseline_mm:.4f} vs {rs['rows_px']} / {rs['baseline_mm']}")
    if sm.n_pairs != rs["n_pairs"]:
        off.append(f"pairs {sm.n_pairs} vs {rs['n_pairs']}")
    for b in sm.bands:
        n, med = rs["bands"][(b["lo"], b["hi"])]
        if b["n"] != n or abs(b["median_pct"] - med) > ENTRY_TOL_PCT:
            off.append(f"band {b['lo']}-{b['hi']}: {b['median_pct']:+.4f} (n {b['n']}) vs {med:+.4f} (n {n})")
    assert not off, "\n".join(off)


def test_summary_and_validate_wrap_pipeline(tmp_path, monkeypatch):
    """livecfg.summary / validate66 hand pipeline a step-66 JSON of the live config; a RADTAN config's step 66 runs
    through livecfg itself (the runner that registers the model), others through step 66 directly"""
    calls = {}

    def fake_summary(calib_json, test_npz, rig, solver):
        calls["summary"] = (json.loads(Path(calib_json).read_text())["model"], "RADTAN" in lensmodel.MODELS)
        calls.setdefault("solver", []).append(solver)
        return "S"

    def fake_validate(calib_json, test_npz, out_dir, rig, fit_report=None, script=None, solver=None, jobs=1):
        calls.setdefault("validate", []).append((json.loads(Path(calib_json).read_text())["model"], script))
        calls.setdefault("solver", []).append(solver)
        return "V"

    monkeypatch.setattr(pipeline, "summary", fake_summary)
    monkeypatch.setattr(pipeline, "validate66", fake_validate)
    rt = livecfg.load(_write(tmp_path, okvis_yaml()))
    assert livecfg.summary(rt, "t.npz", "rig.yaml", tmp_path) == "S"
    assert calls["summary"] == ("RADTAN", True) and "RADTAN" not in lensmodel.MODELS
    assert livecfg.validate66(rt, "t.npz", tmp_path / "v", "rig.yaml") == "V"
    kb = livecfg.load(_write(tmp_path, okvis_yaml("equidistant", (0.01, 0, 0, 0)), "kb.yaml"))
    livecfg.validate66(kb, "t.npz", tmp_path / "w", "rig.yaml")
    (m1, s1), (m2, s2) = calls["validate"]
    assert m1 == "RADTAN" and Path(s1) == Path(livecfg.__file__)
    assert m2 == "KB4" and s2 is None
    assert calls["solver"] == ["fast"] * 3                  # calibtool's default solver, passed on
    livecfg.validate66(kb, "t.npz", tmp_path / "x", "rig.yaml", solver="scipy")
    livecfg.summary(rt, "t.npz", "rig.yaml", tmp_path, solver="scipy")
    assert calls["solver"][3:] == ["scipy", "scipy"]


def test_runner_registers_radtan_for_step66(tmp_path):
    """livecfg run as a script is step 66 with RADTAN registered: a radtan calibration gets past model checks (a
    missing test file then stops it, after the model was accepted)"""
    import subprocess
    import sys
    import os
    rt = livecfg.load(_write(tmp_path, okvis_yaml()))
    cal = rt.write_json(tmp_path / "rt.json")
    code = textwrap.dedent(f"""
        import sys, runpy
        sys.argv = [{livecfg.__file__!r}, "--check-model", {str(cal)!r}]
        runpy.run_path({livecfg.__file__!r}, run_name="__main__")
        """)
    env = dict(os.environ, PYTHONPATH=str(Path(lensmodel.__file__).parent))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "RADTAN ok" in p.stdout
