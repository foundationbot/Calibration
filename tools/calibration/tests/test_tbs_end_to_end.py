"""Steps 75, 81 and 82 end to end on a synthetic neck take: a session.mcap with /joints and the
joint names, an EuRoC export with the head IMU and rendered fisheye images of the AprilGrid from
both cameras, and a head that carries a T_BS 2 deg off the config about Head y. Exercises the
command lines (readers, hold detection, detection, PnP, the fit) the way the robot's take will."""
import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("yaml")
mcap_writer = pytest.importorskip("mcap.writer")

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(HERE))
import neckfk  # noqa: E402
import rigmath as rm  # noqa: E402
from test_calib_pipeline_synthetic import D, K, _render, _texture  # noqa: E402

URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
T0 = 1_792_000_000_000_000_000
NAMES = ["LeftKnee", "NeckPitch", "NeckYaw"]
GRID = [(y, p) for p in (-0.25, 0.0, 0.25) for y in (-0.3, 0.0, 0.3)]
T_SC0 = rm.transform(rm.rot_axis([1, 0, 0], np.radians(-86.1)), [-0.067, 0.147, 0.016])
T_SC1 = T_SC0 @ rm.transform(np.eye(3), [0.0448, 0.0, 0.0])
T_BS_CFG = rm.transform(rm.rot_axis([1, 0, 0], np.radians(-33.8)) @ rm.rot_axis([0, 0, 1], np.radians(-90)),
                        [0.011, -0.046, 0.083])
T_BS_TRUE = rm.transform(rm.rot_axis([0, 1, 0], np.radians(2.0)), [0.0, 0.0, 0.0]) @ T_BS_CFG


def _timeline():
    """[(t_start, t_end, (yaw0, pitch0), (yaw1, pitch1))] segments: 2 s holds joined by 1 s moves,
    then a 4 s pitch sweep and a 4 s yaw sweep, ending still."""
    segs, t, q = [], 0.0, (0.0, 0.0)
    for g in GRID:
        segs.append((t, t + 1.0, q, g)); t += 1.0
        segs.append((t, t + 2.0, g, g)); t += 2.0
        q = g
    segs.append((t, t + 1.0, q, (0.0, 0.0))); t += 1.0
    return segs, t


def _q(tt):
    """Neck angles (yaw, pitch) and their rates at times tt [s]: cosine-eased moves, then sweeps."""
    segs, t_end = _timeline()
    y, p, yd, pd = (np.zeros_like(tt) for _ in range(4))
    for a, b, q0, q1 in segs:
        m = (tt >= a) & (tt < b)
        s = (1 - np.cos(np.pi * (tt[m] - a) / (b - a))) / 2
        sd = np.pi * np.sin(np.pi * (tt[m] - a) / (b - a)) / 2 / (b - a)
        y[m], p[m] = q0[0] + (q1[0] - q0[0]) * s, q0[1] + (q1[1] - q0[1]) * s
        yd[m], pd[m] = (q1[0] - q0[0]) * sd, (q1[1] - q0[1]) * sd
    m = (tt >= t_end) & (tt < t_end + 4)
    p[m], pd[m] = 0.3 * np.sin(np.pi * (tt[m] - t_end)), 0.3 * np.pi * np.cos(np.pi * (tt[m] - t_end))
    m = (tt >= t_end + 4) & (tt < t_end + 8)
    y[m], yd[m] = 0.3 * np.sin(np.pi * (tt[m] - t_end - 4)), 0.3 * np.pi * np.cos(np.pi * (tt[m] - t_end - 4))
    return y, p, yd, pd


@pytest.fixture(scope="module")
def take(tmp_path_factory):
    w = tmp_path_factory.mktemp("necktake")
    shutil.copytree(PKG / "config", w / "config")
    c = neckfk.NeckChain(URDF)
    _, t_end = _timeline()
    dur = t_end + 9.0
    # joints, 467 Hz, in a session.mcap the way the record app writes them
    tj = np.arange(0, dur, 1 / 467)
    yj, pj, _, _ = _q(tj)
    with open(w / "session.mcap", "wb") as f:
        wr = mcap_writer.Writer(f, compression=mcap_writer.CompressionType.NONE)
        wr.start()
        sid = wr.register_schema(name="okvis.JointState", encoding="jsonschema", data=b"{}")
        ch = wr.register_channel(topic="/joints", message_encoding="json", schema_id=sid)
        for t, y, p in zip(tj, yj, pj):
            ns = T0 + int(t * 1e9)
            msg = {"timestamp_ns": ns, "num_joints": 3, "positions": [0.0, float(p), float(y)]}
            wr.add_message(ch, log_time=ns + 30_000_000, publish_time=ns, data=json.dumps(msg).encode())
        wr.add_metadata("robot", {"joint_names": ",".join(NAMES)})
        wr.finish()
    # head IMU, 500 Hz: the gyro sees the true head through T_BS_TRUE
    ex = w / "export"
    (ex / "imu0").mkdir(parents=True)
    ti = np.arange(0.002, dur, 1 / 500)
    y, p, yd, pd = _q(ti)
    wS = c.omega_B(p, yd, pd) @ T_BS_TRUE[:3, :3] + [0.01, -0.012, 0.004]
    rng = np.random.default_rng(3)
    wS += 0.002 * rng.standard_normal(wS.shape)
    with open(ex / "imu0" / "data.csv", "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        for t, g in zip(ti, wS):
            f.write(f"{T0 + int(t * 1e9)},{g[0]},{g[1]},{g[2]},0.0,0.0,9.81\n")
    # two images per hold and camera, rendered from the true head pose
    tex, margin = _texture()
    T_CW0 = rm.transform(rm.rot_axis([1, 0, 0], np.pi), [-0.41, 0.29, 1.1])     # board centred, 1.1 m
    T_TW = c.T_TB(0, 0) @ T_BS_TRUE @ T_SC0 @ T_CW0
    segs, _ = _timeline()
    for cam, T_SC in (("cam1", T_SC0), ("cam2", T_SC1)):
        (ex / cam / "data").mkdir(parents=True)
        rows = []
        for a, b, q0, q1 in segs:
            if q0 != q1 or b - a < 2.0:
                continue
            for t in (a + 0.8, a + 1.2):
                T_CW = rm.inv(c.T_TB(*q0) @ T_BS_TRUE @ T_SC) @ T_TW
                ns = T0 + int(t * 1e9)
                cv2.imwrite(str(ex / cam / "data" / f"{ns}.png"), _render(tex, margin, T_CW[:3, :3], T_CW[:3, 3], ss=1))
                rows.append(f"{ns},{ns}.png")
        (ex / cam / "data.csv").write_text("#timestamp [ns],filename\n" + "\n".join(rows) + "\n")
    # an OKVIS config with the renderer's lens and the config's (2 deg off) T_BS
    flat = lambda T: ", ".join(repr(float(v)) for v in np.asarray(T).ravel())
    cam = lambda T: (f"    - {{T_SC: [{flat(T)}], image_dimension: [1280, 800], distortion_coefficients: "
                     f"[{flat(D)}], distortion_type: equidistant, focal_length: [{K[0, 0]}, {K[1, 1]}], "
                     f"principal_point: [{K[0, 2]}, {K[1, 2]}]}}\n")
    (w / "okvis.yaml").write_text("%YAML:1.0\ncameras:\n" + cam(T_SC0) + cam(T_SC1)
                                  + f"imu_parameters:\n    T_BS: [{flat(T_BS_CFG)}]\n")
    return w


def _main(module, argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [module] + [str(v) for v in argv])
    code = importlib.import_module(module).main()
    return code, capsys.readouterr().out


def _common(w):
    return ["--session", w / "session.mcap", "--export", w / "export", "--config", w / "okvis.yaml",
            "--urdf", URDF, "--rig", w / "config" / "rig-phantom12.yaml"]


def test_step81_measures_the_two_degrees(take, monkeypatch, capsys):
    code, out = _main("81_tbs_handeye", _common(take) + ["--min-verdict-holds", "6", "--out", take / "a.json"], monkeypatch, capsys)
    rep = json.loads((take / "a.json").read_text())
    assert code == 1 and rep["verdict"] == "REPLACE" and rep["n_holds"] == 9, out
    assert np.allclose(rep["vs_config"]["rot_head_deg"], [0, 2.0, 0], atol=0.05), out
    assert rm.rot_angle_deg(np.array(rep["T_BS_measured"])[:3, :3].T @ T_BS_TRUE[:3, :3]) < 0.05
    assert rep["model"] == "pixels" and rep["fit"]["rms_px"] < 0.5 and "VERDICT REPLACE" in out, rep["fit"]["rms_px"]
    assert rep["cameras_apart"]["dist_mm"] < 1.0 and not rep["inconclusive_because"], out
    # what was used: the export route counts its images in /joints gaps (none in this take), the config's KB4 lens
    # (4 distortion terms), corners within the default radius
    assert rep["n_images_in_joint_gaps"] == 0 and "(0 images in /joints gaps left out)" in out, out
    assert rep["lens"]["lens"] == "pinhole-equi" and rep["lens"]["n_dist"] == 4
    assert 0 < rep["max_corner_radius_used_px"] <= rep["max_radius_px"] == 450.0
    code, out = _main("81_tbs_handeye", _common(take) + ["--model", "poses", "--min-verdict-holds", "6", "--out",
                                                         take / "p.json"], monkeypatch, capsys)
    p = json.loads((take / "p.json").read_text())
    assert code == 1 and p["model"] == "poses" and p["verdict"] == "REPLACE" and p["spread_after"]["trans_rms_mm"] < 1.0, out
    # with the default 20 holds the 9 holds allow no cross-check between the cameras: X's sigma is infinite,
    # and the report is still strict JSON (null, never Infinity)
    code, out = _main("81_tbs_handeye", _common(take) + ["--out", take / "n.json"], monkeypatch, capsys)
    n = json.loads((take / "n.json").read_text(), parse_constant=lambda c: pytest.fail(f"{c} in the report"))
    assert code == 2 and n["verdict"] == "INCONCLUSIVE" and n["sigma_mm"] is None, out
    code, out = _main("81_tbs_handeye", _common(take) + ["--min-verdict-holds", "6", "--compare", take / "a.json", "--out", take / "b.json"],
                      monkeypatch, capsys)
    assert json.loads((take / "b.json").read_text())["vs_other_take"]["agree"], out
    # the runner's glue: step 75's verdict and drop list reach step 81, and step 83 will not write the
    # fit of a take step 75 did not pass (a qc.json whose file is missing reads MISSING)
    (take / "qc.json").write_text(json.dumps({"verdict": "RETAKE", "findings": [], "drop_holds": [0]}))
    code, out = _main("81_tbs_handeye", _common(take) + ["--min-verdict-holds", "6", "--drop-holds", "0", "--qc",
                                                         take / "qc.json", "--out", take / "c.json"], monkeypatch, capsys)
    c = json.loads((take / "c.json").read_text())
    assert c["qc_verdict"] == "RETAKE" and c["n_holds"] == 8 and c["take"] == take.name, out
    assert any("step 75 said RETAKE" in w for w in c["warnings"])
    code, out = _main("81_tbs_handeye", _common(take) + ["--min-verdict-holds", "6", "--qc", take / "none.json",
                                                         "--compare", take / "none.json", "--out", take / "d.json"],
                      monkeypatch, capsys)
    d = json.loads((take / "d.json").read_text())
    assert d["qc_verdict"] == "MISSING" and "vs_other_take" not in d and d["verdict"] == "REPLACE"
    s83 = importlib.import_module("83_tbs_apply")
    cfg_T_SC0 = np.array(c["T_SC0"])
    with pytest.raises(ValueError, match="not GOOD"):
        s83.mean_tbs([c, d], cfg_T_SC0)


def test_step82_sees_the_same_rotation(take, monkeypatch, capsys):
    code, out = _main("82_tbs_gyro_axes", _common(take) + ["--out", take / "g.json"], monkeypatch, capsys)
    rep = json.loads((take / "g.json").read_text())
    assert code == 1 and rep["verdict"] == "FAIL" and abs(rep["R_BS_angle_deg"] - 2.0) < 0.2, out
    assert abs(rep["axes"]["inter_axis_deg"] - 90.0) < 0.3 and rep["half_turn_explained_by"] is None


def test_step75_runs_and_reads_the_encoder_signs(take, monkeypatch, capsys):
    code, out = _main("75_qc_necktake", _common(take) + ["--qc-out", take / "qc75.json"], monkeypatch, capsys)
    q = json.loads((take / "qc75.json").read_text())
    assert code == 1 and q["verdict"] == "RETAKE" and isinstance(q["drop_holds"], list), out
    assert "holds: 9 " in out and "IMU is the head's" in out, out
    assert out.count("as the URDF") == 2, out
    assert "QC VERDICT: RETAKE" in out    # 9 holds and 2 images per hold are too few for a real take
