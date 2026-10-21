"""Step 83: a measured T_BS goes into an OKVIS config and nothing else changes."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import okviscfg  # noqa: E402
import rigmath as rm  # noqa: E402

s83 = importlib.import_module("83_tbs_apply")

YAML = """%YAML:1.0
# phantom21's T_BS, unverified
cameras:
    - {T_SC: [1, 0, 0, 0.1,  0, 1, 0, 0.2,  0, 0, 1, 0.3,  0, 0, 0, 1],
       image_dimension: [1280, 800], distortion_coefficients: [0.1, 0.2, 0.3, 0.4],
       distortion_type: equidistant, focal_length: [500, 501], principal_point: [640, 400]}
imu_parameters:
    use: true
    # transform Body-Sensor (IMU)
    T_BS:
        [-0.00188426, 0.831282, 0.555847, 0.0110568,
         -0.999996, -0.00262363, 0.000533834, -0.0457738,
         0.0019021, -0.555844, 0.831284, 0.0826725,
         0.0, 0.0, 0.0, 1.0]

frontend_parameters:
    feature_type: superpoint
"""


I4 = np.eye(4)


def _report(T, take, qc="GOOD", verdict="REPLACE", T_SC0=I4):
    """A step-81 report whose camera pose in the Head is T @ T_SC0 (so its T_BS is T)."""
    return {"T_BS_measured": T.tolist(), "T_BC_measured": (T @ T_SC0).tolist(), "T_SC0": T_SC0.tolist(),
            "take": take, "qc_verdict": qc, "verdict": verdict, "vs_config": {"angle_deg": 1.0, "dist_mm": 5.0}}


def test_only_T_BS_and_one_note_change(tmp_path):
    T = rm.transform(rm.rot_axis([0.1, 1, 0.2], 1.2), [0.012, -0.044, 0.081])
    out = s83.apply(YAML, T, "measured\nsecond line")
    p = tmp_path / "out.yaml"
    p.write_text(out)
    r = okviscfg.load(p)
    assert np.allclose(r.T_BS, T, atol=1e-12) and np.allclose(r.T_SC[0][:3, 3], [0.1, 0.2, 0.3])
    old, new = YAML.splitlines(), out.splitlines()
    assert new[:3] == ["%YAML:1.0", "# measured", "# second line"]
    t_bs = [i for i, l in enumerate(old) if "T_BS:" in l][0]
    assert new[3:3 + t_bs] == old[1:1 + t_bs] and new[-3:] == old[-3:]
    with pytest.raises(ValueError, match="CRLF"):
        s83.apply(YAML.replace("\n", "\r\n"), T, "x")


def test_two_agreeing_takes_are_averaged_and_disagreeing_ones_refused():
    T = rm.transform(rm.rot_axis([0, 1, 0], 0.6), [0.01, -0.04, 0.08])
    T2 = rm.transform(rm.rot_axis([0, 1, 0], np.radians(0.04)), [0.0004, 0, 0]) @ T
    m = s83.mean_tbs([_report(T, "A"), _report(T2, "B")], I4)
    assert rm.rot_angle_deg(m[:3, :3].T @ T[:3, :3]) == pytest.approx(0.02, abs=1e-3)
    assert np.allclose(m[:3, 3], (T[:3, 3] + T2[:3, 3]) / 2, atol=1e-12)          # the translation is the mean too
    far = rm.transform(rm.rot_axis([0, 1, 0], np.radians(0.5)), [0, 0, 0]) @ T
    with pytest.raises(ValueError, match="disagree"):
        s83.mean_tbs([_report(T, "A"), _report(far, "B")], I4)


def test_every_pair_must_agree():
    """0.08 deg either side of the first take: each agrees with it, but they are 0.16 deg apart."""
    T = rm.transform(rm.rot_axis([0, 1, 0], 0.6), [0.01, -0.04, 0.08])
    lo, hi = (rm.transform(rm.rot_axis([0, 1, 0], np.radians(d)), [0, 0, 0]) @ T for d in (-0.08, 0.08))
    with pytest.raises(ValueError, match="takes 1 and 2 disagree"):
        s83.mean_tbs([_report(T, "A"), _report(lo, "B"), _report(hi, "C")], I4)


def test_the_two_take_rule_holds_unless_overridden():
    T = rm.transform(rm.rot_axis([0, 1, 0], 0.6), [0.01, -0.04, 0.08])
    with pytest.raises(ValueError, match="two-take rule"):
        s83.mean_tbs([_report(T, "A")], I4)
    with pytest.raises(ValueError, match="two-take rule"):
        s83.mean_tbs([_report(T, "A"), _report(T, "A")], I4)                     # the same take twice
    assert np.allclose(s83.mean_tbs([_report(T, "A")], I4, single_take=True), T, atol=1e-12)


def test_a_report_from_a_take_step_75_did_not_pass_or_without_a_verdict_is_refused():
    """Take A said REPLACE from a robot that turned 20 deg; a take whose step 75 crashed has no
    qc_verdict at all: neither is a measurement."""
    T = rm.transform(rm.rot_axis([0, 1, 0], 0.6), [0.01, -0.04, 0.08])
    with pytest.raises(ValueError, match="not GOOD"):
        s83.mean_tbs([_report(T, "A"), _report(T, "B", qc="STOP")], I4)
    bare = _report(T, "B")
    del bare["qc_verdict"]
    with pytest.raises(ValueError, match="no qc_verdict"):
        s83.mean_tbs([_report(T, "A"), bare], I4)
    with pytest.raises(ValueError, match="not REPLACE"):
        s83.mean_tbs([_report(T, "A"), _report(T, "B", verdict="INCONCLUSIVE")], I4)
    with pytest.raises(ValueError, match="not REPLACE or IMPROVE"):
        s83.mean_tbs([_report(T, "A"), _report(T, "B", verdict="KEEP")], I4)


def test_improve_is_written_and_the_note_says_what_a_reader_needs():
    """Two takes saying IMPROVE: the note gives each take's sigma and 2 sigma against the gate, the
    cameras' disagreement, the distance between the takes, and that the value is not good to the gate."""
    T = rm.transform(rm.rot_axis([0, 0, 1], 0.01), [0.001, 0, 0])
    T2 = rm.transform(rm.rot_axis([0, 0, 1], 0.0101), [0.0012, 0, 0])
    ca = {"angle_deg": 0.16, "dist_mm": 5.17}
    stat = {"sigma_stat_deg": 0.04, "sigma_stat_mm": 1.5}                    # the note prints X's sigma, not this
    reps = [{**_report(T, "P1", verdict="IMPROVE"), "model": "pixels", "sigma_deg": 0.06, "sigma_mm": 2.0, "cameras_apart": ca, **stat},
            {**_report(T2, "P2", verdict="IMPROVE"), "model": "pixels", "sigma_deg": 0.06, "sigma_mm": 2.0, "cameras_apart": ca, **stat}]
    assert np.abs(s83.mean_tbs(reps, I4)[:3, 3] - [0.0011, 0, 0]).max() < 1e-9
    note = s83.note_for(["P1.json", "P2.json"], T, I4, I4, "1, 0", False, reps)
    assert "P1.json (pixels model): IMPROVE, 1 sigma 0.060 deg / 2.00 mm (2 sigma 0.120 deg / 4.00 mm" in note
    assert "the cameras' own fits 0.160 deg / 5.17 mm apart" in note and "P1.json and P2.json are" in note
    assert "NOT shown good to the gate" in note and note.splitlines()[-1].startswith("the T_BS it replaced")
    plain = s83.note_for(["a", "b"], T, I4, I4, "1, 0", False, [_report(T, "A"), _report(T, "B")])
    assert "NOT shown good" not in plain and "a (poses model): REPLACE" in plain


def test_reports_from_different_models_are_refused():
    T = rm.transform(np.eye(3), [0.001, 0, 0])
    with pytest.raises(ValueError, match="different models"):
        s83.mean_tbs([{**_report(T, "A"), "model": "pixels"}, {**_report(T, "B"), "model": "poses"}], I4)


def test_a_fit_against_another_camera_pose_is_refused_and_T_BS_uses_the_targets():
    T = rm.transform(rm.rot_axis([0, 1, 0], 0.6), [0.01, -0.04, 0.08])
    T_SC0 = rm.transform(rm.rot_axis([1, 0, 0], -1.5), [-0.067, 0.147, 0.016])
    r = [_report(T, "A", T_SC0=T_SC0), _report(T, "B", T_SC0=T_SC0)]
    assert np.allclose(s83.mean_tbs(r, T_SC0), T, atol=1e-12)
    with pytest.raises(ValueError, match="different camera pose"):
        s83.mean_tbs(r, rm.transform(np.eye(3), [0.001, 0, 0]) @ T_SC0)


def test_the_note_measures_against_the_replaced_T_BS_and_keeps_its_text():
    old = s83.old_tbs_text(YAML)
    assert old.startswith("-0.00188426, 0.831282,") and old.endswith("0.0, 0.0, 0.0, 1.0")
    T_old = rm.transform(rm.rot_axis([0, 1, 0], 0.6), [0.01, -0.04, 0.08])
    T_new = rm.transform(rm.rot_axis([0, 1, 0], np.radians(0.5)), [0, 0, 0]) @ T_old
    note = s83.note_for(["a.json", "b.json"], T_new, T_old, I4, old, False)
    assert "moved 0.500 deg" in note and f"[{old}]" in note and "phantom21's or unverified" in note
