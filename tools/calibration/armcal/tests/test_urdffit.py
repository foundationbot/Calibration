"""urdffit.py (stage 2): known URDF errors and a known camera place in the head (position or rotation) are found back
from synthetic captures of both arms; the prediction agrees with B0's own offsets under the URDF the captures were
matched with; the camera term's convention; the candidate set; selection, cross-validation and the writer."""
import numpy as np
import pytest

import synth
from armcal import meshfit, urdffit
from armcal.urdf import Urdf
from rigmath import rot_axis, transform

BASE = synth.URDF
TRUE = {"NeckYaw": {"dxyz": [-8.0, 15.0, -13.0]}, "RightShoulderRoll": {"post_axis_deg": -1.7},
        "LeftShoulderRoll": {"post_axis_deg": 0.5}, "HEAD": {"dxyz": [-3.0, 11.0, -20.0]}}
# the camera turned in the head instead (phantom10 13 Nov: HEADR about -2.7, +2.1, -1.5 deg): 2-3 deg about two axes
TRUE_R = {**{k: v for k, v in TRUE.items() if k != "HEAD"}, "HEAD": {"drot": [2.5, -2.0, 0.0]}}
NECKS = [(-0.2, 0.17), (0.0, 0.38), (0.2, 0.45), (0.1, 0.25)]


def true_D(head):
    """the camera's true move in the head, built WITHOUT urdffit.head_transform: rotation about the head's axes
    (rigmath's Rodrigues of the unit axis and the angle), then the translation"""
    R, t = np.eye(3), np.zeros(3)
    if head and "drot" in head:
        w = np.radians(np.asarray(head["drot"], float))
        R = rot_axis(w / np.linalg.norm(w), float(np.linalg.norm(w)))
    if head and "dxyz" in head:
        t = 1e-3 * np.asarray(head["dxyz"], float)
    return transform(R, t)


def captures(side, change=TRUE, poses=5, noise_mm=0.0, seed=0):
    """Obs of `side` at `poses` arm poses x 4 neck angles, seen through the true (changed) robot"""
    rng = np.random.default_rng(seed)
    ch = dict(change)
    head = ch.pop("HEAD", None)
    true = BASE.changed(ch)
    D = true_D(head)
    q0, _, _, _ = synth.pose(side)
    out = []
    for k in range(poses):
        q = {n: v + rng.normal(0, 0.15) for n, v in q0.items()}
        for i, (yaw, pitch) in enumerate(NECKS):
            T_TC_true = true.chain("Torso", "Head").T({"NeckYaw": yaw, "NeckPitch": pitch}) @ D @ synth.T_BC
            w = true.chain("Torso", side + "WristRollLink").T(q)[:3, 3] + 1e-3 * rng.normal(0, noise_mm, 3)
            m = np.linalg.inv(T_TC_true) @ np.r_[w, 1.0]
            o = urdffit.Obs(f"{k:02d}-p{k:02d}-{i}", side, f"p{k:02d}", m, synth.T_BC, yaw, pitch, q, np.zeros(3))
            o.off = urdffit.offsets([o], BASE)[0]
            out.append(o)
    return out


def test_the_prediction_under_the_matching_urdf_is_b0s_offset():
    q, yaw, pitch, T_TC = synth.pose("Right")
    E = synth.rigid(3.0, (1, 1, 0), (5.0, -4.0, 2.0))
    arm = synth.arm("Right")
    rec = dict(name="07-p09-10-0", clean=True, T_TC=T_TC.tolist(), E=E.tolist(), wrist=arm.wrist_point_T(q).tolist(),
               T_BC=synth.T_BC.tolist(), yaw=yaw, pitch=pitch, q=q,
               off=(1e3 * meshfit.wrist_offset(arm, q, T_TC, E)).tolist())
    (o,), side = urdffit.observations([rec], "Right"), "Right"
    assert o.pose == "p09"
    assert np.allclose(urdffit.offsets([o], BASE)[0], rec["off"], atol=1e-9)
    assert urdffit.observations([dict(rec, clean=False)], side) == []


@pytest.mark.parametrize("model", ["M12+head", "M12+headR", "M12"])
def test_the_true_parameters_are_found(model):
    change = {"M12+head": TRUE, "M12+headR": TRUE_R}.get(model, {k: v for k, v in TRUE.items() if k != "HEAD"})
    R, L = captures("Right", change), captures("Left", change, seed=1)
    f = urdffit.fit(urdffit.MODELS[model], R, L, BASE, model)
    want = {"NY.x": -8.0, "NY.y": 15.0, "NY.z": -13.0, "Rroll0": -1.7, "Lroll0": 0.5,
            "HEAD.x": -3.0, "HEAD.y": 11.0, "HEAD.z": -20.0, "HEADR.x": 2.5, "HEADR.y": -2.0, "HEADR.z": 0.0}
    for n, v in zip(f.names, f.x):
        assert v == pytest.approx(want[n], abs=0.02), n           # mm or deg
    assert f.rms < 0.01 and np.all(f.resid_right < 0.05) and np.all(f.resid_left < 0.05)


# ---------------------------------------------------------------- the camera's rotation in the head (HEADR)


def test_head_transform_turns_about_the_head_axes_then_moves():
    """D = [R | t]: a head-frame point p goes to R p + t (R about the HEAD frame's axes, then t), and the camera's
    place to D @ T_BC; rotation vector in deg, translation in mm"""
    w, t = np.array([2.5, -2.0, 1.0]), np.array([4.0, -7.0, 12.0])
    D = urdffit.head_transform(t, w)
    R = rot_axis(w / np.linalg.norm(w), np.radians(np.linalg.norm(w)))
    assert np.allclose(D[:3, :3], R, atol=1e-15) and np.allclose(D[:3, 3], 1e-3 * t, atol=1e-18)
    assert np.array_equal(D[3], [0, 0, 0, 1])
    p = np.array([0.3, -0.1, 0.2, 1.0])
    assert np.allclose(D @ p, np.r_[R @ p[:3] + 1e-3 * t, 1.0], atol=1e-15)
    assert np.allclose(D, transform(np.eye(3), 1e-3 * t) @ transform(R, np.zeros(3)), atol=1e-15)   # turn, then move
    assert not np.allclose(D, transform(R, np.zeros(3)) @ transform(np.eye(3), 1e-3 * t), atol=1e-6)
    moved = D @ synth.T_BC                                      # the camera turned about the head's origin, not its own
    assert np.allclose(moved[:3, :3], R @ synth.T_BC[:3, :3]) and np.allclose(moved[:3, 3], R @ synth.T_BC[:3, 3]
                                                                               + 1e-3 * t)
    assert np.array_equal(urdffit.head_transform(), np.eye(4))
    assert np.array_equal(urdffit.head_transform(None, [0.0, 0.0, 0.0]), np.eye(4))
    assert np.array_equal(urdffit.head_transform([1.0, 2.0, 3.0])[:3, :3], np.eye(3))
    a = np.radians(3.0)                                       # right-handed: +3 deg about z takes x towards y
    assert np.allclose(urdffit.head_transform(None, [0, 0, 3.0])[:3, 0], [np.cos(a), np.sin(a), 0], atol=1e-15)


def test_off_change_applies_both_camera_terms():
    R = captures("Right", poses=2)
    ch = {"HEAD": {"dxyz": np.array([1.0, -2.0, 3.0]), "drot": np.array([0.5, 1.0, -0.5])}}
    want = urdffit.offsets(R, BASE, [1.0, -2.0, 3.0], [0.5, 1.0, -0.5])
    assert np.array_equal(urdffit.off_change(R, BASE, ch), want)
    assert not np.allclose(want, urdffit.offsets(R, BASE, [1.0, -2.0, 3.0]), atol=1.0)


def test_a_camera_rotation_in_the_head_is_recovered_with_noise():
    """2 mm per axis of noise on every capture (about the matcher's): HEADR comes back within 0.5 deg per axis and
    within three of its standard errors (measured, seeds 0/2/4: worst 0.33 deg, se 0.13-0.25 deg; noise-free: exact
    to 0.02 deg, test_the_true_parameters_are_found); the shoulder and neck terms stay close too"""
    R, L = captures("Right", TRUE_R, noise_mm=2.0), captures("Left", TRUE_R, noise_mm=2.0, seed=1)
    f = urdffit.fit(urdffit.MODELS["M12+headR"], R, L, BASE, "M12+headR")
    got = dict(zip(f.names, zip(f.x, f.se)))
    for i, a in enumerate("xyz"):
        v, se = got["HEADR." + a]
        err = abs(v - TRUE_R["HEAD"]["drot"][i])
        assert err < 0.5 and err < 3 * se, (a, v, se)
    assert abs(got["Rroll0"][0] - (-1.7)) < 0.5 and abs(got["Lroll0"][0] - 0.5) < 0.5
    assert np.linalg.norm([got["NY." + a][0] for a in "xyz"] - np.array(TRUE["NeckYaw"]["dxyz"])) < 5.0
    assert set(f.change["HEAD"]) == {"drot"}


def test_the_default_candidates_choose_the_rotation_when_the_camera_turned():
    R, L = captures("Right", TRUE_R, poses=4, noise_mm=1.0), captures("Left", TRUE_R, poses=4, noise_mm=1.0, seed=1)
    cv = {m: urdffit.cv_summary(urdffit.lopo(spec, R, L, BASE, m)) for m, spec in urdffit.MODELS.items()}
    assert urdffit.choose(cv) == "M12+headR"                  # measured: 1.3 mm pooled; M12+head 4.7, M12 5.5
    assert cv["M12+headR"]["pooled_median"] < 2.5 < 4.0 < min(v["pooled_median"] for m, v in cv.items()
                                                               if m != "M12+headR")


def test_the_candidate_set():
    assert list(urdffit.MODELS) == ["M12+headR", "M12+head", "M12", "M9", "X1"]
    assert urdffit.DEFAULT == "M12+headR" and urdffit.FALLBACK == "M12"
    assert "M12+headRT" not in urdffit.MODELS and len(urdffit.ALL_MODELS["M12+headRT"]) == 11
    assert [n for n, _ in urdffit.MODELS["M12+headR"]][-3:] == ["HEADR.x", "HEADR.y", "HEADR.z"]
    assert urdffit.choose({"M12+headR": {"pooled_median": 4.0}, "M12+head": {"pooled_median": 4.0}}) == "M12+headR"
    with pytest.raises(ValueError, match="unknown models"):
        urdffit.run(None, None, synth.URDF, models=("M13",))


def test_standard_errors_and_conditioning_are_reported():
    R, L = captures("Right", noise_mm=2.0), captures("Left", noise_mm=2.0, seed=1)
    f = urdffit.fit(urdffit.MODELS["M12+head"], R, L, BASE, "M12+head")
    assert np.all(f.se > 0) and np.isfinite(f.cond) and f.cond > 1
    assert np.allclose(np.diag(f.corr), 1.0)
    s = f.summary()
    assert set(s["params"]) == set(f.names) and "HEAD" in s["change"]


def test_a_parameter_whose_error_exceeds_it_is_fixed_and_the_rest_refitted(monkeypatch):
    calls = []

    def fake(spec, R, L, base, model=""):
        names = [n for n, _ in spec]
        calls.append(names)
        x = np.array([0.1 if n == "Lroll0" else 5.0 for n in names])
        return urdffit.Fit(model, names, x, np.ones(len(names)), 1.0, np.eye(len(names)), 1.0, 0, 0, {})

    monkeypatch.setattr(urdffit, "fit", fake)
    f, dropped = urdffit.fit_identified(urdffit.MODELS["M12"], [], [], BASE, "M12")
    assert dropped == ["Lroll0"] and "Lroll0" not in f.names and len(calls) == 2


def test_select_drops_a_capture_far_from_the_arms_median():
    R = captures("Right")
    R[3].off = R[3].off + np.array([40.0, 0, 0])
    keep, drop = urdffit.select(R)
    assert drop == [R[3].name] and len(keep) == len(R) - 1


def test_leave_one_arm_pose_out_holds_out_every_neck_angle_of_a_pose():
    R, L = captures("Right", noise_mm=1.0, poses=3), captures("Left", noise_mm=1.0, poses=3, seed=1)
    folds = urdffit.lopo(urdffit.MODELS["M12+head"], R, L, BASE, "M12+head")
    assert len(folds) == 6 and all(f["n"] == len(NECKS) for f in folds)
    s = urdffit.cv_summary(folds)
    assert s["Right"]["better"] == s["Right"]["n"] and s["pooled_median"] < 5.0


def test_the_written_urdf_carries_the_urdf_terms_and_states_the_head_term(tmp_path):
    R, L = captures("Right"), captures("Left", seed=1)
    f = urdffit.fit(urdffit.MODELS["M12+head"], R, L, BASE, "M12+head")
    out = urdffit.write_urdf(BASE, f, tmp_path / "p.urdf", ["session X"])
    text = out.read_text()
    assert "NOT in this file" in text and "session X" in text and "M12+head" in text
    W = Urdf.load(out)
    no_head = {k: v for k, v in f.change.items() if k != "HEAD"}
    assert np.allclose(W.joint("NeckYaw").origin, BASE.changed(no_head).joint("NeckYaw").origin, atol=1e-8)


def test_the_written_urdf_states_the_camera_rotation(tmp_path):
    R, L = captures("Right", TRUE_R, poses=3), captures("Left", TRUE_R, poses=3, seed=1)
    f = urdffit.fit(urdffit.MODELS["M12+headR"], R, L, BASE, "M12+headR")
    text = urdffit.write_urdf(BASE, f, tmp_path / "r.urdf").read_text()
    assert "camera-in-head rotation HEADR = [2.5, -2.0, " in text and "deg (rotation vector, head frame)" in text
    assert "NOT in this file" in text
    assert "camera-in-head translation" not in text and "T_BS' = D T_BS" in text
