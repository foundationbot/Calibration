"""urdffit.py (stage 2): known URDF errors are found back from synthetic captures of both arms; the prediction agrees
with B0's own offsets under the URDF the captures were matched with; selection, cross-validation and the writer."""
import numpy as np
import pytest

import synth
from armcal import meshfit, urdffit
from armcal.urdf import Urdf

BASE = synth.URDF
TRUE = {"NeckYaw": {"dxyz": [-8.0, 15.0, -13.0]}, "RightShoulderRoll": {"post_axis_deg": -1.7},
        "LeftShoulderRoll": {"post_axis_deg": 0.5}, "HEAD": {"dxyz": [-3.0, 11.0, -20.0]}}
NECKS = [(-0.2, 0.17), (0.0, 0.38), (0.2, 0.45), (0.1, 0.25)]


def captures(side, change=TRUE, poses=5, noise_mm=0.0, seed=0):
    """Obs of `side` at `poses` arm poses x 4 neck angles, seen through the true (changed) robot"""
    rng = np.random.default_rng(seed)
    ch = dict(change)
    head = ch.pop("HEAD", None)
    true = BASE.changed(ch)
    D = np.eye(4)
    if head is not None:
        D[:3, 3] = 1e-3 * np.asarray(head["dxyz"])
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


@pytest.mark.parametrize("model", ["M12+head", "M12"])
def test_the_true_parameters_are_found(model):
    change = TRUE if model == "M12+head" else {k: v for k, v in TRUE.items() if k != "HEAD"}
    R, L = captures("Right", change), captures("Left", change, seed=1)
    f = urdffit.fit(urdffit.MODELS[model], R, L, BASE, model)
    want = {"NY.x": -8.0, "NY.y": 15.0, "NY.z": -13.0, "Rroll0": -1.7, "Lroll0": 0.5,
            "HEAD.x": -3.0, "HEAD.y": 11.0, "HEAD.z": -20.0}
    for n, v in zip(f.names, f.x):
        assert v == pytest.approx(want[n], abs=0.02), n
    assert f.rms < 0.01 and np.all(f.resid_right < 0.05) and np.all(f.resid_left < 0.05)


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
