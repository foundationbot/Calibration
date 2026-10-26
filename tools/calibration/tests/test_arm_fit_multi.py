"""Phase B of docs/superpowers/plans/2026-10-25-tagfree-selfcal.md (stereo-depth-eval branch): several plate sessions,
one plate placement X per session, one shared camera correction dT (B1) or one dT per head-pitch band (B2)
(armfit_multi.py), on synthetic sessions with known answers."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
import armchain  # noqa: E402
import armfit  # noqa: E402
import armfit_multi as am  # noqa: E402
import armplan  # noqa: E402
import neckfk  # noqa: E402
import rigcfg  # noqa: E402
import rigmath as rm  # noqa: E402

URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
GRID = rigcfg.AprilGrid(2, 2, 0.045, 0.2)
INTR = np.array([565.0, 564.0, 640.0, 400.0, -0.008, 0.012, -0.0005, -0.0014, 0.0008, -0.0011])
Q0 = np.radians([-40, -5, 45, -90, -135, 20, 20])
X_TRUE = rm.transform(rm.rot_axis((1, 0, 0), np.pi / 2), (0.03, 0.02, -0.2))
YAW = np.radians(-0.4)
PITCH = [np.radians([23.0]), np.radians([28.0]), np.radians([18.0, 23.0, 28.0]), np.radians([23.0, 28.0])]
# per session, as 1023c, 1023d, rv1024, rl1024: only sessions spanning several pitches tell B2 from B1, since a
# session held at one pitch lets its own X absorb that band's dT
DT = rm.transform(rm.exp_rot(np.radians([0.2, 0.05, -0.1])), (-0.0014, 0.0023, 0.0014))


def _rig():
    """A stereo rig in the Head placed 0.5 m in front of the plate at Q0 with the neck at (YAW, 23 deg)."""
    arm, neck = armchain.ArmChain(URDF), neckfk.NeckChain(URDF)
    P, _ = armplan.plate_points(GRID)
    T_TP = arm.T_base_tip(Q0) @ X_TRUE
    flip = np.diag([1.0, -1.0, -1.0])
    T_TC = T_TP @ rm.transform(flip, P.mean(0) - flip @ np.array([0, 0, 0.5]))
    T_HC0 = rm.inv(neck.T_TB(YAW, np.radians(23.0))) @ T_TC
    T_C0C1 = rm.transform(rm.rot_axis((0, 1, 0), 0.002), (0.0443, 0, 0))
    lens = [("KB4T", INTR), ("KB4T", INTR + np.r_[2.0, 2.0, -3.0, 1.0, np.zeros(6)])]
    return arm, neck, armfit.Rig(T_HC=[T_HC0, T_HC0 @ T_C0C1], lens=lens, size=(1280, 800)), P


def _X(s):
    """Session s's true plate placement: the plate moved a few mm and degrees between sessions."""
    return X_TRUE @ rm.transform(rm.exp_rot(np.radians([0.8 * s, -0.5 * s, 0.3 * s])), (0.002 * s, -0.001 * s, 0.0015 * s))


def _session(arm, neck, rig, P, X, pitches, dT_of_pitch, n_poses=8, frames=4, px=0.3, seed=0, spread=0.25):
    """Frames of n_poses holds per pitch in `pitches`, made with the TRUE X and dT_of_pitch(pitch), px noise."""
    rng = np.random.default_rng(seed)
    obs, h = [], 0
    for pitch in pitches:
        dT = dT_of_pitch(pitch)
        for _ in range(n_poses):
            q = Q0 + rng.uniform(-spread, spread, 7)
            for f in range(frames):
                qf = q + rng.normal(0, 1e-4, 7)
                for c in (0, 1):
                    T_TC = dT @ neck.T_TB(YAW, pitch) @ rig.T_HC[c]
                    Pc = (rm.inv(T_TC) @ arm.T_base_tip(qf) @ X @ np.c_[P, np.ones(len(P))].T).T[:, :3]
                    uv = armplan.lensmodel.project(rig.lens[c][0], rig.lens[c][1], Pc) + rng.normal(0, px, (len(P), 2))
                    obs.append(armfit.Frame(hold=h, cam=c, t=h * 10_000_000_000 + f * 70_000_000, q_arm=qf,
                                            q_neck=(YAW, pitch), P=P, uv=uv))
            h += 1
    return obs


def _sessions(arm, neck, rig, P, dT_of_pitch, px=0.3):
    return [_session(arm, neck, rig, P, _X(s), p, dT_of_pitch, px=px, seed=s) for s, p in enumerate(PITCH)]


def _dist(A, B):
    return np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1e3, rm.rot_angle_deg(A[:3, :3].T @ B[:3, :3])


def test_b1_recovers_the_shared_dT_and_each_sessions_plate():
    arm, neck, rig, P = _rig()
    S = _sessions(arm, neck, rig, P, lambda p: DT)
    fit = am.fit_all(S, arm, neck, rig, X_TRUE)
    mm, deg = _dist(fit["dT"][0], DT)
    assert mm < 0.2 and deg < 0.02
    for s, X in enumerate(fit["X"]):
        assert _dist(X, _X(s))[0] < 0.3
    assert fit["rms_px"] < 0.5


def test_b2_bands_recover_a_pitch_dependent_dT_that_b1_cannot_fit():
    arm, neck, rig, P = _rig()
    shift = {18: (0.0, 0.0, 0.0), 23: (0.002, 0.0, 0.0), 28: (0.004, 0.0, 0.0)}      # 2 mm per band
    true = {b: DT @ rm.transform(np.eye(3), t) for b, t in shift.items()}
    of = lambda p: true[am.band(p)]                                                    # noqa: E731
    S = _sessions(arm, neck, rig, P, of)
    b2 = am.fit_all(S, arm, neck, rig, X_TRUE, bands=True)
    for g, b in enumerate(b2["bands"]):
        assert _dist(b2["dT"][g], true[b])[0] < 0.3
    b1 = am.fit_all(S, arm, neck, rig, X_TRUE)
    assert b1["rms_px"] > b2["rms_px"] + 0.3


def test_a_fixed_dT_refits_only_the_plates():
    arm, neck, rig, P = _rig()
    S = _sessions(arm, neck, rig, P, lambda p: DT)
    fit = am.fit_all(S, arm, neck, rig, X_TRUE, fixed_dT=DT)
    assert np.allclose(fit["dT"][0], DT)
    assert all(_dist(X, _X(s))[0] < 0.3 for s, X in enumerate(fit["X"]))


def test_unseen_errors_are_zero_at_truth_and_read_a_known_shift_in_mm():
    arm, neck, rig, P = _rig()
    S = _sessions(arm, neck, rig, P, lambda p: DT, px=0.02)          # the plumbing, not one hold's noise floor
    truth = {"X": [_X(s) for s in range(len(S))], "dT": [DT], "bands": [None]}
    e0 = am.unseen_errors(S, arm, neck, rig, truth)
    assert max(max(v.values()) for v in e0) < 0.1
    off = dict(truth, dT=[DT @ rm.transform(np.eye(3), (0.0, 0.0, 0.002))])
    e2 = am.unseen_errors(S, arm, neck, rig, off)
    assert all(1.9 < v < 2.1 for d in e2 for v in d.values())


def test_band_is_the_nearest_pitch_band():
    assert [am.band(np.radians(d)) for d in (17.0, 20.4, 20.6, 25.4, 25.6, 30.0)] == [18, 18, 23, 23, 28, 28]
