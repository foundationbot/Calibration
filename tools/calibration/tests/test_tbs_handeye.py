"""Step 81's core: the robot-world hand-eye fit recovers the camera in the Head link."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
import handeye  # noqa: E402
import neckfk  # noqa: E402
import rigmath as rm  # noqa: E402

URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
POSES = [(y, p) for y in (-0.35, -0.175, 0.0, 0.175, 0.35) for p in (-0.3, -0.15, 0.0, 0.15, 0.3)]
# phantom12-like geometry: camera 0 is 142 mm ahead of the Head origin and looks 30 deg down;
# camera 1 sits 44.8 mm to its right; the board is 1.1 m along camera 0's axis at zero neck angles
X_TRUE = rm.transform(rm.rot_axis([0, 1, 0], np.radians(30)) @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.]]),
                      [0.142, 0.021, 0.014])
C1 = rm.transform(rm.rot_axis([0, 1, 0], np.radians(0.3)), [0.0448, 0.0, 0.0002])


def _obs(X_true=X_TRUE, chain=None, noise_rad=0.0, noise_m=0.0, seed=0, cams=(0, 1), poses=POSES):
    chain = chain or neckfk.NeckChain(URDF)
    rng = np.random.default_rng(seed)
    Z = chain.T_TB(0, 0) @ X_true @ rm.transform(np.eye(3), [0, 0, 1.1])
    obs = []
    for y, p in poses:
        A = chain.T_TB(y, p)
        for c in cams:
            C = np.eye(4) if c == 0 else C1
            B = rm.inv(Z) @ A @ X_true @ C                         # camera c in the board frame
            B = B @ rm.transform(rm.exp_rot(noise_rad * rng.standard_normal(3)), noise_m * rng.standard_normal(3))
            obs.append((neckfk.NeckChain(URDF).T_TB(y, p), C, B))   # the fit always uses the URDF
    return obs


def _err(X):
    return rm.rot_angle_deg(X_TRUE[:3, :3].T @ X[:3, :3]), 1e3 * np.linalg.norm(X[:3, 3] - X_TRUE[:3, 3])


@pytest.mark.parametrize("start_deg, axis", [(1.0, [0, 1, 0]), (20.0, [0, 1, 0]), (10.0, [1, 1, 1])])
def test_recovers_X_from_a_wrong_start(start_deg, axis):
    """The config may be 20 deg off (phantom21's T_BS on phantom12): the fit must still land."""
    X0 = rm.transform(rm.rot_axis(axis, np.radians(start_deg)), [0.01, -0.01, 0.005]) @ X_TRUE
    r = handeye.fit(_obs(), X0)
    da, dt = _err(r["X"])
    assert r["converged"] and da < 1e-5 and dt < 1e-4 and r["rms_trans_mm"] < 1e-4


def test_the_answer_is_X_not_its_inverse():
    """X_TRUE is far from symmetric (142 mm, 30 deg): any mix-up of T_BC with T_CB fails here."""
    r = handeye.fit(_obs(), X_TRUE @ rm.transform(rm.rot_axis([1, 0, 0], 0.05), [0.003, 0, 0]))
    assert np.allclose(r["X"], X_TRUE, atol=1e-7)
    chain = neckfk.NeckChain(URDF)                   # and Z, which step 81's play() uses, is the board
    assert np.allclose(r["Z"], chain.T_TB(0, 0) @ X_TRUE @ rm.transform(np.eye(3), [0, 0, 1.1]), atol=1e-7)
    assert r["converged"]


def test_noise_gives_honest_error_bars():
    errs, sig = [], []
    for seed in range(4):
        r = handeye.fit(_obs(noise_rad=np.radians(0.03), noise_m=0.0003, seed=seed), X_TRUE)
        errs.append(_err(r["X"]))
        sig.append((np.linalg.norm(r["sigma_rot_head_deg"]), np.linalg.norm(r["sigma_trans_head_mm"])))
    errs, sig = np.array(errs), np.array(sig)
    assert errs[:, 0].max() < 0.05 and errs[:, 1].max() < 1.0      # well inside the 0.3 deg / 3 mm gate
    assert np.all(errs < 4 * sig + 1e-9)                              # the error bars are not optimistic


def _head_err(X):
    return np.r_[np.degrees(rm.log_rot(X[:3, :3] @ X_TRUE[:3, :3].T)), 1e3 * (X[:3, 3] - X_TRUE[:3, 3])]


def _sigma(r):
    return np.r_[r["sigma_rot_head_deg"], r["sigma_trans_head_mm"]]


@pytest.mark.parametrize("w", [1.0, 10.0])
def test_sigma_matches_the_scatter_per_head_axis(w, monkeypatch):
    """Independent noise, rotation matched to translation through W_ROT_M: over 24 takes the error
    along each Head axis matches the reported 1-sigma (one left in camera axes, taken from Z's
    block, 2x off, or divided by W fails)."""
    monkeypatch.setattr(handeye, "W_ROT_M", w)
    E, S = [], []
    for seed in range(24):
        r = handeye.fit(_obs(noise_rad=3e-4 / w, noise_m=3e-4, seed=seed), X_TRUE)
        E.append(_head_err(r["X"]))
        S.append(_sigma(r))
    ratio = np.sqrt(np.mean(np.square(E), 0)) / np.mean(S, 0)
    assert np.all((ratio > 0.6) & (ratio < 1.6)), ratio


def test_sigma_is_in_head_axes_whatever_the_camera_axes():
    """Re-expressing the same data in rotated camera axes must not change X's sigma in Head axes."""
    Q = rm.transform(rm.rot_axis([1, 2, 3], 1.2), np.zeros(3))
    obs = _obs(noise_rad=np.radians(0.03), noise_m=3e-4)
    a = handeye.fit(obs, X_TRUE)
    b = handeye.fit([(A, rm.inv(Q) @ C @ Q, B @ Q) for A, C, B in obs], X_TRUE @ Q)
    assert np.allclose(b["X"], a["X"] @ Q, atol=1e-9)
    assert np.allclose(a["sigma_rot_head_deg"], b["sigma_rot_head_deg"], rtol=1e-4)
    assert np.allclose(a["sigma_trans_head_mm"], b["sigma_trans_head_mm"], rtol=1e-4)


def _held(seed, joint_deg, frames=2):
    """Each pose held: `frames` images per camera with 0.1 mrad / 0.1 mm of their own noise, and
    the neck joint_deg (1-sigma, per hold) off its encoder reading, an error the hold's images share."""
    chain = neckfk.NeckChain(URDF)
    rng = np.random.default_rng(seed)
    Z = chain.T_TB(0, 0) @ X_TRUE @ rm.transform(np.eye(3), [0, 0, 1.1])
    obs, groups = [], []
    for h, (y, p) in enumerate(POSES):
        ey, ep = np.radians(joint_deg) * rng.standard_normal(2)
        A_true, A = chain.T_TB(y + ey, p + ep), chain.T_TB(y, p)
        for C in (np.eye(4), C1):
            for _ in range(frames):
                B = rm.inv(Z) @ A_true @ X_TRUE @ C
                obs.append((A, C, B @ rm.transform(rm.exp_rot(1e-4 * rng.standard_normal(3)), 1e-4 * rng.standard_normal(3))))
                groups.append(h)
    return obs, groups


def test_sigma_clustered_by_hold_is_honest_when_a_hold_shares_its_error():
    """A 0.05 deg joint error per hold, shared by its 4 images: clustered by hold, the sigma matches
    the scatter over 16 takes on every Head axis (0.7-1.1 measured). Counted as 100 independent
    images it does not: 3.5x too small about Head x, 3-5x too large along Head x, y, z."""
    E, S = [], []
    for seed in range(16):
        obs, groups = _held(seed, 0.05)
        r = handeye.fit(obs, X_TRUE, groups=groups)
        E.append(_head_err(r["X"]))
        S.append(_sigma(r))
    ratio = np.sqrt(np.mean(np.square(E), 0)) / np.mean(S, 0)
    assert np.all((ratio > 0.5) & (ratio < 2.0)), ratio


def test_a_joint_that_never_moved_leaves_X_unmeasured_and_says_so():
    """Pitch held at 0 all take: X's rotation about, and position along, the yaw axis are unseen.
    The fit must say so (condition number, infinite sigma) rather than report the start, 25 mm
    off along Head z, with a sub-millimetre error bar."""
    X0 = rm.transform(np.eye(3), [0, 0, 0.025]) @ X_TRUE
    r = handeye.fit(_obs(poses=[(y, 0.0) for y in (-0.35, -0.175, 0.0, 0.175, 0.35)]), X0)
    assert r["cond"] > handeye.MAX_COND
    assert np.isinf(r["sigma_rot_head_deg"]).all() and np.isinf(r["sigma_trans_head_mm"]).all()
    assert handeye.fit(_obs(), X_TRUE)["cond"] < 1e3                   # the 5 x 5 grid sees all of it


def test_bad_input_is_refused_by_name():
    obs = _obs()
    with pytest.raises(ValueError, match="iters"):
        handeye.fit(obs, X_TRUE, iters=0)
    A, C, B = obs[7]
    B = B.copy()
    B[0, 3] = np.nan
    with pytest.raises(ValueError, match=r"NaN or inf: \[7\]"):
        handeye.fit(obs[:7] + [(A, C, B)] + obs[8:], X_TRUE)
    with pytest.raises(ValueError, match="groups"):
        handeye.fit(obs, X_TRUE, groups=[0] * 3)


def test_a_wrong_urdf_leaves_a_residual():
    """The fit can absorb a T_BS error completely, but not a neck that differs from its URDF."""
    true = neckfk.NeckChain(URDF)
    true.pitch = neckfk.Joint("NeckPitch", true.pitch.parent, true.pitch.child, true.pitch.origin,
                              rm.rot_axis([1, 0, 0], np.radians(0.5)) @ np.array([0, 1, 0.]))
    r = handeye.fit(_obs(chain=true), X_TRUE)
    assert r["rms_rot_deg"] > 0.01 or r["rms_trans_mm"] > 0.1


def test_with_few_holds_the_clustered_sigma_is_not_optimistic():
    """8 holds sharing their joint error (the real takes kept 9-16 after drops): with the small-sample
    inflation G / (G - 7) the scatter is ~0.5 of the sigma; without it ~1.4 (the Task 6 re-review saw
    1.3-1.6 at 9-16 holds with PnP-shaped noise)."""
    E, S = [], []
    for seed in range(24):
        obs, groups = _held(seed, 0.05)
        keep = [i for i, g in enumerate(groups) if g < 8]
        r = handeye.fit([obs[i] for i in keep], X_TRUE, groups=[groups[i] for i in keep])
        E.append(_head_err(r["X"]))
        S.append(_sigma(r))
    ratio = np.sqrt(np.mean(np.square(E), 0)) / np.mean(S, 0)
    assert ratio.max() < 1.0, ratio
