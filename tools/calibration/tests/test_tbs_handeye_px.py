"""Step 81's pixel-level hand-eye fit: the projection, the fit, its error bars and its checks."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
import handeye_px as hp  # noqa: E402
import neckfk  # noqa: E402
import rigmath as rm  # noqa: E402

URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
POSES = [(y, p) for y in (-0.35, -0.175, 0.0, 0.175, 0.35) for p in (-0.3, -0.15, 0.0, 0.15, 0.3)]
# phantom12-like geometry (as test_tbs_handeye.py): camera 0 142 mm ahead of the Head origin looking 30 deg
# down, camera 1 44.8 mm to its right, the board 1.1 m along camera 0's axis at zero neck angles
X_TRUE = rm.transform(rm.rot_axis([0, 1, 0], np.radians(30)) @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.]]),
                      [0.142, 0.021, 0.014])
C = [np.eye(4), rm.transform(rm.rot_axis([0, 1, 0], np.radians(0.3)), [0.0448, 0.0, 0.0002])]
K = [np.array([[585.0, 0, 640.0], [0, 584.0, 400.0], [0, 0, 1.0]]), np.array([[587.0, 0, 636.0], [0, 586.5, 404.0], [0, 0, 1.0]])]
D_EQUI = [np.array([0.05, -0.02, 0.01, -0.003]), np.array([0.04, -0.015, 0.008, -0.002])]
BOARD = np.array([[0.12 * i - 0.3, 0.12 * j - 0.18, 0.0] for i in range(6) for j in range(4)])   # 24 corners, centred


def _images(X_true=X_TRUE, Z=None, jitter_deg=0.0, noise_px=0.0, seed=0, poses=POSES, take=0, cams=(0, 1)):
    """Every pose seen by both cameras, as if the head carried X_true; with jitter, each hold's yaw and
    pitch are off their encoders by a random amount both of its images share; pixel noise on top."""
    chain = neckfk.NeckChain(URDF)
    rng = np.random.default_rng(seed)
    Z = chain.T_TB(0, 0) @ X_true @ rm.transform(np.eye(3), [0, 0, 1.1]) if Z is None else Z
    out = []
    for h, (y, p) in enumerate(poses):
        dy, dp = np.radians(jitter_deg) * rng.standard_normal(2)
        for c in cams:
            T = rm.inv(chain.T_TB(y + dy, p + dp) @ X_true @ C[c]) @ Z
            uv = hp.project(BOARD @ T[:3, :3].T + T[:3, 3], K[c], D_EQUI[c], "equi")
            uv = uv + noise_px * rng.standard_normal(uv.shape)
            out.append(hp.Image(take, c, chain.T_TB(y, p), BOARD.copy(), uv, f"{take}-{h}"))
    return out, Z


def _err(X):
    return rm.rot_angle_deg(X_TRUE[:3, :3].T @ X[:3, :3]), 1e3 * np.linalg.norm(X[:3, 3] - X_TRUE[:3, 3])


def _start(deg=1.0, mm=5.0):
    return rm.transform(rm.rot_axis([1, 1, 0], np.radians(deg)), [mm / 1e3, -mm / 2e3, mm / 3e3]) @ X_TRUE


@pytest.mark.parametrize("kind", ["equi", "radtan"])
def test_the_projection_is_opencvs(kind):
    cv2 = pytest.importorskip("cv2")
    P = np.random.default_rng(1).uniform([-0.6, -0.4, 0.7], [0.6, 0.4, 1.5], (60, 3))
    if kind == "equi":
        D = D_EQUI[0]
        ref, _ = cv2.fisheye.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K[0], D.reshape(4, 1))
    else:
        D = np.array([-0.28, 0.08, 0.0005, -0.0003])
        ref, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K[0], D)
    assert np.abs(hp.project(P, K[0], D, kind) - ref.reshape(-1, 2)).max() < 1e-6
    with pytest.raises(ValueError, match="lens kind"):
        hp.project(P, K[0], D, "kb8")


def test_recovers_X_and_the_board_from_pixels():
    ims, Z = _images()
    r = hp.fit(ims, _start(), [rm.transform(rm.rot_axis([0, 0, 1], 0.02), [0.01, 0.0, 0.0]) @ Z], C, K, D_EQUI, "equi")
    da, dt = _err(r["X"])
    assert r["converged"] and da < 1e-5 and dt < 1e-4 and r["rms_px"] < 1e-4 and r["n_outliers"] == 0
    assert r["trim_rounds"] == 1 and r["trim_converged"]
    assert np.abs(r["Zs"][0] - Z).max() < 1e-6
    assert r["n_images"] == 50 and r["n_corners"] == 50 * 24 and r["n_groups"] == 25


def test_the_error_bars_cover_the_scatter_of_held_poses():
    """Joint error per hold (0.08 deg, shared by both of its images) plus 0.2 px: over seeds, X's errors
    scatter like its clustered sigma says (counted as independent corners they came out ~5x too small)."""
    errs, sig = [], []
    for seed in range(8):
        ims, Z = _images(jitter_deg=0.08, noise_px=0.2, seed=seed)
        r = hp.fit(ims, _start(), [Z], C, K, D_EQUI, "equi")
        errs.append(1e3 * np.linalg.norm(r["X"][:3, 3] - X_TRUE[:3, 3]))
        sig.append(np.linalg.norm(r["sigma_trans_head_mm"]))
    ratio = np.sqrt(np.mean(np.square(errs))) / np.mean(sig)
    assert 0.5 < ratio < 2.0, (errs, sig)


def test_a_mis_refined_corner_is_left_out():
    ims, Z = _images(noise_px=0.1)
    for k in (3, 17, 30):
        ims[k].uv[5] += [25.0, -12.0]
    r = hp.fit(ims, _start(), [Z], C, K, D_EQUI, "equi")
    da, dt = _err(r["X"])
    assert r["n_outliers"] == 3 and r["rms_px"] < 0.2 and da < 0.01 and dt < 0.1


def test_the_cut_is_at_outlier_px_drops_images_left_with_too_few_corners_and_reports_the_share():
    ims, Z = _images(noise_px=0.05)
    ims[4].uv[2] += [5.0, 0.0]                         # 5 px: cut at 3 px (would survive a 10 px cut)
    ims[9].uv[3:] += [20.0, 20.0]                      # 21 of 24 corners off: the image goes, all 24 counted
    r = hp.fit(ims, _start(), [Z], C, K, D_EQUI, "equi")
    assert r["n_outliers"] == 1 + 24 and r["n_images"] == 49
    assert abs(r["outlier_fraction"] - 25 / (50 * 24)) < 1e-12 and r["rms_all_px"] > 5 * r["rms_px"]
    assert r["trim_converged"] and 2 <= r["trim_rounds"] <= hp.TRIM_ROUNDS
    with pytest.raises(ValueError, match="every image lost its corners"):
        hp.fit(ims, _start(), [Z], C, K, D_EQUI, "equi", outlier_px=1e-9)


def test_a_direction_the_data_cannot_see_has_an_infinite_error_bar():
    ims, Z = _images(poses=[(y, 0.0) for y in (-0.35, -0.175, 0.0, 0.175, 0.35)])   # the pitch never moves
    r = hp.fit(ims, X_TRUE, [Z], C, K, D_EQUI, "equi")
    assert r["cond"] > hp.MAX_COND and np.all(np.isinf(r["sigma_trans_head_mm"]))


def test_two_takes_share_X_and_each_has_its_board():
    chain = neckfk.NeckChain(URDF)
    Z2 = chain.T_TB(0.1, -0.05) @ X_TRUE @ rm.transform(rm.rot_axis([1, 0, 0], 0.1), [0.05, 0.02, 1.0])
    a, Z1 = _images(take=0)
    b, _ = _images(Z=Z2, take=1, poses=POSES[::2])
    r = hp.fit(a + b, _start(), [Z1, rm.transform(np.eye(3), [0.01, 0, 0]) @ Z2], C, K, D_EQUI, "equi")
    da, dt = _err(r["X"])
    assert da < 1e-5 and dt < 1e-4 and np.abs(r["Zs"][1] - Z2).max() < 1e-6


def test_one_camera_alone_is_fitted_through_its_stereo_pose():
    ims, Z = _images(cams=(1,))
    r = hp.fit(ims, _start(), [Z], C, K, D_EQUI, "equi")
    da, dt = _err(r["X"])
    assert da < 1e-5 and dt < 1e-4


def test_bad_input_is_refused():
    ims, Z = _images(poses=POSES[:4])
    with pytest.raises(ValueError, match="starting board poses"):
        hp.fit(ims, X_TRUE, [Z, Z], C, K, D_EQUI, "equi")
    bad = hp.Image(0, 0, ims[0].A, ims[0].P[:3], ims[0].uv[:3], "x")
    with pytest.raises(ValueError, match="need >= 4"):
        hp.fit([bad] + ims, X_TRUE, [Z], C, K, D_EQUI, "equi")
    nan = hp.Image(0, 0, ims[0].A, ims[0].P, ims[0].uv * np.nan, "x")
    with pytest.raises(ValueError, match="NaN"):
        hp.fit([nan] + ims, X_TRUE, [Z], C, K, D_EQUI, "equi")
    with pytest.raises(ValueError, match="no images"):
        hp.fit([], X_TRUE, [Z], C, K, D_EQUI, "equi")
    behind = ims[0].A @ X_TRUE @ rm.transform(np.eye(3), [0, 0, -1.1])     # Z0 built from T_CW^-1 (17 Oct slip)
    with pytest.raises(ValueError, match="behind their cameras"):
        hp.fit(ims, X_TRUE, [behind], C, K, D_EQUI, "equi")


def test_a_corner_behind_the_camera_is_a_large_residual():
    ims, Z = _images(poses=POSES[:1], cams=(0,))
    assert np.allclose(hp.residuals(X_TRUE, [Z], ims, C, K, D_EQUI, "equi"), 0, atol=1e-6)   # the true board
    behind = ims[0].A @ X_TRUE @ rm.transform(np.eye(3), [0, 0, -1.1])     # the board 1.1 m behind camera 0
    assert np.all(hp.residuals(X_TRUE, [behind], ims, C, K, D_EQUI, "equi") == hp.BEHIND_PX)


import lensmodel  # noqa: E402

D_KB4T = [np.r_[D_EQUI[0], 0.00085, -0.00111], np.r_[D_EQUI[1], -0.00037, -0.00007]]   # Fit A's tangential terms


def test_kb4t_is_lensmodels_and_differs_from_equi():
    P = np.random.default_rng(3).uniform([-0.6, -0.4, 0.7], [0.6, 0.4, 1.5], (60, 3))
    k = K[0]
    ref = lensmodel.project("KB4T", [k[0, 0], k[1, 1], k[0, 2], k[1, 2], *D_KB4T[0]], P)
    assert np.abs(hp.project(P, K[0], D_KB4T[0], "kb4t") - ref).max() < 1e-12
    assert np.abs(hp.project(P, K[0], D_KB4T[0], "kb4t") - hp.project(P, K[0], D_EQUI[0], "equi")).max() > 0.1


def test_kb4t_refuses_a_skewed_K():
    """KB4T has no skew term (lensmodel.py, as rigcfg.Lens): a K with one is refused, not silently dropped."""
    P = np.random.default_rng(4).uniform([-0.6, -0.4, 0.7], [0.6, 0.4, 1.5], (10, 3))
    skewed = K[0].copy()
    skewed[0, 1] = 0.5
    with pytest.raises(ValueError, match="skew"):
        hp.project(P, skewed, D_KB4T[0], "kb4t")
    assert np.abs(hp.project(P, skewed, D_EQUI[0], "equi") - hp.project(P, K[0], D_EQUI[0], "equi")).max() > 0.01


def _images_kb4t(noise_px=0.0, seed=0):
    chain = neckfk.NeckChain(URDF)
    rng = np.random.default_rng(seed)
    Z = chain.T_TB(0, 0) @ X_TRUE @ rm.transform(np.eye(3), [0, 0, 1.1])
    out = []
    for h, (y, p) in enumerate(POSES):
        for c in (0, 1):
            T = rm.inv(chain.T_TB(y, p) @ X_TRUE @ C[c]) @ Z
            uv = hp.project(BOARD @ T[:3, :3].T + T[:3, 3], K[c], D_KB4T[c], "kb4t")
            out.append(hp.Image(0, c, chain.T_TB(y, p), BOARD.copy(), uv + noise_px * rng.standard_normal(uv.shape),
                                f"0-{h}"))
    return out, Z


def test_a_kb4t_head_is_recovered_with_kb4t_and_misfit_without_it():
    ims, Z = _images_kb4t()
    r = hp.fit(ims, _start(), [Z], C, K, D_KB4T, "kb4t")
    da, dt = _err(r["X"])
    assert r["converged"] and da < 1e-5 and dt < 1e-4 and r["rms_px"] < 1e-4
    wrong = hp.fit(ims, _start(), [Z], C, K, D_EQUI, "equi")       # the tangential terms dropped
    assert wrong["rms_px"] > 1e-3                                    # noise-free: any misfit shows (kb4t: < 1e-4)
