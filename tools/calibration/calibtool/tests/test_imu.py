"""calibtool.imu: camera-to-IMU rotation v1 from an AprilGrid flat on the floor during a neck sweep (design plan 9.3).

Fast tests on synthetic sweeps (neck = Rz(yaw) Ry(pitch), camera fixed in the head, the IMU reads R_SC_true @ up);
stored-data tests (CALIBTOOL_DATA, Moonstone) reproduce phantom10's 6 Nov sweep, its fitted rotation and its
`..._1280_imu.yaml` config."""
import re

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("yaml")

from calibtool import imu  # noqa: E402

R_HC = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # camera (x right, y down, z fwd) in head


def up_cam(pitch_deg, yaw_deg):
    """'up' in the camera frame for a neck pose: head = Rz(yaw) Ry(pitch), pitch > 0 looks down"""
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return R_HC.T @ (Rz @ Ry).T @ np.array([0.0, 0.0, 1.0])


R_CFG = R_HC.T @ imu.axis_rotation([1, 2, 3], 7.0)     # the config's cam0 rotation (any rotation works)


def synthetic(correction=np.eye(3), pitches=(9, 17, 25), yaws=(-30, -15, 0, 15, 30), noise_deg=0.0, seed=3):
    """a sweep whose true cam0 rotation is R_CFG @ correction; normals tilted by ~noise_deg"""
    rng = np.random.default_rng(seed)
    R_true = R_CFG @ correction
    neck, acc, nrm = [], [], []
    for k, p in enumerate(pitches):
        for y in (yaws if k % 2 == 0 else yaws[::-1]):
            n = up_cam(p, y)
            acc.append(9.81 * (R_true @ n))
            if noise_deg:
                n = imu.axis_rotation(rng.normal(size=3), noise_deg * abs(rng.normal())) @ n
            neck.append((p + rng.normal(0, 0.2), y + rng.normal(0, 0.5)))
            nrm.append(n)
    neck, acc, nrm = map(np.array, (neck, acc, nrm))
    return imu.sweep_from_arrays(neck, acc, np.full_like(acc, 0.1), nrm, R_CFG)


# ---------------------------------------------------------------- task 1: sweep, rotations, gap


def test_axis_rotation_and_rot_xz():
    R = imu.axis_rotation([0, 0, 2], 90.0)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])
    a = imu.axis_rotation([1, -2, 0.5], 13.0)
    assert np.allclose(a @ a.T, np.eye(3)) and np.isclose(np.linalg.det(a), 1)
    assert np.isclose(imu.rotation_angle_deg(np.eye(3), a), 13.0)
    rx, rz = np.radians(0.7), np.radians(-1.1)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    assert np.allclose(imu.rot_xz(0.7, -1.1), Rx @ Rz)


def test_gap_zero_when_config_is_true():
    s = synthetic()
    assert len(s) == 15 and s.n_recorded == 15
    assert imu.floor_gap_deg(s, R_CFG).max() < 1e-9


def test_gap_equals_error_angle():
    s = synthetic(imu.rot_xz(0.5, 0.0))
    gap = imu.floor_gap_deg(s, R_CFG)
    seen = 0.5 * np.linalg.norm(np.cross([1.0, 0, 0], s.normal), axis=1)
    assert np.allclose(gap, seen, atol=1e-4)
    assert np.allclose(imu.floor_gap_deg(s, R_CFG, idx=[0, 3]), gap[[0, 3]])


def _npz(tmp_path, s, **over):
    arr = dict(neck=s.neck, acc=s.acc, acc_std=s.acc_std, nb=s.normal, n=s.normal, R_SC=s.R_SC)
    arr.update(over)
    p = tmp_path / "sweep.npz"
    np.savez(p, **arr)
    return p


def test_load_sweep_reads_rscfit_format(tmp_path):
    s = synthetic()
    t = imu.load_sweep(_npz(tmp_path, s))
    assert np.array_equal(t.normal, s.normal) and np.array_equal(t.R_SC, R_CFG) and len(t) == 15


def test_load_sweep_filters_motion(tmp_path):
    s = synthetic()
    std = s.acc_std.copy()
    std[2, 1] = 0.5                                  # pose 3 was moving (rscfit: any axis std >= 0.4)
    nb = s.normal.copy()
    nb[5] = np.nan                                   # pose 6 had no board
    other = s.normal[::-1].copy()
    t = imu.load_sweep(_npz(tmp_path, s, acc_std=std, nb=nb, n=other))
    assert len(t) == 13 and t.n_recorded == 15
    assert np.array_equal(t.neck, np.delete(s.neck, [2, 5], axis=0))
    assert np.array_equal(imu.load_sweep(_npz(tmp_path, s, n=other), key="n").normal, other)


# ---------------------------------------------------------------- task 2: the two-angle fit


def test_fit_recovers_xz_correction():
    s = synthetic(imu.rot_xz(0.32, 0.11))
    f = imu.fit(s)
    assert np.allclose(f.angles_deg, [0.32, 0.11], atol=1e-5)
    assert f.gap_deg.max() < 1e-5 and len(f.idx) == 15
    assert np.allclose(f.correction, imu.rot_xz(*f.angles_deg))


def test_fit_sigma_shrinks_with_noise():
    lo = imu.fit(synthetic(imu.rot_xz(0.32, 0.11), noise_deg=0.02))
    hi = imu.fit(synthetic(imu.rot_xz(0.32, 0.11), noise_deg=0.2))
    assert np.all(lo.sigma_deg > 0) and np.all(lo.sigma_deg < 0.05)
    assert np.all(hi.sigma_deg > lo.sigma_deg)
    assert np.isclose(lo.combined_sigma_deg, np.hypot(*lo.sigma_deg))


def test_fit_soft_l1_bounds_one_bad_pose():
    """soft-L1's influence is bounded: a 12 deg bad pose moves the fit barely more than a 3 deg one (least squares
    would move it 4x as far)"""
    s = synthetic(imu.rot_xz(0.32, 0.11), noise_deg=0.02)
    clean = imu.fit(s).angles_deg
    good = s.normal[7].copy()
    shift = {}
    for deg in (3.0, 12.0):
        s.normal[7] = imu.axis_rotation([1, 0, 0], deg) @ good
        shift[deg] = np.abs(imu.fit(s).angles_deg - clean).max()
    assert shift[12.0] < 1.5 * shift[3.0] and shift[12.0] < 0.15


def test_fit_subset_and_minimum():
    s = synthetic(imu.rot_xz(0.32, 0.11))
    f = imu.fit(s, idx=[0, 4, 9, 14])
    assert np.allclose(f.angles_deg, [0.32, 0.11], atol=1e-5) and list(f.idx) == [0, 4, 9, 14]
    with pytest.raises(ValueError):
        imu.fit(s, idx=[0, 1])


def test_r_new_is_config_times_correction():
    s = synthetic(imu.rot_xz(-0.4, 0.2))
    f = imu.fit(s)
    assert np.allclose(imu.R_new(s, f), R_CFG @ f.correction)
    assert imu.rotation_angle_deg(imu.R_new(s, f), R_CFG @ imu.rot_xz(-0.4, 0.2)) < 1e-5


# ---------------------------------------------------------------- task 3: held-out scoring


def test_clusters_columns_and_rows():
    s = synthetic()
    cols, rows = imu.yaw_columns(s), imu.pitch_rows(s)
    assert sorted(set(cols)) == [0, 1, 2, 3, 4] and sorted(set(rows)) == [0, 1, 2]
    assert np.all(np.diff([s.neck[cols == c, 1].mean() for c in range(5)]) > 0)
    assert np.all(np.diff([s.neck[rows == r, 0].mean() for r in range(3)]) > 0)
    assert list(imu.clusters(np.array([5.0, 0.0, 0.4, 9.9]), 3.0)) == [1, 0, 0, 2]


def test_heldout_folds_partition_the_sweep():
    h = imu.heldout(synthetic(imu.rot_xz(0.5, -0.3), noise_deg=0.02))
    assert len(h.columns) == 5 and len(h.pitch) == 2 and h.n_poses == 15
    tested = np.concatenate([f.test_idx for f in h.columns])
    assert sorted(tested) == list(range(15))
    for f in h.columns + h.pitch:
        assert not set(f.fit_idx) & set(f.test_idx)
        assert len(f.gap_new_deg) == len(f.gap_live_deg) == len(f.test_idx)
    assert {len(f.test_idx) for f in h.pitch} == {5, 10}


def test_heldout_noise_free():
    h = imu.heldout(synthetic(imu.rot_xz(0.5, -0.3)))
    assert h.new_median_deg < 1e-4 and h.new_p90_deg < 1e-4
    assert h.live_median_deg > 0.2 and h.live_p90_deg >= h.live_median_deg


def test_heldout_live_rotation_argument():
    s = synthetic(imu.rot_xz(0.5, -0.3))
    h = imu.heldout(s, R_live=R_CFG @ imu.rot_xz(0.5, -0.3))
    assert h.live_median_deg < 1e-6


def test_heldout_single_column():
    h = imu.heldout(synthetic(imu.rot_xz(0.5, -0.3), yaws=(0,)))
    assert h.columns == [] and np.isnan(h.new_median_deg) and np.isnan(h.live_median_deg)
