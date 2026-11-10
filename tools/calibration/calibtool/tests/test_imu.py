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


# ---------------------------------------------------------------- task 4: the third angle


def test_observability_weak_axis_is_up():
    s = synthetic(noise_deg=0.02)
    ob = imu.observability(s, R_CFG)
    up = s.normal.mean(axis=0)
    up /= np.linalg.norm(up)
    assert np.allclose(ob.up_cam, up)
    assert abs(ob.weak_axis @ up) > 0.99
    assert ob.ratio < 0.3 and ob.singular[0] >= ob.singular[1] >= ob.singular[2]
    assert ob.weak_sigma_deg > 0


def test_single_pitch_row_is_unobservable():
    ob = imu.observability(synthetic(pitches=(17,)), R_CFG)
    assert ob.ratio < 1e-6


def test_gap_change_about_gravity_is_small():
    ob = imu.observability(synthetic(noise_deg=0.02), R_CFG)
    assert set(ob.gap_change_deg) == {"gravity", "camera x", "camera z"}
    assert ob.gap_change_deg["gravity"] < 0.3 * ob.gap_change_deg["camera x"]
    assert ob.gap_change_deg["camera x"] > 0.5


# ---------------------------------------------------------------- task 5: T_SC read and apply


def _block(T, bottom="0, 0, 0, 1"):
    rows = [", ".join(repr(float(v)) for v in T[i]) for i in range(3)] + [bottom]
    return "[" + ",\n          ".join(rows) + "]"


T0_SYN = np.eye(4)
T0_SYN[:3, :3] = R_CFG
T0_SYN[:3, 3] = (-0.0536, 0.1374, 0.0255)
T01_SYN = np.eye(4)
T01_SYN[:3, :3] = imu.axis_rotation([0.2, 1, 0.1], 0.8)
T01_SYN[:3, 3] = (0.044, 0.0003, -0.0002)
T1_SYN = T0_SYN @ T01_SYN


def config_text(blocks=(T0_SYN, T1_SYN)):
    """an OKVIS config in the robot files' layout: flow-mapping camera blocks, a T_BS block, comments naming T_SC"""
    cams = "".join(f"""\
    - {{T_SC:
        {_block(T)},
        image_dimension: [1280, 800],
        distortion_coefficients: [-0.01, 0.002, 0.0003, -0.0001, 1e-05, -2e-05],
        distortion_type: kb4t,
        focal_length: [583.5, 582.9],
        principal_point: [643.37, 388.85]}}
""" for T in blocks)
    return f"""%YAML:1.0
# a note naming T_SC: [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 1] in a comment
cameras:
{cams}
camera_parameters:
    timestamp_tolerance: 0.005
imu_parameters:
    # T_SC in a comment again
    T_BS:
        [3.26794897e-07, 0.84339145, 0.537299603, 0.00954301,
         -1.0, 2.75616021e-07, 1.75586768e-07, -0.0389529,
         0.0, -0.537299603, 0.84339145, 0.08296029,
         0.0, 0.0, 0.0, 1.0]
"""


def _bodies(text):
    return [m for m in re.finditer(r"^(?![ \t]*#)[^\n]*?\bT_SC:\s*\[([^\]]*)\]", text, re.M)]


NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def test_t_sc_reads_two_blocks_skipping_comments():
    T0, T1 = imu.t_sc(config_text())
    assert np.allclose(T0, T0_SYN, atol=0) and np.allclose(T1, T1_SYN, atol=1e-15)
    assert np.array_equal(imu.cam0_rotation(config_text()), T0[:3, :3])


def test_apply_rotates_cam0_keeps_stereo_pose():
    C = imu.rot_xz(0.32, 0.11)
    N0, N1 = imu.t_sc(imu.apply(config_text(), C))
    T0, T1 = imu.t_sc(config_text())
    assert np.allclose(N0[:3, :3], T0[:3, :3] @ C, atol=1e-15)
    assert np.array_equal(N0[:3, 3], T0[:3, 3])
    assert np.allclose(np.linalg.inv(N0) @ N1, np.linalg.inv(T0) @ T1, atol=1e-12)


def test_apply_diff_is_only_those_numbers():
    old = config_text()
    new = imu.apply(old, imu.rot_xz(0.32, 0.11))
    mo, mn = _bodies(old), _bodies(new)
    assert len(mo) == len(mn) == 2
    outside = lambda t, ms: [t[:ms[0].start(1)], t[ms[0].end(1):ms[1].start(1)], t[ms[1].end(1):]]  # noqa: E731
    assert outside(old, mo) == outside(new, mn)
    for a, b, kept in zip(mo, mn, ((3, 7, 11, 12, 13, 14, 15), (12, 13, 14, 15))):
        assert NUM.split(a.group(1)) == NUM.split(b.group(1))           # separators identical
        ta, tb = NUM.findall(a.group(1)), NUM.findall(b.group(1))
        assert [ta[i] for i in kept] == [tb[i] for i in kept]          # untouched numbers keep their text
        assert all(ta[i] != tb[i] for i in range(16) if i not in kept)


def test_apply_identity_keeps_cam0_text():
    old = config_text()
    new = imu.apply(old, np.eye(3))
    assert _bodies(old)[0].group(0) == _bodies(new)[0].group(0)
    assert np.allclose(imu.t_sc(new)[1], T1_SYN, atol=1e-15)


def test_apply_refuses_bad_input():
    with pytest.raises(ValueError):
        imu.apply(config_text(blocks=(T0_SYN,)), np.eye(3))
    with pytest.raises(ValueError):
        imu.apply(config_text(blocks=(T0_SYN, T1_SYN, T1_SYN)), np.eye(3))
    with pytest.raises(ValueError):
        imu.apply(config_text(), np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError):
        imu.apply(config_text(), np.eye(3) * 1.001)


def test_apply_note_after_yaml_header():
    new = imu.apply(config_text(), imu.rot_xz(0.1, 0.1), note="cam0 T_SC rotation re-measured\nsecond line")
    lines = new.splitlines()
    assert lines[:3] == ["%YAML:1.0", "# cam0 T_SC rotation re-measured", "# second line"]
    assert "\n".join(lines[3:]) + "\n" == imu.apply(config_text(), imu.rot_xz(0.1, 0.1)).split("\n", 1)[1]


def test_apply_output_reads_in_okviscfg(tmp_path):
    pytest.importorskip("cv2")
    from calibtool import paths  # noqa: F401  (scripts folder on sys.path)
    import okviscfg
    C = imu.rot_xz(0.32, 0.11)
    p = tmp_path / "new.yaml"
    p.write_text(imu.apply(config_text(), C))
    rig = okviscfg.load(p)
    assert np.allclose(rig.T_SC[0][:3, :3], R_CFG @ C, atol=1e-9)
    assert np.allclose(np.linalg.inv(rig.T_SC[0]) @ rig.T_SC[1], T01_SYN, atol=1e-9)


# ---------------------------------------------------------------- task 6: verdict


def test_verdict_deploys_a_real_error():
    s = synthetic(imu.rot_xz(0.5, 0.3), noise_deg=0.02)
    v = imu.verdict(s, R_live=R_CFG)
    assert v.deploy, v.reasons
    assert v.heldout.live_median_deg - v.heldout.new_median_deg > v.fit.combined_sigma_deg
    new = imu.cam0_rotation(imu.apply(config_text(), v.correction))
    assert imu.rotation_angle_deg(new, R_CFG @ imu.rot_xz(0.5, 0.3)) < 0.05
    assert 0.4 < v.change_vs_live_deg < 0.7


def test_verdict_keeps_a_correct_live():
    s = synthetic(imu.rot_xz(0.5, 0.3), noise_deg=0.02)
    v = imu.verdict(s, R_live=R_CFG @ imu.rot_xz(0.5, 0.3))
    assert not v.deploy
    assert any("combined sigma" in r for r in v.reasons)


@pytest.mark.parametrize("kw, word", [(dict(pitches=(9, 25), yaws=(-30, 0, 30)), "poses"),
                                      (dict(pitches=(5, 9, 13, 17, 21, 25, 29, 33, 37, 41), yaws=(0,)), "yaw columns"),
                                      (dict(pitches=(17,), yaws=(-30, -20, -10, 0, 10, 20, 30, 40, 50)), "pitch span")])
def test_verdict_gates(kw, word):
    v = imu.verdict(synthetic(imu.rot_xz(0.5, 0.3), noise_deg=0.02, **kw), R_live=R_CFG)
    assert not v.deploy
    assert any(word in r and "need" in r for r in v.reasons), v.reasons


def test_verdict_reports_third_angle_and_tbs_note():
    v = imu.verdict(synthetic(imu.rot_xz(0.5, 0.3), noise_deg=0.02), R_live=R_CFG)
    text = "\n".join(v.reasons)
    assert "third angle" in text and "kept from the live config" in text
    assert "T_BS" in text and "Head" in text
    assert v.observability.ratio < 0.3


def test_correction_is_relative_to_live():
    s = synthetic(imu.rot_xz(0.5, 0.3), noise_deg=0.02)
    R_live = R_CFG @ imu.axis_rotation([0.3, 1, -0.2], 1.5)          # live differs from the sweep's config
    v = imu.verdict(s, R_live=R_live)
    assert np.allclose(R_live @ v.correction, v.R_new, atol=1e-12)
    assert np.allclose(v.R_new, R_CFG @ v.fit.correction, atol=1e-12)
    assert any("sweep ran with" in r for r in v.reasons)
