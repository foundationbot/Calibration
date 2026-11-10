"""calibtool.solver: the fast KB4T stereo solve (M3). Derivatives against finite differences, the stereo residuals
against stereofit's, and the solve against scipy on synthetic data (lens_synth, Fit A truth): where scipy stops short
in a flat valley, a cost at or below scipy's and a minimum the fast solve returns to from scipy's result; the
comparison with scipy run tight is in the replay tier (minutes)."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

CAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAL / "tests"))
S = importlib.import_module("lens_synth")
sf = S.sf
lensmodel = S.lensmodel
from calibtool import solver  # noqa: E402

FITA = {"KB4T": [S.I0, S.I1], "KB4": [S.I0[:8], S.I1[:8]]}


def _points():
    """camera-frame points from 0 to 80 deg off the axis, every azimuth, one exactly on the axis"""
    rng = np.random.default_rng(5)
    th = np.radians(rng.uniform(0, 80, 200))
    ph = rng.uniform(-np.pi, np.pi, 200)
    z = rng.uniform(0.3, 3.0, 200)
    P = np.c_[np.tan(th) * np.cos(ph) * z, np.tan(th) * np.sin(ph) * z, z]
    return np.r_[P, [[0.0, 0.0, 1.0], [1e-9, -2e-9, 0.7]]]


def _fd(f, x, h):
    """central differences of f (-> (N, 2)) in each column of x (N, k) -> (N, 2, k)"""
    out = []
    for j in range(x.shape[1]):
        e = np.zeros_like(x)
        e[:, j] = h[j] if np.ndim(h) else h
        out.append((f(x + e) - f(x - e)) / (2 * e[:, j:j + 1]))
    return np.stack(out, axis=-1)


@pytest.mark.parametrize("model", ["KB4T", "KB4"])
def test_projection_and_its_derivatives_match_lensmodel_and_finite_differences(model):
    intr = np.asarray(FITA[model][0])
    P = _points()
    uv, dP, dI = solver.project_jac(model, intr, P)
    assert np.abs(uv - lensmodel.project(model, intr, P)).max() < 1e-9
    fdP = _fd(lambda Q: lensmodel.project(model, intr, Q), P, 1e-6)
    off = np.hypot(P[:, 0], P[:, 1]) > 1e-6                      # finite differences straddle the axis there
    assert np.abs(dP - fdP)[off].max() < 1e-5 * np.abs(fdP).max()
    assert np.abs(dP[~off] - fdP[~off]).max() < 1e-3             # on the axis: the pinhole limit
    h = np.maximum(np.abs(intr), 1e-3) * 1e-6
    fdI = np.stack([(lensmodel.project(model, intr + np.eye(len(intr))[j] * h[j], P)
                     - lensmodel.project(model, intr - np.eye(len(intr))[j] * h[j], P)) / (2 * h[j])
                    for j in range(len(intr))], axis=-1)
    assert dI.shape == (len(P), 2, len(intr))
    assert np.abs(dI - fdI).max() < 1e-5 * np.abs(fdI).max()


# ---------------------------------------------------------------- the stereo problem


def _scene(seed=4, n_holds=9, noise=0.3, C=None):
    d = S.views([S.I0, S.I1], S.C_TRUE if C is None else C, n_holds, seed=seed, noise=noise)
    data = sf.Data(d, S.BOARD)
    poses, ok = sf.pnp_poses(data, "KB4T", [S.I0, S.I1], S.C_TRUE)
    assert ok.all()
    return data, poses


def _start(rng, data, poses):
    """a perturbed calibration and poses, to evaluate the residuals away from their minimum"""
    intr = [S.I0 * (1 + rng.normal(0, 1e-3, 10)), S.I1 * (1 + rng.normal(0, 1e-3, 10))]
    C = S.C_TRUE @ np.r_[np.c_[sf.rotmat(rng.normal(0, 2e-3, 3))[0], rng.normal(0, 1e-3, 3)], [[0, 0, 0, 1]]]
    return intr, C, poses + rng.normal(0, 1e-3, poses.shape), rng.normal(0, 1e-2, data.n)


def test_residuals_are_stereofits_including_the_clamps():
    data, poses = _scene()
    rng = np.random.default_rng(1)
    intr, C, poses, ls = _start(rng, data, poses)
    poses[2, 5] = -1.0                                               # one board behind both cameras: clamped
    prob = solver.Problem(data, "KB4T")
    st = solver.State.start(intr, C, poses, ls)
    x = sf._pack(intr, poses, ls, True, True)
    want = sf.residuals(np.r_[x[:20], np.zeros(6), x[20:]], data, "KB4T", True, True, True, intr, C).reshape(-1, 2)
    got = prob.residuals(st)
    assert (want[data.pair == 2] == 1e3).all()
    assert np.abs(got - want).max() < 1e-9


@pytest.mark.parametrize("free_intr,free_C,scale", [(True, True, True), (True, True, False), (False, True, False),
                                                    (False, False, True)])
def test_jacobian_is_finite_differences_of_the_local_updates(free_intr, free_C, scale):
    data, poses = _scene()
    rng = np.random.default_rng(2)
    intr, C, poses, ls = _start(rng, data, poses)
    poses[3, 5] = -1.0                                               # clamped rows: zero Jacobian
    prob = solver.Problem(data, "KB4T")
    st = solver.State.start(intr, C, poses, ls if scale else None)
    w = prob.valid.copy()
    r, Jg, Jp = prob.linearise(st, free_intr, free_C, scale, w)
    G, k = Jg.shape[-1], Jp.shape[-1]
    assert G == 20 * free_intr + 6 * free_C and k == 6 + scale
    for j in range(G):
        e = np.zeros(G)
        h = 1e-6 * (abs(np.r_[intr[0], intr[1]][j]) if j < 20 * free_intr else 1.0)
        e[j] = h
        rp = prob.linearise(st.step(e, np.zeros((data.n, k)), free_intr, free_C, scale), free_intr, free_C, scale, w,
                            jac=False)
        rm_ = prob.linearise(st.step(-e, np.zeros((data.n, k)), free_intr, free_C, scale), free_intr, free_C, scale,
                             w, jac=False)
        fd = (rp - rm_) / (2 * h)
        assert np.abs(Jg[..., j] - fd).max() <= 1e-5 * max(1.0, np.abs(fd).max()), j
    for j in range(k):
        e = np.zeros((data.n, k))
        e[:, j] = 1e-6
        rp = prob.linearise(st.step(np.zeros(G), e, free_intr, free_C, scale), free_intr, free_C, scale, w, jac=False)
        rm_ = prob.linearise(st.step(np.zeros(G), -e, free_intr, free_C, scale), free_intr, free_C, scale, w,
                             jac=False)
        fd = (rp - rm_) / 2e-6
        assert np.abs(Jp[..., j] - fd).max() <= 1e-5 * max(1.0, np.abs(fd).max()), j
    assert (Jp[3] == 0).all() and (Jg[3] == 0).all()


def test_rotvec_is_rigmaths_log_at_every_angle():
    rm = importlib.import_module("rigmath")
    rng = np.random.default_rng(3)
    ax = rng.normal(size=(50, 3))
    ax /= np.linalg.norm(ax, axis=1)[:, None]
    ang = np.r_[0.0, 1e-9, rng.uniform(0, np.pi, 44), np.pi - 1e-7, np.pi, 1.0, 2.0]
    R = sf.rotmat(ax * ang[:, None])
    got = solver.rotvec(R)
    assert np.abs(sf.rotmat(got) - R).max() < 1e-12
    assert np.abs(got - np.array([rm.log_rot(x) for x in R]))[ang < 3].max() < 1e-12


# ---------------------------------------------------------------- the solve against scipy, tight


def _tight(monkeypatch, tol=1e-12):
    """stereofit.solve with scipy run to ftol = xtol = gtol = tol: the reference (1e-15 x 20000 evaluations took
    > 9 min for the suite; the camera-fixed and stereo-pose-only solves converge well before 1e-12)"""
    import functools
    monkeypatch.setattr(sf, "least_squares", functools.partial(sf.least_squares, ftol=tol, xtol=tol, gtol=tol))


def _grid_diff(intr_a, C_a, intr_b, C_b, model="KB4T"):
    """max pixel difference per camera of camera-0-frame points on a 32 px grid of each image at z 0.5 / 1 / 2 m"""
    W, H = 1280, 800
    g = np.stack(np.meshgrid(np.arange(0, W, 32.0), np.arange(0, H, 32.0)), -1).reshape(-1, 2)
    out = [0.0, 0.0]
    for c in (0, 1):
        ray = lensmodel.unproject(model, intr_a[c], g)
        ray = ray[ray[:, 2] > 0.05]
        for z in (0.5, 1.0, 2.0):
            Pc = ray * (z / ray[:, 2:3])
            P0 = Pc if c == 0 else Pc @ C_a[:3, :3].T + C_a[:3, 3]
            for cc in (0, 1):
                pa = P0 if cc == 0 else (P0 - C_a[:3, 3]) @ C_a[:3, :3]
                pb = P0 if cc == 0 else (P0 - C_b[:3, 3]) @ C_b[:3, :3]
                ua = lensmodel.project(model, intr_a[cc], pa)
                ub = lensmodel.project(model, intr_b[cc], pb)
                m = (pa[:, 2] > 0.05) & (ua[:, 0] >= 0) & (ua[:, 0] < W) & (ua[:, 1] >= 0) & (ua[:, 1] < H)
                out[cc] = max(out[cc], float(np.abs(ub[m] - ua[m]).max()))
    return out


def _pose_diff(Ca, Cb):
    rm = importlib.import_module("rigmath")
    return (1e3 * np.linalg.norm(Ca[:3, 3] - Cb[:3, 3]),
            float(np.degrees(np.linalg.norm(rm.log_rot(Ca[:3, :3] @ Cb[:3, :3].T)))))


@pytest.fixture(scope="module")
def presolved():
    """fit_from_nominal's state just before fit_converged: the PnP start and the stereo-pose presolve (scipy)"""
    data = sf.Data(S.fit_sweep(), S.BOARD)
    intr0, C0 = sf.nominal_start(S.rig12(), "KB4T")
    data, poses, _ = sf._pnp_kept(data, "KB4T", intr0, C0, lambda *a: None)
    pre = sf.solve(data, "KB4T", intr0, C0, poses, free_intr=False, free_C=True, trim_px=np.inf, max_nfev=2000)
    return data, intr0, pre


def test_the_full_fit_ends_below_scipy_on_a_minimum_it_returns_to(presolved):
    """scipy (step 45's tolerances) stops early in the flat valley of the full fit, so it is no fixed reference here
    (measured 9 Nov: cost fast 2109.40725 < scipy 1e-10 2109.40972 < scipy default 2109.41594; scipy at 1e-10 needs
    ~2900 evaluations, 110 s). What the unit tier pins instead: the fast fit converges, its cost is not above
    scipy's, and started from scipy's result it returns to its own solution (a minimum, not a stop in the valley). The
    tight-scipy comparison at the design bound is test_the_full_fit_is_within_the_design_bound_of_tight_scipy"""
    data, intr0, pre = presolved
    fast = solver.solve(data, "KB4T", intr0, pre["C"], pre["poses"], max_nfev=2000)
    ref = sf.solve(data, "KB4T", intr0, pre["C"], pre["poses"], max_nfev=2000)
    assert fast["success"] and fast["nfev"] <= 4 * 2000
    assert set(fast) == set(ref)
    assert fast["poses"].shape == ref["poses"].shape and np.abs(fast["log_scale"]).max() == 0
    assert np.array_equal(fast["kept"], ref["kept"])
    assert sf.fit_cost(fast)[0] <= sf.fit_cost(ref)[0] * (1 + 1e-12)
    back = solver.solve(data, "KB4T", ref["intr"], ref["C"], ref["poses"], max_nfev=2000)
    assert back["success"] and np.array_equal(back["kept"], fast["kept"])
    assert max(_grid_diff(fast["intr"], fast["C"], back["intr"], back["C"])) < 1e-5
    dt, dr = _pose_diff(fast["C"], back["C"])
    assert dt < 1e-5 and dr < 1e-6
    assert abs(sf.fit_cost(back)[0] - sf.fit_cost(fast)[0]) <= 1e-9 * sf.fit_cost(fast)[0]
    assert max(_grid_diff(ref["intr"], ref["C"], fast["intr"], fast["C"])) < 0.05     # same minimum (scipy short)


@pytest.mark.replay
def test_the_full_fit_is_within_the_design_bound_of_tight_scipy(presolved, monkeypatch):
    """design 9.1 on synthetic data: against scipy at ftol = xtol = gtol = 1e-10 (~110 s single thread, hence the
    replay tier) the grid projection differs < 0.02 px per camera, the stereo pose < 0.01 mm / 0.001 deg, and the
    fast cost is not above the reference's"""
    data, intr0, pre = presolved
    fast = solver.solve(data, "KB4T", intr0, pre["C"], pre["poses"], max_nfev=2000)
    _tight(monkeypatch, 1e-10)
    ref = sf.solve(data, "KB4T", intr0, pre["C"], pre["poses"], max_nfev=5000)
    assert fast["success"] and ref["success"] and np.array_equal(fast["kept"], ref["kept"])
    assert sf.fit_cost(fast)[0] <= sf.fit_cost(ref)[0] * (1 + 1e-12)
    assert max(_grid_diff(ref["intr"], ref["C"], fast["intr"], fast["C"])) < 0.02
    dt, dr = _pose_diff(ref["C"], fast["C"])
    assert dt < 0.01 and dr < 0.001


def test_the_stereo_pose_presolve_matches_scipy(monkeypatch):
    data = sf.Data(S.fit_sweep(), S.BOARD)
    intr0, C0 = sf.nominal_start(S.rig12(), "KB4T")
    data, poses, _ = sf._pnp_kept(data, "KB4T", intr0, C0, lambda *a: None)
    fast = solver.solve(data, "KB4T", intr0, C0, poses, free_intr=False, free_C=True, trim_px=np.inf, max_nfev=2000)
    _tight(monkeypatch)
    ref = sf.solve(data, "KB4T", intr0, C0, poses, free_intr=False, free_C=True, trim_px=np.inf, max_nfev=20000)
    assert fast["success"] and fast["kept"].all()
    dt, dr = _pose_diff(ref["C"], fast["C"])
    assert dt < 1e-3 and dr < 1e-4
    assert all(np.array_equal(a, b) for a, b in zip(fast["intr"], intr0))


def test_a_cap_too_small_to_converge_is_not_success(presolved):
    data, intr0, pre = presolved
    r = solver.solve(data, "KB4T", intr0, pre["C"], pre["poses"], max_nfev=2, rounds=1)
    assert not r["success"] and r["nfev"] <= 2


def _pair_cost(data, r):
    """sum of squared residuals per pair over the corners r kept"""
    e = np.where(r["kept"][:, None], r["res"], 0.0)
    return np.bincount(data.pair, weights=(e ** 2).sum(1), minlength=data.n)


def _a_minimum_at_or_below(data, model, intr, C, fast, ref, scale):
    """fast's kept corners are ref's, its cost is not above ref's on any pair, and the fast solve started from ref's
    result returns to fast's own (poses to 1e-7 rad / m, board scales to 1e-7): ref stopped short, not elsewhere"""
    assert fast["success"] and ref["success"] and np.array_equal(fast["kept"], ref["kept"])
    pf, pr = _pair_cost(data, fast), _pair_cost(data, ref)
    assert (pf <= pr + 1e-9 * np.maximum(pr, 1.0)).all(), np.max(pf - pr)
    back = solver.solve(data, model, intr, C, ref["poses"], free_intr=False, free_C=False, scale=scale,
                        max_nfev=2000, ls0=ref["log_scale"] if scale else None)
    assert back["success"] and np.array_equal(back["kept"], fast["kept"])
    assert np.abs(back["poses"][:, 3:] - fast["poses"][:, 3:]).max() < 1e-7
    assert np.abs(sf.rotmat(back["poses"][:, :3]) - sf.rotmat(fast["poses"][:, :3])).max() < 1e-7
    assert np.abs(back["log_scale"] - fast["log_scale"]).max() < 1e-7
    assert np.array_equal(fast["C"], C) and np.abs(fast["log_scale"]).max() * (not scale) == 0


@pytest.mark.parametrize("scale", [False, True])
def test_with_the_camera_fixed_each_pair_is_a_minimum_at_or_below_scipys(scale):
    """the scorer's solve against scipy at stereofit's tolerances. Not "equal to tight scipy per pair": scipy stops
    early on a pair whose pose and board scale are nearly degenerate (holdout_sweep pair 28: board scale off the fast
    result by 8.0e-5 at scipy's defaults, 1.7e-5 at tol 1e-12 where scipy stops unconverged at 2000 evaluations,
    measured 9 Nov), while the fast cost is lower on every pair"""
    data = sf.Data(S.holdout_sweep(), S.BOARD)
    intr, C = [S.I0, S.I1], S.C_TRUE
    poses, ok = sf.pnp_poses(data, "KB4T", intr, C)
    C = C.copy()
    C[:3, 3] *= 1.01                                         # board scales away from 1
    fast = solver.solve(data, "KB4T", intr, C, poses, free_intr=False, free_C=False, scale=scale, max_nfev=2000)
    ref = sf.solve(data, "KB4T", intr, C, poses, free_intr=False, free_C=False, scale=scale, max_nfev=2000)
    _a_minimum_at_or_below(data, "KB4T", intr, C, fast, ref, scale)


def _centre_holds():
    """holdout_sweep()'s centre-target holds (0-11): inside the radius where lens_synth.radtan_like() fits the lens"""
    data = sf.Data(S.holdout_sweep(), S.BOARD)
    return data.select(data.hold < 12)


def test_radtan_with_the_camera_fixed_and_refused_with_free_intrinsics():
    from calibtool import livecfg
    rt = list(S.radtan_like())
    data = _centre_holds()
    with livecfg.radtan_scope():
        poses, ok = sf.pnp_poses(data, "RADTAN", rt, S.C_TRUE)
        d = data.select(ok)
        fast = solver.solve(d, "RADTAN", rt, S.C_TRUE, poses[ok], free_intr=False, free_C=False, scale=True,
                            max_nfev=2000)
        ref = sf.solve(d, "RADTAN", rt, S.C_TRUE, poses[ok], free_intr=False, free_C=False, scale=True,
                       max_nfev=2000)
        assert fast["kept"].mean() > 0.95
        _a_minimum_at_or_below(d, "RADTAN", rt, S.C_TRUE, fast, ref, True)
        with pytest.raises(NotImplementedError):
            solver.solve(d, "RADTAN", rt, S.C_TRUE, poses[ok], free_intr=True)


def test_a_pair_whose_minimum_runs_away_reports_success_as_scipy_does():
    """a lens that does not fit the corners (a made-up RADTAN on the fisheye sweep): trimming leaves pairs with a
    handful of corners whose board scale runs off (log scale > 8) with the cost still falling. scipy stops there on
    its ftol test (relative decrease < 1e-8) and reports success; the fast solve keeps going to its cap for accuracy
    but reports success too, since scipy's test was met (measured 9 Nov: it reported failure, 5197 evaluations)"""
    from calibtool import livecfg
    data = sf.Data(S.holdout_sweep(), S.BOARD)
    rt = [np.r_[565.0, 564.0, 642.0, 385.0, -0.01, 0.002, 0.0005, -0.0003],
          np.r_[567.0, 566.0, 642.5, 383.0, -0.012, 0.003, 0.0001, 0.0002]]
    with livecfg.radtan_scope():
        poses, ok = sf.pnp_poses(data, "RADTAN", rt, S.C_TRUE)
        d = data.select(ok)
        fast = solver.solve(d, "RADTAN", rt, S.C_TRUE, poses[ok], free_intr=False, free_C=False, scale=True,
                            max_nfev=2000)
        ref = sf.solve(d, "RADTAN", rt, S.C_TRUE, poses[ok], free_intr=False, free_C=False, scale=True,
                       max_nfev=2000)
    assert ref["success"] and fast["success"]
    assert fast["kept"].mean() < 0.5 and np.abs(fast["log_scale"]).max() > 5        # the scenario: a runaway pair
