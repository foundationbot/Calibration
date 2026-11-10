"""calibtool.solver: the fast KB4T stereo solve (M3). Derivatives against finite differences, the stereo residuals
against stereofit's, and the solve against scipy run to a tight tolerance on synthetic data (lens_synth, Fit A truth)."""
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
