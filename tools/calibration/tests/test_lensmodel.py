"""lensmodel.py: the KB4 / KB4T lens the package fits and uses (numpy, no OpenCV)."""
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import lensmodel as lm  # noqa: E402

REF = json.loads((HERE / "data" / "kb4t_reference.json").read_text())
FIT_A = REF["intr"]                      # Fit A (20 Oct): [fx, fy, cx, cy, k1..k4, p1, p2] per camera


def _points(n=400, seed=1, max_deg=80.0):
    rng = np.random.default_rng(seed)
    th, ph, r = np.radians(rng.uniform(0, max_deg, n)), rng.uniform(-np.pi, np.pi, n), rng.uniform(0.3, 2.5, n)
    return np.c_[r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)]


def test_kb4_is_opencvs_fisheye():
    cv2 = pytest.importorskip("cv2")
    intr = FIT_A[0][:8]
    K = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1.0]])
    P = _points(max_deg=70)
    ref, _ = cv2.fisheye.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.array(intr[4:]).reshape(4, 1))
    assert np.abs(lm.project("KB4", intr, P) - ref.reshape(-1, 2)).max() < 1e-6


def test_kb4t_without_tangential_terms_is_kb4_exactly():
    P, intr = _points(), list(FIT_A[0][:8])
    assert np.array_equal(lm.project("KB4T", intr + [0.0, 0.0], P), lm.project("KB4", intr, P))


def test_kb4t_is_the_code_that_fitted_fit_a():
    """lensfit.project's pixels for the same points: Fit A must project exactly as it was fitted (a swapped p1/p2
    or a sign error passes the p = 0 test above, not this one)"""
    P = np.array(REF["points"])
    for c in (0, 1):
        assert np.abs(lm.project("KB4T", FIT_A[c], P) - np.array(REF["pixels"][c])).max() < 1e-9


@pytest.mark.parametrize("model", ["KB4", "KB4T"])
def test_unproject_inverts_project(model):
    intr = FIT_A[0] if model == "KB4T" else FIT_A[0][:8]
    P = _points(max_deg=80)
    rays = lm.unproject(model, intr, lm.project(model, intr, P))
    assert np.abs(rays - P / np.linalg.norm(P, axis=1, keepdims=True)).max() < 1e-10
    u, v = np.meshgrid(np.arange(0.0, 1280.0, 16.0), np.arange(0.0, 800.0, 16.0))
    uv = np.c_[u.ravel(), v.ravel()]                          # the whole raw image, corners included
    assert np.abs(lm.project(model, intr, lm.unproject(model, intr, uv)) - uv).max() < 1e-8


def test_unproject_converges_in_two_newton_steps(monkeypatch):
    """The Newton steps' Jacobians are the true ones: two steps already invert KB4T to ~1e-8 px over the whole raw
    image (Fit A camera 0: 8.7e-9). With a wrong-sign j21 the 30 default steps still converge (the test above
    passes) but two leave 1.3e-6 px."""
    monkeypatch.setattr(lm, "NEWTON_STEPS", 2)
    u, v = np.meshgrid(np.arange(0.0, 1280.0, 16.0), np.arange(0.0, 800.0, 16.0))
    uv = np.c_[u.ravel(), v.ravel()]
    for c in (0, 1):
        assert np.abs(lm.project("KB4T", FIT_A[c], lm.unproject("KB4T", FIT_A[c], uv)) - uv).max() < 1e-7, c


def test_the_principal_point_is_the_optical_axis():
    assert np.allclose(lm.unproject("KB4T", FIT_A[1], [[FIT_A[1][2], FIT_A[1][3]]]), [[0.0, 0.0, 1.0]])


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="lens model"):
        lm.project("KB8", FIT_A[0], np.ones((1, 3)))
    with pytest.raises(ValueError, match="takes 10 numbers"):
        lm.project("KB4T", FIT_A[0][:8], np.ones((1, 3)))
