"""Tests of torso_fit.py (route 2), run on Moonstone from the point-cloud code copy (meshfit and _burstfusion are
on feat/pointcloud-agent-burst-fusion, not here; results/ is not collected by CI):
  PYTHONPATH=<copy>/build:<copy>/repo/tools/burst_pointcloud venv/bin/python -m pytest -q check_torso_fit.py
Needs TORSO_MESH_DIR (the MJCF meshes folder holding MK1_1_Torso_Shell.obj)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torso_fit  # noqa: E402
import meshfit  # noqa: E402
import mesh  # noqa: E402
import frames  # noqa: E402

MESH_DIR = Path(os.environ.get("TORSO_MESH_DIR", ""))
K = (466.253, 466.253, 631.797, 376.008)
W, H = 1280, 800


def _rt(rx, ry, rz, t):
    T = np.eye(4)
    T[:3, :3] = meshfit._rot(np.radians([rx, ry, rz]))
    T[:3, 3] = t
    return T


def _head_camera():
    """a camera 0.5 m above the torso origin looking down-forward at the chest, z forward (like neck 28)"""
    T_TC = np.eye(4)
    # camera axes in the torso: x = -torso y, y = down-forward, z = forward-down
    a = np.radians(28 + 40)
    z = np.array([np.cos(a), 0.0, -np.sin(a)])
    x = np.array([0.0, -1.0, 0.0])
    y = np.cross(z, x)
    T_TC[:3, :3] = np.c_[x, y, z]
    T_TC[:3, 3] = (0.08, 0.0, 0.62)
    return T_TC


def test_dT_from_E_inverts_the_plate_convention():
    """T_TC(true) = dT . T_TC(chain); the mesh seen through the true camera is E . T_CT(chain) . p"""
    T_TC = _head_camera()
    dT = _rt(0.19, -0.05, 0.03, (-0.0014, 0.0030, 0.0019))
    T_CT_true = np.linalg.inv(dT @ T_TC)
    E = T_CT_true @ T_TC
    assert np.allclose(torso_fit.dT_from_E(T_TC, E), dT, atol=1e-12)


def test_body_model_takes_the_torso_alone():
    V = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], float)
    F = np.array([[0, 1, 2]])
    m = torso_fit.BodyModel(frames.URDF, {"Torso": (V, F)}, 0.01)
    assert m.links == ["Torso"] and m.side is None
    P, _, lid = m.place({}, np.eye(4))
    assert len(P) > 0 and np.all(lid == 0)
    assert np.allclose(m.wrist_point_T({}), 0.0)


def _box_corner(c, a):
    """the three faces of a cube (edge a) meeting at its corner c facing the camera: well conditioned in all 6 dof"""
    V, F = [], []
    for k in range(3):
        u, v = [j for j in range(3) if j != k]
        base = len(V)
        for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
            p = np.array(c, float)
            p[u] += du * a
            p[v] += dv * a
            V.append(p)
        F += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return np.array(V), np.array(F)


def test_fit_and_conversion_recover_a_known_dT_on_a_box_corner():
    """the code path (BodyModel, fit_pose with one link, dT_from_E) on geometry that pins all 6 dof"""
    T_TC = _head_camera()
    V, F = _box_corner(T_TC[:3, 3] + 0.5 * T_TC[:3, 2] - 0.06, 0.12)
    body = torso_fit.BodyModel(frames.URDF, {"Torso": (V, F)}, 0.001)
    dT = _rt(0.19, -0.05, 0.03, (-0.0014, 0.0030, 0.0019))
    depth = mesh.raster(V, F, np.linalg.inv(dT @ T_TC), K, W, H)
    f = meshfit.fit_pose(body, {}, T_TC, meshfit.unproject(depth, K), K, W, H)
    got = torso_fit.dT_from_E(T_TC, f.delta_c)
    err = np.degrees(np.linalg.norm(torso_fit.rotvec(got[:3, :3] @ dT[:3, :3].T)))
    assert f.ok and err < 0.01 and np.linalg.norm(got[:3, 3] - dT[:3, 3]) < 2e-4, (err, f.rms_mm, f.cond)
    pf = torso_fit.cost_profile(body, T_TC, f.delta_c, meshfit.unproject(depth, K), K, W, H)
    assert pf["flat"] == [], pf["flat"]
    for name, row in pf["axes"].items():
        assert min(v for _, v in row) >= 1.0 - 1e-9, name          # the fit is the minimum along every axis
        assert len(row) == 40 and any(abs(d - 0.05) < 1e-9 for d, _ in row)


def test_profile_flags_a_plane_as_flat_about_its_normal():
    """a single plane is blind to rotation about its normal: the profile must call that axis flat"""
    T_TC = _head_camera()
    c = T_TC[:3, 3] + 0.5 * T_TC[:3, 2]
    V = np.array([c + (-0.2, -0.2, 0), c + (0.2, -0.2, 0), c + (0.2, 0.2, 0), c + (-0.2, 0.2, 0)])
    F = np.array([[0, 1, 2], [0, 2, 3]])
    body = torso_fit.BodyModel(frames.URDF, {"Torso": (V, F)}, 0.002)
    depth = mesh.raster(V, F, np.linalg.inv(T_TC), K, W, H)
    pf = torso_fit.cost_profile(body, T_TC, np.eye(4), meshfit.unproject(depth, K), K, W, H)
    assert "rz" in pf["flat"], pf["flat"]                           # torso z is the plane's normal


def _sess(m0=True, rv=None, d=None, flat=()):
    x = {"m0_pass": m0}
    if rv is not None:
        x.update({"median_rotvec_deg": list(rv), "deg_from_b1": d, "profiles": {"c": {"flat": list(flat)}}})
    return x


def test_verdict_follows_the_plan():
    b1 = torso_fit.B1_ROTVEC_DEG
    ok = _sess(rv=b1 + (0.02, 0, 0), d=0.02)
    assert torso_fit.verdict({"a": ok, "b": _sess(rv=b1, d=0.0), "c": _sess(m0=True)})["route2"] == "USABLE"
    v = torso_fit.verdict({"a": ok, "b": _sess(rv=b1, d=0.0, flat=("rx",))})
    assert v["route2"] == "NOT USABLE" and v["per_session"]["b"].startswith("FLAT")
    v = torso_fit.verdict({"a": ok, "b": _sess(m0=False, rv=b1, d=0.0)})
    assert v["per_session"]["b"] == "M0 FAIL" and v["route2"] == "NOT USABLE"
    v = torso_fit.verdict({"a": _sess(rv=b1 + (0.09, 0, 0), d=0.09), "b": _sess(rv=b1 - (0.09, 0, 0), d=0.09)})
    assert v["passed"] == ["a", "b"] and v["route2"] == "NOT USABLE"   # each within 0.1 of B1 but 0.18 apart
    v = torso_fit.verdict({"a": ok, "b": _sess(rv=b1 + (0.2, 0, 0), d=0.2)})
    assert v["per_session"]["b"].startswith("FAIL") and v["route2"] == "NOT USABLE"
