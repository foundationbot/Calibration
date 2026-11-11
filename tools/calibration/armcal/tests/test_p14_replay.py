"""The port against what ran on phantom14 (4 Nov 2026) and in H-1 (11 Nov), on the stored data ($CALIBTOOL_DATA/p14arm,
skipped without it):
  - the numpy fusion equals the C++ module's stored fusions;
  - read B (burst file -> record, the fast computation) reproduces stored per-capture records of both arms;
  - stage 2 reproduces H-1's M12+head and M12 fits (parameters, standard errors, condition number, the
    leave-one-arm-pose-out folds) and phantom14's deployed M12 URDF."""
import json
from pathlib import Path

import numpy as np
import pytest

from armcal import burst_io, fuse, meshfit, readb, urdffit
from armcal.urdf import Urdf, write_variant

H1 = Path(__file__).resolve().parents[2] / "results" / "handeye-h1-2026-11-11" / "h1b.json"
STORED = {"base_w1.json": "1103a-right", "base_w2.json": "1104a-right", "base_L.json": "1104b-left"}
OFF_TOL_MM, ROT_TOL_DEG = 0.5, 1.0     # the speed rule (H0 task): within these of the stored record, same clean


def stored(p14, f, name):
    D = json.loads((p14 / f).read_text())
    return next(r for r in D["res"] if r["name"] == name), D["side"]


@pytest.mark.parametrize("name", ["p14-armwave-1104a-right-07-p09-10-0", "p14-armwave-1104b-left-13-pL10-22-m12",
                                  "p14-armwave-1103a-right-02-p01"])
def test_numpy_fusion_equals_the_cpp_fusion(p14, name):
    b = burst_io.load(p14 / "bursts" / f"{name}.npz")
    want = np.load(p14 / "fused_cpp" / f"{name}.npz")["depth"]
    got = fuse.fuse_depth(b.depth, b.sigma).depth
    assert got.shape == want.shape
    assert ((got > 0) == (want > 0)).all()
    assert np.abs(got.astype(np.float64) - want).max() < 1e-6       # float32 rounding of a ~1 m depth is 6e-8


@pytest.mark.parametrize("f,name", [("base_w2.json", "07-p09-10-0"), ("base_L.json", "17-pL04-22-0"),
                                    ("base_L.json", "13-pL10-22-m12")])          # the last: not clean (58 deg)
def test_read_b_reproduces_the_stored_record(p14, f, name):
    rec0, side = stored(p14, f, name)
    urdf = Urdf.load(p14 / "phantom_mk1.urdf")
    cap = readb.load_capture(p14 / "bursts" / f"p14-armwave-{STORED[f]}-{name}.npz", urdf, side)
    rec, _ = readb.match(readb.arm_model(urdf, side), cap, meshfit.FAST, view=False)
    assert np.linalg.norm(np.subtract(rec["off"], rec0["off"])) < OFF_TOL_MM
    assert abs(rec["rot"] - rec0["rot"]) < ROT_TOL_DEG
    assert rec["clean"] == rec0["clean"]
    assert np.allclose(rec["T_TC"], rec0["T_TC"], atol=1e-9) and np.allclose(rec["wrist"], rec0["wrist"], atol=1e-12)


@pytest.fixture
def h1_data(p14):
    base = Urdf.load(p14 / "phantom_mk1.urdf")
    R, _ = urdffit.load_records(p14 / "base_w2.json")
    L, _ = urdffit.load_records(p14 / "base_L.json")
    R, dR = urdffit.select(R)
    L, dL = urdffit.select(L)
    return base, R, L, dR, dL, json.loads(H1.read_text())


def test_stage2_selection_is_h1s(h1_data):
    base, R, L, dR, dL, h1 = h1_data
    assert [o.name for o in R] == h1["selection"]["right"] and [o.name for o in L] == h1["selection"]["left"]
    assert dR == h1["selection"]["drop_right"] and dL == h1["selection"]["drop_left"] == ["01-pL07-10-0"]


@pytest.mark.parametrize("model", ["M12+head", "M12"])
def test_stage2_full_fit_reproduces_h1(h1_data, model):
    base, R, L, _, _, h1 = h1_data
    f = urdffit.fit(urdffit.MODELS[model], R, L, base, model)
    want = h1["full"][model]
    assert f.names == want["names"]
    assert np.allclose(f.x, want["x"], atol=0.01)
    assert np.allclose(f.se, want["se"], atol=0.01)
    assert f.cond == pytest.approx(want["cond"], rel=0.01)
    assert f.rms == pytest.approx(want["rms"], abs=0.01)


def test_stage2_cross_validation_reproduces_h1(h1_data):
    base, R, L, _, _, h1 = h1_data
    folds = urdffit.lopo(urdffit.MODELS["M12+head"], R, L, base, "M12+head")
    want = {(r["side"], r["pose"]): r for r in h1["lopo"] if r["model"] == "M12+head"}
    assert len(folds) == len(want) == 10
    for r in folds:
        w = want[r["side"], r["pose"]]
        assert np.allclose(r["new"], w["new"], atol=0.01) and np.allclose(r["base"], w["base"], atol=0.01)
    s = urdffit.cv_summary(folds)
    assert round(s["Right"]["median"], 1) == 3.6 and round(s["Left"]["median"], 1) == 5.3


def test_stage2_reproduces_phantom14s_deployed_urdf(p14, tmp_path):
    """phantom14's twoarm.py: M12 fitted with right poses 03/08/14/15 and left 02/10/15/19 held out, left 01 dropped;
    its URDF (deployed 4 Nov) has these three origins"""
    base = Urdf.load(p14 / "phantom_mk1.urdf")
    R, _ = urdffit.load_records(p14 / "base_w2.json")
    L, _ = urdffit.load_records(p14 / "base_L.json")
    L = [o for o in L if not o.name.startswith("01-")]
    R = [o for o in R if not o.name.startswith(("03-", "08-", "14-", "15-"))]
    L = [o for o in L if not o.name.startswith(("02-", "10-", "15-", "19-"))]
    f = urdffit.fit(urdffit.MODELS["M12"], R, L, base, "M12")
    want = json.loads((p14 / "twoarm_fit_M9_M10_M11_M12.json").read_text())["M12_neckyaw-xyz+Rroll0+Lroll0(5)"]
    assert np.allclose(f.change["NeckYaw"]["dxyz"], want["NeckYaw"]["dxyz"], atol=0.01)
    for j in ("RightShoulderRoll", "LeftShoulderRoll"):
        assert f.change[j]["post_axis_deg"] == pytest.approx(want[j]["post_axis_deg"], abs=0.001)
    out = write_variant(base, want, tmp_path / "p14.urdf", ["replay of phantom14's 4 Nov fit"])
    deployed = Urdf.load(p14 / "phantom14_mk1_handeye-20261104.urdf")
    mine = Urdf.load(out)
    for j in deployed.joints:
        assert np.allclose(mine.joint(j).origin, deployed.joint(j).origin, atol=1e-8), j


# the robust matcher (readb2, 11 Nov 2026): its two passes on stored captures, pinned to the runs reported in
# results/handeye-readb2-2026-11-11/ (r_K1_base / r_K2_base: pass 1 from the encoders, pass 2 from the wave's median)
ROBUST_PINS = [   # burst, wave median of pass 1 (mm), pass 1 offset, pass 2 offset (mm)
    ("p14-armwave-1104a-right-07-p09-10-0", "Right", (13.5082, -23.9855, 20.4139), (12.694, -25.896, 23.142),
     (12.728, -25.867, 23.167)),
    ("p14-armwave-1104b-left-17-pL04-22-0", "Left", (9.9047, -10.3041, 15.8224), (15.01, -17.072, 11.947),
     (14.627, -16.701, 11.859)),
]


@pytest.mark.parametrize("name,side,med,off1,off2", ROBUST_PINS)
def test_robust_passes_reproduce_the_reported_runs(p14, name, side, med, off1, off2):
    urdf = Urdf.load(p14 / "phantom_mk1.urdf")
    cap = readb.load_capture(p14 / "bursts" / f"{name}.npz", urdf, side)
    arm = readb.arm_model(urdf, side, cfg=meshfit.ROBUST)
    r1, _ = readb.match(arm, cap, meshfit.ROBUST, view=False)
    r2, _ = readb.match(arm, cap, meshfit.ROBUST, view=False, start_off=med)
    assert np.linalg.norm(np.subtract(r1["off"], off1)) < 0.05
    assert np.linalg.norm(np.subtract(r2["off"], off2)) < 0.05
    assert r1["clean"] and r2["clean"] and np.linalg.norm(np.subtract(r1["off"], r2["off"])) <= readb.SETTLED_MM
