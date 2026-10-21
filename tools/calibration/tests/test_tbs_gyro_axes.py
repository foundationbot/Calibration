"""Step 82: the rotation of T_BS from the gyro and the neck joint rates alone."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
import neckfk  # noqa: E402
import necktake  # noqa: E402
import rigmath as rm  # noqa: E402

s82 = importlib.import_module("82_tbs_gyro_axes")
URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
R_SB = rm.rot_axis([0.3, -0.8, 0.5], 1.1)
BIAS = np.array([0.011, -0.016, -0.002])
T0 = 1_792_000_000_000_000_000


LSB = 0.0005326401442289352            # phantom12's gyro step [rad/s]


def _take(lag_s=0.0, flip=(1, 1), noise=0.003, seed=1, chain=None, gyro=np.eye(3), coupling=0.0,
          yaw_pitch=0.0, sag=0.0, lsb=None, pitch_only=False, gap=None, tremor=0.0):
    """5 s still, 10 s pitch sweep, 10 s yaw sweep at pitch yaw_pitch, 10 s mixed, 5 s still.
    flip = (yaw, pitch) encoder signs relative to the URDF axes; chain = the neck the head really
    has (default: the URDF's); gyro = the gyro's scale/axis matrix (identity: ideal); coupling = the
    other joint's share in each sweep (a hand never moves one joint alone); sag = pitch drift [rad/s]
    during the first still second-to-fifth (a held head sinking); lsb = gyro quantisation step;
    pitch_only = no yaw anywhere; gap = (t0, t1) [s] with no /joints samples; tremor = 1-sigma of
    gyro noise correlated over 0.1 s [rad/s] (a hand's tremor, which white noise understates)."""
    c = neckfk.NeckChain(URDF)
    true = chain or c
    def q(t):
        ps = np.where((t >= 5) & (t < 15), 0.4 * np.sin(2 * np.pi * 0.5 * (t - 5)), 0.0)
        ys = np.where((t >= 15) & (t < 25), 0.4 * np.sin(2 * np.pi * 0.4 * (t - 15)), 0.0)
        y = ys + coupling * ps + np.where((t >= 25) & (t < 35), 0.3 * np.sin(2 * np.pi * 0.3 * (t - 25)), 0.0)
        p = ps + coupling * ys + np.where((t >= 25) & (t < 35), 0.3 * np.sin(2 * np.pi * 0.45 * (t - 25)), 0.0)
        p = p + yaw_pitch * np.clip(t - 14.5, 0, 0.5) / 0.5 * (t < 25) + yaw_pitch * np.clip(25.5 - t, 0, 0.5) / 0.5 * (t >= 25)
        p = p + sag * np.clip(t - 1, 0, 4) * (t < 5) + sag * 4 * np.clip(5.5 - t, 0, 0.5) / 0.5 * (t >= 5)
        return (0 * y if pitch_only else y), p
    tj = np.arange(0, 40, 1 / 467)
    if gap is not None:
        tj = tj[(tj < gap[0]) | (tj > gap[1])]
    yj, pj = q(tj)
    j = necktake.Joints(T0 + (tj * 1e9).astype(np.int64), flip[0] * yj, flip[1] * pj)
    ti = np.arange(0.1, 39.9, 1 / 500)
    h = 1e-4
    (y0, p0), (y1, p1) = q(ti - lag_s), q(ti - lag_s + h)
    wB = true.omega_B(p0, (y1 - y0) / h, (p1 - p0) / h)
    rng = np.random.default_rng(seed)
    wS = wB @ R_SB.T @ gyro.T + BIAS + noise * rng.standard_normal(wB.shape)
    if tremor:
        k = np.ones(50) / np.sqrt(50)
        wS += tremor * np.stack([np.convolve(rng.standard_normal(len(ti)), k, mode="same") for _ in range(3)], 1)
    if lsb:
        wS = np.round(wS / lsb) * lsb
    return c, j, T0 + (ti * 1e9).astype(np.int64), wS


def test_kabsch_recovers_a_rotation():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 3))
    assert np.allclose(s82.kabsch(a, a @ R_SB.T), R_SB, atol=1e-9)
    planar = a * [1, 1, 0]                          # rank 2: the reflection must still be refused
    R = s82.kabsch(planar, planar @ R_SB.T)
    assert np.allclose(R, R_SB, atol=1e-9) and np.linalg.det(R) > 0


def test_right_config_passes_with_axes_and_bias():
    rep = s82.run(*_take(), R_SB)
    assert rep["verdict"] == "PASS" and rep["R_BS_angle_deg"] < 0.1 and rep["sigma_deg"] < 0.1
    assert np.allclose(rep["R_BS_measured"], R_SB.T, atol=2e-3)      # R_BS, not R_SB, in the report
    assert np.allclose(rep["bias_rad_s"], BIAS, atol=2e-3) and rep["half_turn_explained_by"] is None
    ax = rep["axes"]
    assert ax["pitch_axis_deg"] < 0.2 and ax["yaw_axis_deg"] < 0.2 and abs(ax["inter_axis_deg"] - 90) < 0.2
    assert rep["warnings"] == [] and rep["eig_ratio"] > 0.1


def test_hand_coupled_sweeps_leave_an_exact_neck_at_90_degrees():
    """3 % of the other joint in every sweep (a hand never moves one joint alone): averaging the raw
    samples puts the axes ~1.7 deg apart from 90 and blames the URDF; with the other joint's rate
    subtracted the exact neck reads 90 and no warning."""
    rep = s82.run(*_take(coupling=0.03, noise=0.0), R_SB)
    ax = rep["axes"]
    assert ax["pitch_coupling"] > 0.02 and abs(ax["inter_axis_deg"] - 90) < 0.1 and ax["pitch_axis_deg"] < 0.2
    assert not any("neck geometry" in w for w in rep["warnings"]), rep["warnings"]


def test_a_yaw_sweep_away_from_zero_pitch_still_measures_the_yaw_axis():
    rep = s82.run(*_take(yaw_pitch=0.25), R_SB)
    assert rep["axes"]["yaw_axis_deg"] < 0.2 and abs(rep["axes"]["inter_axis_deg"] - 90) < 0.2


def test_bias_from_sagging_holds_and_a_quantised_gyro():
    """The head sinks 0.1 deg/s while held, and the gyro reads in 0.53 mrad/s steps: the bias is
    what the encoders do not explain, so the sag is not taken for bias."""
    rep = s82.run(*_take(sag=np.radians(0.1) / 1.0, lsb=LSB, noise=0.0008), R_SB)
    assert np.allclose(rep["bias_rad_s"], BIAS, atol=3e-4) and rep["verdict"] == "PASS"


def test_one_joint_only_is_inconclusive_not_a_failure():
    rep = s82.run(*_take(pitch_only=True), R_SB)
    assert rep["verdict"] == "INCONCLUSIVE" and rep["eig_ratio"] < 0.1
    assert any("barely moved" in w for w in rep["warnings"])


def test_a_joints_gap_is_masked():
    """1 s without /joints in the middle of the yaw sweep: the rates there would be a straight line
    across the gap, so those IMU samples (and 20 ms either side) are left out."""
    rep = s82.run(*_take(gap=(19.0, 20.0), noise=0.0), R_SB)
    assert 500 <= rep["n_gap_masked"] <= 530 and rep["R_BS_angle_deg"] < 0.05


def test_the_verdict_needs_the_error_bar_inside_the_gate():
    assert s82.verdict(0.6, 0.1, 0.5) == "PASS"
    assert s82.verdict(0.9, 0.1, 0.5) == "INCONCLUSIVE"
    assert s82.verdict(1.3, 0.1, 0.5) == "FAIL"
    assert s82.verdict(0.2, 0.01, 0.05) == "INCONCLUSIVE"


def test_a_config_two_degrees_off_fails():
    rep = s82.run(*_take(), R_SB @ rm.rot_axis([0, 1, 0], np.radians(2.0)))
    assert rep["verdict"] == "FAIL" and abs(rep["R_BS_angle_deg"] - 2.0) < 0.15


def test_fit_finds_a_joint_clock_lag():
    c, j, ti, wS = _take(lag_s=0.012)
    r = s82.fit(c, j, ti, wS)
    assert abs(r["lag_s"] - 0.012) <= 0.004 and not r["lag_at_edge"]
    rep = s82.run(*_take(lag_s=0.08), R_SB)                    # beyond the +-50 ms search
    assert rep["lag_at_edge"] and any("edge of the search" in w for w in rep["warnings"])


def test_a_tilted_urdf_axis_is_named_not_blamed_on_T_BS():
    """The real pitch axis 0.5 deg off Head y: the joint fit blames T_BS (~0.3 deg); the angle
    between the measured axes shows it is the neck geometry."""
    true = neckfk.NeckChain(URDF)
    true.pitch = neckfk.Joint("NeckPitch", true.pitch.parent, true.pitch.child, true.pitch.origin,
                              rm.rot_axis([1, 0, 0], np.radians(0.5)) @ np.array([0, 1, 0.]))
    rep = s82.run(*_take(chain=true, noise=0.0), R_SB)
    assert abs(rep["axes"]["inter_axis_deg"] - 89.5) < 0.05
    assert any("neck geometry" in w for w in rep["warnings"])


def test_a_non_ideal_gyro_moves_the_answer_by_tenths_only():
    """1 % scale mismatch: ~0.4 deg, inside the 1 deg gate (why the gate is not 0.3 deg)."""
    rep = s82.run(*_take(gyro=np.diag([1.0, 1.01, 0.99])), R_SB)
    assert 0.1 < rep["R_BS_angle_deg"] < 0.6 and rep["verdict"] == "PASS"


@pytest.mark.parametrize("flip, which", [((1, -1), "pitch"), ((-1, 1), "yaw"), ((-1, -1), "yaw and pitch")])
def test_a_reversed_encoder_reads_as_a_half_turn(flip, which):
    """The gyro cannot tell a reversed encoder from an IMU mounted half a turn round: the fit is
    as good as with the right sign, R_SB comes out 180 deg off about one Head axis, and that axis
    names the joint."""
    right = s82.fit(*_take())
    c, j, ti, wS = _take(flip=flip)
    r = s82.fit(c, j, ti, wS)
    assert abs(r["residual_rad_s"] - right["residual_rad_s"]) < 1e-3
    assert rm.rot_angle_deg(r["R_SB"].T @ R_SB) > 179.0
    assert s82.flip_diagnosis(r["R_SB"], R_SB) == which


def test_the_error_bars_match_the_scatter_under_tremor():
    """8 takes with 0.1 s-correlated tremor and 3 % coupling: the R_BS error and the inter-axis error
    scatter as their sigmas say (rms error over mean sigma within 0.4-2.5), and an exact neck does
    not warn about the URDF."""
    e_r, s_r, e_i, s_i, warned = [], [], [], [], 0
    for seed in range(8):
        rep = s82.run(*_take(coupling=0.03, tremor=0.02, seed=seed), R_SB)
        e_r.append(rep["R_BS_angle_deg"])
        s_r.append(rep["sigma_deg"])
        e_i.append(rep["axes"]["inter_axis_deg"] - 90.0)
        s_i.append(rep["axes"]["inter_axis_sigma_deg"])
        warned += any("neck geometry" in w for w in rep["warnings"])
    for e, sg in ((e_r, s_r), (e_i, s_i)):
        ratio = np.sqrt(np.mean(np.square(e))) / np.mean(sg)
        assert 0.4 < ratio < 2.5, (ratio, e, sg)
    assert warned == 0
