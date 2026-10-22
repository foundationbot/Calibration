"""driftcheck: camera 1's roll and pitch against a calibration from one still board's epipolar coplanarity; the depth
error at the board; coverage; the verdict (plan 2026-10-22-phantom12-drift-check.md)."""
import importlib
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = importlib.import_module("lens_synth")
dc = importlib.import_module("driftcheck")
I = [S.I0, S.I1]
CENTRE = [(643, 385)]


def turned(C, rotvec_deg):
    """C with camera 1 turned by rotvec (camera-0 axes, deg): R' = Exp(rotvec) R"""
    out = np.array(C, float)
    out[:3, :3] = S.sf.rotmat(np.radians([rotvec_deg]))[0] @ out[:3, :3]
    return out


def board(truth, n=12, seed=3, z=(0.7, 0.9), targets=CENTRE):
    return S.views(I, truth, n, seed=seed, z_range=z, noise=0.15, targets=targets, jitter=(40, 30))


def test_a_planted_roll_and_pitch_are_read_back():
    d = board(turned(S.C_TRUE, [0.03, 0.0, -0.36]))
    r = dc.roll_pitch("KB4T", I, S.C_TRUE, d, S.BOARD)
    assert r["roll_deg"] == pytest.approx(-0.36, abs=0.02) and r["pitch_deg"] == pytest.approx(0.03, abs=0.01)
    assert r["pairs"] == 12 and r["roll_repeat_p10_p90"][0] <= r["roll_deg"] <= r["roll_repeat_p10_p90"][1]


def test_the_true_calibration_reads_zero_and_yaw_is_not_mistaken_for_roll():
    for truth in (S.C_TRUE, turned(S.C_TRUE, [0.0, 0.10, 0.0])):      # a 0.1 deg yaw: not a roll, not a pitch
        r = dc.roll_pitch("KB4T", I, S.C_TRUE, board(truth), S.BOARD)
        assert abs(r["roll_deg"]) < 0.02 and abs(r["pitch_deg"]) < 0.01


def test_the_verdict_flags_the_19_oct_roll_and_passes_the_truth():
    ok = dc.check("KB4T", I, S.C_TRUE, board(S.C_TRUE), S.BOARD)
    bad = dc.check("KB4T", I, S.C_TRUE, board(turned(S.C_TRUE, [0, 0, -0.36])), S.BOARD)
    assert ok["verdict"] == "OK" and bad["verdict"] == "DRIFTED" and any("roll" in w for w in bad["reasons"])
    json.dumps(ok, allow_nan=False)                                   # the report is strict JSON


def test_a_pitch_over_003_deg_and_a_depth_error_over_1_percent_are_drift():
    p = dc.check("KB4T", I, S.C_TRUE, board(turned(S.C_TRUE, [0.05, 0, 0])), S.BOARD)
    long = S.C_TRUE.copy()
    long[:3, 3] *= 1.02                                              # baseline 2 % long: depth 2 % off, no roll
    z = dc.check("KB4T", I, S.C_TRUE, board(long), S.BOARD)
    assert p["verdict"] == "DRIFTED" and any("pitch" in w for w in p["reasons"])
    assert z["verdict"] == "DRIFTED" and any("depth" in w for w in z["reasons"]) and abs(z["roll_deg"]) < 0.02


def test_a_board_in_a_strip_at_the_image_top_is_no_result():
    d = board(S.C_TRUE, targets=[(643, 90)])                         # the 18:11 geometry
    r = dc.check("KB4T", I, S.C_TRUE, d, S.BOARD)
    assert r["verdict"] == "NO RESULT" and any("centre" in w or "span" in w for w in r["reasons"])


def test_too_few_pairs_is_no_result_and_a_board_outside_the_depth_band_skips_only_the_depth(monkeypatch):
    few = dc.check("KB4T", I, S.C_TRUE, board(S.C_TRUE, n=5), S.BOARD)
    assert few["verdict"] == "NO RESULT"
    long = S.C_TRUE.copy()
    long[:3, 3] *= 1.02                                              # a depth error the band now hides
    monkeypatch.setattr(dc, "DEPTH_BAND_M", (0.4, 0.6))              # phantom12's board fails the span rule before
    out = dc.check("KB4T", I, S.C_TRUE, board(long), S.BOARD)       # 1.3 m (400 px at ~1.19 m), so the band is
    assert out["verdict"] == "OK" and out["depth_judged"] is False   # tested by narrowing it


def test_a_far_board_is_no_result_by_its_span():
    far = dc.check("KB4T", I, S.C_TRUE, board(S.C_TRUE, z=(2.1, 2.3)), S.BOARD)
    assert far["verdict"] == "NO RESULT" and any("span" in w for w in far["reasons"])


m67 = importlib.import_module("67_drift_check")


def test_load_okvis_reads_the_calibration_okvis_runs(tmp_path):
    """an OKVIS kb4t config (step 71's surgery) gives back the fit's lenses and C = inv(T_SC0) T_SC1"""
    m71 = importlib.import_module("71_export_lens")
    tpl_mod = importlib.import_module("test_calib_make_yaml")
    tpl = tmp_path / "template.yaml"
    tpl.write_text(tpl_mod.TEMPLATE.replace("{tbs}", tpl_mod.TBS))
    C = turned(S.C_TRUE, [0.0, 0.0, -0.36])
    cfg = tmp_path / "okvis.yaml"
    cfg.write_text(m71.okvis_text(tpl, S.rig12(), I, C, ["test"], kb4t=True))
    model, intr, C2 = dc.load_okvis(cfg)
    assert model == "KB4T" and all(np.allclose(a, b, rtol=1e-14, atol=0) for a, b in zip(intr, I))
    assert np.allclose(C2, C, atol=1e-12)


@pytest.mark.parametrize("truth, targets, code", [(None, CENTRE, 0), ("roll", CENTRE, 1), (None, [(643, 90)], 2)])
def test_the_cli_exits_0_ok_1_drifted_2_no_result_and_writes_the_report(tmp_path, truth, targets, code, capsys):
    C = turned(S.C_TRUE, [0, 0, -0.36]) if truth == "roll" else S.C_TRUE
    np.savez(tmp_path / "c.npz", **board(C, targets=targets))
    S.write_calibration(tmp_path / "cal.json", "KB4T", I, S.C_TRUE)
    out = tmp_path / "r.json"
    assert m67.main(["--corners", str(tmp_path / "c.npz"), "--calibration", str(tmp_path / "cal.json"),
                     "--rig", str(S.RIG12), "--out", str(out)]) == code
    rep = json.loads(out.read_text())
    assert rep["verdict"] == ("OK", "DRIFTED", "NO RESULT")[code] and rep["calibration"]["source"].endswith("cal.json")
    assert any(ln.startswith(rep["verdict"] + ":") for ln in capsys.readouterr().out.splitlines())


def test_a_session_is_dumped_with_one_long_hold(tmp_path, monkeypatch):
    seen = {}
    d = board(S.C_TRUE)

    def fake_dump(path, b, out, still_px=1.0, per_hold=3, **kw):
        seen.update(path=str(path), per_hold=per_hold)
        np.savez(out, **d)
        return len(d["img_t0"])

    monkeypatch.setattr(m67, "_dump", fake_dump)
    S.write_calibration(tmp_path / "cal.json", "KB4T", I, S.C_TRUE)
    assert m67.main(["--session", str(tmp_path / "session.mcap"), "--calibration", str(tmp_path / "cal.json"),
                     "--rig", str(S.RIG12), "--out", str(tmp_path / "r.json")]) == 0
    assert seen == {"path": str(tmp_path / "session.mcap"), "per_hold": m67.PER_HOLD} and m67.PER_HOLD == 30


# --- the code review's I1-I4 and minors (plan revision 2): a truly still board, every limit pinned on both sides ---

def still(truth, n=30, target=(643, 385), z=0.8, seed=5):
    """ONE board pose and n noise realisations of it: what a 30 s still recording gives"""
    return S.views(I, truth, 1, seed=seed, z_range=(z, z + 0.001), noise=0.2, per_hold=n, targets=[target],
                   jitter=(0, 0))


@pytest.mark.parametrize("rot, expect", [([0, 0, 0.08], "OK"), ([0, 0, 0.12], "DRIFTED"), ([0, 0, -0.12], "DRIFTED"),
                                         ([0.02, 0, 0], "OK"), ([0.04, 0, 0], "DRIFTED"), ([-0.04, 0, 0], "DRIFTED")])
def test_the_roll_and_pitch_limits_on_both_sides(rot, expect):
    r = dc.check("KB4T", I, S.C_TRUE, still(turned(S.C_TRUE, rot)), S.BOARD)
    assert r["verdict"] == expect, (rot, r["roll_deg"], r["pitch_deg"], r["reasons"])


@pytest.mark.parametrize("k, expect", [(1.008, "OK"), (1.012, "DRIFTED")])
def test_the_depth_limit_on_both_sides(k, expect):
    C = S.C_TRUE.copy()
    C[:3, 3] *= k
    r = dc.check("KB4T", I, S.C_TRUE, still(C), S.BOARD)
    assert r["verdict"] == expect and r["depth_judged"], (k, r.get("depth_pct"), r["reasons"])


def test_the_centre_box_is_150_by_100_px():
    inside = dc.check("KB4T", I, S.C_TRUE, still(S.C_TRUE, target=(643 + 120, 385 + 80)), S.BOARD)
    right = dc.check("KB4T", I, S.C_TRUE, still(S.C_TRUE, target=(643 + 200, 385)), S.BOARD)
    low = dc.check("KB4T", I, S.C_TRUE, still(S.C_TRUE, target=(643, 385 + 150)), S.BOARD)
    assert inside["verdict"] == "OK" and right["verdict"] == low["verdict"] == "NO RESULT"


def test_nine_pairs_are_no_result_and_ten_are_judged():
    assert dc.check("KB4T", I, S.C_TRUE, still(S.C_TRUE, n=9), S.BOARD)["verdict"] == "NO RESULT"
    assert dc.check("KB4T", I, S.C_TRUE, still(S.C_TRUE, n=10), S.BOARD)["verdict"] == "OK"


def test_an_unconverged_roll_fit_is_no_result_and_an_unconverged_depth_is_not_judged(monkeypatch):
    d = still(S.C_TRUE)
    ls = importlib.import_module("lensscore")
    monkeypatch.setattr(ls, "solved", lambda solves: False)
    r = dc.check("KB4T", I, S.C_TRUE, d, S.BOARD)
    assert r["verdict"] == "OK" and r["depth_judged"] is False and "converge" in r["depth_note"]
    monkeypatch.undo()
    real = dc._fit
    monkeypatch.setattr(dc, "_fit", lambda *a: (real(*a)[0], False))
    r = dc.check("KB4T", I, S.C_TRUE, d, S.BOARD)
    assert r["verdict"] == "NO RESULT" and any("converge" in w for w in r["reasons"])


def test_load_okvis_with_a_turned_camera_0_and_the_okvis_order(tmp_path):
    m70 = importlib.import_module("70_make_mk13_yaml")
    m71 = importlib.import_module("71_export_lens")
    tpl_mod = importlib.import_module("test_calib_make_yaml")
    tpl = tmp_path / "template.yaml"
    tpl.write_text(tpl_mod.TEMPLATE.replace("{tbs}", tpl_mod.TBS))
    C = turned(S.C_TRUE, [0.0, 0.0, -0.36])
    T0 = S.stereo_pose([-91.0, 2.0, -88.0], [0.03, -0.01, 0.02])     # camera 0 in the IMU: not the identity
    text = m71.okvis_text(tpl, S.rig12(), I, C, ["test"], kb4t=True)
    cfg = tmp_path / "okvis.yaml"
    cfg.write_text(m70.replace_tsc(text, [T0, T0 @ C]))
    model, intr, C2 = dc.load_okvis(cfg)
    assert np.allclose(C2, C, atol=1e-9)
    with pytest.raises(ValueError, match="okvis_order"):
        dc.load_okvis(cfg, okvis_order=[1, 0])


def test_errors_exit_3_never_1(tmp_path):
    np.savez(tmp_path / "c.npz", **still(S.C_TRUE))
    S.write_calibration(tmp_path / "cal.json", "KB4T", I, S.C_TRUE)
    charuco = Path(__file__).resolve().parent.parent / "config" / "rig.yaml"     # the package default: a ChArUco board
    args = ["--corners", str(tmp_path / "c.npz"), "--calibration", str(tmp_path / "cal.json")]
    assert m67.main(args + ["--rig", str(charuco)]) == 3                          # not an AprilGrid: refused
    assert m67.main(["--corners", str(tmp_path / "missing.npz"), "--calibration", str(tmp_path / "cal.json"),
                     "--rig", str(S.RIG12)]) == 3                                  # any error: 3


def test_the_u_span_limit_is_250_px_and_separate_from_the_v_limit():
    """rev 3: the real board spans 280-394 raw px at 0.7-1.2 m (partly cut or turned), so 400 px was unreachable;
    250 px keeps roll's lever arm. Synthetic face-on board: ~280 x ~200 px at 1.62 m, ~220 x ~155 px at 2.1 m"""
    ok = dc.check("KB4T", I, S.C_TRUE, still(S.C_TRUE, z=1.62), S.BOARD)
    far = dc.check("KB4T", I, S.C_TRUE, still(S.C_TRUE, z=2.1), S.BOARD)
    assert ok["span_px"][0] >= 250 > far["span_px"][0] and far["span_px"][1] >= 150, (ok["span_px"], far["span_px"])
    assert ok["verdict"] == "OK" and ok["depth_judged"] is False                    # 1.62 m: beyond the depth band
    assert far["verdict"] == "NO RESULT" and any("span" in w for w in far["reasons"])
