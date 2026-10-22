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
    assert r["pairs"] == 12 and r["roll_p10_p90"][0] <= r["roll_deg"] <= r["roll_p10_p90"][1]


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
    far = dc.check("KB4T", I, S.C_TRUE, board(S.C_TRUE, z=(1.5, 1.7)), S.BOARD)
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
