"""calibtool.paths, calibtool.sessions (the stored-session registry) and calibtool.session (the Session data model).

Registry facts are checked without data; everything that reads a stored session is skipped without CALIBTOOL_DATA."""
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from calibtool import paths, sessions  # noqa: E402

CAL = Path(__file__).resolve().parents[2]

P21_TAKES = ["d050", "d050b", "d110", "d160", "d200", "dnear", "dleft", "dtiltleft"]
P10_TAKES = ["n6_075", "n6_110", "n6_160", "n6_200", "n6_far", "n6_litR", "n6_near", "n6_nearR", "n6_right",
             "n6_tiltL", "n6_tiltR"]


# ---------------------------------------------------------------- A1: paths


def test_scripts_dir_and_import_script():
    assert paths.SCRIPTS == CAL / "scripts"
    assert paths.CAL == CAL
    d16 = paths.import_script("16_split_distance")
    assert callable(d16.split)
    assert paths.script("66_validate_lens") == CAL / "scripts" / "66_validate_lens.py"


def test_data_root_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CALIBTOOL_DATA", str(tmp_path))
    assert paths.data_root() == tmp_path
    assert paths.have_data()


def test_data_root_unset_raises(monkeypatch):
    monkeypatch.delenv("CALIBTOOL_DATA", raising=False)
    assert not paths.have_data()
    with pytest.raises(paths.DataRootUnset):
        paths.data_root()


# ---------------------------------------------------------------- A1: registry


def test_registry_names():
    assert {"p21_1105", "p10_1106"} <= set(sessions.names())


def test_unknown_session_is_keyerror():
    with pytest.raises(KeyError):
        sessions.get("p99_0101")


def test_p21_registry_entry():
    s = sessions.get("p21_1105")
    assert isinstance(s, sessions.StoredSession)
    assert s.name == "p21_1105"
    assert s.robot == "phantom21"
    assert s.rig_path == CAL / "config" / "rig-phantom21.yaml"
    assert s.rig_path.is_file()
    assert s.take_order == P21_TAKES
    assert s.live_model == "RADTAN"
    r = s.recorded
    assert (r.fit_pairs_16, r.test_pairs_16) == (512, 499)
    assert r.fit_used_pairs == 503
    assert r.fit_rms_px == pytest.approx(0.367, abs=5e-4)
    assert r.baseline_mm == pytest.approx(43.92, abs=5e-3)
    assert r.new_verdict == "PASS"
    assert r.new_entries == {("left", (0.9, 1.3)): 0.11, ("centre", (0.4, 0.6)): 0.03, ("centre", (0.6, 0.9)): -0.03,
                             ("right", (0.4, 0.6)): 0.14, ("right", (0.6, 0.9)): 0.10, ("bottom", (0.6, 0.9)): -0.32,
                             ("bottom", (0.9, 1.3)): -0.17}
    assert r.live_verdict == "FAIL"
    assert r.live_n_validated == 6
    assert r.live_failing == {("right", (0.6, 0.9)): 0.54, ("bottom", (0.9, 1.3)): 3.99}
    assert (r.summary_new.median_pct, r.summary_new.iqr_pct, r.summary_new.rows_px) == (0.04, (-0.19, 0.28), 0.207)
    assert (r.summary_live.median_pct, r.summary_live.iqr_pct, r.summary_live.rows_px) == (0.77, (0.07, 1.77), 0.820)
    assert r.source == "p21/kit/final/final21.log"


def test_p10_registry_entry():
    s = sessions.get("p10_1106")
    assert s.robot == "phantom10"
    assert s.rig_path == CAL / "config" / "rig-phantom11.yaml"
    assert s.take_order == P10_TAKES
    assert s.live_model == "KB4T"
    r = s.recorded
    assert (r.fit_pairs_16, r.test_pairs_16) == (587, 572)
    assert r.fit_used_pairs == 578
    assert r.fit_rms_px == pytest.approx(0.417, abs=5e-4)
    assert r.baseline_mm == pytest.approx(44.06, abs=5e-3)
    assert r.new_verdict == "PASS" and len(r.new_entries) == 9
    assert r.live_verdict == "FAIL" and r.live_n_validated == 7 and len(r.live_failing) == 7
    assert (r.summary_new.median_pct, r.summary_new.iqr_pct, r.summary_new.rows_px) == (-0.04, (-0.45, 0.33), 0.279)
    assert (r.summary_live.median_pct, r.summary_live.iqr_pct, r.summary_live.rows_px) == (0.06, (-2.49, 1.59), 3.967)


def test_recorded_entries_use_step66_bins():
    import lenscells
    for name in ("p21_1105", "p10_1106"):
        r = sessions.get(name).recorded
        for cell, dist in list(r.new_entries) + list(r.live_failing):
            assert cell in lenscells.CELLS and dist in lenscells.DIST_BINS, (name, cell, dist)


def test_paths_need_data_root(monkeypatch):
    monkeypatch.delenv("CALIBTOOL_DATA", raising=False)
    s = sessions.get("p21_1105")
    with pytest.raises(paths.DataRootUnset):
        s.corner_files


@pytest.mark.parametrize("name", ["p21_1105", "p10_1106"])
def test_stored_files_resolve(name, data_root):
    s = sessions.get(name)
    files = s.corner_files
    assert [f.stem for f in files] == s.take_order
    assert all(f.is_file() and f.is_relative_to(data_root) for f in files)
    assert [d.name for d in s.raw_dirs] == s.take_order
    assert s.live_config.is_file() and s.live_config.is_relative_to(data_root)
    assert (data_root / s.recorded.source).is_file()


# ---------------------------------------------------------------- A2: Session (synthetic takes)

PAIR15B = ("img_hold", "img_t0", "img_t1", "img_yaw", "img_pitch", "img_ay", "img_ap")
DET = ("det_img", "det_cam", "det_tag", "det_k", "det_u", "det_v")


def synth_take(poses, t_start, per_pose=2, n_corners=16, seed=0):
    """a 15b-format corner dict: pose indices `poses` (gaps allowed: a pose without the board writes nothing),
    per_pose pairs each, n_corners per camera per pair"""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in PAIR15B + DET}
    t = t_start
    for p in poses:
        for _ in range(per_pose):
            j = len(out["img_t0"])
            out["img_hold"].append(p)
            out["img_t0"].append(t)
            out["img_t1"].append(t + 5)
            out["img_yaw"].append(0.01 * p)
            out["img_pitch"].append(-0.02 * p)
            out["img_ay"].append(1)
            out["img_ap"].append(0)
            t += 1000
            for cam in (0, 1):
                n = n_corners - cam                      # cam1 sees one corner fewer: counts must differ
                for c in range(n):
                    out["det_img"].append(j)
                    out["det_cam"].append(cam)
                    out["det_tag"].append(c // 4)
                    out["det_k"].append(c % 4)
                    out["det_u"].append(float(rng.uniform(0, 1280)))
                    out["det_v"].append(float(rng.uniform(0, 800)))
        t += 10_000
    ints = {"img_hold", "img_t0", "img_t1", "img_ay", "img_ap", "det_img", "det_cam", "det_tag", "det_k"}
    return {k: np.asarray(v, np.int64 if k in ints else float) for k, v in out.items()}


@pytest.fixture
def synth_files(tmp_path):
    """take B recorded first in time but listed second: take order and capture order differ"""
    a = synth_take([0, 1, 3], t_start=1_000_000, seed=1)       # max pose 3 -> take B's holds start at 4
    b = synth_take([0, 2], t_start=10, seed=2)
    fa, fb = tmp_path / "A.npz", tmp_path / "B.npz"
    np.savez_compressed(fa, **a)
    np.savez_compressed(fb, **b)
    return [("A", fa), ("B", fb)], {"A": a, "B": b}


def test_session_keeps_take_order_and_offsets_holds(synth_files):
    from calibtool.session import Session
    files, raw = synth_files
    s = Session.from_files(files)
    assert s.take_order == ["A", "B"]
    p = s.pairs
    assert list(p["take"]) == ["A"] * 6 + ["B"] * 4
    assert list(p["pose"]) == [0, 0, 1, 1, 3, 3, 0, 0, 2, 2]
    assert list(p["hold"]) == [0, 0, 1, 1, 3, 3, 4, 4, 6, 6]          # B offset by max(A's pose) + 1, as step 16
    assert s.hold_offset == {"A": 0, "B": 4}
    assert list(p["t0"]) == list(raw["A"]["img_t0"]) + list(raw["B"]["img_t0"])
    assert np.allclose(p["neck_yaw_rad"][:6], raw["A"]["img_yaw"])
    assert np.allclose(p["neck_pitch_rad"][6:], raw["B"]["img_pitch"])
    assert list(p["n_cam0"]) == [16] * 10 and list(p["n_cam1"]) == [15] * 10
    for k in ("t1", "approach_yaw", "approach_pitch"):
        assert len(p[k]) == 10
    assert s.n_pairs == 10


def test_session_holds_in_capture_order(synth_files):
    from calibtool.session import Session
    s = Session.from_files(synth_files[0])
    hs = list(s.holds())
    assert [h.hold for h in hs] == [4, 6, 0, 1, 3]                     # take B was recorded first
    assert [h.take for h in hs] == ["B", "B", "A", "A", "A"]
    assert [h.pose for h in hs] == [0, 2, 0, 1, 3]
    t0 = [h.t0 for h in hs]
    assert t0 == sorted(t0)
    assert sorted(i for h in hs for i in h.pairs) == list(range(s.n_pairs))
    assert all((s.pairs["hold"][h.pairs] == h.hold).all() for h in hs)


def test_to_npz_dict_round_trips_a_take(synth_files):
    from calibtool.session import Session
    files, raw = synth_files
    s = Session.from_files(files)
    for take in ("A", "B"):
        d = s.to_npz_dict(s.pairs["take"] == take, hold="pose")
        assert set(d) == set(PAIR15B + DET)
        for k in PAIR15B + DET:
            assert np.array_equal(d[k], raw[take][k]), (take, k)
            assert d[k].dtype == raw[take][k].dtype, (take, k)


def test_to_npz_dict_global_holds_and_subset(synth_files):
    from calibtool.session import Session
    s = Session.from_files(synth_files[0])
    m = np.zeros(s.n_pairs, bool)
    m[[1, 7, 8]] = True
    d = s.to_npz_dict(m)
    assert list(d["img_hold"]) == [0, 4, 6]
    assert list(d["img_t0"]) == list(s.pairs["t0"][m])
    assert sorted(set(d["det_img"].tolist())) == [0, 1, 2]
    assert (np.bincount(d["det_img"]) == 31).all()
    full = s.to_npz_dict()
    assert len(full["img_t0"]) == s.n_pairs and len(full["det_u"]) == 31 * s.n_pairs


def test_to_npz_dict_loads_in_stereofit(synth_files):
    pytest.importorskip("yaml")
    pytest.importorskip("cv2")
    import rigcfg
    import stereofit
    from calibtool.session import Session
    rig = rigcfg.load(sessions.get("p21_1105").rig_path)
    s = Session.from_files(synth_files[0])
    data = stereofit.Data(s.to_npz_dict(), rig.board)
    assert data.n == s.n_pairs
    assert list(data.hold) == list(s.pairs["hold"])


def test_index_of_t0(synth_files):
    from calibtool.session import Session
    s = Session.from_files(synth_files[0])
    t0 = s.pairs["t0"]
    assert list(s.index_of_t0(t0[[5, 0, 9]])) == [5, 0, 9]
    with pytest.raises(KeyError):
        s.index_of_t0([12345])


def test_duplicate_t0_refused(tmp_path):
    from calibtool.session import Session
    a = synth_take([0, 1], t_start=0)
    f1, f2 = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez_compressed(f1, **a)
    np.savez_compressed(f2, **a)
    with pytest.raises(ValueError):
        Session.from_files([("a", f1), ("b", f2)])


# ---------------------------------------------------------------- A2: Session (stored sessions)


@pytest.mark.parametrize("name", ["p21_1105", "p10_1106"])
def test_stored_session_round_trips_every_take(name, data_root):
    from calibtool.session import Session
    st = sessions.get(name)
    s = Session.load(st)
    assert s.take_order == st.take_order
    for take, f in zip(st.take_order, st.corner_files):
        raw = dict(np.load(f))
        d = s.to_npz_dict(s.pairs["take"] == take, hold="pose")
        for k in raw:
            assert np.array_equal(d[k], raw[k]), (take, k)


@pytest.mark.parametrize("name", ["p21_1105", "p10_1106"])
def test_step16_split_joins_back_by_t0(name, data_root):
    """16's split of the stored corner files (the recorded options) joins back to the Session by img_t0: every pair
    is found, its hold id is the Session's global hold (same offset rule), and every TEST pair gets its neck angles"""
    pytest.importorskip("cv2")
    pytest.importorskip("yaml")
    import rigcfg
    from calibtool.session import Session
    st = sessions.get(name)
    rig = rigcfg.load(st.rig_path)
    d16 = paths.import_script("16_split_distance")
    fit, test = d16.split([str(f) for f in st.corner_files], rig.board, rig.nominal_focal_px, rig.resolution,
                          3.0, 2)
    s = Session.load(st)
    assert (len(fit["img_t0"]), len(test["img_t0"])) == (st.recorded.fit_pairs_16, st.recorded.test_pairs_16)
    for part in (fit, test):
        idx = s.index_of_t0(part["img_t0"])
        assert np.array_equal(s.pairs["hold"][idx], part["img_hold"])
        assert np.isfinite(s.pairs["neck_yaw_rad"][idx]).all() and np.isfinite(s.pairs["neck_pitch_rad"][idx]).all()
    both = np.r_[fit["img_t0"], test["img_t0"]]
    assert len(set(both.tolist())) == len(both) <= s.n_pairs
