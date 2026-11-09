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
