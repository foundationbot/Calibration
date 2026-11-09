"""DEPS.txt is the devel-compatibility contract: every tools/calibration module calibtool imports is listed there, and
every listed module must exist in this tree (devel_compat.sh then checks which of them devel lacks)."""
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
CAL = PKG.parent


def deps():
    lines = (PKG / "DEPS.txt").read_text().splitlines()
    return [ln.split("#")[0].strip() for ln in lines if ln.split("#")[0].strip()]


def test_deps_listed():
    assert deps(), "DEPS.txt lists no modules"


def test_every_dep_exists_in_this_tree():
    missing = [d for d in deps() if not (CAL / d).is_file()]
    assert not missing, f"DEPS.txt names files that do not exist under tools/calibration: {missing}"


def test_deps_are_unique_relative_python_paths():
    d = deps()
    assert len(d) == len(set(d)), "duplicate entries"
    assert all(x.endswith(".py") and not x.startswith("/") for x in d), d
