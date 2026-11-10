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


def test_deps_are_unique_relative_paths_of_modules_or_configs():
    d = deps()
    assert len(d) == len(set(d)), "duplicate entries"
    ok = lambda x: not x.startswith("/") and (x.endswith(".py") or (x.startswith("config/") and x.endswith(".yaml")))
    assert all(ok(x) for x in d), d


def test_rig_files_the_stored_sessions_and_profiles_use_are_listed():
    """devel has only rig.yaml/rig-phantom09/rig-phantom12: the rig files calibtool loads must travel with it (the
    devel-compat run on 9 Nov failed on a missing rig-phantom21.yaml)."""
    import re
    used = set()
    for src in list(PKG.glob("*.py")) + list((PKG / "config").glob("*.yaml")):
        used |= set(re.findall(r"rig-phantom[0-9a-z-]+\.yaml", src.read_text()))
    listed = {x.removeprefix("config/") for x in deps() if x.startswith("config/rig-")}
    assert used and used <= listed, f"rig files used but not in DEPS.txt: {sorted(used - listed)}"
