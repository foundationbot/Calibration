"""Every robot rig file loads, and its robot host matches the robot it names (rig-phantom21.yaml once carried
phantom11's host after being copied from it). phantom11 was renamed phantom10 on 2026-11-09: its rig files keep the old
file name (stored takes and configs use it) but point at the new host."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("yaml")
CAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAL / "scripts"))
import rigcfg  # noqa: E402

RENAMED = {"phantom11": "phantom10"}
RIGS = sorted((CAL / "config").glob("rig-phantom*.yaml"))


def test_robot_rigs_present():
    names = {p.stem for p in RIGS}
    assert {"rig-phantom09", "rig-phantom11", "rig-phantom15", "rig-phantom21"} <= names, names


@pytest.mark.parametrize("path", RIGS, ids=[p.stem for p in RIGS])
def test_rig_loads_and_host_matches(path):
    rig = rigcfg.load(path)
    robot = path.stem.removeprefix("rig-").split("-")[0]
    assert rig.robot["host"] == f"root@{RENAMED.get(robot, robot)}"
