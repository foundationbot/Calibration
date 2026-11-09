"""calibtool tests: `import calibtool` from tools/calibration, the `replay` marker, the stored-data fixture.

`replay` tests replay a stored session through the slow pipeline steps (a step-45 fit takes 5-10 min) and need
CALIBTOOL_DATA (Moonstone: ~/calibtool/data). They are left out of every run that does not select them by marker
(`-m replay`), so the default suite (and devel's CI) never starts one. Stdlib only here: devel's pytest-only CI job
loads this file without numpy."""
import os
import sys
from pathlib import Path

import pytest

CAL = Path(__file__).resolve().parents[2]          # tools/calibration
if str(CAL) not in sys.path:
    sys.path.insert(0, str(CAL))


def pytest_configure(config):
    config.addinivalue_line("markers", "replay: slow replay of a stored calibration session (needs CALIBTOOL_DATA); "
                                       "runs only when selected with -m replay")


def pytest_collection_modifyitems(config, items):
    if "replay" in (config.getoption("markexpr", "") or ""):
        return
    dropped = [it for it in items if it.get_closest_marker("replay")]
    if dropped:
        items[:] = [it for it in items if not it.get_closest_marker("replay")]
        config.hook.pytest_deselected(items=dropped)


@pytest.fixture
def data_root():
    """the stored-session data root (CALIBTOOL_DATA); the test is skipped without it"""
    d = os.environ.get("CALIBTOOL_DATA")
    if not d:
        pytest.skip("CALIBTOOL_DATA unset: needs the stored session data (Moonstone ~/calibtool/data)")
    return Path(d).expanduser()
