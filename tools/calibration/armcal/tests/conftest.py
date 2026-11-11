"""armcal tests: `import armcal` from tools/calibration; the stored phantom14 data fixture (skips without it)."""
import os
import sys
from pathlib import Path

import pytest

CAL = Path(__file__).resolve().parents[2]          # tools/calibration
if str(CAL) not in sys.path:
    sys.path.insert(0, str(CAL))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


@pytest.fixture
def p14():
    """$CALIBTOOL_DATA/p14arm: phantom14's arm waves of 3-4 Nov 2026 (bursts/, the C++ fusions fused_cpp/, the
    stored per-capture records base_w1/base_w2/base_L.json, the base and the deployed URDF, the twoarm M12 fit);
    the test is skipped without it"""
    d = os.environ.get("CALIBTOOL_DATA")
    if not d:
        pytest.skip("CALIBTOOL_DATA unset: needs the stored phantom14 arm data (Moonstone ~/calibtool/data/p14arm)")
    p = Path(d).expanduser() / "p14arm"
    if not (p / "base_w2.json").exists():
        pytest.skip(f"{p} has no stored phantom14 arm data")
    return p
