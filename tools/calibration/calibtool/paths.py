"""Where calibtool finds the calibration package and the stored session data.

The pipeline steps live in tools/calibration/scripts/ as scripts with numeric names (16_split_distance.py); they are
imported, never copied: import_script("16_split_distance") puts the scripts folder on sys.path and imports by
importlib (a numeric name is not an identifier). Stored sessions (corner files, raw takes, recorded runs, robot OKVIS
configs) live under the data root, $CALIBTOOL_DATA (Moonstone: ~/calibtool/data). Stdlib only."""
import importlib
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent              # tools/calibration/calibtool
CAL = PKG.parent                                    # tools/calibration
SCRIPTS = CAL / "scripts"
CONFIG = CAL / "config"
DATA_ENV = "CALIBTOOL_DATA"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class DataRootUnset(RuntimeError):
    """the stored-session data root ($CALIBTOOL_DATA) is not set"""


def have_data() -> bool:
    return bool(os.environ.get(DATA_ENV))


def data_root() -> Path:
    """$CALIBTOOL_DATA (with ~ expanded); DataRootUnset when it is not set"""
    d = os.environ.get(DATA_ENV)
    if not d:
        raise DataRootUnset(f"${DATA_ENV} is not set: it names the stored-session data root (Moonstone: "
                            "~/calibtool/data)")
    return Path(d).expanduser()


def import_script(name: str):
    """a pipeline script by module name, e.g. import_script("16_split_distance")"""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(name)


def script(name: str) -> Path:
    """the path of pipeline script `name` (no .py), for running it as its own process"""
    return SCRIPTS / f"{name}.py"
