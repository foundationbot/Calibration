"""lenscells: the image cells (20 Oct corner definition) and distance bins of the KB4T lens steps."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import lenscells as lc  # noqa: E402


def test_corners_by_quadrant_image_y_down():
    got = lc.cell_of([-300, 300, -300, 300], [-200, -200, 200, 200]).tolist()
    assert got == ["top-left", "top-right", "bottom-left", "bottom-right"]


def test_corner_needs_both_offsets():
    assert lc.cell_of(250, 150)[0] == "bottom-right"          # both limits are inclusive
    assert lc.cell_of(249, 200)[0] == "right"                 # |du| short of 250: a side, |du| >= |dv|
    assert lc.cell_of(400, 149)[0] == "right"                 # |dv| short of 150


def test_centre_then_sides():
    assert lc.cell_of(100, 100)[0] == "centre"                # r 141 < 150
    assert lc.cell_of(0, -150)[0] == "top"                    # r = 150 is not the centre
    assert lc.cell_of(-160, 0)[0] == "left"
    assert lc.cell_of(0, 160)[0] == "bottom"
    assert lc.cell_of(120, -120)[0] == "right"                # a tie goes to left / right


def test_every_cell_name_is_listed():
    du, dv = np.meshgrid(np.arange(-640, 641, 20), np.arange(-400, 401, 20))
    assert set(lc.cell_of(du.ravel(), dv.ravel())) == set(lc.CELLS)


def test_distance_bins():
    assert lc.dist_bin([0.39, 0.4, 0.59, 0.6, 0.9, 1.29, 1.3]).tolist() == [-1, 0, 0, 1, 2, 2, -1]
