"""Step 20's per-image corner extraction for both target types."""
import importlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
cv2 = pytest.importorskip("cv2")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import rigcfg  # noqa: E402
from test_calib_board import APRILGRID, CHECKER, _render_grid  # noqa: E402

coverage = importlib.import_module("20_check_coverage")


def test_aprilgrid_image_yields_every_corner():
    img, _ = _render_grid()
    pts = coverage.corners_from_image(rigcfg.board_from_target(APRILGRID), img)
    assert pts.shape == (140, 2)


def test_a_partial_aprilgrid_view_still_counts():
    # Partial views are the AprilGrid's point: tags at the image edge carry the
    # distortion information a checkerboard view would have to drop.
    img, origin = _render_grid()
    x_cut = origin[3][0]                                 # keep tag columns 0..2
    pts = coverage.corners_from_image(rigcfg.board_from_target(APRILGRID), img[:, :x_cut])
    assert pts is not None and pts.shape == (15 * 4, 2)


def test_blank_image_yields_nothing():
    blank = np.full((400, 600), 255, np.uint8)
    for target in (APRILGRID, CHECKER):
        assert coverage.corners_from_image(rigcfg.board_from_target(target), blank) is None


def test_checkerboard_image_yields_its_internal_corner_grid():
    sq, cols, rows = 30, 9, 12                           # 9x12 squares -> 8x11 corners
    img = np.full((rows * sq + 80, cols * sq + 80), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[40 + r * sq:40 + (r + 1) * sq, 40 + c * sq:40 + (c + 1) * sq] = 0
    pts = coverage.corners_from_image(rigcfg.board_from_target(CHECKER), img)
    assert pts.shape == (88, 2)
