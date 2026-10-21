"""Board handling in rigcfg: Kalibr checkerboard and aprilgrid targets."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import rigcfg  # noqa: E402

APRILGRID = {"target_type": "aprilgrid", "tagCols": 7, "tagRows": 5,
             "tagSize": 0.1, "tagSpacing": 0.2}
CHECKER = {"target_type": "checkerboard", "targetCols": 8, "targetRows": 11,
           "rowSpacingMeters": 0.06, "colSpacingMeters": 0.06}


def test_checkerboard_target_keeps_its_corner_grid():
    b = rigcfg.board_from_target(CHECKER)
    assert b.kind == "checkerboard"
    assert b.size == (8, 11)
    assert b.ambiguous_orientation  # 180 deg symmetric: step 30 must relabel


def test_aprilgrid_target_reads_kalibr_keys():
    b = rigcfg.board_from_target(APRILGRID)
    assert b.kind == "aprilgrid"
    assert (b.cols, b.rows, b.n_tags) == (7, 5, 35)
    assert b.pitch == pytest.approx(0.12)          # tagSize * (1 + tagSpacing)
    assert not b.ambiguous_orientation             # every tag carries its ID


def test_aprilgrid_tag_ids_run_along_rows():
    b = rigcfg.board_from_target(APRILGRID)
    assert b.tag_rc(0) == (0, 0)
    assert b.tag_rc(6) == (0, 6)
    assert b.tag_rc(8) == (1, 1)


def test_aprilgrid_same_corner_of_neighbouring_tags_is_one_pitch_apart():
    b = rigcfg.board_from_target(APRILGRID)
    for k in range(4):
        right = np.linalg.norm(b.obj_points(9)[k] - b.obj_points(8)[k])
        up = np.linalg.norm(b.obj_points(8 + 7)[k] - b.obj_points(8)[k])
        assert right == pytest.approx(0.12) and up == pytest.approx(0.12)


def test_aprilgrid_tag_corners_span_one_tag_edge():
    b = rigcfg.board_from_target(APRILGRID)
    c = b.obj_points(3)
    edges = [np.linalg.norm(c[(k + 1) % 4] - c[k]) for k in range(4)]
    assert edges == pytest.approx([0.1] * 4)


def _render_grid(cols=7, rows=5, side=60, gap=24, margin=40):
    """A synthetic AprilGrid: DICT_APRILTAG_36h11 markers with a 2-cell border,
    ids row-major. Returns the image and each tag's top-left pixel."""
    cv2 = pytest.importorskip("cv2")
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    h = 2 * margin + rows * side + (rows - 1) * gap
    w = 2 * margin + cols * side + (cols - 1) * gap
    img = np.full((h, w), 255, np.uint8)
    origin = {}
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        x, y = margin + c * (side + gap), margin + r * (side + gap)
        img[y:y + side, x:x + side] = cv2.aruco.generateImageMarker(d, i, side, borderBits=2)
        origin[i] = (x, y)
    return img, origin


def test_aprilgrid_detects_every_tag_corner_of_a_rendered_grid():
    img, origin = _render_grid()
    b = rigcfg.board_from_target(APRILGRID)
    found = b.detect(img)
    assert {tid for tid, _ in found} == set(range(35))
    assert len(found) == 140
    for i, (x, y) in origin.items():
        tl = np.array(found[(i, 0)])                # OpenCV order starts top-left
        assert np.allclose(tl, (x - 0.5, y - 0.5), atol=1.0)
