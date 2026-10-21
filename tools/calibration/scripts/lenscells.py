"""Image cells and distance bins of the KB4T lens steps (06 coverage, 46 repeatability, 66 validation).

A board, or a map point, is placed by its camera-0 pixel relative to the principal point, du = u - cx, dv = v - cy
(image y points down):
  corners  |du| >= 250 px and |dv| >= 150 px, one cell per quadrant (the corner definition the user approved on 20 Oct)
  centre   otherwise r = hypot(du, dv) < 150 px
  sides    otherwise left / right when |du| >= |dv|, else top / bottom
Distance bins: the board centre's depth z in camera 0, 0.4-0.6, 0.6-0.9 and 0.9-1.3 m. Beyond 1.3 m a board is in no
bin: phantom12's far boards (20 Oct, 1.4-2.0 m) fit a rigid board 2-4x worse than near ones and read -25..+40 % per hold.
The pixel limits are for 1280 x 800 heads (phantom12, mk13). numpy only, so the live coverage step can use it."""
import numpy as np

CELLS = ("top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right")
CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")
CENTRE_R_PX, CORNER_DU_PX, CORNER_DV_PX = 150.0, 250.0, 150.0
DIST_BINS = ((0.4, 0.6), (0.6, 0.9), (0.9, 1.3))


def cell_of(du, dv):
    """cell name per point (object array of str) from its offset to the principal point (px)"""
    du, dv = np.atleast_1d(np.asarray(du, float)), np.atleast_1d(np.asarray(dv, float))
    vert = np.where(dv < 0, "top", "bottom")
    horiz = np.where(du < 0, "left", "right")
    side = np.where(np.abs(du) >= np.abs(dv), horiz, vert)
    out = np.where(np.hypot(du, dv) < CENTRE_R_PX, "centre", side).astype(object)
    corner = (np.abs(du) >= CORNER_DU_PX) & (np.abs(dv) >= CORNER_DV_PX)
    out[corner] = np.char.add(np.char.add(vert[corner], "-"), horiz[corner])
    return out


def dist_bin(z):
    """distance bin index per board depth z (m); -1 outside every bin"""
    z = np.atleast_1d(np.asarray(z, float))
    out = np.full(z.shape, -1, int)
    for i, (lo, hi) in enumerate(DIST_BINS):
        out[(z >= lo) & (z < hi)] = i
    return out
