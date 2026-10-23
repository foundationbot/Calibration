"""Arm session, Task 2: what the cameras see of the palm plate, how much a pose tells the fit, and the pose list
(armplan.py; decisions T2-2..T2-9 of docs/superpowers/plans/2026-10-22-phantom12-arm-session.md rev 3)."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
import armchain  # noqa: E402
import armplan  # noqa: E402
import rigcfg  # noqa: E402
import rigmath as rm  # noqa: E402

URDF = ROOT / "config/foundation_rig/20250206_phantom_mk1.urdf"
INTR = np.array([565.0, 564.0, 640.0, 400.0, -0.008, 0.012, -0.0005, -0.0014, 0.0008, -0.0011])
GRID = rigcfg.AprilGrid(2, 2, 0.045, 0.2)


def _cam(T_TC=np.eye(4)):
    return armplan.Camera(T_TC=T_TC, model="KB4T", intr=INTR, size=(1280, 800))


def test_plate_points_are_the_grids_16_corners():
    P, keys = armplan.plate_points(GRID)
    assert P.shape == (16, 3) and len(keys) == 16 and len(set(keys)) == 16
    assert np.isclose(P[:, 0].max() - P[:, 0].min(), 0.054 + 0.045)   # two tags and one gap across


def test_a_plate_square_to_the_camera_faces_it_and_its_tags_are_measured():
    """A plate 0.5 m ahead, square to the optical axis, its printed side towards the camera: facing 0 deg, range
    0.5 m, a 45 mm tag ~ f * 0.045 / 0.5 = 51 px wide, and the margin is its distance to the nearest image edge."""
    P, _ = armplan.plate_points(GRID)
    c = P.mean(0)
    T_CP = rm.transform(np.diag([1.0, -1.0, -1.0]), np.array([0, 0, 0.5]) - np.diag([1.0, -1.0, -1.0]) @ c)
    v = armplan.view([_cam()], T_CP[None], P, GRID)
    assert v["facing_deg"].shape == (1, 1)
    assert v["facing_deg"][0, 0] < 1e-6
    assert np.isclose(v["range_m"][0, 0], 0.5, atol=1e-9)
    assert 49 < v["tag_px"][0, 0] < 52
    uv = v["uv"][0, 0]
    assert np.isclose(v["margin_px"][0, 0], min(uv[:, 0].min(), 1279 - uv[:, 0].max(), uv[:, 1].min(),
                                                 799 - uv[:, 1].max()))
    assert armplan.visible(v).tolist() == [True]


def test_the_back_of_the_plate_and_a_far_or_oblique_plate_are_not_visible():
    P, _ = armplan.plate_points(GRID)
    c = P.mean(0)
    def at(R, z):
        return rm.transform(R, np.array([0, 0, z]) - R @ c)
    flip = np.diag([1.0, -1.0, -1.0])
    poses = np.stack([at(np.eye(3), 0.5),                                  # the blank back towards the camera
                      at(flip, 1.0),                                       # too far (and its tags too small)
                      at(flip @ rm.rot_axis((0, 1, 0), np.radians(60)), 0.5)])  # 60 deg oblique
    v = armplan.view([_cam()], poses, P, GRID)
    assert v["facing_deg"][0, 0] > 170 and np.isclose(v["facing_deg"][2, 0], 60, atol=1e-6)
    assert armplan.visible(v).tolist() == [False, False, False]


Q0 = np.radians([-40, -5, 45, -90, -135, 20, 20])
X_SYN = rm.transform(rm.rot_axis((1, 0, 0), np.pi / 2), (0.03, 0.02, -0.2))


def _chain_and_cams():
    """The real arm, and a stereo pair 0.5 m in front of the plate's printed side at the pose Q0."""
    arm = armchain.ArmChain(URDF)
    P, _ = armplan.plate_points(GRID)
    T_TP = arm.T_base_tip(Q0) @ X_SYN
    flip = np.diag([1.0, -1.0, -1.0])                      # camera axes in the plate: looking at the print
    T_PC = rm.transform(flip, P.mean(0) - flip @ np.array([0, 0, 0.5]))
    T01 = rm.transform(rm.rot_axis((0, 1, 0), 0.002), (0.0443, 0, 0))
    return arm, [_cam(T_TP @ T_PC), _cam(T_TP @ T_PC @ T01)]


def test_the_first_and_last_joint_offsets_are_unobservable_and_the_inner_five_are():
    """T2-2: at one neck pose a shoulder-pitch zero offset is ΔT and a gripper-yaw one is X; with all 7 offsets the
    information has a 2-dimensional null space, with the inner 5 it has none."""
    arm, cams = _chain_and_cams()
    P, _ = armplan.plate_points(GRID)
    X = X_SYN
    Q = np.random.default_rng(3).uniform(-0.15, 0.15, (12, 7)) + Q0
    assert np.isfinite(armplan.pixels(arm, cams, X, Q, P)).all()
    full = armplan.information(arm, cams, X, Q, P, sigma_px=0.5, offsets=range(7)).sum(0)
    inner = armplan.information(arm, cams, X, Q, P, sigma_px=0.5).sum(0)
    assert full.shape == (19, 19) and inner.shape == (17, 17)
    ev_full = np.linalg.eigvalsh(full)
    ev_inner = np.linalg.eigvalsh(inner)
    assert ev_full[1] / ev_full[-1] < 1e-9          # two zero directions
    assert ev_inner[0] / ev_inner[-1] > 1e-9        # none


def test_information_matches_a_finite_difference_of_the_pixels():
    arm, cams = _chain_and_cams()
    P, _ = armplan.plate_points(GRID)
    X = X_SYN
    q = Q0[None]
    J = armplan.jacobian(arm, cams, X, q, P)[0]
    p = np.zeros(17)
    p[14] = 1e-4                                    # the elbow's offset (inner joint 3 -> parameter 12 + 2)
    d = (armplan.pixels(arm, cams, X, q, P, p) - armplan.pixels(arm, cams, X, q, P, -p))[0] / 2e-4
    assert np.allclose(J[:, 14], d, atol=1e-3)
    assert np.allclose(armplan.information(arm, cams, X, q, P, sigma_px=2.0)[0], J.T @ J / 4.0)


def test_plate_sigma_shrinks_with_closer_plates():
    arm, cams = _chain_and_cams()
    P, _ = armplan.plate_points(GRID)
    flip = np.diag([1.0, -1.0, -1.0])
    near = rm.transform(flip, (0, 0, 0.4) - flip @ P.mean(0))
    far = rm.transform(flip, (0, 0, 0.8) - flip @ P.mean(0))
    s = armplan.plate_sigma_mm([_cam(), _cam(rm.transform(np.eye(3), (0.0443, 0, 0)))], np.stack([near, far]), P,
                               sigma_px=0.5)
    assert 0 < s[0] < s[1] < 5


def test_select_meets_the_coverage_rules_before_the_information():
    rng = np.random.default_rng(4)
    n = 300
    J = rng.normal(size=(n, 6, 4))
    info = np.einsum("nij,nik->njk", J, J)
    cell = rng.integers(0, 9, n)
    third = rng.integers(0, 3, n)
    Q = rng.uniform(-1.5, 1.5, (n, 7))
    sel = armplan.select(info, cell, third, Q, n_pick=12, min_per_cell=1, min_bottom=2, min_per_third=3,
                         prior=np.eye(4) * 1e-6)
    assert len(sel) == len(set(sel)) == 12
    assert set(cell[sel]) == set(range(9))
    assert np.isin(cell[sel], armplan.BOTTOM_CORNERS).sum() >= 2
    assert all((third[sel] == t).sum() >= 3 for t in range(3))


def test_select_skips_a_cell_the_candidates_cannot_fill():
    rng = np.random.default_rng(5)
    info = np.stack([np.eye(3)] * 40)
    cell = np.r_[np.zeros(39, int), 4]                         # cell 4 has one candidate, cells 1-3, 5-8 none
    sel = armplan.select(info, cell, np.zeros(40, int), rng.uniform(-1.5, 1.5, (40, 7)), n_pick=5,
                         min_per_cell=2, min_bottom=0, min_per_third=0, prior=np.eye(3))
    assert len(sel) == 5      # cell 4 reaches fewer than min_per_cell candidates: not required, no failure


def test_select_keeps_poses_apart_in_joint_space():
    info = np.stack([np.eye(2)] * 3)
    Q = np.radians([[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0, 0]])
    sel = armplan.select(info, np.zeros(3, int), np.zeros(3, int), Q, n_pick=3, min_per_cell=0, min_bottom=0,
                         min_per_third=0, prior=np.eye(2), min_sep_deg=5.0)
    assert sorted(sel) in ([0, 2], [1, 2])


def test_tour_starts_and_ends_at_the_start_and_two_opt_never_worsens():
    rng = np.random.default_rng(6)
    Q = np.radians(rng.uniform(-60, 60, (15, 7)))
    start = np.zeros(7)
    order = armplan.tour(Q, start)
    assert sorted(order) == list(range(15))
    nn = armplan.tour(Q, start, two_opt=False)
    assert armplan.tour_cost(Q, start, order) <= armplan.tour_cost(Q, start, nn) + 1e-12


def test_tour_avoids_a_forbidden_move():
    Q = np.radians([[10, 0, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0, 0], [30, 0, 0, 0, 0, 0, 0]])
    bad = {(0, 1), (1, 0)}
    order = armplan.tour(Q, np.zeros(7), allowed=lambda a, b: (a, b) not in bad)
    assert all({order[i], order[i + 1]} != {0, 1} for i in range(len(order) - 1))


def test_stops_hold_out_every_fourth_and_add_the_repeats_with_their_pre_poses():
    Q = np.radians(np.arange(26)[:, None] * np.array([2.0, 1, 1, -2, 0, 0, 0]) + np.array([-30, 0, 20, -60, -135, 20, 20]))
    start = np.radians([-42.6, -4.8, 47.5, -92.3, -135.7, 21.2, 19.8])
    order = list(range(26))
    st = armplan.stops(Q, order, start, n_same=3, n_opp=4, pre_deg=6.0)
    kinds = [s["kind"] for s in st]
    assert kinds.count("fit") == 20 and kinds.count("heldout") == 6
    assert kinds.count("repeat_same") == 3 and kinds.count("repeat_opp") == 4
    assert kinds.count("pre") == 7 and kinds[-1] == "return"
    ho = [s["pose"] for s in st if s["kind"] == "heldout"]
    assert ho == [3, 7, 11, 15, 19, 23]
    for i, s in enumerate(st):
        if s["kind"].startswith("repeat"):
            pre, k = st[i - 1], s["pose"]
            assert pre["kind"] == "pre"
            first = next(j for j, t in enumerate(st) if t.get("pose") == k)
            prev = st[first - 1]["q"]
            side = np.sign(np.array(s["q"])[:4] - np.array(prev)[:4])
            step = np.array(s["q"])[:4] - np.array(pre["q"])[:4]
            want = side if s["kind"] == "repeat_same" else -side
            assert np.allclose(step, want * np.radians(6.0))


def test_ramp_is_six_seconds_or_the_largest_change_at_point_three_rad_per_second():
    a = np.zeros(7)
    assert armplan.ramp_s(a, np.radians([10, 0, 0, 0, 0, 0, 0])) == 6.0
    assert np.isclose(armplan.ramp_s(a, np.array([0, 0, 0, 3.0, 0, 0, 0])), 10.0)


def test_list_gate_names_each_failing_parameter():
    info = np.diag([1e8] * 6 + [1e8] * 6 + [1e8] * 4 + [1.0])     # the last joint offset barely seen
    g = armplan.list_gate(info, plate_sigma_mm=np.array([0.3, 0.6]))
    assert not g["pass"]
    assert g["fails"] == ["offset RightWristRoll"]
    ok = armplan.list_gate(np.diag([1e10] * 17), plate_sigma_mm=np.array([0.3]))
    assert ok["pass"] and ok["fails"] == []


def test_stops_repeat_only_the_poses_allowed():
    Q = np.radians(np.arange(26)[:, None] * np.array([2.0, 1, 1, -2, 0, 0, 0]) + np.array([-30, 0, 20, -60, -135, 20, 20]))
    allowed = {0, 1, 2, 4, 5, 6, 8, 9}
    st = armplan.stops(Q, list(range(26)), np.zeros(7), repeatable=allowed)
    reps = [s["pose"] for s in st if s["kind"].startswith("repeat")]
    assert len(reps) == 7 and set(reps) <= allowed
    with pytest.raises(ValueError, match="repeat"):
        armplan.stops(Q, list(range(26)), np.zeros(7), repeatable={0, 1})


def test_workspace_keeps_the_arm_low_and_forward():
    arm = armchain.ArmChain(URDF)
    low = np.radians([-42.6, -4.8, 47.5, -92.3, -135.7, 21.2, 19.8])    # 22 Oct's held pose: forearm across the chest
    up = np.radians([-160.0, -4.8, 0.0, -10.0, 0.0, 0.0, 0.0])          # the arm raised overhead
    back = np.radians([30.0, -4.8, 0.0, -90.0, 0.0, 0.0, 0.0])          # shoulder pitched back
    rolled = np.radians([-30.0, -100.0, 0.0, -90.0, 0.0, 0.0, 0.0])     # shoulder rolled out past 60 deg
    assert armplan.workspace_ok(arm, np.stack([low, up, back, rolled])).tolist() == [True, False, False, False]


def _paste_grid(img, cols, rows, side, gap, x0, y0, ids):
    """Kalibr-layout tags (tag 0 bottom-left, each marker rotated 180 deg as Kalibr prints them) at (x0, y0) = the
    grid's top-left pixel. -> {id: (x, y) of its top-left pixel}."""
    cv2 = pytest.importorskip("cv2")
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    at = {}
    for i in ids:
        r, c = divmod(i, cols)
        x, y = x0 + c * (side + gap), y0 + (rows - 1 - r) * (side + gap)
        img[y:y + side, x:x + side] = cv2.rotate(cv2.aruco.generateImageMarker(d, i, side, borderBits=2),
                                                 cv2.ROTATE_180)
        at[i] = (x, y)
    return at


def _scene(plate=True):
    img = np.full((800, 1280), 255, np.uint8)
    board = _paste_grid(img, 7, 2, 40, 12, 300, 20, range(14))               # the 7x5 board's bottom rows
    pl = _paste_grid(img, 2, 2, 90, 18, 250, 400, range(4)) if plate else {}  # the palm plate, nearer, larger
    return img, board, pl


def test_the_plain_detector_mixes_the_boards_tags_into_the_plate():
    img, board, _ = _scene()
    found = GRID.detect(img)
    assert any(v < 200 for (_, _), (u, v) in found.items())          # a corner taken from the board strip


def test_detect_plate_keeps_only_the_plates_tags():
    img, board, pl = _scene()
    found = armplan.detect_plate(GRID, img)
    assert len(found) == 16
    assert all(v > 380 for (_, _), (u, v) in found.items())
    for t, (x, y) in pl.items():
        us = [found[(t, k)][0] for k in range(4)]
        assert min(us) > x - 3 and max(us) < x + 93


def test_detect_plate_finds_nothing_when_only_the_board_is_in_view():
    img, _, _ = _scene(plate=False)
    assert armplan.detect_plate(GRID, img) == {}
