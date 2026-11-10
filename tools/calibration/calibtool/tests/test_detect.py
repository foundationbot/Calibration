"""calibtool.detect: AprilGrid detection for live capture. Tags are FOUND on a contrast-stretched copy (design D9: a
dark board is still found) but every corner is REFINED on the raw image; the worker pool is sized to the robot's free
CPU; board_obs gives the board's centroid and depth for placement guidance; hold_dict turns detected pairs into the
step-15b corner dict replay.Collection takes."""
import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("yaml")

from calibtool import detect, paths  # noqa: E402

rigcfg = paths.import_script("rigcfg")
F = 568.0
SIZE = (1280, 800)
PINHOLE_KB4 = [1 / 3, 2 / 15, 17 / 315, 62 / 2835]       # tan(t) = t (1 + t^2/3 + 2 t^4/15 + ...): a pinhole as KB4


def board():
    return rigcfg.AprilGrid(7, 5, 0.1, 0.2)


def render(b, z_m=1.2, centre=(640.0, 400.0), lo=0, hi=255, hires=2000.0):
    """a fronto-parallel Kalibr aprilgrid (tags drawn rotated 180 deg, black squares in the gap crossings) at depth
    z_m in a pinhole camera of focal F: -> (image, project) where project(obj points) gives the true pixels"""
    g = b.tag_size * b.spacing
    W = (b.cols - 1) * b.pitch + b.tag_size
    H = (b.rows - 1) * b.pitch + b.tag_size
    m = 2 * g
    R = hires
    cw, ch = int(round((W + 2 * m) * R)), int(round((H + 2 * m) * R))
    hi_img = np.full((ch, cw), 255, np.uint8)

    def px(x, y):                       # board metres (y up) -> hi-res pixels
        return (m + x) * R, (m + H - y) * R

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    s = int(round(b.tag_size * R))
    for tid in range(b.n_tags):
        r, c = b.tag_rc(tid)
        mk = np.rot90(cv2.aruco.generateImageMarker(d, tid, s, borderBits=2), 2)
        x0, y0 = px(c * b.pitch, r * b.pitch + b.tag_size)
        hi_img[int(round(y0)):int(round(y0)) + s, int(round(x0)):int(round(x0)) + s] = mk
    gs = int(round(g * R))
    for c in range(b.cols + 1):
        for r in range(b.rows + 1):
            x0, y0 = px(c * b.pitch - g, r * b.pitch)
            hi_img[int(round(y0)):int(round(y0)) + gs, int(round(x0)):int(round(x0)) + gs] = 0
    k = F / z_m / R                     # hi-res px -> image px
    ox = centre[0] - k * (m + W / 2) * R
    oy = centre[1] - k * (m + H / 2) * R
    A = np.array([[k, 0, ox], [0, k, oy]], float)
    img = cv2.warpAffine(hi_img, A, SIZE, flags=cv2.INTER_AREA, borderValue=255)
    img = (lo + (img.astype(float) / 255.0) * (hi - lo)).round().astype(np.uint8)

    def project(obj):
        obj = np.asarray(obj, float).reshape(-1, 3)
        hx, hy = px(obj[:, 0], obj[:, 1])
        return np.c_[k * np.asarray(hx) + ox, k * np.asarray(hy) + oy]
    return img, project


def test_render_matches_the_boards_corner_convention():
    b = board()
    img, proj = render(b)
    det = b.detect(img)
    assert len({t for t, _ in det}) == b.n_tags
    err = [np.hypot(*(np.asarray(uv) - proj(b.obj_points(t)[k])[0])) for (t, k), uv in det.items()]
    assert np.median(err) < 1.0


# ------------------------------------------------------------------ stretch

def test_stretch_maps_percentiles_to_full_range():
    g = np.tile(np.arange(100, 131, dtype=np.uint8), (10, 1))
    s = detect.stretch(g, 0, 100)
    assert s.dtype == np.uint8 and s.min() == 0 and s.max() == 255


def test_stretch_leaves_a_flat_image_alone():
    g = np.full((5, 5), 77, np.uint8)
    assert np.array_equal(detect.stretch(g), g)


# ------------------------------------------------------------------ find on stretched, refine on raw

def test_low_contrast_board_is_found_only_through_the_stretched_copy():
    b = board()
    img, _ = render(b, lo=118, hi=126)
    plain = len({t for t, _ in b.detect(img)})
    found = detect.find_refine(b, img)
    assert len({t for t, _ in found}) > plain
    assert len({t for t, _ in found}) >= 30


def test_corners_are_refined_on_the_raw_image():
    b = board()
    img, _ = render(b, lo=60, hi=200)
    found = detect.find_refine(b, img)
    quads = detect.quads(b, detect.stretch(img))
    expect = {}
    for tid, q in quads:
        for k, uv in b.refine(img, q).items():
            expect[(tid, k)] = uv
    assert found == expect


def test_without_stretch_equals_the_boards_own_detect():
    b = board()
    img, _ = render(b)
    assert detect.find_refine(b, img, use_stretch=False) == b.detect(img)


# ------------------------------------------------------------------ pool size

@pytest.mark.parametrize("ncpu,load,want", [(9, 9.0, 1), (9, 0.5, 4), (32, 2.0, 4), (8, 3.5, 2), (4, 10.0, 1)])
def test_pool_size_from_free_cpu(ncpu, load, want):
    assert detect.pool_size(ncpu, load) == want


def test_pool_size_shrinks_with_load():
    sizes = [detect.pool_size(12, ld) for ld in np.arange(0, 14, 0.5)]
    assert sizes == sorted(sizes, reverse=True) and min(sizes) == 1


def test_detect_pool_gives_the_same_corners():
    b = board()
    img, _ = render(b)
    with detect.DetectPool(b, 2) as pool:
        dl, dr = pool.submit_pair(img, img).result(timeout=60)
    assert dl == detect.find_refine(b, img) and dr == dl


# ------------------------------------------------------------------ board_obs

def test_board_obs_depth_and_centroid():
    b = board()
    img, _ = render(b, z_m=1.2, centre=(700.0, 380.0))
    det = detect.find_refine(b, img)
    o = detect.board_obs(det, b, intr=[F, F, 640.0, 400.0] + PINHOLE_KB4, size=SIZE)
    assert o.z == pytest.approx(1.2, rel=0.01)
    assert np.allclose(o.centroid, (700.0, 380.0), atol=3.0)
    assert o.n_tags == 35 and not o.border and o.cut == ()


def test_board_obs_needs_enough_tags():
    b = board()
    img, _ = render(b)
    det = detect.find_refine(b, img)
    few = {k: v for k, v in det.items() if k[0] < 5}
    assert detect.board_obs(few, b, intr=[F, F, 640.0, 400.0] + PINHOLE_KB4, size=SIZE) is None


def test_board_obs_flags_a_board_cut_by_the_image_edge():
    b = board()
    img, _ = render(b, z_m=1.0, centre=(1100.0, 400.0))
    det = detect.find_refine(b, img)
    o = detect.board_obs(det, b, intr=[F, F, 640.0, 400.0] + PINHOLE_KB4, size=SIZE)
    assert o is not None and o.border and o.cut == ("right",)


def test_board_obs_default_camera_is_the_nominal_equidistant_one():
    b = board()
    img, _ = render(b, z_m=1.2)
    det = detect.find_refine(b, img)
    o = detect.board_obs(det, b, focal=F, size=SIZE)
    assert o.z == pytest.approx(1.2, rel=0.05)


# ------------------------------------------------------------------ hold_dict

def test_hold_dict_is_a_15b_corner_dict_replay_accepts():
    from calibtool import assign, replay
    b = board()
    img, _ = render(b, z_m=0.75)
    det = detect.find_refine(b, img)
    pairs = [(1000, 1010, det, det), (2000, 2010, det, det)]
    d = detect.hold_dict(pairs, hold=7, neck_rad=(0.1, -0.2))
    for k in ("img_hold", "img_t0", "img_t1", "det_img", "det_cam", "det_tag", "det_k", "det_u", "det_v", "img_yaw",
              "img_pitch", "img_ay", "img_ap"):
        assert k in d, k
    assert list(d["img_hold"]) == [7, 7] and list(d["img_t0"]) == [1000, 2000]
    assert set(d["det_cam"]) == {0, 1} and (d["det_cam"] == 0).sum() == 2 * len(det)
    assert d["img_pitch"][0] == pytest.approx(0.1) and d["img_yaw"][0] == pytest.approx(-0.2)
    rig = paths.CONFIG / "rig-phantom11.yaml"
    col = replay.Collection(rig, replay.capture_labeler(rig))
    lab = col.add(assign.CaptureHold(hold=7, placement="trial", t0=1000, data=d))
    assert lab in ("FIT", "TEST")
    assert col.counts()["pairs"] == 2


def test_hold_dict_of_no_pairs_is_empty():
    d = detect.hold_dict([], hold=0)
    assert len(d["img_t0"]) == 0 and len(d["det_u"]) == 0


def test_pool_size_counts_only_the_cpus_this_process_may_use():
    """found on phantom10 9 Nov: os.cpu_count() reports every core, the process may run on 9"""
    import os
    assert detect.usable_cpus() == len(os.sched_getaffinity(0))


def _stored_frame(data_root, take="n6_110", pose=24):
    f = data_root / "p10" / "kit" / "takes" / take / f"pose_{pose:03d}.npz"
    if not f.is_file():
        pytest.skip(f"{f} not in the stored data")
    z = np.load(f)
    return z["L"][0], z["R"][0]


def test_live_detection_on_a_stored_phantom10_frame(data_root):
    """a real 1280x800 frame of phantom10's 6 Nov session (board at ~1.1 m, centre pose): the stretched-copy finder
    sees every tag the plain detector sees, the corners they share agree (both refined on the raw frame), and the
    board's depth reads in the 1.1 m band"""
    rig = paths.import_script("rigcfg").load(paths.CONFIG / "rig-phantom11.yaml")
    yl, _ = _stored_frame(data_root)
    plain = rig.board.detect(yl)
    found = detect.find_refine(rig.board, yl)
    assert {t for t, _ in found} >= {t for t, _ in plain}
    common = set(plain) & set(found)
    assert len(common) >= 0.9 * len(plain)
    d = [np.hypot(*np.subtract(plain[k], found[k])) for k in common]
    assert np.median(d) < 0.05
    o = detect.board_obs(found, rig.board, focal=rig.nominal_focal_px, size=SIZE)
    assert 0.925 <= o.z < 1.35


def test_a_dark_stored_frame_is_still_found(data_root):
    """the same frame at 1/8 of its brightness (a dark room, D9): the plain detector loses tags, the stretched copy
    finds them, corners still refined on the dark raw frame"""
    rig = paths.import_script("rigcfg").load(paths.CONFIG / "rig-phantom11.yaml")
    yl, _ = _stored_frame(data_root)
    dark = (yl.astype(np.float32) / 8).astype(np.uint8)
    n_plain = len({t for t, _ in rig.board.detect(dark)})
    n_found = len({t for t, _ in detect.find_refine(rig.board, dark)})
    n_bright = len({t for t, _ in rig.board.detect(yl)})
    assert n_found > n_plain
    assert n_found >= 0.8 * n_bright


def test_guide_corners_match_full_detection_on_a_stored_phantom10_frame():
    """the fast guidance path (half size, no refinement, left only) must put the board where full detection does"""
    from calibtool import paths
    if not paths.have_data():
        pytest.skip("CALIBTOOL_DATA not set")
    pose = paths.data_root() / "p10" / "kit" / "takes" / "n6_110" / "pose_024.npz"
    if not pose.is_file():
        pytest.skip(f"{pose} missing")
    from calibtool import pipeline
    rig = pipeline.load_rig(paths.CONFIG / "rig-phantom11.yaml")
    y = np.load(pose)["L"][0]
    full = detect.find_refine(rig.board, y)
    fast = detect.guide_corners(rig.board, y)
    assert len(fast) >= 0.8 * len(full) > 0
    a = detect.board_obs(full, rig.board, focal=rig.nominal_focal_px)
    b = detect.board_obs(fast, rig.board, focal=rig.nominal_focal_px)
    assert a is not None and b is not None
    assert np.hypot(*(a.centroid - b.centroid)) < 3.0 and abs(b.z / a.z - 1) < 0.02
