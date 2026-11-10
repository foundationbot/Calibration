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
