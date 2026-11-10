"""calibtool.protocol: the collection-protocol measurements (design plan 13.1): every frame of a raw neck-grid take
detected, step 15b's still-pair selection reproduced from those detections (per_hold variants without re-detecting),
corner noise across the still frames of a pose, grid/band bookkeeping, and the poses each coverage target costs.

Synthetic tests pin the arithmetic and 15b's selection rule (against 15b itself on fake pose files); the stored-data
test checks the selection against a recorded corner file (needs CALIBTOOL_DATA)."""
import pytest

np = pytest.importorskip("numpy")

from calibtool import paths, protocol as pr  # noqa: E402

W, H = 1280, 800


# ---------------------------------------------------------------- grid and bands


def test_grid_index_nearest_pose():
    assert pr.grid_index(-22.6, -33.2) == 0             # measured, sagging: still the first grid pose
    assert pr.grid_index(25.5, 32.9) == 48
    assert pr.grid_index(0.4, -0.4) == 24
    assert pr.grid_index(-6.4, -11.4) == 2 * 7 + 2      # pitch -8 (row 2), yaw -11.33 (col 2)


def test_pose_cmds_from_log():
    text = ("body pinned at: x\n"
            "09:49:14 pose 1/49 cmd -24/-34 meas -22.6/-33.2 pairs 5 j0 ss_err +0.08 (3.2 s)\n"
            "09:49:17 pose 2/49 cmd -24/-23 meas -22.6/-23.6 pairs 5 j0 ss_err +0.09 (3.1 s)\n"
            "09:49:20 pose 3/49 cmd +8/+11 meas 9.0/10.1 pairs 5 (3.1 s)\n")
    cmds, total = pr.pose_cmds_from_log(text)
    assert cmds == {0: (-24.0, -34.0), 1: (-24.0, -23.0), 2: (8.0, 11.0)}
    assert total == 49


def test_band_of_nominal_distances():
    z = np.array([0.55, 0.7, 1.0, 1.5, 2.15, 2.8, np.nan, 0.3])
    b = pr.band_of(z)
    assert list(b[:6]) == [0.5, 0.75, 1.1, 1.6, 2.0, 3.0]
    assert np.isnan(b[6])
    assert b[7] == 0.5                                  # nearer than every band: the nearest one


def test_swept_poses_counts_every_pose_of_earlier_takes():
    order = ["a", "b", "c"]
    n = {"a": 49, "b": 20, "c": 49}
    assert pr.swept(order, n, "a", 0) == 1
    assert pr.swept(order, n, "b", 4) == 49 + 5
    assert pr.swept(order, n, "c", 48) == 49 + 20 + 49


def test_met_at_is_when_the_target_was_met_for_the_last_time():
    swept = [3, 5, 9, 12, 20]
    assert pr.met_at([False, True, False, True, True], swept) == 12
    assert pr.met_at([True] * 5, swept) == 3
    assert pr.met_at([True, True, True, True, False], swept) is None


def test_session_minutes():
    m = pr.session_minutes([49, 30, 21], sec_per_pose=3.0, placement_s=60.0)
    assert m == pytest.approx((100 * 3.0 + 3 * 60.0) / 60.0)


# ---------------------------------------------------------------- 15b's selection


def test_per_hold_slice_takes_the_middle_like_15b():
    k = [0, 1, 2, 3, 4]
    assert pr.per_hold_slice(k, 1) == [2]
    assert pr.per_hold_slice(k, 2) == [1, 2]
    assert pr.per_hold_slice(k, 3) == [1, 2, 3]
    assert pr.per_hold_slice(k, 5) == k
    assert pr.per_hold_slice([7, 9], 3) == [7, 9]
    assert pr.per_hold_slice([], 3) == []


def _tags(n, du=0.0, dv=0.0, base=0):
    """n whole tags (4 corners each) of a fake detection, shifted by (du, dv)"""
    return {(base + t, k): (100.0 + 30 * t + 5 * k + du, 200.0 + 3 * k + dv) for t in range(n) for k in range(4)}


class _FakeBoard:
    """detect(img) returns a prepared detection keyed by the image's first pixel"""

    def __init__(self, table):
        self.table = table

    def detect(self, img):
        return dict(self.table[int(np.asarray(img).ravel()[0])])


def _fake_take(tmp_path):
    """3 poses x 5 frames: pose 0 all still; pose 1 frame 0 too few left tags, frame 2 moved 3 px, frame 4 right
    fails; pose 2 nothing detected. Image value = key into the fake board's table"""
    table, poses = {0: {}}, []
    key = 1
    for p in range(3):
        L, R = [], []
        for f in range(5):
            if p == 2:
                dl, dr = {}, {}
            elif p == 1 and f == 0:
                dl, dr = _tags(4), _tags(8)
            elif p == 1 and f == 2:
                dl, dr = _tags(8, du=3.0), _tags(8)
            elif p == 1 and f == 4:
                dl, dr = _tags(8, du=0.2), _tags(3)
            else:
                dl, dr = _tags(8, du=0.1 * f, dv=-0.05 * f), _tags(7, du=0.1 * f)
            table[key], table[key + 1] = dl, dr
            L.append(key)
            R.append(key + 1)
            key += 2
        d = tmp_path / f"pose_{p:03d}.npz"
        np.savez(d, cmd=np.array([-24.0 + 8 * p, -34.0]), neck=np.array([-22.6 + 8 * p, -33.2]),
                 tl=np.arange(5, dtype=np.int64) * 66_666_667 + p * 3_100_000_000,
                 tr=np.arange(5, dtype=np.int64) * 66_666_667 + p * 3_100_000_000 + 15_000,
                 L=np.array(L, np.uint8).reshape(5, 1, 1), R=np.array(R, np.uint8).reshape(5, 1, 1))
        poses.append(d)
    return _FakeBoard(table)


@pytest.mark.parametrize("per_hold", [1, 2, 3, 5])
def test_corner_dict_from_detections_equals_15b(tmp_path, per_hold):
    board = _fake_take(tmp_path)
    s15b = paths.import_script("15b_dump_neck_take")
    out = tmp_path / "15b.npz"
    s15b.dump_take(str(tmp_path), board, out, still_px=1.0, per_hold=per_hold, min_tags=6, procs=1)
    ref = dict(np.load(out))
    frames = pr.detect_take(tmp_path, board, procs=1)
    got = pr.corner_dict(frames, per_hold=per_hold, still_px=1.0, min_tags=6)
    assert set(got) == set(ref)
    for k in ref:
        assert np.array_equal(np.asarray(got[k]), ref[k]), k


def test_still_frames_rule(tmp_path):
    board = _fake_take(tmp_path)
    frames = pr.detect_take(tmp_path, board, procs=1)
    assert pr.still_frames(frames[0]) == [0, 1, 2, 3, 4]
    assert pr.still_frames(frames[1]) == [1, 3]          # 0: 4 tags; 2: moved 3 px; 4: right camera 3 tags
    assert pr.still_frames(frames[2]) == []
    assert pr.pose_detected(frames[1]) and not pr.pose_detected(frames[2])


def test_frames_save_load_roundtrip(tmp_path):
    board = _fake_take(tmp_path)
    frames = pr.detect_take(tmp_path, board, procs=1)
    f = tmp_path / "frames.npz"
    pr.save_frames(frames, f)
    back = pr.load_frames(f)
    assert len(back) == len(frames)
    for a, b in zip(frames, back):
        assert np.array_equal(a.cmd, b.cmd) and np.array_equal(a.neck, b.neck)
        assert np.array_equal(a.tl, b.tl) and np.array_equal(a.tr, b.tr)
        assert a.dets == b.dets


# ---------------------------------------------------------------- noise, tag size, timing, bearing


def test_corner_noise_per_corner_sd_common_shift_and_jumps():
    rng = np.random.default_rng(1)
    base = _tags(30)
    dets = []
    for f in range(5):
        shift = (0.5 * f, -0.3 * f)                    # the head still settling: the whole board moves
        dets.append({k: (u + shift[0] + rng.normal(0, 0.2), v + shift[1] + rng.normal(0, 0.2))
                     for k, (u, v) in base.items()})
    jump = (3, 2)                                       # one corner jumps 2 px in one frame (seen on stored takes)
    dets[4][jump] = (dets[4][jump][0] + 2.0, dets[4][jump][1])
    n = pr.corner_noise(dets, remove_shift=True)
    assert n.n == 120
    assert n.median_sd == pytest.approx(0.2, rel=0.15)
    assert n.unstable == pytest.approx(1 / 120)        # per-corner sd > UNSTABLE_PX
    assert n.pooled > n.median_sd                       # the jump weighs on the pooled sd only
    raw = pr.corner_noise(dets, remove_shift=False)
    assert raw.median_sd > 0.4
    assert np.isnan(pr.corner_noise(dets[:1]).median_sd)     # one frame: nan
    assert pr.corner_noise(dets[:1]).n == 0


def test_tag_px_mean_side():
    d = {(0, 0): (100.0, 100.0), (0, 1): (120.0, 100.0), (0, 2): (120.0, 120.0), (0, 3): (100.0, 120.0),
         (1, 0): (0.0, 0.0), (1, 1): (10.0, 0.0)}                     # tag 1 not whole: ignored
    assert pr.tag_px(d) == pytest.approx(20.0)
    assert np.isnan(pr.tag_px({}))


def test_timing_from_stamps():
    period = 3_100_000_000
    tl = [np.arange(5, dtype=np.int64) * 66_666_667 + i * period for i in range(4)]
    t = pr.timing(tl)
    assert t["frame_ms"] == pytest.approx(66.67, abs=0.01)
    assert t["window_ms"] == pytest.approx(4 * 66.67, abs=0.05)
    assert t["period_s"] == pytest.approx(3.1)


def test_bearing_sign_and_size():
    f = 568.0
    # board centred at neck yaw +10 deg (head turned to the robot's left): bearing +10
    assert pr.bearing_deg([np.radians(10.0)], [W / 2], f, W) == pytest.approx(10.0)
    # at home, board 5 deg right of the image centre (equidistant: r = f theta) = toward the robot's right: -5
    assert pr.bearing_deg([0.0], [W / 2 + f * np.radians(5.0)], f, W) == pytest.approx(-5.0)


# ---------------------------------------------------------------- pose lists


def test_propose_keeps_poses_seen_in_at_least_half_the_takes():
    det = np.zeros((3, 49), bool)                        # 3 takes of one band
    det[:, 24] = True
    det[:2, 0] = True                                    # 2 of 3
    det[:1, 48] = True                                   # 1 of 3
    keep = pr.propose(det, min_rate=0.5)
    assert keep == [0, 24]


def test_grid_text_lays_out_pitch_rows_and_yaw_columns():
    vals = [str(i) for i in range(49)]
    txt = pr.grid_text(vals)
    rows = [r for r in txt.splitlines() if r.startswith("|") and "---" not in r]
    assert len(rows) == 8                                # header + 7 pitch rows
    assert rows[1].split("|")[2].strip() == "0" and rows[7].split("|")[8].strip() == "48"


# ---------------------------------------------------------------- stored data


def test_selection_reproduces_a_recorded_corner_file(data_root):
    """15b's rule from protocol's detections == the recorded p10 corner file (per_hold 3, still 1 px, 6 tags): the same
    pairs; the corners to 0.01 px (the robot's OpenCV and Moonstone's 5.0.0 differ by <= 0.0014 px and by 4 corners
    in 19,456 on this take, measured 9 Nov 2026)"""
    import contextlib
    import io
    import rigcfg
    with contextlib.redirect_stdout(io.StringIO()):
        rig = rigcfg.load(paths.CONFIG / "rig-phantom11.yaml")
    take = data_root / "p10/kit/takes/n6_200"
    ref = dict(np.load(data_root / "p10/kit/out1280/n6_200.npz"))
    got = pr.corner_dict(pr.detect_take(take, rig.board, procs=1), per_hold=3, still_px=1.0, min_tags=6)
    for k in ("img_hold", "img_t0", "img_t1", "img_yaw", "img_pitch", "img_ay", "img_ap"):
        assert np.array_equal(np.asarray(got[k]), ref[k]), k

    def corners(z):
        return {(i, c, t, k): (u, v) for i, c, t, k, u, v in zip(*(np.asarray(z[f"det_{x}"]) for x in
                                                                    ("img", "cam", "tag", "k", "u", "v")))}
    a, b = corners(got), corners(ref)
    common = set(a) & set(b)
    assert len(common) >= 0.999 * max(len(a), len(b))
    assert max(np.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1]) for k in common) < 0.01


# ---------------------------------------------------------------- analysis helpers


def test_take_kind_by_name():
    assert pr.take_kind("n6_tiltL") == "tilted" and pr.take_kind("dtiltleft") == "tilted"
    assert pr.take_kind("d075floordn") == "floor"
    assert pr.take_kind("n6_110") == "upright" and pr.take_kind("dleft") == "upright"


def test_cell_abbrev_and_edge_letters():
    assert pr.cell_abbrev("top-left") == "tl" and pr.cell_abbrev("centre") == "c" and pr.cell_abbrev("bottom") == "b"
    row = {"edges0": {"left": 3, "right": 0, "top": 0, "bottom": 0},
           "edges1": {"left": 0, "right": 0, "top": 2, "bottom": 0}}
    assert pr.edge_letters(row) == "LT"
    assert pr.edge_letters({}) == ""


def _corner_file():
    """2 detected poses (0 and 2) of a 3-pose take, 2 pairs each; 8 tags of a 7x5 board at about 1 m"""
    import contextlib
    import io
    import rigcfg
    with contextlib.redirect_stdout(io.StringIO()):
        rig = rigcfg.load(paths.CONFIG / "rig-phantom11.yaml")
    rows, hold, t0 = [], [], []
    for j, (h, du) in enumerate([(0, 0.0), (0, 0.1), (2, 400.0), (2, 400.1)]):
        hold.append(h)
        t0.append(1000 + j)
        for cam in (0, 1):
            for (t, k), (u, v) in _tags(8, du=du + 40 * cam).items():
                rows.append((j, cam, t, k, u, v))
    r = np.array(rows, float)
    z = dict(img_hold=np.array(hold), img_t0=np.array(t0), img_t1=np.array(t0) + 1, det_img=r[:, 0].astype(int),
             det_cam=r[:, 1].astype(int), det_tag=r[:, 2].astype(int), det_k=r[:, 3].astype(int), det_u=r[:, 4],
             det_v=r[:, 5], img_pitch=np.radians([-22.6, -22.6, -22.6, -22.6]),
             img_yaw=np.radians([-33.2, -33.2, -11.0, -11.0]))
    return z, rig


def test_rows_from_corners_marks_missing_poses_undetected():
    z, rig = _corner_file()
    cmds = {0: (-24.0, -34.0), 1: (-24.0, -23.0), 2: (-24.0, -11.0)}
    rows = pr.rows_from_corners(z, rig, "t", cmds, 3)
    assert [r["detected"] for r in rows] == [True, False, True]
    assert [r["grid"] for r in rows] == [0, 1, 2]
    assert rows[0]["edges0"]["left"] == 0 and rows[0]["cell"] in ("top-left", "left", "top")
    assert np.isnan(rows[1]["z"])


def test_far_table_buckets_by_distance():
    rows = [{"z": 1.05, "detected": True, "tags_l": 35, "tag_px": 60.0, "noise0": 0.05, "unstable0": 0.01},
            {"z": 1.10, "detected": False, "tags_l": 3, "tag_px": 58.0, "noise0": np.nan, "unstable0": np.nan},
            {"z": 2.30, "detected": True, "tags_l": 20, "tag_px": 27.0, "noise0": 0.15, "unstable0": 0.05},
            {"z": np.nan, "detected": False, "tags_l": 0, "tag_px": np.nan, "noise0": np.nan, "unstable0": np.nan}]
    t = pr.far_table(rows, edges=(1.0, 1.25, 2.25, 2.5))
    assert [b["n"] for b in t] == [2, 0, 1]
    assert t[0]["detected"] == 1 and t[0]["tags_med"] == pytest.approx(19.0)
    assert t[2]["noise_med"] == pytest.approx(0.15) and t[2]["tag_px_med"] == pytest.approx(27.0)
    assert t[2]["unstable_med"] == pytest.approx(0.05)


def test_met_index_matches_met_at():
    assert pr.met_index([False, True, False, True, True]) == 3
    assert pr.met_index([True, True]) == 0
    assert pr.met_index([True, False]) is None and pr.met_index([]) is None


def test_centre_tags_median_over_the_nine_central_poses():
    rows = [{"grid": g, "tags_l": (30 if g in pr.CENTRE_GRID else 2)} for g in range(49)]
    assert sorted(pr.CENTRE_GRID) == [16, 17, 18, 23, 24, 25, 30, 31, 32]
    assert pr.centre_tags(rows) == 30
    assert np.isnan(pr.centre_tags([]))


def test_propose_keeps_rare_poses_that_fill_an_edge_strip():
    det = np.zeros((3, 49), bool)
    det[:, 24] = True
    det[:1, 48] = True                                   # 1 of 3, fills an edge strip: kept
    det[:1, 47] = True                                   # 1 of 3, adds no edge: dropped
    adds = np.zeros(49, bool)
    adds[[48, 10]] = True                                # 10: never detected, never kept
    assert pr.propose(det, min_rate=0.5, adds_edge=adds) == [24, 48]
