"""calibtool.postcheck: the checks after a deploy (design plan 6.5, 13.2): depth first and always (new vs old on the
same frames), OKVIS second and optional (its log, latency, rectified-pair rows new vs old by the same method, floor
tilt), the auto-rollback rule (crash loop or new worse than old by the same method; everything else reported) and the
row-health reference. Synthetic scores and images only."""
import pytest

np = pytest.importorskip("numpy")

from calibtool import decide, postcheck as pc  # noqa: E402


def scores(err, dy, n=20, seed=0, z=1.6):
    rng = np.random.default_rng(seed)
    src = np.arange(n)
    return decide.Scores(src, src // 2, np.array(["post"] * n), err + rng.normal(0, 0.02, n), np.full(n, z),
                         np.repeat(src, 30), dy + rng.normal(0, 0.02, n * 30))


# ---------------------------------------------------------------- E1: depth check + decision


def test_sync_first():
    d = pc.depth_check(0.6, scores(0.2, 0.2), scores(0.9, 0.8))
    assert d.status == "not_checked" and "sync" in d.reasons[0]
    assert not pc.decide(d).rollback


def test_new_better_is_ok():
    d = pc.depth_check(0.02, scores(0.2, 0.2), scores(0.9, 0.8))
    assert d.status == "ok" and d.new["depth"] < d.old["depth"]
    assert not pc.decide(d).rollback


@pytest.mark.parametrize("new,old", [((0.5, 0.2), (0.2, 0.2)), ((0.2, 0.5), (0.2, 0.2))])
def test_new_worse_by_the_same_method_rolls_back(new, old):
    d = pc.depth_check(0.02, scores(*new), scores(*old))
    assert d.status == "worse"
    dec = pc.decide(d)
    assert dec.rollback and any("worse" in r for r in dec.reasons)


def test_small_differences_are_not_worse():
    d = pc.depth_check(0.02, scores(0.25, 0.22, seed=1), scores(0.2, 0.2, seed=2))
    assert d.status == "ok"


@pytest.mark.parametrize("dd,dr,status", [(0.12, 0.0, "ok"), (0.20, 0.0, "worse"), (0.0, 0.12, "ok"),
                                          (0.0, 0.20, "worse")])
def test_worse_means_more_than_0p15_on_depth_or_rows(dd, dr, status):
    """the thresholds agreed with the user on 10 Nov: roll back when new is worse than old by more than 0.15 % depth or
    0.15 px rows (same frames, same method); the same noise on both sides, so the differences are exact"""
    d = pc.depth_check(0.02, scores(0.2 + dd, 0.2 + dr), scores(0.2, 0.2))
    assert d.status == status, d.reasons


def test_depth_compares_the_same_pairs_only():
    new, old = scores(0.2, 0.2), scores(0.2, 0.2)
    keep = old.src < 10
    old2 = decide.Scores(old.src[keep], old.hold[keep], old.placement[keep], old.err_pct[keep] + 5, old.z[keep],
                         old.corner_src[np.isin(old.corner_src, old.src[keep])],
                         old.corner_dy[np.isin(old.corner_src, old.src[keep])])
    d = pc.depth_check(0.02, new, old2)
    assert d.new["pairs"] == 10 and d.old["pairs"] == 10


def test_crash_loop_rolls_back_alone():
    dec = pc.decide(None, None, crash=True)
    assert dec.rollback and "crash" in dec.reasons[0]


def test_okvis_findings_are_reported_not_rolled_back():
    d = pc.depth_check(0.02, scores(0.2, 0.2), scores(0.9, 0.8))
    ok = pc.okvis_check(["[entrypoint] config: x.yaml explicit path /etc/okvis/x.yaml (abcdef123456)"],
                        "abcdef123456", latency_s=0.5, baseline_latency_s=0.13, floor_tilt_deg=2.6)
    assert ok.kb4t_line is False and ok.config_sha and not ok.latency_ok
    dec = pc.decide(d, ok)
    assert not dec.rollback
    txt = " ".join(dec.reasons)
    assert "KB4T" in txt and "latency" in txt and "tilt" in txt


def test_okvis_rect_rows_worse_rolls_back():
    d = pc.depth_check(0.02, scores(0.2, 0.2), scores(0.9, 0.8))
    old = pc.RowHealth("rowfeat-orb", 400, 1.78, {"left": 1.7, "centre": 1.8, "right": 1.8}, {})
    new = pc.RowHealth("rowfeat-orb", 400, 2.6, {"left": 2.5, "centre": 2.6, "right": 2.7}, {})
    ok = pc.okvis_check(["KB4T pinhole camera 0", "(abcdef123456)"], "abcdef123456", rect_new=new, rect_old=old)
    assert ok.rect_worse
    assert pc.decide(d, ok).rollback
    same = pc.okvis_check(["KB4T pinhole camera 0", "(abcdef123456)"], "abcdef123456", rect_new=old, rect_old=old)
    assert not same.rect_worse and not pc.decide(d, same).rollback


@pytest.mark.parametrize("new_dy,worse", [(2.18, False), (2.38, True)])
def test_okvis_rect_rows_worse_means_more_than_0p5(new_dy, worse):
    """agreed 10 Nov: OKVIS's rectified-pair rows count as worse when new exceeds old by more than 0.5 px"""
    old = pc.RowHealth("rowfeat-orb", 400, 1.78, {}, {})
    new = pc.RowHealth("rowfeat-orb", 400, new_dy, {}, {})
    ok = pc.okvis_check(["KB4T pinhole camera 0", "(abcdef123456)"], "abcdef123456", rect_new=new, rect_old=old)
    assert bool(ok.rect_worse) is worse


def test_different_methods_are_not_compared():
    old = pc.RowHealth("okvis-rect-jpg", 400, 0.2, {}, {})
    new = pc.RowHealth("rowfeat-orb", 400, 3.0, {}, {})
    ok = pc.okvis_check(["KB4T pinhole camera 0"], "x", rect_new=new, rect_old=old)
    assert ok.rect_worse is None and any("method" in r for r in ok.reasons)


def test_crash_loop_from_pods():
    assert pc.crash_loop([{"waiting": "CrashLoopBackOff", "restarts": 1}])
    assert pc.crash_loop([{"waiting": None, "restarts": 2}])
    assert not pc.crash_loop([{"waiting": None, "restarts": 0}])


# ---------------------------------------------------------------- E2: row features, reference, scoring input


def _texture(w=640, h=400, seed=3):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w)).astype(np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 1.2)


def test_rowfeat_measures_a_known_row_offset():
    cv2 = pytest.importorskip("cv2")
    a = _texture()
    M = np.float32([[1, 0, -12], [0, 1, 2]])
    b = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REFLECT)
    r = pc.rowfeat(a, b)
    assert r is not None and r.n >= 50 and r.median_dy == pytest.approx(2.0, abs=0.2)
    assert set(r.thirds) == {"left", "centre", "right"}
    assert all(v == pytest.approx(2.0, abs=0.3) for v in r.thirds.values() if v is not None)


def test_rowfeat_flat_image_gives_none():
    pytest.importorskip("cv2")
    flat = np.full((400, 640), 128, np.uint8)
    assert pc.rowfeat(flat, flat) is None


def test_row_reference_records_method_pattern_conditions():
    r = pc.RowHealth("rowfeat-orb", 512, 1.78, {"left": 1.7, "centre": 1.8, "right": 1.9},
                     {"top": 1.6, "bottom": 1.9})
    ref = pc.row_reference(r, {"head_still": True, "sync_ms": 0.014}, "abcdef123456", config_name="x.yaml")
    assert ref["method"] == "rowfeat-orb" and ref["median_dy_px"] == 1.78 and ref["thirds"]["right"] == 1.9
    assert ref["edges"]["bottom"] == 1.9 and ref["conditions"]["sync_ms"] == 0.014
    assert ref["config_sha12"] == "abcdef123456" and ref["config"] == "x.yaml" and "taken_at" in ref


def test_score_holds_writes_one_npz_of_the_holds(tmp_path):
    from types import SimpleNamespace

    def hold(hid, n):
        return SimpleNamespace(hold=hid, data={
            "img_hold": np.full(n, hid), "img_t0": np.arange(n) + 100 * hid, "img_t1": np.arange(n) + 100 * hid + 1,
            "det_img": np.repeat(np.arange(n), 2), "det_cam": np.tile([0, 1], n), "det_tag": np.zeros(2 * n, int),
            "det_k": np.zeros(2 * n, int), "det_u": np.ones(2 * n), "det_v": np.ones(2 * n),
            "img_yaw": np.zeros(n), "img_pitch": np.zeros(n), "img_ay": np.zeros(n, int),
            "img_ap": np.zeros(n, int)})
    got = {}

    def scorer(calib_json, npz, rig, placement):
        z = np.load(npz)
        got.update(n=len(z["img_t0"]), det_img=z["det_img"].tolist(), pl=list(placement), cal=calib_json)
        return "scores"
    out = pc.score_holds("new.json", [hold(1, 2), hold(2, 3)], "rig", tmp_path, scorer=scorer)
    assert out == "scores" and got["n"] == 5 and got["cal"] == "new.json"
    assert got["det_img"] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] and got["pl"] == ["postcheck"] * 5
