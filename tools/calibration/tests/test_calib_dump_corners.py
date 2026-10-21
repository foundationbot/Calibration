"""Step 15: stereo pairs from a record-app MCAP, corners of both images, still pairs only, in the npz layout
lensfit.Data reads. The MCAP is built in memory the way okvis2x_app_record writes it (uncompressed chunk,
image then meta per frame, publish_time = capture time), from rendered board images."""
import importlib
import struct
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

PKG = Path(__file__).resolve().parent.parent
ROOT = PKG.parents[1]
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "tests"))
sys.path.insert(0, str(ROOT / "interface"))
dc = importlib.import_module("15_dump_corners")
synth = importlib.import_module("test_calib_pipeline_synthetic")
rigcfg = importlib.import_module("rigcfg")

MAGIC = b"\x89MCAP0\r\n"


def _s(t):
    b = t.encode()
    return struct.pack("<I", len(b)) + b


def _rec(op, body):
    return bytes([op]) + struct.pack("<Q", len(body)) + body


def _write(path, frames):
    """frames: [(cam, capture_ns, png bytes)] in file order -> a record-app-like MCAP"""
    recs = bytearray()
    cid = {1: 1, 2: 2}
    for cam, c in cid.items():
        recs += _rec(0x04, struct.pack("<HH", c, 0) + _s(f"/cam{cam}/image") + _s("png") + struct.pack("<I", 0))
    for seq, (cam, ts, png) in enumerate(frames):
        recs += _rec(0x05, struct.pack("<HI", cid[cam], seq) + struct.pack("<QQ", ts, ts) + png)
    head = struct.pack("<QQ", 0, 0) + struct.pack("<Q", len(recs)) + struct.pack("<I", 0)
    path.write_bytes(MAGIC + _rec(0x01, _s("") + _s("")) +
                     _rec(0x06, head + _s("") + struct.pack("<Q", len(recs)) + bytes(recs)))


def _png3(gray):
    """a 3-channel PNG, as okvis2x_app_record writes its frames"""
    return cv2.imencode(".png", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))[1].tobytes()


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    tex, margin = synth._texture()
    views = []
    for k in range(12):                                     # 6 frames still at pose A, 6 moving
        moving = k >= 6
        R = cv2.Rodrigues(np.radians(np.array([180.0, 8.0 + (3 * k if moving else 0), 0.0])))[0]
        t = np.array([-0.40 + (0.02 * k if moving else 0), -0.28, 0.85])
        R1, t1 = synth.T10[:3, :3] @ R, synth.T10[:3, :3] @ t + synth.T10[:3, 3]
        ts = 1_792_000_000_000_000_000 + k * 66_666_667
        views.append((1, ts, _png3(synth._render(tex, margin, R, t))))
        views.append((2, ts + 30_000, _png3(synth._render(tex, margin, R1, t1))))
    views.append((2, 1_792_000_000_000_000_000 + 20 * 66_666_667, views[-1][2]))   # a right frame with no left
    p = tmp_path_factory.mktemp("mcap") / "session.mcap"
    _write(p, views)
    return p


def test_pairs_are_matched_within_two_ms_and_unpaired_frames_dropped(session):
    pairs = dc.pair_frames(session, ("/cam1/image", "/cam2/image"), max_dt_ns=2_000_000)
    assert len(pairs) == 12
    assert all(abs(p[1] - p[0]) <= 2_000_000 for p in pairs)


def test_still_pairs_only_and_the_layout_lensfit_reads(session, tmp_path):
    board = rigcfg.AprilGrid(synth.COLS, synth.ROWS, synth.TAG, synth.SPACING)
    out = tmp_path / "c.npz"
    n = dc.dump(session, board, out, still_px=1.0, per_hold=3)
    d = np.load(out)
    for k in ("img_hold", "img_t0", "img_t1", "det_img", "det_cam", "det_tag", "det_k", "det_u", "det_v"):
        assert k in d.files
    assert n == len(d["img_hold"]) == 3                     # the still pose gives one hold, 3 pairs kept
    assert set(np.unique(d["det_cam"]).tolist()) == {0, 1}
    assert (np.bincount(d["det_img"]) >= 40).all()          # both cameras' corners for every kept pair
    for j in range(n):                                      # camera 0 = /cam1 = left: the board sits further
        u0 = d["det_u"][(d["det_img"] == j) & (d["det_cam"] == 0)].mean()   # right in it than in camera 1
        u1 = d["det_u"][(d["det_img"] == j) & (d["det_cam"] == 1)].mean()
        assert u0 > u1 + 20


def test_frames_are_read_as_the_export_route_reads_them(tmp_path):
    """the record app writes 3-channel PNGs: _gray must equal cv2.imread(png, IMREAD_GRAYSCALE), not
    cvtColor(BGR2GRAY), which rounds differently"""
    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, (80, 120, 3), dtype=np.uint8)
    png = cv2.imencode(".png", bgr)[1].tobytes()
    f = tmp_path / "frame.bin"
    f.write_bytes(b"xx" + png)
    with open(f, "rb") as fh:
        g = dc._gray(fh, 2, len(png))
    p = tmp_path / "frame.png"
    p.write_bytes(png)
    assert np.array_equal(g, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))


class _ScriptedBoard:
    """detect() returns the corners scripted for the frame id stored in every pixel of the frame"""

    def __init__(self, by_id):
        self.by_id = by_id

    def detect(self, gray):
        return dict(self.by_id.get(int(gray[0, 0]), {}))


def _corners(cx, cy, n_tags=8):
    return {(t, k): (cx + 12.0 * t + 3.0 * (k % 2), cy + 3.0 * (k // 2)) for t in range(n_tags) for k in range(4)}


def test_hold_rules_pairing_and_cameras_with_a_scripted_detector(tmp_path):
    """frames 1-19 at 15 fps, left frame id k, right frame id 100 + k (3-channel PNGs, every pixel = the id):
    1-5 still (frame 1's right frame 2.5 ms late: unpaired), 6-8 still (frame 7's right image shows 4 tags),
    9-10 still (2 pairs: too short), 11-18 drifting 0.4 px a frame (splits after > 1 px from a hold's first
    pair), 19 no board; frames 8 and 9 written to the file in swapped order. Expected, from the rules: holds
    [2-5], [6-8], [11-13], [14-16]; kept the middle 3 of each (minus frame 7): 3, 4, 5, 6, 8, 11, 12, 13, 14,
    15, 16; camera 0 = the left (/cam1) frames' corners"""
    centre = {k: (500.0, 300.0) for k in range(1, 6)} | {k: (700.0, 300.0) for k in range(6, 9)} \
        | {k: (900.0, 300.0) for k in (9, 10)} | {k: (300.0 + 0.4 * (k - 11), 500.0) for k in range(11, 19)}
    by_id = {k: _corners(*c) for k, c in centre.items()}
    by_id |= {100 + k: _corners(c[0] - 30.0, c[1], 4 if k == 7 else 8) for k, c in centre.items()}
    t0 = 1_792_000_000_000_000_000
    frames = []
    for k in range(1, 20):
        ts = t0 + k * 66_666_667
        img = np.full((8, 8), k, np.uint8)
        frames.append((k, [(1, ts, _png3(img)), (2, ts + (2_500_000 if k == 1 else 30_000), _png3(img + 100))]))
    order = [f for k, f in frames if k not in (8, 9)]
    order[7:7] = [frames[8][1], frames[7][1]]               # 9 before 8 in the file
    path = tmp_path / "session.mcap"
    _write(path, [v for f in order for v in f])
    pairs = dc.pair_frames(path)
    assert [p[0] for p in pairs] == [t0 + k * 66_666_667 for k in range(2, 20)]   # sorted, frame 1 unpaired
    out = tmp_path / "c.npz"
    n = dc.dump(path, _ScriptedBoard(by_id), out, still_px=1.0, per_hold=3)
    d = np.load(out)
    kept = [3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16]
    assert n == len(kept)
    assert d["img_t0"].tolist() == [t0 + k * 66_666_667 for k in kept]
    assert d["img_hold"].tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3]
    for j, k in enumerate(kept):
        m0 = (d["det_img"] == j) & (d["det_cam"] == 0)
        m1 = (d["det_img"] == j) & (d["det_cam"] == 1)
        assert np.allclose(d["det_u"][m0].mean(), np.mean([u for u, _ in by_id[k].values()]))
        assert np.allclose(d["det_u"][m1].mean(), np.mean([u for u, _ in by_id[100 + k].values()]))


@pytest.mark.parametrize("record_dirs, queues, cam0_left", [
    (None, None, True),                     # phantom12 as shipped: camera 0 = queue 1 = /cam1, the left eye
    (("left", "right"), (1, 2), True),      # record dirs named otherwise: the channels still follow the queues
    (("left", "right"), (2, 1), False),     # queues swapped: camera 0 = /cam2, the right eye
])
def test_main_reads_each_camera_from_its_dma_queue_channel(session, tmp_path, record_dirs, queues, cam0_left):
    """camera 0 is the rig file's kalibr cam0, and the record app names a camera's MCAP channels after its DMA queue
    (okvis2x_app_record.cpp: "/cam" + the --cameras index, a DMA queue number), not after the rig file's record_dir
    (an export directory's name)"""
    import shutil
    yaml = pytest.importorskip("yaml")
    cfg = tmp_path / "config"
    shutil.copytree(PKG / "config", cfg)
    p = cfg / "rig-phantom12.yaml"
    rig = yaml.safe_load(p.read_text())
    for c, rd, q in zip(rig["cameras"]["kalibr"], record_dirs or (None, None), queues or (None, None)):
        if rd is not None:
            c["record_dir"], c["dma_queue"] = rd, q
    p.write_text(yaml.safe_dump(rig))
    out = tmp_path / "c.npz"
    dc.main([str(session), str(out), "--rig", str(p)])
    d = np.load(out)
    assert len(d["img_hold"]) == 3                          # the still hold's 3 pairs: both channels were found
    u0 = d["det_u"][(d["det_img"] == 0) & (d["det_cam"] == 0)].mean()
    u1 = d["det_u"][(d["det_img"] == 0) & (d["det_cam"] == 1)].mean()
    assert (u0 > u1 + 20) if cam0_left else (u0 < u1 - 20)  # the left eye sees the board further right
