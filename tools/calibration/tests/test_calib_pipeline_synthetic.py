"""Steps 60 and 65 end to end on rendered fisheye stereo images of an aprilgrid.

The board is drawn the way OpenCV sees a Kalibr-printed aprilgrid (each tag turned
180 deg, tag 0 bottom-left, ids along x then y), which is the convention
AprilGrid.obj_points encodes; step 60's mono reprojection RMS therefore also checks
that convention against the detector.
"""
import importlib
import shutil
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")
cv2 = pytest.importorskip("cv2")

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "scripts"))
validate = importlib.import_module("60_validate_stereo")
scale = importlib.import_module("65_scale_check")

RES = (1280, 800)
K = np.array([[568.0, 0, 641.5], [0, 568.5, 398.0], [0, 0, 1.0]])
D = np.array([-0.02, 0.045, -0.045, 0.017]).reshape(4, 1)   # mk13-like equidistant
T10 = np.eye(4)
T10[:3, :3] = cv2.Rodrigues(np.radians([0.3, -0.4, 0.2]))[0]
T10[:3, 3] = [-0.080, 0.0005, 0.001]                        # LEFT cam0, 80 mm baseline
MM = 0.001                                                   # texture: 1 px = 1 mm
COLS, ROWS, TAG, SPACING = 7, 5, 0.1, 0.2


def _texture():
    pitch = TAG * (1 + SPACING)
    margin = 60
    w = int(round((COLS * pitch - TAG * SPACING) / MM)) + 2 * margin
    h = int(round((ROWS * pitch - TAG * SPACING) / MM)) + 2 * margin
    tex = np.full((h, w), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    side = int(round(TAG / MM))
    for i in range(COLS * ROWS):
        r, c = divmod(i, COLS)
        m = cv2.rotate(cv2.aruco.generateImageMarker(d, i, side, borderBits=2), cv2.ROTATE_180)
        x0 = margin + int(round(c * pitch / MM))
        y_top = h - margin - int(round((r * pitch + TAG) / MM))   # board y is up
        tex[y_top:y_top + side, x0:x0 + side] = m
    # Kalibr prints a black square (spacing x tagSize) in the gap at every tag
    # corner, turning each tag corner into an X-junction that sub-pixel corner
    # refinement localises precisely; a bare-tag board leaves L-corners instead.
    sq = int(round(TAG * SPACING / MM))
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            x = margin + int(round(c * pitch / MM)) - sq
            y = h - margin - int(round(r * pitch / MM))
            tex[max(y, 0):max(y, 0) + sq, max(x, 0):max(x, 0) + sq] = 0
    return tex, margin


def _render(tex, margin, R_cb, t_cb, ss=2):
    """Ray-cast every pixel onto the board plane; ss x ss supersampled, then
    area-averaged, so tag edges are antialiased like a real exposure."""
    W, H = RES[0] * ss, RES[1] * ss
    u, v = np.meshgrid((np.arange(W) + 0.5) / ss - 0.5, (np.arange(H) + 0.5) / ss - 0.5)
    px = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
    ray = cv2.fisheye.undistortPoints(px, K, D).reshape(-1, 2)
    d = np.hstack([ray, np.ones((len(ray), 1))])
    n = R_cb[:, 2]
    lam = (n @ t_cb) / (d @ n)
    Xb = (d * lam[:, None] - t_cb) @ R_cb                       # R_cb^T (X - t)
    h = tex.shape[0]
    # texture pixel i covers board [i, i+1) mm; remap samples pixel centres at i
    mx = (Xb[:, 0] / MM + margin - 0.5).astype(np.float32).reshape(H, W)
    my = (h - margin - Xb[:, 1] / MM - 0.5).astype(np.float32).reshape(H, W)
    bad = (lam <= 0).reshape(H, W)
    mx[bad] = -1
    hi = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderValue=255)
    return cv2.resize(hi, RES, interpolation=cv2.INTER_AREA)


POSES = [((180, 8, 0), (-0.40, -0.28, 0.85)), ((175, -12, 4), (-0.45, -0.25, 1.0)),
         ((195, 5, -3), (-0.35, -0.32, 0.75)), ((185, 15, 8), (-0.50, -0.30, 1.1))]


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    w = tmp_path_factory.mktemp("synthetic")
    cfg = w / "config"
    shutil.copytree(PKG / "config", cfg)
    tex, margin = _texture()
    for n, (rot_deg, t) in enumerate(POSES):
        R0 = cv2.Rodrigues(np.radians(np.array(rot_deg, float)))[0]
        t0 = np.array(t, float)
        R1, t1 = T10[:3, :3] @ R0, T10[:3, :3] @ t0 + T10[:3, 3]
        ts = 1_792_020_000_000_000_000 + n * 66_666_667
        for cam, (R, tt) in (("cam0", (R0, t0)), ("cam1", (R1, t1))):
            (w / "heldout2" / cam).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(w / "heldout2" / cam / f"{ts}.png"), _render(tex, margin, R, tt))
    cam = lambda: {"camera_model": "pinhole", "distortion_model": "equidistant",
                   "distortion_coeffs": D.ravel().tolist(),
                   "intrinsics": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])], "resolution": list(RES)}
    (w / "camchain.yaml").write_text(yaml.safe_dump(
        {"cam0": cam(), "cam1": {**cam(), "T_cn_cnm1": T10.tolist()}}))
    return w


def _run(module, monkeypatch, argv, capsys):
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return capsys.readouterr().out


def test_step60_passes_the_true_calibration_on_rendered_images(workdir, monkeypatch, capsys):
    out = _run(validate, monkeypatch,
               ["60", str(workdir / "camchain.yaml"), "--heldout", str(workdir / "heldout2"),
                "--rig", str(workdir / "config" / "rig-phantom12.yaml")], capsys)
    assert "pairs used 4" in out
    assert "VERDICT: PASS" in out
    med = float(out.split("epipolar |dy|")[1].split("median")[1].split()[0])
    assert med < 0.15        # detector + renderer floor, well under the 0.5 px gate
    mono = float(out.split("mono reproj RMS")[1].split("mean")[1].split()[0])
    assert mono < 0.5        # obj_points convention agrees with the detector


def test_step65_recovers_the_board_pitch(workdir, monkeypatch, capsys):
    out = _run(scale, monkeypatch,
               ["65", str(workdir / "camchain.yaml"), "--tree", str(workdir / "heldout2"),
                "--pitch-m", "0.12", "--rig", str(workdir / "config" / "rig-phantom12.yaml")],
               capsys)
    assert "VERDICT: PASS" in out
    med = float(out.split("pitch: median")[1].split()[0])
    assert med == pytest.approx(1.0, abs=0.003)


def test_step65_catches_a_one_percent_wrong_board_size(workdir, monkeypatch, capsys):
    # the board was really 1 % larger than declared -> the fit's scale is 1 % short
    with pytest.raises(SystemExit) as e:
        _run(scale, monkeypatch,
             ["65", str(workdir / "camchain.yaml"), "--tree", str(workdir / "heldout2"),
              "--pitch-m", str(0.12 * 1.01),
              "--rig", str(workdir / "config" / "rig-phantom12.yaml")], capsys)
    assert e.value.code == 1


def test_detected_corners_land_on_the_true_x_junctions():
    # OpenCV's ArUco quads merge each tag's border with Kalibr's corner squares and
    # miss the corner by ~2 px; refining at the X-junction must bring the typical
    # error to a small fraction of a pixel.
    rigcfg = importlib.import_module("rigcfg")
    board = rigcfg.board_from_target({"target_type": "aprilgrid", "tagCols": COLS,
                                      "tagRows": ROWS, "tagSize": TAG, "tagSpacing": SPACING})
    tex, margin = _texture()
    errs = []
    for rot_deg, t in POSES[:2]:
        R0 = cv2.Rodrigues(np.radians(np.array(rot_deg, float)))[0]
        t0 = np.array(t, float)
        found = board.detect(_render(tex, margin, R0, t0))
        assert len({tid for tid, _ in found}) >= 25
        for (tid, k), uv in found.items():
            truth = cv2.fisheye.projectPoints(board.obj_points(tid)[k].reshape(1, 1, 3),
                                              cv2.Rodrigues(R0)[0], t0.reshape(3, 1), K, D)[0]
            errs.append(np.linalg.norm(np.array(uv) - truth.ravel()))
    # ArUco quads alone: ~2 px. The bound is detector noise well under step 60's
    # 0.5 px gate; ~0.1 px is also this renderer's floor at 2x supersampling.
    assert np.median(errs) < 0.15
    assert np.percentile(errs, 90) < 0.3
