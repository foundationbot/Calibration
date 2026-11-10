"""calibtool.placement: the board-placement model. From one neck pose's corners the board's pose in the torso is fixed;
every other neck pose then predicts where the board appears in each camera (neck chain from the repo URDF, camera in
Head X = T_BS . T_SC0 from the session's OKVIS config).

Synthetic tests pin the geometry; the stored-session test measures the predicted vs measured camera-0 board centroid
on every pose of every stored take (needs CALIBTOOL_DATA)."""
import contextlib
import io

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from calibtool import paths, placement as pl  # noqa: E402

W, H = 1280, 800


def _rig():
    import rigcfg
    with contextlib.redirect_stdout(io.StringIO()):
        return rigcfg.load(paths.CONFIG / "rig-phantom21.yaml")


def _head():
    """a forward-looking camera 9 cm ahead of the Head origin, 10 deg nose-down; cam1 4.4 cm to its right"""
    import rigmath
    c, s = np.cos(np.radians(10)), np.sin(np.radians(10))
    # camera axes in Head: x = robot's right (-y), y = down (-z), z = forward (+x), then pitched down 10 deg
    R0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    Rp = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])       # about Head y (left): +angle tips forward down
    X = rigmath.transform(Rp @ R0, [0.09, 0.0, 0.2])
    cam0 = pl.Camera("pinhole-equi", np.array([[565.0, 0, 640], [0, 565.0, 400], [0, 0, 1]]), np.zeros(4), (W, H))
    C01 = rigmath.transform(np.eye(3), [0.044, 0, 0])
    return pl.Head(X=X, cams=[cam0, cam0], C=[np.eye(4), C01])


def _board_T(chain, head, dist=1.0, bearing_deg=0.0, dz=-0.3):
    """a board facing the camera at home, its centre `dist` m out along the floor at bearing (+: robot's left), dz
    above the camera"""
    return pl.facing_pose(chain, head, _rig().board, dist, bearing_deg, dz)


def test_facing_pose_distance_is_along_the_floor_and_faces_the_camera():
    chain, head, rig = pl.chain(), _head(), _rig()
    T = pl.facing_pose(chain, head, rig.board, 0.75, 30.0, -1.0)
    m = pl.BoardModel(T, rig.board, head, chain)
    cam = (chain.T_TB(0.0, 0.0) @ head.X)[:3, 3]
    c = T[:3, :3] @ m.centre + T[:3, 3]
    d = c - cam
    assert np.hypot(d[0], d[1]) == pytest.approx(0.75, abs=1e-9)
    assert d[2] == pytest.approx(-1.0, abs=1e-9)
    assert np.degrees(np.arctan2(d[1], d[0])) == pytest.approx(30.0, abs=1e-9)
    assert T[:3, 2] @ (-d) / np.linalg.norm(d) == pytest.approx(1.0, abs=1e-9)       # +z of the board: to the camera
    assert abs(T[2, 0]) < 1e-9                                                       # rows level


def _obs_at(model, pitch, yaw):
    p = model.predict(pitch, yaw)
    return pl.HoldObs(tag=p.keys[:, 0], k=p.keys[:, 1], u=p.uv[:, 0], v=p.uv[:, 1], neck_pitch=pitch, neck_yaw=yaw)


def test_from_first_pose_recovers_board_and_predicts_other_poses():
    chain, head, rig = pl.chain(), _head(), _rig()
    T = _board_T(chain, head, 1.1, 10.0)
    truth = pl.BoardModel(T, rig.board, head, chain)
    obs = _obs_at(truth, np.radians(-8), np.radians(5))
    assert len(obs.u) == 140                        # 35 tags x 4: all seen
    m = pl.BoardModel.from_first_pose(obs, rig, head.cams[0], head.X, chain)
    assert np.abs(m.T_TW[:3, 3] - T[:3, 3]).max() < 1e-3
    for q in [(-0.3, 0.4), (0.2, -0.5), (0.0, 0.0)]:
        a, b = truth.predict(*q), m.predict(*q)
        assert a.detected == b.detected
        if a.detected:
            assert np.linalg.norm(a.centroid - b.centroid) < 0.5


def test_board_toward_robots_left_appears_image_left():
    """image x grows toward the robot's right: a board to the robot's left (+bearing) sits left of the centre"""
    chain, head, rig = pl.chain(), _head(), _rig()
    left = pl.BoardModel(_board_T(chain, head, 1.5, 25.0), rig.board, head, chain).predict(0.0, 0.0)
    right = pl.BoardModel(_board_T(chain, head, 1.5, -25.0), rig.board, head, chain).predict(0.0, 0.0)
    assert left.centroid[0] < 640 - 200 and right.centroid[0] > 640 + 200
    # turning the head toward the board (+yaw = to the robot's left) moves it back toward the image centre
    assert abs(pl.BoardModel(_board_T(chain, head, 1.5, 25.0), rig.board, head, chain)
               .predict(0.0, np.radians(25)).centroid[0] - 640) < 60


def test_pitch_down_moves_board_up_in_image():
    chain, head, rig = pl.chain(), _head(), _rig()
    m = pl.BoardModel(_board_T(chain, head, 1.2, 0.0, 0.0), rig.board, head, chain)
    v0 = m.predict(0.0, 0.0).centroid[1]
    assert m.predict(np.radians(15), 0.0).centroid[1] < v0 - 100          # +pitch = nose down: board rises
    assert m.predict(np.radians(-15), 0.0).centroid[1] > v0 + 100


def test_visible_fraction_and_out_of_view():
    chain, head, rig = pl.chain(), _head(), _rig()
    m = pl.BoardModel(_board_T(chain, head, 1.5, 0.0), rig.board, head, chain)
    p = m.predict(0.0, 0.0)
    assert p.visible_frac == 1.0 and p.detected and p.n_tags == 35
    assert 1.3 < p.z < 1.7
    # a board 110 deg to the side is out of view at home (the lens reaches ~65 deg)
    q = pl.BoardModel(_board_T(chain, head, 1.5, 110.0), rig.board, head, chain).predict(0.0, 0.0)
    assert q.visible_frac == 0.0 and not q.detected and np.isnan(q.centroid).all()


def test_partial_view_counts_whole_tags_only():
    chain, head, rig = pl.chain(), _head(), _rig()
    m = pl.BoardModel(_board_T(chain, head, 0.8, 52.0), rig.board, head, chain)
    p = m.predict(0.0, 0.0)
    assert 0.0 < p.visible_frac < 1.0
    assert len(p.uv) % 4 == 0 and len(p.uv) == 4 * p.n_tags
    assert (p.uv[:, 0] >= 0).all() and (p.uv[:, 0] < W).all()


def test_second_camera_sits_right_of_first():
    """cam1 is the right camera: the same board appears farther LEFT in it (positive disparity u0 - u1)"""
    chain, head, rig = pl.chain(), _head(), _rig()
    m = pl.BoardModel(_board_T(chain, head, 1.0, 0.0), rig.board, head, chain)
    a, b = m.predict(0.0, 0.0, cam=0), m.predict(0.0, 0.0, cam=1)
    assert 10 < a.centroid[0] - b.centroid[0] < 40


def test_head_from_okvis_x_is_tbs_tsc0(data_root):
    import okviscfg
    from calibtool import sessions
    st = sessions.get("p10_1106")
    head = pl.head_from_okvis(st.live_config, (W, H))
    cfg = okviscfg.load(st.live_config)
    assert np.allclose(head.X, cfg.T_BS @ cfg.T_SC[0])
    assert np.allclose(head.C[1], np.linalg.inv(cfg.T_SC[0]) @ cfg.T_SC[1])
    assert head.cams[0].lens.kind == "kb4t"


# ---------------------------------------------------------------- stored takes: predicted vs measured centroid


@pytest.mark.parametrize("name", ["p10_1106", "p21_1105"])
def test_stored_takes_centroid_error(data_root, name):
    from calibtool import pipeline, sessions
    from calibtool.session import Session
    st = sessions.get(name)
    s, rig = Session.load(st), pipeline.load_rig(st.rig_path)
    head = pl.head_from_okvis(st.live_config, rig.resolution)
    rep = pl.take_errors(s, rig, head, pl.chain())
    assert set(rep) == set(st.take_order)
    allv = np.concatenate([r["err_px"] for r in rep.values()])
    assert len(allv) > 300
    for t, r in rep.items():
        print(f"{name} {t:10s} poses {len(r['err_px']):3d} median {np.median(r['err_px']):6.1f} "
              f"p90 {np.percentile(r['err_px'], 90):6.1f} px (same corners: median {np.median(r['err_same_px']):6.1f})")
    med, p90 = np.median(allv), np.percentile(allv, 90)
    print(f"{name} ALL poses {len(allv)} median {med:.1f} p90 {p90:.1f} px")
    assert med < 60
