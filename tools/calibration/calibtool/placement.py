"""The board-placement model: where a board standing still in the room appears in each head camera at any neck pose.

From one neck pose (the first pose of a take, or the home pose of a live placement) the board's pose in the camera
(PnP of the camera-0 corners, refined on pixels) and the neck chain fix the board in the torso frame:

    T_TW = T_TB(q0) . X . T_C0W          X = camera 0 in the Head link = T_BS . T_SC0 (the session's OKVIS config)

and any other neck pose q predicts T_CcW = (T_TB(q) . X . T_C0Cc)^-1 . T_TW, projected with the camera's lens. The
neck chain is the repo URDF (neckfk.NeckChain, measured angles in rad with the URDF's signs: +yaw turns the head to
the robot's left, +pitch tips the nose down). The torso is assumed still.

Which corners count as seen (the detection model): whole tags only, every corner of the tag inside the image (MARGIN_PX
from the border), the tag at least MIN_TAG_PX across and seen at most MAX_VIEW_DEG off its normal; the board counts as
detected with at least MIN_TAGS tags (the rig plan's >= 6 tags)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibtool import paths

URDF = paths.CAL.parents[1] / "config" / "foundation_rig" / "20250206_phantom_mk1.urdf"
MIN_TAGS = 6
MARGIN_PX = 2.0
MIN_TAG_PX = 10.0
MAX_VIEW_DEG = 75.0


def chain(urdf=None):
    """neckfk.NeckChain of the repo URDF (or of `urdf`)"""
    import neckfk
    return neckfk.NeckChain(Path(urdf) if urdf else URDF)


class Camera:
    """one camera's lens: rigcfg.Lens name ("pinhole-equi" | "pinhole-kb4t" | "pinhole-radtan"), K, D, size (W, H)"""

    def __init__(self, lens, K, D, size):
        import rigcfg
        self.lens = lens if isinstance(lens, rigcfg.Lens) else rigcfg.Lens(lens)
        self.K = np.asarray(K, float)
        self.D = np.asarray(D, float).reshape(-1, 1)
        self.size = tuple(int(v) for v in size)

    def project(self, Pc) -> np.ndarray:
        """camera-frame points (N, 3), in front of the camera -> pixels (N, 2)"""
        Pc = np.asarray(Pc, float).reshape(-1, 1, 3)
        if not len(Pc):
            return np.zeros((0, 2))
        uv, _ = self.lens.project(Pc, np.zeros((3, 1)), np.zeros((3, 1)), self.K, self.D)
        return np.asarray(uv, float).reshape(-1, 2)

    def normalised(self, uv) -> np.ndarray:
        """pixels (N, 2) -> pinhole-normalised points (N, 2)"""
        return np.asarray(self.lens.undistort(np.asarray(uv, float).reshape(-1, 1, 2), self.K, self.D)).reshape(-1, 2)


@dataclass
class Head:
    X: np.ndarray       # camera 0 in the Head link (4x4)
    cams: list          # [Camera, Camera]: OKVIS camera 0 (left), 1 (right)
    C: list             # [I, T_C0C1]: each camera in camera 0


def head_from_okvis(path, size) -> Head:
    """the head cameras of an OKVIS config: X = T_BS . T_SC0, lenses, camera 1 in camera 0"""
    import okviscfg
    import rigmath
    cfg = okviscfg.load(path)
    cams = [Camera(cfg.lens, cfg.K[i], cfg.D[i], size) for i in range(2)]
    C = [np.eye(4), rigmath.inv(cfg.T_SC[0]) @ cfg.T_SC[1]]
    return Head(X=cfg.T_BS @ cfg.T_SC[0], cams=cams, C=C)


@dataclass
class HoldObs:
    """camera-0 corners seen at one neck pose (one or more pairs of one hold) and the neck angles (rad)"""
    tag: np.ndarray
    k: np.ndarray
    u: np.ndarray
    v: np.ndarray
    neck_pitch: float
    neck_yaw: float

    @classmethod
    def from_pairs(cls, session, rows, cam=0):
        """the camera's corners of the pair rows (one hold), neck angles averaged over them"""
        rows = np.atleast_1d(rows)
        C = session.corners
        m = np.isin(C["pair"], rows) & (C["cam"] == cam)
        return cls(C["tag"][m], C["k"][m], C["u"][m], C["v"][m],
                   float(np.mean(session.pairs["neck_pitch_rad"][rows])),
                   float(np.mean(session.pairs["neck_yaw_rad"][rows])))

    @property
    def n_tags(self) -> int:
        return len(np.unique(self.tag))


@dataclass
class Prediction:
    centroid: np.ndarray    # (2,) mean pixel of the seen corners; nan if none
    visible_frac: float     # seen corners / board corners
    uv: np.ndarray          # (n, 2) seen corners' pixels
    keys: np.ndarray        # (n, 2) their (tag, k)
    n_tags: int
    z: float                # camera-frame depth (m) of the seen corners' centre (the board's centre if none seen)
    detected: bool          # n_tags >= MIN_TAGS


def board_corners(board):
    """every corner of the board: keys (n, 2) (tag, k) and object points (n, 3), tag-major"""
    keys = np.array([(t, k) for t in range(board.n_tags) for k in range(4)], np.int64)
    obj = np.concatenate([board.obj_points(t) for t in range(board.n_tags)])
    return keys, obj


def pnp(obs: HoldObs, board, cam: Camera) -> np.ndarray:
    """board pose in the camera (T_CW, 4x4) from its corners: PnP on normalised points, refined on pixels (Gauss-Newton,
    numeric Jacobian)"""
    import cv2
    import rigmath
    obj = np.array([board.obj_points(int(t))[int(k)] for t, k in zip(obs.tag, obs.k)], float)
    uv = np.c_[obs.u, obs.v].astype(float)
    ok, rv, tv = cv2.solvePnP(obj, cam.normalised(uv), np.eye(3), None)
    if not ok:
        raise ValueError("PnP failed")
    p = np.r_[np.ravel(rv), np.ravel(tv)]

    def res(p):
        R = rigmath.exp_rot(p[:3])
        return (cam.project(obj @ R.T + p[3:]) - uv).ravel()

    for _ in range(10):
        r = res(p)
        J = np.empty((r.size, 6))
        for i in range(6):
            d = np.zeros(6)
            d[i] = 1e-7
            J[:, i] = (res(p + d) - res(p - d)) / 2e-7
        dp = -np.linalg.lstsq(J, r, rcond=None)[0]
        p = p + dp
        if np.linalg.norm(dp) < 1e-10:
            break
    return rigmath.transform(rigmath.exp_rot(p[:3]), p[3:])


def facing_pose(chain_, head: Head, board, ahead_m, bearing_deg, dz, at=(0.0, 0.0)) -> np.ndarray:
    """T_TW of a board whose centre is `ahead_m` m from camera 0 along the floor (the torso's horizontal plane), at
    `bearing_deg` from straight ahead (+: toward the robot's left) and dz m above the camera (negative: below), camera 0
    taken at neck pose at=(pitch, yaw) rad; its face (+z, as measured on the stored takes) turned to the camera, its
    rows level"""
    import rigmath
    cam = (chain_.T_TB(at[1], at[0]) @ head.X)[:3, 3]
    b = np.radians(bearing_deg)
    centre = cam + np.array([ahead_m * np.cos(b), ahead_m * np.sin(b), dz])
    n = (cam - centre) / np.linalg.norm(cam - centre)             # the board's +z faces the camera (as measured)
    up = np.array([0.0, 0.0, 1.0])
    y = up - (up @ n) * n
    y = y / np.linalg.norm(y) if np.linalg.norm(y) > 1e-9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(y, n)
    R = np.c_[x, np.cross(n, x), n]
    _, obj = board_corners(board)
    return rigmath.transform(R, centre - R @ obj.mean(0))


class BoardModel:
    """a board still in the torso frame (T_TW) seen by the head's cameras through the neck chain"""

    def __init__(self, T_TW, board, head: Head, chain_):
        self.T_TW = np.asarray(T_TW, float)
        self.board = board
        self.head = head
        self.chain = chain_
        self.keys, self.obj = board_corners(board)
        self.centre = self.obj.mean(0)
        self.n_tags_board = board.n_tags

    @classmethod
    def from_first_pose(cls, hold: HoldObs, rig, lens: Camera, X, chain_, head: Head = None):
        """the board fixed by one pose's camera-0 corners (hold), lens = camera 0, X = camera 0 in the Head link.
        head (optional): the full head, for predicting camera 1 too (its X and camera 0 must be these)"""
        import rigmath  # noqa: F401
        T_C0W = pnp(hold, rig.board, lens)
        T_TW = chain_.T_TB(hold.neck_yaw, hold.neck_pitch) @ np.asarray(X, float) @ T_C0W
        if head is None:
            head = Head(X=np.asarray(X, float), cams=[lens], C=[np.eye(4)])
        return cls(T_TW, rig.board, head, chain_)

    def T_CW(self, neck_pitch, neck_yaw, cam=0) -> np.ndarray:
        import rigmath
        return rigmath.inv(self.chain.T_TB(neck_yaw, neck_pitch) @ self.head.X @ self.head.C[cam]) @ self.T_TW

    def project_keys(self, neck_pitch, neck_yaw, keys, cam=0) -> np.ndarray:
        """pixels of given corners (tag, k) at a neck pose (nan behind the camera)"""
        T = self.T_CW(neck_pitch, neck_yaw, cam)
        obj = np.array([self.board.obj_points(int(t))[int(k)] for t, k in np.asarray(keys).reshape(-1, 2)], float)
        Pc = obj @ T[:3, :3].T + T[:3, 3]
        out = np.full((len(Pc), 2), np.nan)
        f = Pc[:, 2] > 0.05
        out[f] = self.head.cams[cam].project(Pc[f])
        return out

    def predict(self, neck_pitch, neck_yaw, cam=0) -> Prediction:
        """the board as camera `cam` sees it at neck angles (rad)"""
        T = self.T_CW(neck_pitch, neck_yaw, cam)
        camera = self.head.cams[cam]
        W, H = camera.size
        Pc = self.obj @ T[:3, :3].T + T[:3, 3]
        uv = np.full((len(Pc), 2), np.nan)
        front = Pc[:, 2] > 0.05
        uv[front] = camera.project(Pc[front])
        inside = (front & np.isfinite(uv).all(1) & (uv[:, 0] >= MARGIN_PX) & (uv[:, 0] <= W - 1 - MARGIN_PX)
                  & (uv[:, 1] >= MARGIN_PX) & (uv[:, 1] <= H - 1 - MARGIN_PX))
        nt = self.n_tags_board
        tin = inside.reshape(nt, 4).all(1)
        q = np.where(inside[:, None], uv, 0.0).reshape(nt, 4, 2)
        size = np.linalg.norm(q - np.roll(q, 1, axis=1), axis=2).mean(1)
        n_c = T[:3, 2]
        ctr = Pc.reshape(nt, 4, 3).mean(1)
        cosang = np.abs(ctr @ n_c) / np.maximum(np.linalg.norm(ctr, axis=1), 1e-9)
        ok = tin & (size >= MIN_TAG_PX) & (cosang >= np.cos(np.radians(MAX_VIEW_DEG)))
        seen = np.repeat(ok, 4)
        n_tags = int(ok.sum())
        if seen.any():
            centroid = uv[seen].mean(0)
            z = float(Pc[seen].mean(0)[2])
        else:
            centroid = np.full(2, np.nan)
            z = float((T[:3, :3] @ self.centre + T[:3, 3])[2])
        return Prediction(centroid, float(seen.mean()), uv[seen], self.keys[seen], n_tags, z, n_tags >= MIN_TAGS)


def first_usable_hold(session, take, min_tags=MIN_TAGS):
    """the take's first hold (capture order) whose first pair shows >= min_tags camera-0 tags; None if none"""
    C = session.corners
    for h in session.holds():
        if h.take != take:
            continue
        i = h.pairs[0]
        m = (C["pair"] == i) & (C["cam"] == 0)
        if len(np.unique(C["tag"][m])) >= min_tags:
            return h
    return None


def take_errors(session, rig, head: Head, chain_, min_tags=MIN_TAGS) -> dict:
    """per take: the model fixed on its first usable pose, then for every later pose (each hold's first pair) the
    camera-0 centroid error (px) of the prediction vs the measured corners. -> {take: {"err_px": predicted seen
    corners' centroid vs measured centroid, "err_same_px": the measured corners themselves predicted, "missed": poses
    with corners measured but none predicted seen, "first_hold": hold id}}"""
    out = {}
    C = session.corners
    for take in session.take_order:
        h0 = first_usable_hold(session, take, min_tags)
        if h0 is None:
            continue
        model = BoardModel.from_first_pose(HoldObs.from_pairs(session, h0.pairs[:1]), rig, head.cams[0], head.X,
                                           chain_, head)
        err, same, missed = [], [], 0
        for h in session.holds():
            if h.take != take or h.hold == h0.hold:
                continue
            obs = HoldObs.from_pairs(session, h.pairs[:1])
            if not len(obs.u):
                continue
            meas = np.array([obs.u.mean(), obs.v.mean()])
            p = model.predict(obs.neck_pitch, obs.neck_yaw)
            if p.n_tags == 0:
                missed += 1
                continue
            err.append(float(np.linalg.norm(p.centroid - meas)))
            pk = model.project_keys(obs.neck_pitch, obs.neck_yaw, np.c_[obs.tag, obs.k])
            same.append(float(np.linalg.norm(np.nanmean(pk, 0) - meas)))
        out[take] = {"err_px": np.array(err), "err_same_px": np.array(same), "missed": missed, "first_hold": h0.hold}
    return out
