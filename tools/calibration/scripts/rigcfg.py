"""Shared loader for config/rig.yaml — the one place rig-specific inputs live.

Every step imports this instead of carrying its own constants. Resolution order
for the file: --rig PATH, then $CALIB_RIG, then <package>/config/rig.yaml.

Also holds the two things that used to be copy-pasted per script:
  Board  the corner grid, read from the Kalibr target file
  Lens   equidistant vs radial-tangential, so cv2.fisheye.* is not hard-wired
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

PKG = Path(__file__).resolve().parent.parent
DEFAULT_RIG = PKG / "config" / "rig.yaml"

# Kalibr, the Kalibr camchain and OKVIS name the same two models differently. kb4t (equidistant fisheye plus
# tangential p1, p2: lensmodel.py) has no Kalibr or OKVIS name: OpenCV has no such model either.
_LENS = {"pinhole-equi": "equi", "equidistant": "equi", "equi": "equi",
         "pinhole-radtan": "radtan", "radtan": "radtan", "radialtangential": "radtan",
         "pinhole-kb4t": "kb4t", "kb4t": "kb4t"}
OKVIS_NAME = {"equi": "equidistant", "radtan": "radialtangential"}


def add_arg(ap) -> None:
    ap.add_argument("--rig", type=Path, default=None,
                    help=f"rig config (default: $CALIB_RIG or {DEFAULT_RIG})")


class Lens:
    """The projection calls for one lens model. cv2 is imported lazily so a
    script that never projects (step 70) does not need it. kb4t goes through
    lensmodel.py (numpy) with OpenCV's call shapes, so the steps' PnP code is unchanged."""

    def __init__(self, name: str):
        if name not in _LENS:
            sys.exit(f"ERROR: lens model {name!r} is not supported. This package "
                     f"handles pinhole-equi (equidistant fisheye), pinhole-kb4t (equidistant "
                     f"fisheye with tangential terms) and pinhole-radtan.")
        self.kind = _LENS[name]
        self.fisheye = self.kind == "equi"

    def __repr__(self) -> str:
        return f"Lens({self.kind})"

    @staticmethod
    def _kb4t_intr(K, D) -> list:
        K, D = np.asarray(K, float), np.asarray(D, float).ravel()
        if D.size != 6:
            raise ValueError(f"kb4t lens: 6 distortion numbers (k1..k4, p1, p2), got {D.size}")
        if abs(K[0, 1]) > 1e-12:
            raise ValueError(f"kb4t lens: no skew (K[0, 1] = {K[0, 1]})")
        return [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D]

    def undistort(self, pts, K, D, R=None, P=None):
        if self.kind == "kb4t":
            import lensmodel
            rays = lensmodel.unproject("KB4T", self._kb4t_intr(K, D), np.asarray(pts, float).reshape(-1, 2))
            if R is not None:
                rays = rays @ np.asarray(R, float)[:3, :3].T
            x = rays[:, :2] / rays[:, 2:3]
            if P is not None:
                h = np.c_[x, np.ones(len(x))] @ np.asarray(P, float)[:3, :3].T
                x = h[:, :2] / h[:, 2:3]
            return x.reshape(-1, 1, 2)
        import cv2
        kw = {}
        if R is not None:
            kw["R"] = R
        if P is not None:
            kw["P"] = P
        f = cv2.fisheye.undistortPoints if self.fisheye else cv2.undistortPoints
        return f(pts, K, D, **kw)

    def project(self, obj, rvec, tvec, K, D):
        if self.kind == "kb4t":
            import lensmodel
            from rigmath import exp_rot
            Pc = np.asarray(obj, float).reshape(-1, 3) @ exp_rot(np.ravel(rvec)).T + np.ravel(tvec)
            return lensmodel.project("KB4T", self._kb4t_intr(K, D), Pc).reshape(-1, 1, 2), None
        import cv2
        f = cv2.fisheye.projectPoints if self.fisheye else cv2.projectPoints
        return f(obj, rvec, tvec, K, D)

    def stereo_rectify(self, K0, D0, K1, D1, size, R, t, flags):
        """-> (R0, R1, P0, P1)"""
        if self.kind == "kb4t":
            raise NotImplementedError("kb4t: OpenCV has no rectification for this lens; the S2M2 plan's "
                                      "tools/depth_eval builds the map from lensmodel")
        import cv2
        f = cv2.fisheye.stereoRectify if self.fisheye else cv2.stereoRectify
        return f(K0, D0, K1, D1, tuple(size), R, t, flags=flags)[:4]

    def off_axis_rad(self, r, f):
        """nominal off-axis angle of image radius r for focal f."""
        return r / f if self.kind in ("equi", "kb4t") else np.arctan(r / f)


class Board:
    """A checkerboard's INTERNAL corner grid, in cv2.findChessboardCorners order
    (k = row * cols + col)."""

    kind = "checkerboard"
    # A rectangular corner grid is symmetric under a 180 deg turn, so the
    # detector's corner order can flip between frames (step 30 exists for this).
    ambiguous_orientation = True

    def __init__(self, cols: int, rows: int, dx: float, dy: float):
        self.cols, self.rows, self.dx, self.dy = cols, rows, dx, dy
        self.size = (cols, rows)              # patternSize for findChessboardCorners
        self.n = cols * rows

    def obj(self) -> np.ndarray:
        return np.array([[c * self.dx, r * self.dy, 0.0]
                         for r in range(self.rows) for c in range(self.cols)], np.float64)

    def __repr__(self) -> str:
        return f"{self.cols}x{self.rows} corners @ {self.dx*1000:g}x{self.dy*1000:g} mm"


class AprilGrid:
    """A Kalibr aprilgrid: cols x rows AprilTag 36h11 tags, ids row-major.

    Every corner is identified by (tag id, k), k in OpenCV's ArUco corner order,
    so there is no orientation ambiguity and partial views are usable.

    obj_points uses the Kalibr board frame (tag 0 bottom-left, y up). OpenCV sees
    a Kalibr tag rotated 180 deg relative to Kalibr's own corner order, hence the
    rotated list; this is the mapping tools/mk2_xone_stereo_extrinsic.py uses on
    the same board family. The steps here mostly need only the (id, k) identity
    and the pitch, which do not depend on that convention.
    """

    kind = "aprilgrid"
    ambiguous_orientation = False

    def __init__(self, cols: int, rows: int, tag_size: float, spacing: float,
                 border_bits: int = 2):
        self.cols, self.rows = cols, rows
        self.tag_size, self.spacing = tag_size, spacing
        self.pitch = tag_size * (1.0 + spacing)      # Kalibr: tagSpacing is a ratio
        self.n_tags = cols * rows
        self.border_bits = border_bits                # Kalibr tags: 2-cell black border
        self._det = None

    def tag_rc(self, tag_id: int) -> tuple[int, int]:
        return divmod(int(tag_id), self.cols)

    def obj_points(self, tag_id: int) -> np.ndarray:
        r, c = self.tag_rc(tag_id)
        x0, y0, s = c * self.pitch, r * self.pitch, self.tag_size
        cs = [[x0, y0 + s, 0.0], [x0 + s, y0 + s, 0.0], [x0 + s, y0, 0.0], [x0, y0, 0.0]]
        return np.array(cs[2:] + cs[:2], np.float64)

    def detect(self, gray) -> dict:
        """{(tag_id, k): (u, v)} for every corner of every tag of this grid found.

        OpenCV's ArUco detector only IDENTIFIES the tags here. Its quad corners on a
        Kalibr board are off by ~2 px (the tag border merges with Kalibr's corner
        squares; on phantom21's board its corners sat on a 92.3 mm square for a
        100 mm tag). Each corner is therefore re-found by cornerSubPix at the
        X-junction a tag corner forms with its corner square, as Kalibr itself
        does, with a window scaled to the square's size in the image: a larger
        window reaches the tag's inner edges and diverges, one smaller than ArUco's
        miss (3 px) leaves the corner unrefined. A corner whose refinement wanders
        further than the window is dropped.
        """
        import cv2
        if self._det is None:
            p = cv2.aruco.DetectorParameters()
            p.markerBorderBits = self.border_bits
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
            self._det = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)
        corners, ids, _ = self._det.detectMarkers(gray)
        out = {}
        if ids is None:
            return out
        for quad, tid in zip(corners, ids.ravel()):
            if not 0 <= int(tid) < self.n_tags:
                continue
            for k, uv in self.refine(gray, quad).items():
                out[(int(tid), k)] = uv
        return out

    def refine(self, gray, quad) -> dict:
        """{k: (u, v)}: one ArUco quad's corners re-found at their X-junctions (detect's docstring)."""
        import cv2
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
        q = np.asarray(quad).reshape(4, 2).astype(np.float32)
        edge = float(np.mean(np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)))
        # the window must reach the junction from ArUco's guess (~4 % of the tag
        # edge off on a Kalibr board) yet stay inside the corner square
        win = int(np.clip(round(0.3 * edge * self.spacing), 3, 10))
        ref = cv2.cornerSubPix(gray, q.reshape(-1, 1, 2).copy(), (win, win), (-1, -1),
                               crit).reshape(4, 2)
        moved = np.linalg.norm(ref - q, axis=1)
        return {k: (float(ref[k, 0]), float(ref[k, 1])) for k in range(4) if moved[k] <= 2 * win + 1}

    def __repr__(self) -> str:
        return (f"aprilgrid {self.cols}x{self.rows} tags, {self.tag_size*1000:g} mm, "
                f"pitch {self.pitch*1000:g} mm")


def board_from_target(t: dict):
    """Board (checkerboard) or AprilGrid from a parsed Kalibr target file."""
    kind = t.get("target_type")
    if kind == "checkerboard":
        return Board(int(t["targetCols"]), int(t["targetRows"]),
                     float(t["colSpacingMeters"]), float(t["rowSpacingMeters"]))
    if kind == "aprilgrid":
        return AprilGrid(int(t["tagCols"]), int(t["tagRows"]),
                         float(t["tagSize"]), float(t["tagSpacing"]))
    sys.exit(f"ERROR: target_type {kind!r} is not supported (checkerboard or aprilgrid)")


# KB4T lens steps: defaults of the rig file's gates (values of 20 Oct, calibration.md section 4b)
LENS_GATES = {
    "lens_hold_rms_max_px": 0.6,    # a hold's rigid-board pixel rms above this: dropped from the fit sweep (step
                                    # 45), flagged and still judged on the test sweep (step 66) (phantom12: near
                                    # boards 0.30-0.42 px, the bad far boards 0.67-1.13 px)
    "lens_bad_hold_max_frac": 0.20,  # more of the fit sweep's holds dropped than this: step 45 rejects the recording
    "lens_repeat_max_pct": 0.5,     # split-half fits differing by more at 1 m: the cell is "unconstrained"
    "lens_pass_pct": 0.5,           # step 66: |median depth error| limit per cell x distance entry
    "lens_pass_corner_pct": 1.0,    # the same in the four corner cells
    "lens_min_holds": 8,            # holds a cell x distance entry needs to be judged (step 66); step 06 calls a
                                    # cell done at --target holds, by default this + 4 (its live count needs a margin)
}


class Rig:
    def __init__(self, path: Path):
        self.path = path.resolve()
        if not self.path.is_file():
            sys.exit(f"ERROR: rig config {self.path} not found")
        d = yaml.safe_load(self.path.read_text()) or {}
        self.raw = d
        try:
            self.name = d["rig_name"]
            self.okvis_yaml_name = d.get("okvis_yaml_name", "rig.yaml")
            cam = d["cameras"]
            self.model = cam["model"]
            self.lens = Lens(self.model)
            self.resolution = tuple(int(v) for v in cam["resolution"])
            self.nominal_focal_px = float(cam["nominal_focal_px"])
            # stereo baseline before any calibration (CAD or a past fit), m: only the start of the KB4T fit
            # (step 45), which refines it. None when absent; step 45 then refuses.
            nb = cam.get("nominal_baseline_m")
            self.nominal_baseline_m = None if nb is None else float(nb)
            self.kalibr = cam["kalibr"]
            self.okvis_order = [int(v) for v in cam["okvis_order"]]
            self.imu = d["imu"]
            self.robot = d["robot"]
            self.capture = d["capture"]
            self.gates = d["gates"]
            self.charuco = d.get("charuco", {})
            self.target_path = self.resolve(d["target"])
            # false = the target file still holds placeholder board dimensions
            self.target_measured = bool(d.get("target_measured", True))
        except KeyError as e:
            sys.exit(f"ERROR: {self.path}: missing key {e}")
        self._validate()
        self.board = self._load_board()

    # -- helpers --------------------------------------------------------------
    def resolve(self, p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.path.parent / p)

    def _validate(self) -> None:
        bad = lambda m: sys.exit(f"ERROR: {self.path}: {m}")
        if len(self.kalibr) != 2:
            bad(f"cameras.kalibr has {len(self.kalibr)} entries; this package is "
                "stereo-only and needs exactly 2")
        for i, c in enumerate(self.kalibr):
            for k in ("record_dir", "dma_queue", "side"):
                if k not in c:
                    bad(f"cameras.kalibr[{i}] has no '{k}'")
        if sorted(c["side"] for c in self.kalibr) != ["left", "right"]:
            bad("cameras.kalibr sides must be one 'left' and one 'right'")
        if len({c["record_dir"] for c in self.kalibr}) != 2:
            bad("cameras.kalibr record_dir values must differ")
        if len({int(c["dma_queue"]) for c in self.kalibr}) != 2:
            bad("cameras.kalibr dma_queue values must differ")
        if sorted(self.okvis_order) != [0, 1]:
            bad("cameras.okvis_order must be [0, 1] or [1, 0]")

    def _load_board(self):
        if not self.target_path.is_file():
            sys.exit(f"ERROR: target file {self.target_path} (from {self.path}) not found")
        return board_from_target(yaml.safe_load(self.target_path.read_text()))

    # -- derived views --------------------------------------------------------
    def cam_map(self) -> list[tuple[str, str]]:
        """[(record_dir, kalibr name)] in kalibr order."""
        return [(c["record_dir"], f"cam{i}") for i, c in enumerate(self.kalibr)]

    def okvis_from_kalibr(self) -> list[str]:
        """kalibr camera name feeding OKVIS cam i."""
        return [f"cam{k}" for k in self.okvis_order]

    def okvis_queues(self) -> list[int]:
        return [int(self.kalibr[k]["dma_queue"]) for k in self.okvis_order]

    def okvis_sides(self) -> list[str]:
        return [self.kalibr[k]["side"] for k in self.okvis_order]

    def expected_dx_sign(self) -> int:
        """sign of median(u_kalibr0 - u_kalibr1) after rectification: positive
        when kalibr cam0 is the physical LEFT eye."""
        return +1 if self.kalibr[0]["side"] == "left" else -1

    def camchain_lens(self, cc: dict, name: str) -> Lens:
        """The lens model a camchain camera declares, checked against this rig."""
        c = cc[name]
        if c.get("camera_model", "pinhole") != "pinhole":
            sys.exit(f"ERROR: camchain {name}: camera_model {c.get('camera_model')!r} "
                     "is not supported (pinhole only)")
        lens = Lens(c.get("distortion_model", self.model))
        if lens.kind != self.lens.kind:
            sys.exit(f"ERROR: camchain {name} is {c.get('distortion_model')!r} but "
                     f"{self.path} says cameras.model = {self.model!r}. Fix whichever "
                     "is wrong; the projection calls depend on it.")
        res = c.get("resolution")
        if res and tuple(int(v) for v in res) != self.resolution:
            sys.exit(f"ERROR: camchain {name} resolution {res} != "
                     f"cameras.resolution {list(self.resolution)} in {self.path}")
        return lens

    def lens_gates(self) -> dict:
        """the KB4T lens steps' thresholds (06, 45, 46, 66): this rig file's gates: over the 20 Oct defaults"""
        return {k: type(v)(self.gates.get(k, v)) for k, v in LENS_GATES.items()}

    def t_bs_default_tokens(self) -> list[str]:
        """the 16 numbers of the fallback T_BS, as written in their source file
        (OpenCV-YAML, so parsed by regex rather than PyYAML)."""
        if not self.imu.get("T_BS_default_file"):
            sys.exit(f"ERROR: {self.path} sets no imu.T_BS_default_file: this rig borrows "
                     "no other robot's IMU-in-Head transform, so the step-70 template "
                     "must carry this robot's own T_BS.")
        p = self.resolve(self.imu["T_BS_default_file"])
        key = self.imu.get("T_BS_default_key", "T_BS_DEFAULT")
        m = re.search(r"^" + re.escape(key) + r":\s*\[([^\]]+)\]", p.read_text(), re.M)
        if not m:
            sys.exit(f"ERROR: no '{key}: [...]' in {p}")
        tok = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))
        if len(tok) != 16:
            sys.exit(f"ERROR: {p} {key} has {len(tok)} numbers, expected 16")
        return tok


def load(path: Path | None = None) -> Rig:
    p = path or os.environ.get("CALIB_RIG") or DEFAULT_RIG
    rig = Rig(Path(p))
    print(f"rig config: {rig.path}  ({rig.name}; {rig.model} "
          f"{rig.resolution[0]}x{rig.resolution[1]}; board {rig.board})")
    if not rig.target_measured:
        print(f"  ⚠ MEASURE the board: {rig.target_path.name} holds placeholder dimensions "
              "(target_measured: false).\n    The declared size scales the baseline and "
              "every depth: 1 % size error = 1 % depth error.")
    return rig


if __name__ == "__main__":
    # `python3 scripts/rigcfg.py [rig.yaml]` — validate a rig file and show
    # what every step will derive from it.
    r = load(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    print(f"  record -> kalibr      : {r.cam_map()}")
    for i, (k, q, s) in enumerate(zip(r.okvis_from_kalibr(), r.okvis_queues(), r.okvis_sides())):
        print(f"  OKVIS cam{i}            : kalibr {k}, dma_queue {q}, physical {s.upper()}")
    print(f"  expected disparity    : median(u_kalibr0 - u_kalibr1) "
          f"{'> 0' if r.expected_dx_sign() > 0 else '< 0'}")
    print(f"  IMU                   : {r.imu['name']} index {r.imu['index']}, "
          f"Kalibr model {r.resolve(r.imu['kalibr_yaml'])}")
    print(f"  fallback T_BS         : "
          + (f"{' '.join(r.t_bs_default_tokens()[:4])} ..." if r.imu.get("T_BS_default_file")
             else "none; the step-70 template must carry T_BS"))
    print(f"  robot                 : {r.robot['host']}")
