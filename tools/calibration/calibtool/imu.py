"""Camera-to-IMU rotation v1 from an AprilGrid lying flat on the floor during a neck sweep (design plan 9.3).

Data (`Sweep`): per still neck pose, the IMU slot's mean acceleration (the reaction to gravity: "up" in the IMU frame)
and the board's normal from PnP in camera 0, flipped toward the camera (the floor's "up" in the camera frame). With
the config's cam0 rotation R_SC (camera -> IMU), R_SC.T @ up_imu should equal the board normal; the angle between them
is the pose's floor-vs-gravity gap.

Model (the kit's rscfit.py, reproduced): R_new = R_SC @ Rx(ax) @ Rz(az), two angles about camera x and z; robust
soft-L1 least squares on the stacked cross products (f_scale 0.02) from (0, 0); formal 1-sigma from pinv(J^T J) x
median(r^2). The third angle, about the gravity direction, is (nearly) unobservable from floor sweeps: neck yaw turns
about a near-vertical axis and changes neither gravity in the IMU frame nor the floor normal in the camera frame, so
only the pitch span separates the up directions (one pitch row makes it exactly unobservable). `observability`
measures it; v1 keeps the live config's value.

Scoring is on held-out poses (fitting and scoring on the same poses would make "improves" circular): leave one yaw
column out (fit on the other columns, score the left-out one), pooled over the columns; the two pitch halves (fit on
one, score the other) are reported too. `verdict` deploys the rotation only if the pooled held-out median gap improves
on the live config's, on the same poses, by more than k x the fit's combined sigma (hypot of the two formal sigmas);
the thresholds (`Gates`) are named placeholders until fixed with the user.

`apply` is the kit's mkfinal.py rule: cam0 T_SC rotation R0 -> R0 @ R_correction, cam1 = new cam0 x the config's own
stereo pose inv(T0) @ T1; cam0 translation, T_BS and every other byte kept. Only the changed numbers are rewritten,
each in its own place (repr of the float), so a diff shows nothing else. The kit's 83_tbs_apply.py is not used (it
writes T_BS). Changing T_SC with T_BS kept moves the camera in the Head frame for other consumers (camera-to-arm): the
verdict says so."""
import re
from dataclasses import dataclass, field

import numpy as np

MAX_ACC_STD = 0.4        # m/s^2: a pose whose IMU std reaches this on any axis was not still (rscfit)
SOFT_L1_SCALE = 0.02     # rscfit's f_scale, in |cross| units (sin of the gap: ~1.1 deg)
COLUMN_GAP_DEG = 5.0     # neck yaw values further apart than this start a new yaw column
ROW_GAP_DEG = 3.0        # neck pitch values further apart than this start a new pitch row
MIN_FIT_POSES = 3        # two angles from fewer poses is not a fit


# ---------------------------------------------------------------- rotations


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_xz(ax_deg, az_deg):
    """rscfit's correction: Rx(ax) @ Rz(az), angles in degrees about camera x and z"""
    return _rx(np.radians(ax_deg)) @ _rz(np.radians(az_deg))


def axis_rotation(axis, deg):
    """rotation by `deg` about `axis` (any length)"""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def rotation_angle_deg(Ra, Rb):
    """the angle of the rotation taking Ra to Rb"""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# ---------------------------------------------------------------- the sweep


@dataclass
class Sweep:
    """still neck poses kept for the fit (rscfit's filter applied)"""
    neck: np.ndarray        # (N, 2) measured neck pitch, yaw (deg)
    acc: np.ndarray         # (N, 3) IMU acceleration mean (IMU frame, m/s^2): "up" when still
    acc_std: np.ndarray     # (N, 3) its standard deviation over the window
    normal: np.ndarray      # (N, 3) board normal in camera 0, unit, toward the camera ("up")
    R_SC: np.ndarray        # (3, 3) the cam0 rotation of the config the sweep ran with
    n_recorded: int = 0     # poses recorded before the filter

    @property
    def up_imu(self):
        return self.acc / np.linalg.norm(self.acc, axis=1, keepdims=True)

    def __len__(self):
        return len(self.normal)


def sweep_from_arrays(neck, acc, acc_std, normal, R_SC, max_acc_std=MAX_ACC_STD) -> Sweep:
    """keep the poses rscfit keeps: every acc_std axis below max_acc_std and a finite normal"""
    neck, acc, acc_std, normal = (np.asarray(a, float).reshape(-1, k) for a, k in
                                  ((neck, 2), (acc, 3), (acc_std, 3), (normal, 3)))
    ok = (acc_std.max(axis=1) < max_acc_std) & np.isfinite(normal).all(axis=1) & np.isfinite(acc).all(axis=1)
    return Sweep(neck[ok], acc[ok], acc_std[ok], normal[ok], np.asarray(R_SC, float).reshape(3, 3), int(len(ok)))


def load_sweep(path, key="nb", max_acc_std=MAX_ACC_STD) -> Sweep:
    """an imucam_fast.py / imusweep.save_sweep npz (neck, acc, acc_std, nb, n, R_SC); `key` picks the normals"""
    with np.load(path) as z:
        return sweep_from_arrays(z["neck"], z["acc"], z["acc_std"], z[key], z["R_SC"], max_acc_std)


def _idx(sweep, idx):
    return np.arange(len(sweep)) if idx is None else np.asarray(idx, int)


def floor_gap_deg(sweep: Sweep, R, idx=None):
    """per pose: the angle between the board normal and R.T @ up_imu (deg)"""
    idx = _idx(sweep, idx)
    v = sweep.up_imu[idx] @ np.asarray(R, float)            # rows (R.T @ u).T
    n = sweep.normal[idx]
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(v, n), axis=1), np.einsum("ij,ij->i", v, n)))


# ---------------------------------------------------------------- the fit


@dataclass
class Fit:
    """rscfit's two-angle correction fitted on the poses `idx`"""
    angles_deg: np.ndarray      # (2,) about camera x, about camera z
    sigma_deg: np.ndarray       # (2,) formal 1-sigma
    gap_deg: np.ndarray         # per fitted pose, after the correction
    idx: np.ndarray

    @property
    def correction(self):
        """R_SC -> R_SC @ correction"""
        return rot_xz(*self.angles_deg)

    @property
    def combined_sigma_deg(self):
        return float(np.hypot(*self.sigma_deg))


def fit(sweep: Sweep, idx=None) -> Fit:
    """rscfit.fit: soft-L1 least squares on the cross products of R_new.T @ up_imu and the board normal"""
    from scipy.optimize import least_squares
    idx = _idx(sweep, idx)
    if len(idx) < MIN_FIT_POSES:
        raise ValueError(f"{len(idx)} poses: the two-angle fit needs at least {MIN_FIT_POSES}")
    U, N, R = sweep.up_imu[idx], sweep.normal[idx], sweep.R_SC

    def f(p):
        return np.cross(U @ (R @ _rx(p[0]) @ _rz(p[1])), N).ravel()

    s = least_squares(f, [0.0, 0.0], loss="soft_l1", f_scale=SOFT_L1_SCALE)
    cov = np.linalg.pinv(s.jac.T @ s.jac) * np.median(s.fun ** 2)
    angles = np.degrees(s.x)
    return Fit(angles, np.degrees(np.sqrt(np.diag(cov))), floor_gap_deg(sweep, R @ rot_xz(*angles), idx), idx)


def R_new(sweep: Sweep, f: Fit):
    """the cam0 rotation the fit gives"""
    return sweep.R_SC @ f.correction


# ---------------------------------------------------------------- held-out scoring


def clusters(values, gap):
    """labels 0, 1, ... in ascending order of value; a step larger than `gap` between sorted neighbours starts a new
    label (the commanded grid is not stored, only the measured neck angles)"""
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), int)
    k = 0
    for j, i in enumerate(order):
        if j and values[i] - values[order[j - 1]] > gap:
            k += 1
        labels[i] = k
    return labels


def yaw_columns(sweep: Sweep, gap=COLUMN_GAP_DEG):
    return clusters(sweep.neck[:, 1], gap)


def pitch_rows(sweep: Sweep, gap=ROW_GAP_DEG):
    return clusters(sweep.neck[:, 0], gap)


@dataclass
class Fold:
    """fit on fit_idx, score on test_idx: the new rotation's gaps and the live rotation's on the same poses"""
    name: str
    fit_idx: np.ndarray
    test_idx: np.ndarray
    angles_deg: np.ndarray
    gap_new_deg: np.ndarray
    gap_live_deg: np.ndarray


@dataclass
class HeldOut:
    """columns: leave-one-yaw-column-out folds (pooled into the medians); pitch: fit one pitch half, score the other
    (reported only: it extrapolates in pitch)"""
    columns: list = field(default_factory=list)
    pitch: list = field(default_factory=list)
    new_median_deg: float = float("nan")
    live_median_deg: float = float("nan")
    new_p90_deg: float = float("nan")
    live_p90_deg: float = float("nan")
    n_poses: int = 0


def _fold(sweep, name, fit_idx, test_idx, R_live):
    if len(fit_idx) < MIN_FIT_POSES or len(test_idx) == 0:
        return None
    f = fit(sweep, fit_idx)
    return Fold(name, np.asarray(fit_idx), np.asarray(test_idx), f.angles_deg,
                floor_gap_deg(sweep, R_new(sweep, f), test_idx), floor_gap_deg(sweep, R_live, test_idx))


def heldout(sweep: Sweep, R_live=None) -> HeldOut:
    """held-out floor-vs-gravity gaps of the fitted rotation vs the live rotation (default: the sweep's R_SC)"""
    R_live = sweep.R_SC if R_live is None else np.asarray(R_live, float)
    all_idx = np.arange(len(sweep))
    cols, rows = yaw_columns(sweep), pitch_rows(sweep)
    out = HeldOut()
    if len(set(cols)) > 1:
        for c in sorted(set(cols)):
            fd = _fold(sweep, f"yaw column {sweep.neck[cols == c, 1].mean():+.0f} deg", all_idx[cols != c],
                       all_idx[cols == c], R_live)
            if fd is not None:
                out.columns.append(fd)
    n_rows = len(set(rows))
    if n_rows > 1:
        low = rows < n_rows // 2
        for name, fi, ti in (("fit high pitch, score low", all_idx[~low], all_idx[low]),
                             ("fit low pitch, score high", all_idx[low], all_idx[~low])):
            fd = _fold(sweep, name, fi, ti, R_live)
            if fd is not None:
                out.pitch.append(fd)
    if out.columns:
        new = np.concatenate([f.gap_new_deg for f in out.columns])
        live = np.concatenate([f.gap_live_deg for f in out.columns])
        out.new_median_deg, out.live_median_deg = float(np.median(new)), float(np.median(live))
        out.new_p90_deg, out.live_p90_deg = float(np.percentile(new, 90)), float(np.percentile(live, 90))
        out.n_poses = int(len(new))
    return out


# ---------------------------------------------------------------- the third angle


@dataclass
class Observability:
    """how well the sweep sees rotations of R about each camera-frame axis (small rotations R -> R @ Exp(w))"""
    up_cam: np.ndarray          # mean board normal ("up") in camera 0
    singular: np.ndarray        # singular values of d(stacked cross products)/dw, w in rad, largest first
    weak_axis: np.ndarray       # the least observable rotation axis (camera frame), oriented along up_cam
    ratio: float                # smallest / largest singular value: 0 = unobservable
    weak_sigma_deg: float       # formal 1-sigma of a rotation about weak_axis, if it were fitted
    gap_change_deg: dict        # median gap change for +step_deg about "gravity", "camera x", "camera z"


def _cross_residuals(sweep, R):
    return np.cross(sweep.up_imu @ R, sweep.normal).ravel()


def observability(sweep: Sweep, R, step_deg=1.0, eps=1e-6) -> Observability:
    """the third angle, measured: SVD of the 3-angle Jacobian at R and the median-gap response to +step_deg about up
    (gravity), camera x and camera z"""
    R = np.asarray(R, float)
    J = np.empty((3 * len(sweep), 3))
    for k in range(3):
        ax = np.eye(3)[k]
        Rp = R @ axis_rotation(ax, np.degrees(eps))
        Rm = R @ axis_rotation(ax, -np.degrees(eps))
        J[:, k] = (_cross_residuals(sweep, Rp) - _cross_residuals(sweep, Rm)) / (2 * eps)
    _, sv, vt = np.linalg.svd(J, full_matrices=False)
    up = sweep.normal.mean(axis=0)
    up = up / np.linalg.norm(up)
    weak = vt[-1] if vt[-1] @ up >= 0 else -vt[-1]
    rms = np.sqrt(np.median(_cross_residuals(sweep, R) ** 2))
    weak_sigma = float(np.degrees(rms / sv[-1])) if sv[-1] > 0 else float("inf")
    base = np.median(floor_gap_deg(sweep, R))
    change = {name: float(np.median(floor_gap_deg(sweep, R @ axis_rotation(ax, step_deg))) - base)
              for name, ax in (("gravity", up), ("camera x", [1.0, 0, 0]), ("camera z", [0, 0, 1.0]))}
    return Observability(up, sv, weak, float(sv[-1] / sv[0]), weak_sigma, change)


# ---------------------------------------------------------------- the config: T_SC read and surgery

_TSC = re.compile(r"^[^#\n]*?\bT_SC:\s*\[(?P<body>[^\]]*)\]", re.M)    # never inside a comment
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
ROTATION_TOL = 1e-9
CAM0_ROTATION = (0, 1, 2, 4, 5, 6, 8, 9, 10)     # row-major indices apply() may rewrite in cam0's block
CAM1_TOP = tuple(range(12))                      # ... and in cam1's (its translation follows the stereo pose)


def _blocks(text):
    ms = list(_TSC.finditer(text))
    if len(ms) != 2:
        raise ValueError(f"{len(ms)} T_SC blocks: a stereo OKVIS config has exactly 2")
    for m in ms:
        body = m.group("body")
        if len(_NUM.findall(body)) != 16 or _NUM.sub("", body).strip(", \t\r\n"):
            raise ValueError(f"T_SC block is not 16 numbers: [{body.strip()[:80]}]")
    return ms


def _matrix(m):
    return np.array([float(x) for x in _NUM.findall(m.group("body"))]).reshape(4, 4)


def t_sc(config_text):
    """(T_SC cam0, T_SC cam1) as written in the file (no cleaning: the surgery works on the file's numbers)"""
    return tuple(_matrix(m) for m in _blocks(config_text))


def cam0_rotation(config_text):
    return t_sc(config_text)[0][:3, :3]


def _rewrite(body, values, which):
    """the block's text with the numbers at `which` replaced by values where they changed; all else kept"""
    out, pos = [], 0
    for i, t in enumerate(_NUM.finditer(body)):
        out.append(body[pos:t.start()])
        v = float(values[i])
        out.append(repr(v) if i in which and v != float(t.group()) else t.group())
        pos = t.end()
    out.append(body[pos:])
    return "".join(out)


def apply(config_text, R_correction, note=None):
    """the kit's mkfinal.py rule on an OKVIS config's text: cam0 rotation R0 -> R0 @ R_correction, cam1 = new cam0 x
    inv(T0) @ T1 (the config's stereo pose); cam0 translation, T_BS and all other text kept. `note` (lines) goes in as
    comments right after the %YAML header"""
    C = np.asarray(R_correction, float)
    if C.shape != (3, 3) or np.abs(C @ C.T - np.eye(3)).max() > ROTATION_TOL or np.linalg.det(C) <= 0:
        raise ValueError("R_correction is not a rotation")
    ms = _blocks(config_text)
    T0, T1 = _matrix(ms[0]), _matrix(ms[1])
    N0 = T0.copy()
    N0[:3, :3] = T0[:3, :3] @ C
    N1 = N0 @ (np.linalg.inv(T0) @ T1)
    out, pos = [], 0
    for m, N, which in ((ms[0], N0, CAM0_ROTATION), (ms[1], N1, CAM1_TOP)):
        out += [config_text[pos:m.start("body")], _rewrite(m.group("body"), N.ravel(), which)]
        pos = m.end("body")
    text = "".join(out) + config_text[pos:]
    if note:
        head = "%YAML:1.0\n" if text.startswith("%YAML:1.0\n") else ""
        text = head + "".join(f"# {ln}\n" for ln in note.splitlines()) + text[len(head):]
    return text


# ---------------------------------------------------------------- verdict

SWEEP_LIVE_TOL_DEG = 0.01       # the sweep's config rotation vs live: above this the report says they differ


@dataclass(frozen=True)
class Gates:
    """PLACEHOLDERS until fixed with the user (as decide.Gates was for the lens in M1 C3)"""
    k_sigma: float = 1.0                 # held-out gain must exceed k_sigma x the fit's combined sigma
    min_poses: int = 9                   # still poses with a board, after the filter
    min_yaw_columns: int = 3             # leave-one-column-out needs columns to leave out
    min_pitch_span_deg: float = 10.0     # the pitch span is what separates the up directions


@dataclass
class Verdict:
    deploy: bool
    reasons: list
    fit: Fit
    heldout: HeldOut
    observability: Observability
    R_new: np.ndarray               # the cam0 rotation the full-sweep fit gives
    correction: np.ndarray          # relative to the live cam0 rotation: apply(live_text, correction)
    change_vs_live_deg: float


def verdict(sweep: Sweep, R_live=None, gates: Gates = Gates()) -> Verdict:
    """deploy the fitted rotation only if the gates hold and the pooled held-out median gap beats the live rotation's
    on the same poses by more than gates.k_sigma x the fit's combined sigma"""
    R_live = sweep.R_SC if R_live is None else np.asarray(R_live, float)
    f = fit(sweep)
    Rn = R_new(sweep, f)
    h = heldout(sweep, R_live)
    ob = observability(sweep, Rn)
    reasons, ok = [], True
    n_cols = len(set(yaw_columns(sweep)))
    span = float(np.ptp(sweep.neck[:, 0])) if len(sweep) else 0.0
    for bad, why in ((len(sweep) < gates.min_poses, f"{len(sweep)} still poses with a board (of {sweep.n_recorded}); "
                                                     f"need {gates.min_poses}"),
                     (n_cols < gates.min_yaw_columns, f"{n_cols} yaw columns; need {gates.min_yaw_columns}"),
                     (span < gates.min_pitch_span_deg, f"pitch span {span:.1f} deg; need "
                                                       f"{gates.min_pitch_span_deg:.1f}")):
        if bad:
            ok = False
            reasons.append(f"not enough sweep: {why}")
    gain = h.live_median_deg - h.new_median_deg
    bar = gates.k_sigma * f.combined_sigma_deg
    if not np.isfinite(gain):
        ok = False
        reasons.append("no held-out score (no yaw column could be left out)")
    else:
        better = gain > bar
        ok = ok and better
        reasons.append(f"held-out median gap {h.new_median_deg:.3f} deg (new) vs {h.live_median_deg:.3f} deg (live) on "
                       f"{h.n_poses} poses: gain {gain:+.3f} deg {'>' if better else '<='} {gates.k_sigma:g} x combined "
                       f"sigma {f.combined_sigma_deg:.3f} deg")
    for fd in h.pitch:
        reasons.append(f"  {fd.name}: median {np.median(fd.gap_new_deg):.3f} deg (new) vs "
                       f"{np.median(fd.gap_live_deg):.3f} deg (live)")
    reasons.append(f"fit: {f.angles_deg[0]:+.3f} deg about camera x, {f.angles_deg[1]:+.3f} deg about camera z "
                   f"(1-sigma {f.sigma_deg[0]:.3f} / {f.sigma_deg[1]:.3f}) vs the sweep's config; rotation change vs "
                   f"live {rotation_angle_deg(R_live, Rn):.3f} deg")
    d = rotation_angle_deg(sweep.R_SC, R_live)
    if d > SWEEP_LIVE_TOL_DEG:
        reasons.append(f"the sweep ran with a cam0 rotation {d:.3f} deg from live's; the correction is taken vs live")
    reasons.append(f"third angle (about gravity, camera-frame axis {np.array2string(ob.up_cam, precision=3)}) kept from "
                   f"the live config: observable at {100 * ob.ratio:.0f} % of the best axis (1-sigma "
                   f"{ob.weak_sigma_deg:.2f} deg if fitted); +1 deg about it moves the median gap "
                   f"{ob.gap_change_deg['gravity']:+.3f} deg vs {ob.gap_change_deg['camera x']:+.3f} about camera x")
    reasons.append("T_SC changes with T_BS kept: the camera moves in the Head frame for every other consumer of "
                   "camera-in-Head (camera-to-arm)")
    return Verdict(bool(ok), reasons, f, h, ob, Rn, R_live.T @ Rn, rotation_angle_deg(R_live, Rn))
