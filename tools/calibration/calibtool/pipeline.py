"""The pipeline steps calibtool drives, as calls with parsed results: step 16 (split), 45 (fit), 66 (validate) and
run_lens_kb4t's summary row (TEST depth error and row error of one calibration).

16 and the summary run in this process (fast); 45 and 66 run as their own processes, exactly as the recorded runs
called them, and are read from the JSON they write: step 45's fit JSON and step 66's report.json. Exit codes are never
the verdict (the p21 kit's step-66 runner exited 0 on a FAIL). A step that leaves no JSON raises PipelineError with
the tail of its log."""
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from calibtool import paths

# run_lens_kb4t.BANDS: the summary row's distance bands, (lo, hi] m; step 66 bins are [lo, hi)
BANDS = ((0.0, 1.0), (1.0, 1.3), (1.3, 1.7), (1.7, 9.9))
VERDICTS = ("PASS", "FAIL", "NO RESULT")
# the least squares behind 45 / 66 / the summary: "fast" (calibtool.solver, M3) is calibtool's default; "scipy" is
# the standalone scripts' default (and the recorded runs'). Steps 45 and 66 get it as CALIB_SOLVER in their own
# environment, the in-process scoring through stereofit.solver_scope: the caller's environment is never changed
SOLVERS = ("fast", "scipy")
DEFAULT_SOLVER = "fast"
SOLVER_ENV = "CALIB_SOLVER"


def check_solver(solver):
    if solver not in SOLVERS:
        raise ValueError(f"solver {solver!r}: one of {SOLVERS}")
    return solver


class PipelineError(RuntimeError):
    """a step left no result to read"""


def load_rig(rig):
    """rigcfg.Rig from a path (rigcfg.load prints to stdout: silenced); a Rig is returned as it is"""
    if not isinstance(rig, (str, Path)):
        return rig
    import rigcfg
    with contextlib.redirect_stdout(io.StringIO()):
        return rigcfg.load(Path(rig))


def _rig_path(rig) -> Path:
    return Path(rig) if isinstance(rig, (str, Path)) else Path(rig.path)


def _run(script, args, log, solver=DEFAULT_SOLVER):
    """script (a path) with args as its own process, stdout+stderr to log, CALIB_SOLVER=solver in its environment
    -> (exit code, seconds)"""
    env = dict(os.environ)
    env[SOLVER_ENV] = check_solver(solver)
    env["PYTHONPATH"] = os.pathsep.join([str(paths.SCRIPTS)] + [p for p in [env.get("PYTHONPATH")] if p])
    t = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([sys.executable, str(script)] + [str(a) for a in args], stdout=fh,
                            stderr=subprocess.STDOUT, env=env).returncode
    return rc, time.time() - t


def _tail(log, n=15):
    try:
        lines = [ln for ln in Path(log).read_text().splitlines() if "RuntimeWarning" not in ln]
    except OSError:
        return "(no log)"
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------- step 16


@dataclass
class Split:
    fit_npz: Path
    test_npz: Path
    fit_pairs: int
    test_pairs: int
    near_test_pairs: int
    far_test_pairs: int
    take_files: list            # the per-take corner files given to step 16, in take order


def split16(session, out_dir, rig, max_fit_z=3.0, holdout_every=2) -> Split:
    """step 16 on the session's takes: each take written back as its own corner file (pose hold ids, the layout of
    the stored files) and given to 16.split in the session's take order, so hold ids and the every-N-th holdout are
    16's own. -> out_dir/fit.npz, out_dir/test.npz"""
    out_dir = Path(out_dir)
    (out_dir / "takes").mkdir(parents=True, exist_ok=True)
    rig = load_rig(rig)
    files = []
    for take in session.take_order:
        f = out_dir / "takes" / f"{take}.npz"
        np.savez_compressed(f, **session.to_npz_dict(session.pairs["take"] == take, hold="pose"))
        files.append(f)
    d16 = paths.import_script("16_split_distance")
    fit, test = d16.split([str(f) for f in files], rig.board, rig.nominal_focal_px, rig.resolution, max_fit_z,
                          holdout_every)
    fit_npz, test_npz = out_dir / "fit.npz", out_dir / "test.npz"
    np.savez_compressed(fit_npz, **fit)
    np.savez_compressed(test_npz, **test)
    far = np.nan_to_num(test["img_z"], nan=np.inf) > max_fit_z
    return Split(fit_npz, test_npz, len(fit["img_t0"]), len(test["img_t0"]), int((~far).sum()), int(far.sum()), files)


# ---------------------------------------------------------------- step 45


@dataclass
class FitResult:
    json_path: Path
    log_path: Path
    verdict: str                # step 45's verdict line ("converged: rms ...", "rejected: ...", ...)
    converged: bool
    rms_px: float
    baseline_mm: float
    fit_pairs: int
    fit_holds: int
    returncode: int
    seconds: float
    entry: dict = field(repr=False)

    def as_dict(self):
        d = asdict(self)
        d.pop("entry")
        return d


def fit45(fit_npz, rig, out_dir, max_z=3.0, model="KB4T", script=None, extra=(), solver=DEFAULT_SOLVER) -> FitResult:
    """step 45 (--model, --max-z, --rig) on fit_npz -> out_dir/fit-<model>.json, log out_dir/45.log. script: run this
    instead of 45_fit_lens.py with the same arguments (tests); extra: more arguments; solver: SOLVERS"""
    check_solver(solver)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out, log = out_dir / f"fit-{model.lower()}.json", out_dir / "45.log"
    args = [fit_npz, out, "--model", model, "--max-z", max_z, "--rig", _rig_path(rig)] + list(extra)
    rc, sec = _run(script or paths.script("45_fit_lens"), args, log, solver)
    if not out.is_file():
        raise PipelineError(f"step 45 (exit {rc}) wrote no {out}:\n{_tail(log)}")
    e = json.loads(out.read_text())
    v = str(e.get("verdict"))
    num = lambda k: float(e[k]) if e.get(k) is not None else float("nan")   # noqa: E731
    return FitResult(out, log, v, v.startswith("converged"), num("rms_px"), num("baseline_mm"),
                     int(e.get("fit_pairs") or 0), int(e.get("fit_holds") or 0), rc, sec, e)


# ---------------------------------------------------------------- step 66


def _key(r):
    return (str(r["cell"]), tuple(float(x) for x in r["dist_m"]))


@dataclass
class ValidateResult:
    verdict: str                # "PASS" | "FAIL" | "NO RESULT", from report.json
    reason: str
    text: str                   # the verdict line step 66 printed
    entries: dict               # validated entries {(cell, (lo, hi)): median depth error %}
    failing: dict               # the validated entries outside their limits
    not_validated: list         # [(cell, (lo, hi))] with data but fewer than lens_min_holds holds
    n_validated: int
    report_path: Path
    log_path: Path
    returncode: int
    seconds: float
    report: dict = field(repr=False)

    def as_dict(self):
        d = asdict(self)
        d.pop("report")
        for k in ("entries", "failing"):
            d[k] = {f"{c} {lo:g}-{hi:g}": v for (c, (lo, hi)), v in d[k].items()}
        d["not_validated"] = [f"{c} {lo:g}-{hi:g}" for c, (lo, hi) in d["not_validated"]]
        return d


def validate66(calib_json, test_npz, out_dir, rig, fit_report=None, script=None,
               solver=DEFAULT_SOLVER) -> ValidateResult:
    """step 66 of calib_json on test_npz -> out_dir/report.json (+ report.md), log out_dir.log. The verdict is read
    from report.json and checked against the printed verdict line. script: run this instead of 66_validate_lens.py
    with the same arguments (a runner that registers a lens model first, or a test's fake); solver: SOLVERS"""
    check_solver(solver)
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    log = out_dir.parent / f"{out_dir.name}.log"
    args = [calib_json, test_npz, out_dir] + (["--fit-report", fit_report] if fit_report else [])
    args += ["--rig", _rig_path(rig)]
    rc, sec = _run(script or paths.script("66_validate_lens"), args, log, solver)
    rp = out_dir / "report.json"
    if not rp.is_file():
        raise PipelineError(f"step 66 (exit {rc}) wrote no {rp}:\n{_tail(log)}")
    rep = json.loads(rp.read_text())
    lines = [ln for ln in Path(log).read_text().splitlines() if ln.startswith(VERDICTS)]
    text = lines[-1] if lines else ""
    said = next((v for v in ("NO RESULT", "PASS", "FAIL") if text.startswith(v)), None)
    if said != rep.get("verdict"):
        raise PipelineError(f"step 66's report.json ({rep.get('verdict')}) and its verdict line ({text!r}) disagree")
    table = rep.get("table", [])
    val = [r for r in table if "ok" in r]
    return ValidateResult(rep["verdict"], rep.get("reason", ""), text,
                          {_key(r): float(r["median_pct"]) for r in val},
                          {_key(r): float(r["median_pct"]) for r in val if not r["ok"]},
                          [_key(r) for r in table if "ok" not in r], len(val), rp, log, rc, sec, rep)


# ---------------------------------------------------------------- summary row


@dataclass
class SummaryResult:
    model: str
    baseline_mm: float
    n_pairs: int                # pairs scored (lensscore records)
    median_pct: float           # TEST depth error, + = reads too far
    iqr_pct: tuple
    rows_px: float              # median |dy| after rectification with the calibration's stereo pose, at nominal focal
    bands: list                 # [{"lo", "hi", "n", "median_pct" (None below 5 pairs)}], (lo, hi] m
    solves: dict
    records: dict = field(repr=False)     # lensscore.score's per-pair records: src, hold, scale, z, u, v
    corner_src: np.ndarray = field(default=None, repr=False)     # row_errors: pair per corner
    corner_dy: np.ndarray = field(default=None, repr=False)      # row_errors: dy per corner (px)

    def as_dict(self, records=False):
        d = asdict(self)
        d.pop("corner_src")
        d.pop("corner_dy")
        if not records:
            d.pop("records")
        return d


def summary(calib_json, test_npz, rig, bands=BANDS, solver=DEFAULT_SOLVER) -> SummaryResult:
    """run_lens_kb4t.summary_row's numbers (same math), as data: the TEST depth error of the calibration (lensscore
    board scale - 1: median, IQR, per distance band) and its row error. The lens model must be known to lensmodel
    (a radtan config needs its model registered first). solver: SOLVERS, for this call (stereofit.solver_scope)"""
    import lensscore
    import stereofit as sf
    rig = load_rig(rig)
    model, intr, C = sf.load_calibration(str(calib_json))
    data = sf.Data(str(test_npz), rig.board)
    with sf.solver_scope(check_solver(solver)):
        rec, _, solves = lensscore.score(model, intr, C, data)
    sc = np.array(rec["scale"]) - 1
    zz = np.array(rec["z"])
    csrc, dy = row_errors(model, intr, C, test_npz, rig)
    out_bands = []
    for lo, hi in bands:
        m = (zz > lo) & (zz <= hi)
        out_bands.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                          "median_pct": float(100 * np.median(sc[m])) if m.sum() >= 5 else None})
    return SummaryResult(model, float(1e3 * np.linalg.norm(C[:3, 3])), int(len(sc)), float(100 * np.median(sc)),
                         (float(100 * np.percentile(sc, 25)), float(100 * np.percentile(sc, 75))),
                         float(np.median(np.abs(dy if len(dy) else np.zeros(1)))), out_bands, solves, rec, csrc, dy)


def row_errors(model, intr, C, test_npz, rig):
    """the row error per corner seen by both cameras: rectified with the calibration's own stereo pose, dy (px) at
    the rig's nominal focal -> (pair index in the TEST file per corner, dy per corner), pairs in file order and a
    pair's corners by (tag, corner). Each array of the file read once, every corner unprojected in one call per
    camera. The lens model must be known to lensmodel (livecfg.model_scope)"""
    import cv2
    import lensmodel
    rig = load_rig(rig)
    T10 = np.linalg.inv(np.asarray(C, float))
    R0, R1, *_ = cv2.stereoRectify(np.eye(3), np.zeros(5), np.eye(3), np.zeros(5), tuple(rig.resolution),
                                   np.ascontiguousarray(T10[:3, :3]), np.ascontiguousarray(T10[:3, 3].reshape(3, 1)),
                                   flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    with np.load(test_npz) as z:
        img, cam, tag, k = (np.asarray(z[n], np.int64) for n in ("det_img", "det_cam", "det_tag", "det_k"))
        uv = np.c_[z["det_u"], z["det_v"]].astype(float)
    if not len(img):
        return np.zeros(0, int), np.zeros(0)
    nk = int(k.max()) + 1
    key = (img * (int(tag.max()) + 1) + tag) * nk + k               # (pair, tag, corner), sorts as the old loop did
    i0, i1 = np.flatnonzero(cam == 0), np.flatnonzero(cam == 1)
    # a corner listed twice in one camera: the last one counts (the old loop's dict)
    u0 = len(i0) - 1 - np.unique(key[i0][::-1], return_index=True)[1]
    u1 = len(i1) - 1 - np.unique(key[i1][::-1], return_index=True)[1]
    i0, i1 = i0[u0], i1[u1]
    common, a, b = np.intersect1d(key[i0], key[i1], assume_unique=True, return_indices=True)
    if not len(common):
        return np.zeros(0, int), np.zeros(0)
    b0 = lensmodel.unproject(model, intr[0], uv[i0[a]]) @ R0.T
    b1 = lensmodel.unproject(model, intr[1], uv[i1[b]]) @ R1.T
    return img[i0[a]], rig.nominal_focal_px * (b0[:, 1] / b0[:, 2] - b1[:, 1] / b1[:, 2])
