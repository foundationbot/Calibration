"""The robot's live OKVIS config as a calibration the package's scorer (lensscore, step 66, pipeline.summary) can judge
on the same TEST pairs as a new fit.

load(okvis_yaml) reads the lens model from each camera block's `distortion_type` and from nothing else:
radialtangential -> "RADTAN", equidistant -> "KB4", kb4t -> "KB4T". The coefficient count never decides it
(run_lens_kb4t.okvis_calibration maps 4 numbers to "KB4", which reads a radtan config as a fisheye). The stereo pose
is C = inv(T_SC0) @ T_SC1 (camera 1 in camera 0), from okviscfg (rotations cleaned as OKVIS's own files need).
A config written for another image size is brought back to the scoring size with scaling.unscale_config.

RADTAN (OKVIS's / OpenCV's radial-tangential, k1 k2 p1 p2, pinhole) is not one of lensmodel's models: the package
fits only KB4/KB4T. radtan_scope() registers it in lensmodel for as long as the block runs (lensmodel.MODELS gets
"RADTAN", lensmodel.project/unproject learn it) and puts lensmodel back exactly as it was on the way out, exception or
not. The registration is process-wide while active (other threads see it too); scopes nest.

Step 66 runs as its own process: for a RADTAN config it runs through this file as a script (`python livecfg.py
<step-66 arguments>`), which enters radtan_scope and runs 66_validate_lens.py's main in the same process."""
import contextlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):                       # run as a script (step 66's RADTAN runner)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from calibtool import paths  # noqa: E402,F401  (puts the scripts folder on sys.path)
import lensmodel  # noqa: E402

DISTORTION_MODELS = {"radialtangential": "RADTAN", "equidistant": "KB4", "kb4t": "KB4T"}
RADTAN_N = 4                    # k1, k2, p1, p2: what OKVIS's radialtangential camera reads
NEWTON_STEPS = 30


# ---------------------------------------------------------------- the RADTAN lens


def _radtan_distort(x, y, d):
    k1, k2, p1, p2 = d
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    return x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y


def _radtan_intr(intr):
    intr = np.asarray(intr, float).ravel()
    if intr.size != 4 + RADTAN_N:
        raise ValueError(f"RADTAN takes {4 + RADTAN_N} numbers (fx fy cx cy k1 k2 p1 p2), got {intr.size}")
    return intr


def radtan_project(intr, P):
    """camera-frame points (N, 3) -> pixels (N, 2), OpenCV's projectPoints with (k1, k2, p1, p2)"""
    intr = _radtan_intr(intr)
    X, Y, Z = np.asarray(P, float).reshape(-1, 3).T
    with np.errstate(divide="ignore", invalid="ignore"):
        xd, yd = _radtan_distort(X / Z, Y / Z, intr[4:])
    return np.c_[intr[0] * xd + intr[2], intr[1] * yd + intr[3]]


def radtan_unproject(intr, uv):
    """pixels (N, 2) -> unit rays (N, 3): radtan_project's inverse by Newton steps with the analytic Jacobian"""
    intr = _radtan_intr(intr)
    k1, k2, p1, p2 = intr[4:]
    uv = np.asarray(uv, float).reshape(-1, 2)
    mx, my = (uv[:, 0] - intr[2]) / intr[0], (uv[:, 1] - intr[3]) / intr[1]
    x, y = mx.copy(), my.copy()
    for _ in range(NEWTON_STEPS):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        g = 2 * k1 + 4 * k2 * r2
        fx, fy = _radtan_distort(x, y, intr[4:])
        ex, ey = fx - mx, fy - my
        j11 = rad + g * x * x + 2 * p1 * y + 6 * p2 * x
        j12 = j21 = g * x * y + 2 * p1 * x + 2 * p2 * y
        j22 = rad + g * y * y + 6 * p1 * y + 2 * p2 * x
        det = j11 * j22 - j12 * j21
        x, y = x - (j22 * ex - j12 * ey) / det, y - (j11 * ey - j21 * ex) / det
    r = np.c_[x, y, np.ones_like(x)]
    return r / np.linalg.norm(r, axis=1, keepdims=True)


@contextlib.contextmanager
def radtan_scope():
    """lensmodel knows "RADTAN" inside the block, and is exactly as before after it (MODELS the same dict with the same
    entries, project/unproject the same functions), whether the block returns or raises"""
    models, saved_models = lensmodel.MODELS, dict(lensmodel.MODELS)
    proj, unproj = lensmodel.project, lensmodel.unproject

    def project(model, intr, P):
        return radtan_project(intr, P) if model == "RADTAN" else proj(model, intr, P)

    def unproject(model, intr, uv):
        return radtan_unproject(intr, uv) if model == "RADTAN" else unproj(model, intr, uv)

    models["RADTAN"] = RADTAN_N
    lensmodel.project, lensmodel.unproject = project, unproject
    try:
        yield
    finally:
        lensmodel.project, lensmodel.unproject = proj, unproj
        models.clear()
        models.update(saved_models)


def model_scope(model):
    """radtan_scope() for "RADTAN", else a no-op block"""
    return radtan_scope() if model == "RADTAN" else contextlib.nullcontext()


# ---------------------------------------------------------------- the live config


@dataclass
class LiveCalib:
    model: str                  # "RADTAN" | "KB4" | "KB4T", from distortion_type
    intr: list                  # per camera [fx, fy, cx, cy, *dist]
    C: np.ndarray               # 4x4, camera 1 in camera 0
    size: tuple                 # (W, H) the intrinsics are for
    distortion_type: str        # the file's word
    source: str                 # the config file
    scaled_from: tuple = None   # (W, H) of the file when load() brought it to another size

    def entry(self):
        """the step-66 calibration entry (model, intr, C), no step-45 verdict"""
        return {"model": self.model, "intr": [np.asarray(v, float).tolist() for v in self.intr],
                "C": np.asarray(self.C, float).tolist(), "source": self.source, "size": list(self.size)}

    def write_json(self, path):
        path = Path(path)
        path.write_text(json.dumps(self.entry(), indent=1))
        return path


def _fs_read(text):
    import cv2
    fs = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    cams = fs.getNode("cameras")
    if cams.empty() or cams.size() == 0:
        raise ValueError("no cameras")
    types, sizes = [], []
    for i in range(cams.size()):
        c = cams.at(i)
        t = c.getNode("distortion_type")
        if t.empty() or not t.string():
            raise ValueError(f"cameras[{i}].distortion_type: missing")
        types.append(t.string())
        d = c.getNode("image_dimension")
        if d.empty() or d.size() != 2:
            raise ValueError(f"cameras[{i}].image_dimension: missing")
        sizes.append((int(d.at(0).real()), int(d.at(1).real())))
    return types, sizes


def load(okvis_yaml, at_size=None) -> LiveCalib:
    """the stereo calibration of an OKVIS config (cameras 0 and 1). at_size=(W, H): a config written for another size is
    unscaled to it first (scaling.unscale_config: focal / s, principal point (c + 0.5) / s - 0.5), resize_factor left
    as it is (vio=None): a config without one (VIO at the image size, as scale_config writes 640x400) loads too"""
    import okviscfg
    path = Path(okvis_yaml)
    text = path.read_text()
    try:
        types, sizes = _fs_read(text)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    if len(set(types)) != 1 or types[0] not in DISTORTION_MODELS:
        raise ValueError(f"{path}: distortion_type {sorted(set(types))}; known: {sorted(DISTORTION_MODELS)}")
    if len(sizes) < 2 or sizes[0] != sizes[1]:
        raise ValueError(f"{path}: need two cameras of one image size, got {sizes}")
    scaled_from = None
    if at_size is not None and tuple(at_size) != sizes[0]:
        from calibtool import scaling
        s = scaling.factor(sizes[0], at_size)
        text = scaling.unscale_config(text, s, vio=None)    # the lens only: scoring never reads resize_factor
        scaled_from = sizes[0]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / path.name
        p.write_text(text)
        cfg = okviscfg.load(p)
    intr = [np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], np.ravel(D)] for K, D in zip(cfg.K[:2], cfg.D[:2])]
    C = np.linalg.inv(cfg.T_SC[0]) @ cfg.T_SC[1]
    return LiveCalib(DISTORTION_MODELS[types[0]], intr, C, tuple(at_size or sizes[0]), types[0], str(path),
                     scaled_from)


# ---------------------------------------------------------------- scoring


def summary(live: LiveCalib, test_npz, rig, work_dir):
    """pipeline.summary of the live config (its JSON written to work_dir/live-<model>.json), RADTAN registered while
    it runs"""
    from calibtool import pipeline
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    js = live.write_json(Path(work_dir) / f"live-{live.model.lower()}.json")
    with model_scope(live.model):
        return pipeline.summary(js, test_npz, rig)


def validate66(live: LiveCalib, test_npz, out_dir, rig):
    """pipeline.validate66 of the live config (its JSON beside out_dir); a RADTAN config runs through this file's
    runner, which registers the model in step 66's process"""
    from calibtool import pipeline
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    js = live.write_json(out_dir.parent / f"{out_dir.name}-{live.model.lower()}.json")
    return pipeline.validate66(js, test_npz, out_dir, rig, script=Path(__file__) if live.model == "RADTAN" else None)


def _main(argv):
    """step 66 (66_validate_lens.py's arguments) with RADTAN registered; --check-model calib.json: load and project
    one point with it, print "<model> ok" (a check of this runner)"""
    with radtan_scope():
        if argv[:1] == ["--check-model"]:
            import stereofit as sf
            model, intr, _ = sf.load_calibration(argv[1])
            uv = lensmodel.project(model, intr[0], [[0.1, -0.05, 1.0]])
            assert np.all(np.isfinite(uv))
            print(f"{model} ok")
            return 0
        v66 = paths.import_script("66_validate_lens")
        return v66.main(argv)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
