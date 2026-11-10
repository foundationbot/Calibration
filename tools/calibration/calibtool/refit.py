"""The live refit during collection (design plan 4.3: "refit in the background after each grid; live score in the
TUI"): after each placement, the FIT corners captured so far fitted again, warm from the previous placement's
solution, so the operator sees the fit converge while the board is being moved and the final fit is nearly free.

One refit (LiveRefit.update), on the FIT corners so far (step 15's corner dict, global hold ids):
  start   the first refit: stereofit.fit_from_nominal (PnP with the rig's nominal camera, the stereo pose alone, then
          everything), as step 45. Later ones: the previous refit's camera and stereo pose; its board poses for the
          pairs it already had (matched by left stamp); PnP with its camera, then the pose alone with the camera
          fixed, for the new pairs only (pairs whose PnP fails are left out, as step 45 does)
  fit     stereofit.fit_converged (step 45's trim rounds, evaluation cap and attempts)
  gate    step 45's quality gate (45_fit_lens.quality_gate: holds over gates.lens_hold_rms_max_px or beyond max_z
          dropped, then fit_converged again from the result) and its rejection share (more than
          gates.lens_bad_hold_max_frac of the holds bad: rejected=True, the numbers kept for display)
The gate judges every hold again at each refit, on that refit's fit, as step 45 does on its data (a hold dropped
while the camera was still rough comes back once it fits).
Final fit = the last live refit (10 Nov 2026): once the collection ends, LiveFit refits once more on the final
split's FIT corners (the held-back holds released), while the IMU sweep runs; at FIT that refit IS step 45's fit
(final_fit: written as step 45's JSON, "source" naming the refit) when it covers exactly the final split's FIT corners
(corner_digest: the same pairs, corners and hold grouping) and passed step 45's checks (converged, not rejected by
the gate, camera sides as the rig file says); otherwise step 45 runs as before. On the stored sessions the refit on the
final split equals step 45's fit from nominal within 2e-12 px.

BackgroundRefit: the same in one spawned process of its own (design 10.1: the fit in its own process, never in the
capture's), submit() after each placement -> a Future of the Refit; the state stays in that process."""
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from calibtool import paths, pipeline

MAX_Z = 3.0                 # step 45's --max-z as calibtool runs it


@dataclass
class Refit:
    index: int                  # refits so far (1, 2, ...)
    placement: str
    pairs: int                  # FIT pairs fitted (after PnP and the gate)
    holds: int
    new_pairs: int              # pairs the previous refit did not have
    warm: bool                  # started from the previous refit (False: from nominal)
    converged: bool             # every solve reached the solver's convergence
    rejected: bool              # the gate dropped more than gates.lens_bad_hold_max_frac of the holds
    rms_px: float
    baseline_mm: float
    dropped_holds: list         # by the gate, this refit (rms or depth)
    pnp_failed: int             # new pairs left out: their PnP start failed
    nfev: int
    seconds: float
    model: str
    intr: list = field(repr=False)
    C: np.ndarray = field(repr=False)
    pid: int = 0                # the process that fitted it
    settled: bool = True        # the last fit's trim settled (stereofit.trim_loop)
    digest: str = ""            # corner_digest of the corners it was given
    step45: dict = field(default_factory=dict, repr=False)  # step 45's JSON fields of this fit (verdict included)

    @property
    def verdict(self) -> str:
        return str(self.step45.get("verdict", ""))

    def as_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()
                if k not in ("intr", "C", "step45")} | {"intr": [np.asarray(v).tolist() for v in self.intr],
                                                         "C": np.asarray(self.C).tolist(), "verdict": self.verdict}


def corner_digest(d) -> str:
    """sha256 of a corner dict's content, whatever its layout: per pair (by left stamp) its right stamp and its
    corners (camera, tag, corner, u, v) in a fixed order, and the pairs' grouping into holds (not the hold ids). Two
    dicts with the same digest give step 45 the same fit problem"""
    d = {k: np.asarray(v) for k, v in dict(d).items()}
    t0, t1 = d["img_t0"].astype(np.int64), d["img_t1"].astype(np.int64)
    order = np.argsort(t0, kind="stable")
    rank = np.empty(len(t0), np.int64)
    rank[order] = np.arange(len(t0))
    det = np.c_[rank[d["det_img"].astype(np.int64)], d["det_cam"], d["det_tag"], d["det_k"]].astype(np.int64)
    co = np.lexsort(det.T[::-1])
    groups = {}
    for t, h in zip(t0.tolist(), d["img_hold"].tolist()):
        groups.setdefault(int(h), []).append(int(t))
    h = hashlib.sha256()
    h.update(t0[order].tobytes() + t1[order].tobytes())
    h.update(det[co].tobytes())
    h.update(np.c_[d["det_u"], d["det_v"]].astype(float)[co].tobytes())
    h.update(json.dumps(sorted(sorted(g) for g in groups.values())).encode())
    return h.hexdigest()


class LiveRefit:
    """refits of a growing FIT set, each warm from the last. rig: path or rigcfg.Rig; solver: pipeline.SOLVERS (for
    this object's solves only: stereofit.solver_scope)"""

    def __init__(self, rig, model="KB4T", max_z=MAX_Z, solver=pipeline.DEFAULT_SOLVER, log=None):
        import stereofit as sf
        self.sf = sf
        self.rig = pipeline.load_rig(rig)
        self.model = model
        self.solver = pipeline.check_solver(solver)
        self.gates = self.rig.lens_gates()
        self.gates["lens_hold_max_z_m"] = float(max_z)
        self.log = log or (lambda *a: None)
        self._v45 = paths.import_script("45_fit_lens")
        self.intr = self.C = None
        self.poses = {}             # left stamp -> board pose (6,) of the last refit
        self.history = []

    def update(self, fit_corners, placement="") -> Refit:
        """refit on fit_corners (step 15's corner dict of every FIT pair so far) -> Refit (also in .history)"""
        sf = self.sf
        t = time.time()
        stamps = np.asarray(fit_corners["img_t0"], np.int64)
        with sf.solver_scope(self.solver):
            data = sf.Data(dict(fit_corners), self.rig.board)
            known = np.isin(stamps[data.src_pairs], list(self.poses))
            warm = self.intr is not None
            if not warm:
                data, r = sf.fit_from_nominal(data, self.model, *sf.nominal_start(self.rig, self.model), log=self.log)
                failed, new = r["pnp_failed_pairs"], int(len(known))
            else:
                data, poses, failed = self._warm_start(data, stamps, known)
                new = int((~known).sum())
                r = sf.fit_converged(data, self.model, self.intr, self.C, poses, log=self.log)
            nfev = r["nfev"]
            v45 = self._v45
            rms, misfit, z, far = v45.quality_gate(data, r, self.gates)
            gate = v45.gate_fields(rms, misfit, z, far)
            bad = sorted(set(misfit) | set(far))
            rejected = len(bad) > self.gates["lens_bad_hold_max_frac"] * len(rms)
            if bad and not rejected:
                m = ~np.isin(data.hold, bad)
                data = data.select(m)
                r = sf.fit_converged(data, self.model, r["intr"], r["C"], r["poses"][m], log=self.log)
                nfev += r["nfev"]
            fields, warn = v45.fit_fields(r, data, self.rig)
            if not r["success"]:
                verdict = f"unconverged: {v45.unconverged_text(r)}"
            elif rejected:
                verdict = f"rejected: {len(bad)} of {len(rms)} holds over step 45's gate"
            else:
                verdict = v45.sides_text(self.rig, r["C"]) or v45.converged_text(r, data)
            if warn and verdict.startswith("converged"):
                verdict = f"{verdict}; WARNING: {warn[0]}"
            step45 = {"model": self.model, "gates": dict(self.gates), "pnp_failed_pairs": int(failed), **gate,
                      **fields, "verdict": verdict} | ({"warnings": warn} if warn else {})
        self.intr, self.C = [np.array(v) for v in r["intr"]], np.array(r["C"])
        self.poses = {int(a): p for a, p in zip(stamps[data.src_pairs], r["poses"])}
        out = Refit(len(self.history) + 1, str(placement), int(data.n), int(len(np.unique(data.hold))), new, warm,
                    bool(r["success"]), bool(rejected), float(sf.fit_cost(r)[1]),
                    float(1e3 * np.linalg.norm(r["C"][:3, 3])), [int(h) for h in bad], int(failed), int(nfev),
                    time.time() - t, self.model, self.intr, self.C, os.getpid(), bool(r["settled"]),
                    corner_digest(fit_corners), step45)
        self.history.append(out)
        return out

    def _warm_start(self, data, stamps, known):
        """board poses: the last refit's for known pairs; for new pairs PnP with the last camera, then the pose alone
        (camera fixed). Pairs whose PnP failed are left out -> (data, poses, PnP failures)"""
        sf = self.sf
        poses = np.zeros((data.n, 6))
        poses[known] = [self.poses[int(x)] for x in stamps[data.src_pairs[known]]]
        failed = 0
        if (~known).any():
            dn = data.select(~known)
            p, ok = sf.pnp_poses(dn, self.model, self.intr, self.C)
            failed = int((~ok).sum())
            if ok.any():
                p[ok] = sf.solve(dn.select(ok), self.model, self.intr, self.C, p[ok], free_intr=False, free_C=False,
                                 max_nfev=sf.MAX_NFEV)["poses"]
            idx = np.flatnonzero(~known)
            poses[idx] = p
            keep = np.ones(data.n, bool)
            keep[idx[~ok]] = False
            data, poses = data.select(keep), poses[keep]
        return data, poses, failed


# ---------------------------------------------------------------- the final fit


def final_fit_refusal(r: Refit, fit_npz) -> str | None:
    """None when the refit r may stand for step 45 on fit_npz, else why not"""
    if r is None:
        return "no live refit"
    if not r.verdict.startswith("converged"):
        return f"the live refit's verdict: {r.verdict or 'none'}"
    if not r.converged or r.rejected:
        return "the live refit did not pass step 45's gate (unconverged or rejected)"
    with np.load(fit_npz) as z:
        want = corner_digest({k: z[k] for k in z.files})
    if r.digest != want:
        return f"the live refit ({r.placement}, {r.pairs} pairs) does not cover exactly the final FIT split"
    return None


def final_fit(r: Refit, fit_npz, out_dir, rig):
    """the refit r as step 45's result on fit_npz: out_dir/fit-<model>.json in step 45's format (45_fit_lens.header +
    the refit's fields, "source" naming the refit), out_dir/45.log saying so. -> (pipeline.FitResult, None), or
    (None, why) when r may not stand for step 45 (final_fit_refusal)"""
    from pathlib import Path
    why = final_fit_refusal(r, fit_npz)
    if why:
        return None, why
    import stereofit as sf
    rig = pipeline.load_rig(rig)
    v45 = paths.import_script("45_fit_lens")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = v45.header(rig, r.model, r.step45.get("gates"), fit_npz)
    out.update(r.step45)
    out["source"] = {"live_refit": r.index, "placement": r.placement, "warm": r.warm, "seconds": r.seconds,
                     "pid": r.pid, "corner_digest": r.digest}
    out["seconds"] = 0
    js, log = out_dir / f"fit-{r.model.lower()}.json", out_dir / "45.log"
    sf.write_json(js, out)
    log.write_text(f"step 45 not run: live refit {r.index} ({r.placement}) covers exactly {fit_npz} "
                   f"(corner digest {r.digest[:12]}) and passed step 45's checks\n{r.verdict} -> {js}\n")
    return pipeline.FitResult(js, log, r.verdict, True, float(out["rms_px"]), float(out["baseline_mm"]),
                              int(out["fit_pairs"]), int(out["fit_holds"]), 0, 0.0, out), None


class LiveFit:
    """the live refit as the full session drives it: placement_done() after each placement (a refit on the FIT pairs
    so far, capture-time labels), collection_done() once the collection ends (the split written to
    <out_dir>/live-split and a refit on its FIT file, while the IMU sweep runs), final(fit_npz, out_dir) at FIT (the
    last refit as step 45's result, or None and why). make() -> the background (BackgroundRefit(rig)); to_session:
    the flow's holds -> a Session (flowrobot.holds_session). Nothing here touches the robot"""

    def __init__(self, rig, solver=pipeline.DEFAULT_SOLVER, make=None, to_session=None, wait_s=600.0):
        self.rig = rig
        self.make = make or (lambda: BackgroundRefit(rig, solver=solver))
        self.to_session = to_session
        self.wait_s = float(wait_s)
        self.bg = None
        self.last = None            # the newest Future
        self.final_submitted = False

    def _session(self, holds):
        if self.to_session is not None:
            return self.to_session(holds)
        from calibtool import flowrobot
        return flowrobot.holds_session(holds)

    def _submit(self, corners, placement):
        if self.bg is None:
            self.bg = self.make()
        self.last = self.bg.submit(corners, placement)
        return self.last

    def placement_done(self, collection, holds, placement):
        """a refit on the FIT pairs so far (capture-time labels) -> its Future, or None without FIT pairs"""
        from calibtool import assign
        sess = self._session(holds)
        lab = collection.labeler.pair_labels()
        mask = np.array([lab.get(int(t)) == assign.FIT for t in sess.pairs["t0"]], bool)
        if not mask.any():
            return None
        self.final_submitted = False
        return self._submit(sess.to_npz_dict(mask), placement)

    def collection_done(self, collection, holds, out_dir):
        """the final split written (out_dir/live-split) and a refit on its FIT corners -> its Future"""
        from pathlib import Path
        sess = self._session(holds)
        split = collection.labeler.write_split(sess, Path(out_dir) / "live-split")
        with np.load(split.fit_npz) as z:
            corners = {k: z[k] for k in z.files}
        f = self._submit(corners, "final split")
        self.final_submitted = True
        return f

    def final(self, fit_npz, out_dir):
        """-> (pipeline.FitResult, None) from the last refit, or (None, why): no final-split refit, it failed or
        took longer than wait_s, or final_fit refuses it"""
        if self.last is None or not self.final_submitted:
            return None, "no live refit on the final split"
        try:
            r = self.last.result(timeout=self.wait_s)
        except Exception as e:  # noqa: BLE001  (any failure of the refit: step 45 runs instead)
            return None, f"the live refit on the final split failed: {type(e).__name__}: {e}"
        return final_fit(r, fit_npz, out_dir, self.rig)

    def close(self):
        if self.bg is not None:
            self.bg.close()
            self.bg = None


# ---------------------------------------------------------------- in a process of its own

_W = {}


def _init(rig_path, model, max_z, solver):
    _W["lr"] = LiveRefit(rig_path, model, max_z, solver)


def _update(fit_corners, placement):
    return _W["lr"].update(fit_corners, placement)


class BackgroundRefit:
    """LiveRefit in one spawned process: submit(fit_corners, placement) -> Future[Refit], in submission order, the
    state kept in that process. A context manager (the process ends on exit)"""

    def __init__(self, rig, model="KB4T", max_z=MAX_Z, solver=pipeline.DEFAULT_SOLVER):
        import multiprocessing as mp
        rig_path = str(rig if isinstance(rig, (str, os.PathLike)) else rig.path)
        self.ex = ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"), initializer=_init,
                                      initargs=(rig_path, model, max_z, pipeline.check_solver(solver)))

    def submit(self, fit_corners, placement=""):
        return self.ex.submit(_update, {k: np.asarray(v) for k, v in dict(fit_corners).items()}, placement)

    def close(self):
        self.ex.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------- a stored session, placement by placement


def on_session(stored, out_dir, solver=pipeline.DEFAULT_SOLVER, max_z=MAX_Z, log=None) -> dict:
    """the live refit on a stored session (sessions.StoredSession or anything replay.run takes), as the live tool would
    run it: holds through the capture-time assignment (assign.Assigner) in capture order; after each placement a
    refit on the FIT pairs so far (their capture-time labels; a pair beyond max_z is TEST); at the end the split
    written (out_dir/split, held-back holds released) and one more refit on its FIT file, "final split".
    -> {"session", "refits": [Refit.as_dict()], "split": {fit_npz, test_npz, fit_pairs, test_pairs}}"""
    from pathlib import Path
    from calibtool import assign, replay
    out_dir = Path(out_dir)
    session = replay.load_session(stored)
    rig = pipeline.load_rig(stored.rig_path)
    col = replay.Collection.for_session(session, rig, "capture", max_fit_z=max_z)
    lr = LiveRefit(stored.rig_path, max_z=max_z, solver=solver)
    t0 = session.pairs["t0"]
    done = []

    def refit_now(placement):
        lab = col.labeler.pair_labels()
        mask = np.array([lab.get(int(t)) == assign.FIT for t in t0], bool)
        r = lr.update(session.to_npz_dict(mask), placement)
        done.append(r)
        if log:
            log(f"refit {r.index} after {placement}: {r.pairs} pairs ({r.new_pairs} new), {r.seconds:.1f} s, "
                f"rms {r.rms_px:.3f} px, baseline {r.baseline_mm:.2f} mm")

    cur = None
    for h in assign.capture_holds(session):
        if cur is not None and h.placement != cur:
            col.end_placement(cur)
            refit_now(cur)
        col.add(h)
        cur = h.placement
    if cur is not None:
        col.end_placement(cur)
        refit_now(cur)
    split = col.labeler.write_split(session, out_dir / "split")
    r = lr.update(dict(np.load(split.fit_npz)), "final split")
    done.append(r)
    if log:
        log(f"refit {r.index} on the final split: {r.pairs} pairs, {r.seconds:.1f} s")
    return {"session": getattr(stored, "name", ""), "refits": [x.as_dict() for x in done],
            "split": {"fit_npz": str(split.fit_npz), "test_npz": str(split.test_npz), "fit_pairs": split.fit_pairs,
                      "test_pairs": split.test_pairs}}
