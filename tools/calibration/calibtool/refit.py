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
A hold dropped by the gate stays out of later refits (as it would be dropped from step 45's), unless it is gone.
The deciding fit is still step 45 (from nominal, on the final split): the refit shows progress and gives the
operator a fit to score while collecting; on the stored sessions the last refit reaches step 45's numbers.

BackgroundRefit: the same in one spawned process of its own (design 10.1: the fit in its own process, never in the
capture's), submit() after each placement -> a Future of the Refit; the state stays in that process."""
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

    def as_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()
                if k not in ("intr", "C")} | {"intr": [np.asarray(v).tolist() for v in self.intr],
                                               "C": np.asarray(self.C).tolist()}


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
        self.dropped = set()        # holds the gate dropped
        self.history = []

    def update(self, fit_corners, placement="") -> Refit:
        """refit on fit_corners (step 15's corner dict of every FIT pair so far) -> Refit (also in .history)"""
        sf = self.sf
        t = time.time()
        stamps = np.asarray(fit_corners["img_t0"], np.int64)
        with sf.solver_scope(self.solver):
            data = sf.Data(dict(fit_corners), self.rig.board)
            data = data.select(~np.isin(data.hold, list(self.dropped)))
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
            rms, misfit, z, far = self._v45.quality_gate(data, r, self.gates)
            bad = sorted(set(misfit) | set(far))
            rejected = len(bad) > self.gates["lens_bad_hold_max_frac"] * len(rms)
            if bad and not rejected:
                m = ~np.isin(data.hold, bad)
                data = data.select(m)
                r = sf.fit_converged(data, self.model, r["intr"], r["C"], r["poses"][m], log=self.log)
                nfev += r["nfev"]
                self.dropped |= set(int(h) for h in bad)
        self.intr, self.C = [np.array(v) for v in r["intr"]], np.array(r["C"])
        self.poses = {int(a): p for a, p in zip(stamps[data.src_pairs], r["poses"])}
        out = Refit(len(self.history) + 1, str(placement), int(data.n), int(len(np.unique(data.hold))), new, warm,
                    bool(r["success"]), bool(rejected), float(sf.fit_cost(r)[1]),
                    float(1e3 * np.linalg.norm(r["C"][:3, 3])), [int(h) for h in bad], int(failed), int(nfev),
                    time.time() - t, self.model, self.intr, self.C, os.getpid())
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
