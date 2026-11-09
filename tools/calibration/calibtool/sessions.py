"""The stored calibration sessions calibtool replays: one registry entry per session, with what the recorded run of it
concluded (the numbers a replay must reproduce). Paths of data are relative to the data root (paths.data_root()) and
resolved only when asked for, so the registry itself is usable without data. Stdlib only.

Recorded numbers are quoted from the recorded logs (2 decimals as printed) except the step-45 fit's rms and baseline,
which are the exact values of the recorded fit JSON (source given per field)."""
from dataclasses import dataclass, field
from pathlib import Path

from calibtool import paths


@dataclass(frozen=True)
class Summary:
    """one row of run_lens_kb4t.summary_row as printed: TEST depth error median and IQR (%), row error median (px)"""
    median_pct: float
    iqr_pct: tuple
    rows_px: float


@dataclass(frozen=True)
class Recorded:
    """what the recorded run concluded. Entries are {(cell, (lo, hi) m): median depth error %} (step 66's cells and
    [lo, hi) distance bins)."""
    source: str                     # the recorded run's log, relative to the data root
    fit_pairs_16: int               # step 16: FIT pairs
    test_pairs_16: int              # step 16: TEST pairs
    fit_rms_px: float               # step 45
    fit_used_pairs: int             # step 45: pairs the converged fit used
    baseline_mm: float              # step 45
    new_verdict: str                # step 66, the new fit on TEST
    new_entries: dict               # step 66, the new fit: every validated entry
    live_verdict: str               # step 66, the live config on the same TEST
    live_n_validated: int
    live_failing: dict              # step 66, the live config: the validated entries outside the limits
    summary_new: Summary
    summary_live: Summary


@dataclass(frozen=True)
class StoredSession:
    name: str
    robot: str
    rig_path: Path                  # absolute: this repo's tools/calibration/config/<rig>.yaml
    take_order: list                # the order the recorded run gave step 16 (hold ids are offset in this order)
    corner_rel: str                 # folder of the per-take corner files (step 15b), relative to the data root
    raw_rel: str                    # folder of the raw take folders (pose_NNN.npz)
    live_rel: str                   # the robot's live OKVIS config at the time, relative to the data root
    live_model: str                 # its lens model: "RADTAN" | "KB4T" | "KB4"
    recorded: Recorded
    notes: str = field(default="", compare=False)

    @property
    def corner_files(self) -> list:
        root = paths.data_root()
        return [root / self.corner_rel / f"{t}.npz" for t in self.take_order]

    @property
    def raw_dirs(self) -> list:
        root = paths.data_root()
        return [root / self.raw_rel / t for t in self.take_order]

    @property
    def live_config(self) -> Path:
        return paths.data_root() / self.live_rel


_REG = {}


def _add(s: StoredSession):
    _REG[s.name] = s


_add(StoredSession(
    name="p21_1105", robot="phantom21", rig_path=paths.CONFIG / "rig-phantom21.yaml",
    take_order=["d050", "d050b", "d110", "d160", "d200", "dnear", "dleft", "dtiltleft"],
    corner_rel="p21/kit/out", raw_rel="p21/kit/takes",
    live_rel="p21/etc-okvis/mk12_stereo_holevisor-radtan-pre-daksh.yaml", live_model="RADTAN",
    recorded=Recorded(
        source="p21/kit/final/final21.log",
        fit_pairs_16=512, test_pairs_16=499,
        fit_rms_px=0.3671852349927913, baseline_mm=43.91544486205204,    # p21/kit/final/fit-kb4t.json
        fit_used_pairs=503,
        new_verdict="PASS",
        new_entries={("left", (0.9, 1.3)): 0.11, ("centre", (0.4, 0.6)): 0.03, ("centre", (0.6, 0.9)): -0.03,
                     ("right", (0.4, 0.6)): 0.14, ("right", (0.6, 0.9)): 0.10, ("bottom", (0.6, 0.9)): -0.32,
                     ("bottom", (0.9, 1.3)): -0.17},
        live_verdict="FAIL", live_n_validated=6,
        live_failing={("right", (0.6, 0.9)): 0.54, ("bottom", (0.9, 1.3)): 3.99},
        summary_new=Summary(0.04, (-0.19, 0.28), 0.207),
        summary_live=Summary(0.77, (0.07, 1.77), 0.820)),
    notes="phantom21, 5 Nov 2026: 8 neck-grid takes at 1280x800; live config radtan (scored through the kit's "
          "radtan_shim.py, which hard-codes /root/p21-cal paths)"))

_add(StoredSession(
    name="p10_1106", robot="phantom10", rig_path=paths.CONFIG / "rig-phantom11.yaml",
    take_order=["n6_075", "n6_110", "n6_160", "n6_200", "n6_far", "n6_litR", "n6_near", "n6_nearR", "n6_right",
                "n6_tiltL", "n6_tiltR"],
    corner_rel="p10/kit/out1280", raw_rel="p10/kit/takes",
    live_rel="p10/etc-okvis/phantom11_stereo_kb4t_20261105_1280_rscfix.yaml", live_model="KB4T",
    recorded=Recorded(
        source="p10/kit/fit1280/final1280.log",
        fit_pairs_16=587, test_pairs_16=572,
        fit_rms_px=0.4166901893946995, baseline_mm=44.05653407291519,    # p10/kit/fit1280/fit-kb4t.json
        fit_used_pairs=578,
        new_verdict="PASS",
        new_entries={("top", (0.9, 1.3)): 0.45, ("left", (0.6, 0.9)): -0.18, ("left", (0.9, 1.3)): -0.25,   # cells.txt
                     ("centre", (0.9, 1.3)): 0.29, ("right", (0.6, 0.9)): 0.15, ("right", (0.9, 1.3)): -0.30,
                     ("bottom-left", (0.9, 1.3)): -0.76, ("bottom", (0.4, 0.6)): -0.26,
                     ("bottom", (0.9, 1.3)): 0.04},
        live_verdict="FAIL", live_n_validated=7,
        live_failing={("top", (0.9, 1.3)): 0.91, ("left", (0.6, 0.9)): -2.12, ("left", (0.9, 1.3)): -6.56,
                      ("centre", (0.9, 1.3)): 0.51, ("right", (0.6, 0.9)): 2.69, ("right", (0.9, 1.3)): 2.14,
                      ("bottom", (0.9, 1.3)): -1.16},
        summary_new=Summary(-0.04, (-0.45, 0.33), 0.279),
        summary_live=Summary(0.06, (-2.49, 1.59), 3.967)),
    notes="phantom10 (phantom11 until 9 Nov 2026), 6 Nov: 11 neck-grid takes n6_* at 1280x800 (out1280/rejected/ "
          "left out); live config KB4T at 1280; the rig file keeps the old robot name"))


def names() -> list:
    return sorted(_REG)


def get(name: str) -> StoredSession:
    """the registry entry; KeyError for an unknown name"""
    try:
        return _REG[name]
    except KeyError:
        raise KeyError(f"unknown stored session {name!r}; known: {', '.join(names())}") from None
