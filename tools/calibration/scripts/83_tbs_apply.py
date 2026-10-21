#!/usr/bin/env python3
"""Step 83 — write a measured T_BS into an OKVIS config, changing nothing else.

    83_tbs_apply.py <in.yaml> <out.yaml> --report A-handeye.json --report B-handeye.json [--single-take]

Used only when step 81 says REPLACE (the fit good to the gate) or IMPROVE (the config shown outside
the gate, the fit better but not good to it) on two takes that step 75 passed; the note says which,
with each fit's error bar. Every check fails closed: each report must say verdict REPLACE or IMPROVE
and qc_verdict GOOD (no defaults: a report from a take whose step 75 crashed has neither) and come from
the same model (pixels or poses); the reports must come from different takes (--single-take overrides
and says so in the config); every pair must agree as step 81's compare_takes says (within AGREE_DEG /
AGREE_MM, or within 2 combined sigma when each sigma is within the gate and the takes are within the
gate of each other); and the
camera pose T_SC0 each fit used must be the target config's, because T_BS = mean(X) · T_SC0^-1 with
the target's T_SC0. Text surgery like imucam/deploy/transplant.py: only the imu_parameters T_BS
block and provenance lines under the %YAML header change (the date, the takes, the change against
the config, and the previous T_BS for a rollback), so the same tool serves the results file
(phantom12_stereo.yaml) and the deployed one (phantom12_stereo_dense_thor.yaml)."""
from __future__ import annotations

import importlib
import itertools
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rigmath as rm  # noqa: E402


def check_reports(reports, single_take: bool = False) -> None:
    for i, r in enumerate(reports):
        if r.get("qc_verdict") != "GOOD":
            raise ValueError(f"report {i}: step 75 said {r.get('qc_verdict', 'nothing (no qc_verdict)')}, not GOOD: "
                             "not a measurement")
        if r.get("verdict") not in ("REPLACE", "IMPROVE"):
            raise ValueError(f"report {i} says {r.get('verdict', 'no verdict')}, not REPLACE or IMPROVE: nothing "
                             "measured to write")
    models = {r.get("model", "poses") for r in reports}
    if len(models) > 1:
        raise ValueError(f"reports from different models {sorted(models)}: refit the takes with one")
    takes = [r.get("take") for r in reports]
    if not single_take and (len(reports) < 2 or None in takes or len(set(takes)) < len(takes)):
        raise ValueError(f"the two-take rule needs reports from two or more different takes (got {takes}); "
                         "--single-take overrides and is written into the config")


def mean_tbs(reports, T_SC0, single_take: bool = False) -> np.ndarray:
    """T_BS = mean(T_BC_measured) · T_SC0^-1 over the reports, after check_reports, a T_SC0 match
    against the target (1e-9) and agreement of every pair within step 81's limits."""
    s81 = importlib.import_module("81_tbs_handeye")
    check_reports(reports, single_take)
    for i, r in enumerate(reports):
        if "T_SC0" not in r or not np.allclose(np.array(r["T_SC0"]), T_SC0, atol=1e-9):
            raise ValueError(f"report {i} was fitted with a different camera pose T_SC0 than the target config's: "
                             "refit against this config")
    for (i, a), (k, b) in itertools.combinations(enumerate(reports), 2):
        c = s81.compare_takes(a, b)
        if not c["agree"]:
            raise ValueError(f"takes {i} and {k} disagree by {c['angle_deg']:.3f} deg / {c['dist_mm']:.2f} mm "
                             f"(limits {s81.AGREE_DEG} deg / {s81.AGREE_MM} mm, or 2 combined sigma when each "
                             f"sigma and the distance are within the gate {s81.GATE_DEG} deg / {s81.GATE_MM} mm): "
                             "record another take")
    return rm.mean_pose([np.array(r["T_BC_measured"]) for r in reports]) @ rm.inv(T_SC0)


def format_tbs(T, indent: str) -> str:
    rows = [", ".join(repr(float(v)) for v in T[i]) for i in range(4)]
    return f"{indent}[" + f",\n{indent} ".join(rows) + "]"


def apply(text: str, T: np.ndarray, note: str) -> str:
    if not text.startswith("%YAML:1.0\n"):
        raise ValueError("the config does not start with a %YAML:1.0 line (CRLF line ends?)")
    m = re.search(r"\n(\s*)T_BS:\s*\n(\s*)\[[^\]]*\]", text)
    if not m:
        raise ValueError("no imu_parameters T_BS block")
    new = text[:m.start()] + f"\n{m.group(1)}T_BS:\n" + format_tbs(T, m.group(2)) + text[m.end():]
    return "%YAML:1.0\n" + "".join(f"# {line}\n" for line in note.splitlines()) + new[len("%YAML:1.0\n"):]


def old_tbs_text(text: str) -> str:
    """The T_BS block's numbers exactly as the file has them (a rollback restores the text, not a
    cleaned matrix), on one line."""
    m = re.search(r"\n\s*T_BS:\s*\n\s*\[([^\]]*)\]", text)
    if not m:
        raise ValueError("no imu_parameters T_BS block")
    return " ".join(m.group(1).split())


def note_for(names, T_new, T_old, T_SC0, old_text, single_take, reports=()) -> str:
    """Provenance lines: when, from which takes, how far the camera moves against the T_BS replaced
    (in the Head link, through the target's T_SC0), each report's verdict and 1-sigma, and the
    replaced T_BS's text for a rollback."""
    s81 = importlib.import_module("81_tbs_handeye")
    c = s81.compare(T_new @ T_SC0, T_old @ T_SC0)
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"T_BS below was MEASURED on phantom12 by step 81 (hand-eye fit), {when}, the mean of "
             f"{', '.join(names)}{' (SINGLE TAKE, two-take rule overridden)' if single_take else ''};",
             f"camera 0 in the Head moved {c['angle_deg']:.3f} deg / {c['dist_mm']:.2f} mm against the T_BS it replaced; "
             "lines below that call T_BS phantom21's or unverified describe that replaced value;"]
    nan = float("nan")
    for name, r in zip(names, reports):
        sd, sm = r.get("sigma_deg", nan), r.get("sigma_mm", nan)
        ca = r.get("cameras_apart")
        lines.append(f"{name} ({r.get('model', 'poses')} model): {r.get('verdict')}, 1 sigma {sd:.3f} deg / {sm:.2f} mm "
                     f"(2 sigma {2 * sd:.3f} deg / {2 * sm:.2f} mm against the gate {s81.GATE_DEG} deg / {s81.GATE_MM} mm)"
                     + (f", the cameras' own fits {ca['angle_deg']:.3f} deg / {ca['dist_mm']:.2f} mm apart" if ca else "")
                     + ";")
    for (i, a), (k, b) in itertools.combinations(enumerate(reports), 2):
        d = s81.compare_takes(a, b)
        lines.append(f"{names[i]} and {names[k]} are {d['angle_deg']:.3f} deg / {d['dist_mm']:.2f} mm apart;")
    if any(r.get("verdict") == "IMPROVE" for r in reports):
        lines.append("IMPROVE: better than the T_BS it replaced, NOT shown good to the gate at 2 sigma;")
    lines.append(f"the T_BS it replaced, as written (for a rollback): [{old_text}]")
    return "\n".join(lines)


def main() -> int:
    import argparse
    import okviscfg
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--report", type=Path, action="append", required=True, help="a step-81 report")
    ap.add_argument("--single-take", action="store_true", help="accept one take (written into the config)")
    a = ap.parse_args()
    cfg = okviscfg.load(a.src)
    reports = [json.loads(p.read_text()) for p in a.report]
    T = mean_tbs(reports, cfg.T_SC[0], a.single_take)
    text = a.src.read_text()
    note = note_for([p.name for p in a.report], T, cfg.T_BS, cfg.T_SC[0], old_tbs_text(text), a.single_take,
                    reports)
    a.dst.write_text(apply(text, T, note))
    print(f"wrote {a.dst}: T_BS from {len(reports)} report(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
