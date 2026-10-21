"""The candidate phantom12 OKVIS config with Fit A's lens (distortion_type kb4t) and the measured T_BS. NOT DEPLOYED.

Task 6 of docs/superpowers/plans/2026-10-21-okvis-kb4t-depth-rectification.md. The source is the config phantom12 runs,
/etc/phantomos/okvis/phantom12_stereo_dense_thor.yaml (the 16 Oct transplant, sha256 7dd63554...); the repo holds the
same bytes in results/phantom12-2026-10-15/imucam/deploy/, the default source.

    cd tools/calibration/results/phantom12-2026-10-21-okvis-kb4t
    uv run --with numpy --with scipy --with opencv-python-headless --with pyyaml python make_candidate.py [source.yaml] \\
        > make_candidate.out 2>&1

1. The source's sha256 must start 7dd63554, else stop.
2. Lens, by the package's own surgery (step 71's okvis_text with kb4t=True, i.e. step 70's replace_tsc, repl_seq and
   replace_distortion_type) with Fit A (results/phantom12-2026-10-20-lens/fit-a-kb4t.json KB4T@all@800) and
   config/rig-phantom12.yaml (OKVIS block order [0, 1]): both camera blocks get Fit A's focal length, principal point,
   distortion_type kb4t and [k1, k2, k3, k4, p1, p2]; camera 1's T_SC = T_SC0 * C. okvis_text rewrites T_SC0 as
   okviscfg reads it (cleaned to the nearest rotation, ~1e-16 from the file's digits); the source's T_SC0 text is put
   back, so camera 0's pose is the deployed file's byte for byte.
3. T_BS: step 83 (83_tbs_apply.py, run as a program) on that file with the hand-eye reports
   results/phantom12-2026-10-20-handeye/fita450-P{1,2}.json: T_BS = mean(X) * T_SC0^-1.
4. Checks (any failure stops; the candidate is written only after all pass):
   (a) okviscfg loads it as pinhole-kb4t with Fit A's 10 numbers per camera;
   (b) camera 0's T_SC is the source's (text identical);
   (c) inv(T_SC0) T_SC1 == Fit A's C to 1e-12;
   (d) T_BS T_SC0 == X, the hand-eye mean (T_BC-fita.json), to 1e-9;
   (e) every value OKVIS reads (OpenCV FileStorage leaves) is the source's except the lens keys of both cameras,
       camera 1's T_SC and T_BS;
   (f) every changed text line is a header comment or one of those keys' lines.
Writes phantom12_stereo_dense_thor-fita.yaml and candidate.diff (unified diff against the source) beside this file."""
from __future__ import annotations

import contextlib
import difflib
import hashlib
import importlib
import io
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CAL = REPO / "tools/calibration"
SCRIPTS = CAL / "scripts"
sys.path.insert(0, str(SCRIPTS))
import okviscfg  # noqa: E402
import rigcfg  # noqa: E402
import rigmath as rm  # noqa: E402
import stereofit as sf  # noqa: E402

m70 = importlib.import_module("70_make_mk13_yaml")
m71 = importlib.import_module("71_export_lens")

SOURCE_SHA = "7dd63554"      # the 16 Oct transplant, deployed on phantom12
REPO_COPY = CAL / "results/phantom12-2026-10-15/imucam/deploy/phantom12_stereo_dense_thor.yaml"
FIT = CAL / "results/phantom12-2026-10-20-lens/fit-a-kb4t.json"
FIT_KEY = "KB4T@all@800"
HANDEYE = CAL / "results/phantom12-2026-10-20-handeye"
REPORTS = [HANDEYE / "fita450-P1.json", HANDEYE / "fita450-P2.json"]
T_BC_JSON = HANDEYE / "T_BC-fita.json"
RIG = CAL / "config/rig-phantom12.yaml"
OUT = HERE / "phantom12_stereo_dense_thor-fita.yaml"
DIFF = HERE / "candidate.diff"
PLAN = "docs/superpowers/plans/2026-10-21-okvis-kb4t-depth-rectification.md"
LENS_KEYS = ("focal_length", "principal_point", "distortion_coefficients", "distortion_type")
TSC = re.compile(r"T_SC:\s*\[[^\]]+\]", re.S)             # step 70's replace_tsc pattern
TBS = re.compile(r"\n(\s*)T_BS:\s*\n(\s*)\[[^\]]*\]")     # step 83's apply pattern
TOL_C, TOL_X, RTOL_LENS = 1e-12, 1e-9, 1e-14


def stop(msg: str) -> None:
    print("STOP:", msg)
    sys.exit(1)


def rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name


def scrub(s: str, *dirs) -> str:
    """printed output without local absolute paths"""
    for d, name in [(REPO, "<repo>"), *dirs, (Path.home(), "~")]:
        s = s.replace(str(d), name)
    return s


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def keep_tsc0(source: str, text: str) -> str:
    """text with its first T_SC block (OKVIS camera 0) replaced by the source's, verbatim"""
    s, t = list(TSC.finditer(source)), list(TSC.finditer(text))
    if len(s) != 2 or len(t) != 2:
        stop(f"T_SC blocks: {len(s)} in the source, {len(t)} after the surgery (want 2 and 2)")
    return text[:t[0].start()] + s[0].group(0) + text[t[0].end():]


def fs_leaves(path: Path) -> dict:
    """every value OKVIS's parser reads, by path (as transplant.py's --check)"""
    import cv2

    def leaves(node, where):
        if node.isMap():
            return {k2: v for k in node.keys() for k2, v in leaves(node.getNode(k), f"{where}.{k}").items()}
        if node.isSeq():
            return {k2: v for i in range(node.size()) for k2, v in leaves(node.at(i), f"{where}[{i}]").items()}
        if node.isString():
            return {where: node.string()}
        return {where: node.real() if (node.isReal() or node.isInt()) else None}

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        stop(f"FileStorage cannot open {rel(path)}")
    r = fs.root()
    return {k2: v for k in r.keys() for k2, v in leaves(r.getNode(k), k).items()}


def leaf_allowed(key: str) -> bool:
    if re.fullmatch(r"cameras\[[01]\]\.(" + "|".join(LENS_KEYS) + r")(\[\d+\])?", key):
        return True
    return bool(re.fullmatch(r"cameras\[1\]\.T_SC\[\d+\]", key) or re.fullmatch(r"imu_parameters\.T_BS\[\d+\]", key))


def allowed_lines(text: str) -> set:
    """line numbers (0-based) that may change: the header comments, the lens keys' lines, camera 1's T_SC block and
    the T_BS block"""
    lines = text.split("\n")
    line_of = lambda pos: text.count("\n", 0, pos)  # noqa: E731
    ok = set()
    for i, ln in enumerate(lines):
        if ln.startswith("cameras:"):
            break
        if i > 0 and ln.startswith("#"):
            ok.add(i)
    ok |= {i for i, ln in enumerate(lines) if ln.strip().startswith(tuple(k + ":" for k in LENS_KEYS))}
    tsc = list(TSC.finditer(text))
    if len(tsc) != 2:
        stop(f"{len(tsc)} T_SC blocks")
    ok |= set(range(line_of(tsc[1].start()), line_of(tsc[1].end()) + 1))
    m = TBS.search(text)
    if not m:
        stop("no T_BS block")
    ok |= set(range(line_of(m.start() + 1), line_of(m.end()) + 1))
    return ok


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_COPY
    digest = sha256(src)
    same_as_repo = src.read_bytes() == REPO_COPY.read_bytes()
    print(f"source: {src.name}, sha256 {digest}; byte-identical to the repo copy {rel(REPO_COPY)}: "
          f"{'yes' if same_as_repo else 'NO'}")
    if not digest.startswith(SOURCE_SHA):
        stop(f"the source's sha256 does not start {SOURCE_SHA} (the 16 Oct transplant): not the file this task "
             "starts from")
    source = src.read_text()

    with contextlib.redirect_stdout(io.StringIO()) as said:
        rig = rigcfg.load(RIG)
    print(scrub(said.getvalue()).rstrip())
    if rig.okvis_order != [0, 1]:
        stop(f"{rig.path.name}: okvis_order {rig.okvis_order}, this task was planned for [0, 1]")
    model, intr, C = sf.load_calibration(f"{FIT}:{FIT_KEY}")
    if model != "KB4T":
        stop(f"{rel(FIT)}:{FIT_KEY} is {model}")
    cal_sha = sf.calibration_digest(model, intr, C)
    baseline_mm = 1e3 * float(np.linalg.norm(C[:3, 3]))
    print(f"Fit A: {rel(FIT)}:{FIT_KEY}, calibration sha256 {cal_sha[:12]}, baseline {baseline_mm:.2f} mm")

    cost = [m71.p_drop_cost(intr[c], rig.resolution) for c in (0, 1)]
    vio_cost = "; ".join(f"camera {c} {v['within_350px']:.2f} px within 350 px, {v['whole_image']:.2f} px at the edge"
                         for c, v in enumerate(cost))
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    notes = [f"{OUT.name}: CANDIDATE, NOT DEPLOYED ({when}, {rel(Path(__file__))}, Task 6 of {PLAN}).",
             f"Made from phantom12_stereo_dense_thor.yaml (sha256 {digest[:8]}, the 16 Oct transplant, deployed) "
             "with ONLY these replaced: both cameras' lens, camera 1's T_SC, T_BS. Everything else is that file's.",
             f"Lens: Fit A ({rel(FIT)} {FIT_KEY}, 20 Oct; calibration sha256 {cal_sha[:12]}, baseline "
             f"{baseline_mm:.2f} mm). distortion_type kb4t, distortion_coefficients [k1, k2, k3, k4, p1, p2]: depth "
             "uses the full KB4T lens (the stereo network's rectification map); VIO k1..k4 (p1, p2 dropped there, "
             f"pixel cost {vio_cost}).",
             f"Needs an OKVIS image with the kb4t change ({PLAN}); older images refuse this file. Deploy check: OKVIS "
             "logs 'KB4T rectification (stereo network)' at startup.",
             "Fit A is not validated in the image corners, the bottom of the image or beyond 1.3 m "
             f"({rel(FIT.parent / 'README.md')}).",
             "camera 1's T_SC = T_SC0 * C (Fit A's stereo pose); camera 0's T_SC is the source's (Kalibr camera-IMU, "
             "15 Oct, fitted with the 15 Oct lens). Lines further below that call the lens equidistant or Kalibr's "
             "describe the lens this file replaced."]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stage1, cand = tmp / "stage1-lens.yaml", tmp / OUT.name
        with contextlib.redirect_stdout(io.StringIO()) as said:
            text = m71.okvis_text(src, rig, intr, C, notes, kb4t=True)
        if said.getvalue().strip():
            print(scrub(said.getvalue()).rstrip())
        stage1.write_text(keep_tsc0(source, text))
        p = subprocess.run([sys.executable, str(SCRIPTS / "83_tbs_apply.py"), str(stage1), str(cand),
                            *[a for r in REPORTS for a in ("--report", str(r))]], capture_output=True, text=True)
        out83 = scrub(p.stdout + p.stderr, (tmp, "<tmp>")).rstrip()
        print(f"step 83 (exit {p.returncode}): {out83}")
        if p.returncode != 0:
            stop("step 83 failed")
        cand_text = cand.read_text()

        # (a) the lens as okviscfg reads it
        cfg, base = okviscfg.load(cand), okviscfg.load(src)
        if cfg.lens != "pinhole-kb4t":
            stop(f"okviscfg reads lens {cfg.lens!r}, not pinhole-kb4t")
        worst = 0.0
        for i, k in enumerate(rig.okvis_order):
            K, D = cfg.K[i], cfg.D[i].ravel()
            got = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D])
            if got.size != 10:
                stop(f"camera {i}: {got.size} lens numbers, want 10")
            rd = np.abs(got - intr[k]) / np.maximum(np.abs(intr[k]), 1e-300)
            worst = max(worst, float(rd.max()))
            if not np.allclose(got, intr[k], rtol=RTOL_LENS, atol=0):
                stop(f"camera {i}: lens {got.tolist()} != Fit A's camera {k} {intr[k].tolist()}")
        print(f"(a) okviscfg: lens {cfg.lens}, {len(cfg.D)} cameras x 10 numbers = Fit A's "
              f"(largest relative difference {worst:.1e}, the 16-digit print; limit {RTOL_LENS:g})")

        # (b) camera 0's pose kept
        t0_src, t0_cand = TSC.search(source).group(0), TSC.search(cand_text).group(0)
        if t0_src != t0_cand or not np.array_equal(cfg.T_SC[0], base.T_SC[0]):
            stop("camera 0's T_SC changed")
        print("(b) camera 0's T_SC: the source's text, byte for byte")

        # (c) the stereo pose
        dC = float(np.abs(rm.inv(cfg.T_SC[0]) @ cfg.T_SC[1] - C).max())
        if dC > TOL_C:
            stop(f"inv(T_SC0) T_SC1 differs from C by {dC:.2e} > {TOL_C:g}")
        print(f"(c) inv(T_SC0) T_SC1 - C: largest entry {dC:.2e} (limit {TOL_C:g}); baseline "
              f"{1e3 * np.linalg.norm((rm.inv(cfg.T_SC[0]) @ cfg.T_SC[1])[:3, 3]):.4f} mm")

        # (d) the camera in the Head
        X = np.array(json.loads(T_BC_JSON.read_text())["T_BC"], float)
        dX = float(np.abs(cfg.T_BS @ cfg.T_SC[0] - X).max())
        if dX > TOL_X:
            stop(f"T_BS T_SC0 differs from {T_BC_JSON.name}'s X by {dX:.2e} > {TOL_X:g}")
        s81 = importlib.import_module("81_tbs_handeye")
        moved = s81.compare(cfg.T_BS @ cfg.T_SC[0], base.T_BS @ base.T_SC[0])
        print(f"(d) T_BS T_SC0 - X ({rel(T_BC_JSON)}): largest entry {dX:.2e} (limit {TOL_X:g}); camera 0 in the "
              f"Head moves {moved['angle_deg']:.3f} deg / {moved['dist_mm']:.2f} mm against the deployed T_BS")

        # (e) what OKVIS reads
        L, T = fs_leaves(src), fs_leaves(cand)
        changed = sorted(k for k in set(L) | set(T) if L.get(k) != T.get(k))
        bad = [k for k in changed if not leaf_allowed(k)]
        if bad:
            stop(f"values changed outside the lens keys, camera 1's T_SC and T_BS: {bad}")
        added = sorted(set(T) - set(L))
        groups = sorted({re.sub(r"\[\d+\]$", "", k) for k in changed})
        print(f"(e) FileStorage: {len(L)} values in the source, {len(changed)} differ ({len(added)} added: "
              f"{', '.join(added)}), {len(set(L) & set(T)) - (len(changed) - len(added))} identical; "
              f"the changed ones are {', '.join(groups)}")

        # (f) what the text changed
        a, b = source.split("\n"), cand_text.split("\n")
        ok_a, ok_b = allowed_lines(source), allowed_lines(cand_text)
        n_a = n_b = 0
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            out_a, out_b = set(range(i1, i2)) - ok_a, set(range(j1, j2)) - ok_b
            if out_a or out_b:
                stop("text changed outside the header, the lens keys, camera 1's T_SC and T_BS: source lines "
                     f"{sorted(i + 1 for i in out_a)}, candidate lines {sorted(j + 1 for j in out_b)}")
            n_a += i2 - i1
            n_b += j2 - j1
        print(f"(f) text: {n_a} source lines replaced by {n_b}, all header comments or lens / camera-1 T_SC / T_BS "
              "lines")

        OUT.write_text(cand_text)
        DIFF.write_text("".join(difflib.unified_diff(source.splitlines(keepends=True),
                                                     cand_text.splitlines(keepends=True),
                                                     fromfile=f"{src.name} (deployed, sha256 {digest[:8]})",
                                                     tofile=OUT.name)))
    print(f"wrote {rel(OUT)} (sha256 {sha256(OUT)}) and {rel(DIFF)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
