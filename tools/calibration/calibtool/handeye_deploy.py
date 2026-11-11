"""The hand-eye deploy (milestone H4; hand-eye plan, docs/plans/2026-11-11-handeye-plan.md, section 6): a fitted URDF
correction (armcal.urdffit) and its HEAD term put on a robot as ONE transaction, with a rollback.

What it does
------------
The fit (a folder: fit.json, geometry.json, readb-right.json / readb-left.json) corrects the URDF OKVIS reads now
(NeckYaw origin, the shoulder-roll zeros) and, for M12+head, says where the camera sits in the head (HEAD, mm, head
frame). The deploy writes, next to the robot's files in /etc/phantomos/okvis (= /etc/okvis in the OKVIS pod):
  - <robot>_mk1_handeye_<date>.urdf: the robot's own URDF text with only the fitted <origin> lines changed and a
    provenance header (base md5, the live config's sha, the new T_BS's sha, the fit, the session);
  - <live config>_handeye<date>.yaml: a copy of the LIVE config (whatever robot.env names at the deploy, not the one
    the fit ran with) where only (a) the three translation numbers of T_BS change, by HEAD applied exactly once
    (T_BS' = D(HEAD) T_BS, rotation kept), (b) `dma: urdf:` names the new URDF, (c) header lines say what and why
    (a machine-readable `# handeye: {...}` line among them); every other byte is kept;
then points robot.env's OKVIS_CALIBRATION at the new config, restarts OKVIS by the path detected at the snapshot (the
DaemonSet's `rollout restart`, or okvis.sh on the robots with a bare okvis/okvis2x pod; never a zero grace period) and
checks OKVIS loaded both files: its log names the new config's sha256 prefix (the entrypoint) and says
`[Urdf] loaded '<new URDF>'` (UrdfTfModel::load; `Failed to load URDF '...'` fails it at once).
The transaction itself is calibtool.deploy.Deployer (the head calibration's: backup, rollback.sh first, atomic
writes, drift checks, the load check), given the URDF as an extra file.

Model only (mode "model_only"): T_BS is kept; only the URDF changes. The default model is then M12 (fitted without the
head term): M12+head's URDF terms were fitted together with HEAD (correlations -0.8 to -0.97), so dropping HEAD from
it would leave them biased; asking model-only for a model with a HEAD term is refused.

The checks (all before anything is written; any one refuses)
-----------------------------------------------------------
  identity      the robot's /etc/machine-id (and host name) name the fit's robot in config/robots.yaml
  drift         robot.env, the live config, the URDF and the OKVIS image are what the snapshot read
  base URDF     the URDF OKVIS reads now (the live config's `urdf:`) has the md5 the captures were matched with;
                OKVIS_URDF in robot.env (it would override `urdf:`) is refused
  geometry      the live config's T_BS and cameras' T_SC equal the ones the captures were taken with (geometry.json;
                T_SC from the fit-time config, read on the robot and its sha256 checked): else "hand-eye stale for
                this camera geometry" -- a head calibration changed it since, run the hand-eye again
  once          the live config does not carry this fit already (`# handeye:` line): HEAD is never applied twice (a
                second deploy would also fail the T_BS and base-URDF checks)
  the fit       its records share one URDF md5 and one camera-in-head (and agree with geometry.json); HEAD under
                MAX_HEAD_MM; the new URDF and T_BS reproduce the fit's prediction for every fitted capture (FK of the
                written files, to PREDICT_TOL_MM)
  restart path  OKVIS can be restarted on this robot

Rolling back
------------
Automatic when OKVIS crash-loops or the load check fails (config sha or the URDF line missing). By hand: R in the
caller (HandeyeDeployer.rollback) or `bash <deploy dir>/rollback.sh` (robot.env as saved, OKVIS restarted). The old
config and the old URDF are never changed, so robot.env back is the whole rollback; the new files stay (inert).

Also here: the post-check rule (postcheck_scores / postcheck_verdict: paired, no single-capture threshold), the head
tool's coupling (stale / stale_line: a deploy that changes T_BS or T_SC makes the hand-eye stale), the consumers
(config/handeye_consumers.yaml: OKVIS switched here, the point cloud published as a file, cuRobo and box-pick listed
as not updated), and a dry run (dry_run_remote: every robot command recorded, nothing sent)."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath

import numpy as np

from calibtool import deploy as dp, paths, robots

CONSUMERS_YAML = paths.PKG / "config" / "handeye_consumers.yaml"
KINDS = ("okvis", "offline_file", "not_updated")
T_TOL = 1e-9                    # T_BS / T_SC: the same text parses to the same floats
T_BC_TOL = 1e-6                 # a record's T_BC against T_BS @ T_SC_depth (the burst's T_SC is float32)
PREDICT_TOL_MM = 1e-3           # written files vs the fit's in-memory prediction
MAX_HEAD_MM = 40.0              # a camera does not sit 4 cm from where the head's T_BS puts it: a units/sign slip
MARKER = "handeye:"
GATE_MM = 6.0                   # the user's aim (plan 9.1 #4): held-out / post-check median at the wrist
ALPHA = 0.05
MIN_POSTCHECK = 6


# ================================================================ the OKVIS config text

_TBS = re.compile(r"^[^#\n]*?\bT_BS:\s*\[(?P<body>[^\]]*)\]", re.M)       # never inside a comment
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
TBS_TRANSLATION = (3, 7, 11)


def _tbs_match(text):
    ms = list(_TBS.finditer(text or ""))
    if len(ms) != 1:
        raise ValueError(f"{len(ms)} T_BS blocks in the config: want exactly 1")
    body = ms[0].group("body")
    if len(_NUM.findall(body)) != 16 or _NUM.sub("", body).strip(", \t\r\n"):
        raise ValueError(f"T_BS block is not 16 numbers: [{body.strip()[:80]}]")
    return ms[0]


def t_bs(text) -> np.ndarray:
    """imu_parameters T_BS (IMU S -> Head link B) as written (no cleaning)"""
    return np.array([float(x) for x in _NUM.findall(_tbs_match(text).group("body"))]).reshape(4, 4)


def t_sc(text) -> list:
    """the cameras' T_SC as written (calibtool.imu's reader: exactly two blocks)"""
    from calibtool import imu
    return [np.asarray(T) for T in imu.t_sc(text)]


def apply_head(text, head_mm) -> str:
    """T_BS' = D(HEAD) T_BS: the three translation numbers + HEAD (mm -> m), each rewritten in place (repr of the
    float); the rotation and every other byte kept"""
    from calibtool import imu
    m = _tbs_match(text)
    T = t_bs(text)
    T[:3, 3] = T[:3, 3] + 1e-3 * np.asarray(head_mm, float)
    body = imu._rewrite(m.group("body"), T.ravel(), set(TBS_TRANSLATION))
    return text[:m.start("body")] + body + text[m.end("body"):]


_DMA = re.compile(r"^dma\s*:\s*(#.*)?$")
_URDF_LINE = re.compile(r"^(?P<pre>\s+urdf\s*:\s*)(?P<q>[\"']?)(?P<path>[^\"'#\n]*?)(?P=q)(?P<post>\s*(?:#.*)?)$")


def _dma_urdf_line(text):
    """(line index, match) of the `urdf:` key in the top-level `dma:` section; None when absent"""
    lines = (text or "").split("\n")
    start = next((i for i, ln in enumerate(lines) if _DMA.match(ln)), None)
    if start is None:
        return None, lines
    hits = []
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln and not ln[0].isspace() and not ln.startswith("#"):
            break
        m = _URDF_LINE.match(ln)
        if m:
            hits.append((i, m))
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} `urdf:` keys under dma: want one")
    return (hits[0] if hits else None), lines


def dma_urdf(text):
    """the config's `dma: urdf:` path (the pod's path), or None"""
    hit, _ = _dma_urdf_line(text)
    return hit[1].group("path").strip() if hit else None


def set_urdf(text, path) -> str:
    """`dma: urdf:` pointed at path, its quoting and comment kept"""
    hit, lines = _dma_urdf_line(text)
    if not hit:
        raise ValueError("the config has no `urdf:` under dma:")
    i, m = hit
    lines[i] = f"{m.group('pre')}{m.group('q')}{path}{m.group('q')}{m.group('post')}"
    return "\n".join(lines)


def add_note(text, lines) -> str:
    """comment lines right after the %YAML header (at the top when there is none)"""
    note = "".join(f"# {ln}\n" for ln in lines)
    head = "%YAML:1.0\n" if text.startswith("%YAML:1.0\n") else ""
    return head + note + text[len(head):]


def image_size(text) -> tuple:
    """cam0's image_dimension (the frames OKVIS gets)"""
    items = dp._camera_items(text)
    d = re.search(r"\bimage_dimension\s*:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", items[0]) if items else None
    if not d:
        raise ValueError("no image_dimension in camera 0")
    return int(float(d.group(1))), int(float(d.group(2)))


def geom_sha(T) -> str:
    """a short sha of transforms' numbers (not their text: a reformatted file keeps it)"""
    a = np.concatenate([np.asarray(x, float).ravel() for x in (T if isinstance(T, (list, tuple)) else [T])])
    return hashlib.sha256(",".join(f"{v:.12g}" for v in a).encode()).hexdigest()[:12]


def marker(text):
    """the `# handeye: {...}` record of the config's deployed hand-eye, or None"""
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s.startswith("#") and s[1:].strip().startswith(MARKER):
            return json.loads(s[1:].strip()[len(MARKER):])
    return None


def _md5(text) -> str:
    return hashlib.md5((text or "").encode()).hexdigest()


def _sha256(text) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


# ================================================================ the fit


@dataclass
class FitGeometry:
    """what the captures were taken with: the burst's info okvis_config / okvis_config_sha256, its T_BS (and the depth
    frames' T_SC), and optionally the config's cameras' T_SC (else read from the fit-time config on the robot)"""
    robot: str
    session: str
    okvis_config: str
    okvis_config_sha256: str
    T_BS: np.ndarray
    T_SC_depth: object = None
    T_SC: object = None
    source: str = ""

    @classmethod
    def load(cls, path) -> "FitGeometry":
        d = json.loads(Path(path).read_text())
        arr = lambda v: None if v is None else np.asarray(v, float)  # noqa: E731
        return cls(str(d["robot"]), str(d["session"]), str(d["okvis_config"]), str(d["okvis_config_sha256"]),
                   arr(d["T_BS"]), arr(d.get("T_SC_depth")),
                   None if d.get("T_SC") is None else [np.asarray(T, float) for T in d["T_SC"]], d.get("source", ""))

    def as_dict(self) -> dict:
        tl = lambda v: None if v is None else np.asarray(v).tolist()  # noqa: E731
        return dict(robot=self.robot, session=self.session, source=self.source, okvis_config=self.okvis_config,
                    okvis_config_sha256=self.okvis_config_sha256, T_BS=tl(self.T_BS), T_SC_depth=tl(self.T_SC_depth),
                    T_SC=None if self.T_SC is None else [tl(T) for T in self.T_SC])


def geometry_from_bursts(burst_paths, robot, session) -> FitGeometry:
    """geometry.json from a session's bursts; ValueError when they do not share one geometry"""
    from armcal import burst_io
    ref = None
    for p in burst_paths:
        b = burst_io.load(p)
        if b.T_BS is None:
            raise ValueError(f"{p}: no T_BS in the burst (captured before Plan 4)")
        k = (b.info.get("okvis_config"), b.info.get("okvis_config_sha256"), np.asarray(b.T_BS, float),
             np.asarray(b.T_SC, float))
        if ref is None:
            ref = k
        elif k[:2] != ref[:2] or not (np.array_equal(k[2], ref[2]) and np.array_equal(k[3], ref[3])):
            raise ValueError(f"{p}: another geometry than the first burst's: one fit, one geometry")
    if ref is None:
        raise ValueError("no bursts")
    if not ref[0] or not ref[1]:
        raise ValueError("the bursts do not name their OKVIS config (info okvis_config / okvis_config_sha256)")
    return FitGeometry(str(robot), str(session), ref[0], ref[1], ref[2], ref[3], None,
                       f"{len(list(burst_paths))} bursts")


@dataclass
class HandeyeFit:
    """a fit folder: fit.json (urdffit's report), geometry.json, readb-right.json / readb-left.json"""
    path: str
    report: dict
    geometry: FitGeometry
    records: dict                   # side ("Right" / "Left") -> readb output
    sha12: str

    @classmethod
    def load(cls, d) -> "HandeyeFit":
        d = Path(d)
        text = (d / "fit.json").read_text()
        recs = {}
        for name in ("readb-right.json", "readb-left.json"):
            if (d / name).exists():
                r = json.loads((d / name).read_text())
                recs[r["side"]] = r
        return cls(str(d), json.loads(text), FitGeometry.load(d / "geometry.json"), recs, dp.sha12(text))

    @property
    def fit_id(self) -> str:
        return f"{self.geometry.session}/{self.sha12}"

    @property
    def base_md5(self):
        md5s = {r.get("urdf_md5") for r in self.records.values()}
        return md5s.pop() if len(md5s) == 1 else None

    def selected(self, side) -> list:
        """the fitted captures of `side` (fit.json selection) as urdffit observations"""
        from armcal import urdffit
        names = set((self.report.get("selection") or {}).get(side.lower(), []))
        r = self.records.get(side)
        return [o for o in urdffit.observations(r["res"], side) if o.name in names] if r else []

    def problems(self) -> list:
        out = []
        md5s = {r.get("urdf_md5") for r in self.records.values()}
        if not self.records:
            out.append("the fit has no readb records")
        elif len(md5s) != 1 or None in md5s:
            out.append(f"the arms' records were matched with different URDFs ({sorted(map(str, md5s))})")
        if self.report.get("chosen") not in (self.report.get("models") or {}):
            out.append("fit.json names no chosen model")
        T_BC = [np.asarray(x["T_BC"], float) for r in self.records.values() for x in r["res"]]
        if T_BC and max(np.abs(T - T_BC[0]).max() for T in T_BC) > T_BC_TOL:
            out.append("the captures were taken with more than one camera-in-head (T_BC): one fit, one geometry")
        g = self.geometry
        if T_BC and g.T_SC_depth is not None and np.abs(g.T_BS @ g.T_SC_depth - T_BC[0]).max() > T_BC_TOL:
            out.append("geometry.json's T_BS @ T_SC_depth is not the records' camera-in-head: wrong geometry.json")
        return out


def choose(fit: HandeyeFit, mode="head", model=None):
    """(model, URDF change without HEAD, HEAD mm or None)"""
    if mode not in ("head", "model_only"):
        raise ValueError(f"mode must be 'head' or 'model_only', got {mode!r}")
    models = fit.report.get("models") or {}
    if model is None:
        model = fit.report.get("chosen") if mode == "head" else ("M12" if "M12" in models else fit.report.get("chosen"))
    if model not in models:
        raise ValueError(f"model {model!r} is not in fit.json (have {sorted(models)})")
    change = json.loads(json.dumps(models[model]["change"]))
    head = change.pop("HEAD", None)
    if mode == "model_only" and head is not None:
        raise ValueError(f"model-only with {model}: its URDF terms were fitted together with the head term, so without "
                         f"it they are biased; deploy {model} with the head term, or model-only with M12")
    if not change:
        raise ValueError(f"model {model} changes no URDF origin")
    head_mm = None if head is None else np.asarray(head["dxyz"], float)
    if head_mm is not None and np.linalg.norm(head_mm) > MAX_HEAD_MM:
        raise ValueError(f"HEAD {np.round(head_mm, 1).tolist()} mm is over {MAX_HEAD_MM:g} mm: a units or sign slip, "
                         "not a camera position; not deploying it")
    return model, change, head_mm


# ================================================================ the snapshot

HOSTNAME, MACHINE_ID = "/etc/hostname", "/etc/machine-id"


@dataclass
class HandeyeSnapshot:
    deploy: dp.DeploySnapshot
    hostname: object
    machine_id: object
    env_urdf: object                # OKVIS_URDF in robot.env (overrides `urdf:`), or None
    urdf: object                    # the live config's `urdf:` (the pod's path)
    urdf_host: object
    urdf_text: object
    fit_config_text: object         # the fit-time config read on the robot (T_SC), or None
    listing: list = field(default_factory=list)     # file names in /etc/phantomos/okvis

    def expected(self) -> list:
        return self.deploy.expected() + ([("the URDF OKVIS reads", "file", self.urdf_host, self.urdf_text)]
                                         if self.urdf_host else [])

    def as_dict(self) -> dict:
        d = asdict(self)
        d["deploy"] = self.deploy.as_dict()
        return d

    @classmethod
    def from_dict(cls, d) -> "HandeyeSnapshot":
        d = dict(d)
        ds = dict(d.pop("deploy"))
        ds["dependents"] = [dp.Dependent(**x) for x in ds.get("dependents") or []]
        return cls(dp.DeploySnapshot(**ds), **d)


def take_snapshot(remote: dp.Remote, fit: HandeyeFit, clock=time.time, origin="calibtool handeye") -> HandeyeSnapshot:
    ds = dp.take_snapshot(remote, producer_yaml=None, clock=clock, origin=origin)
    env = dp.parse_env(ds.robot_env)
    urdf = dma_urdf(ds.config_text) if ds.config_text else None
    host = dp.host_path(env.get("OKVIS_URDF") or urdf) if (env.get("OKVIS_URDF") or urdf) else None
    fit_cfg = None
    if fit.geometry.T_SC is None and fit.geometry.okvis_config:
        fit_cfg = remote.read(dp.host_path(fit.geometry.okvis_config))
    rc, out = remote.call(["ls", "-1", dp.HOST_CFG_DIR])
    return HandeyeSnapshot(ds, (remote.read(HOSTNAME) or "").strip() or None,
                           (remote.read(MACHINE_ID) or "").strip() or None, env.get("OKVIS_URDF"), urdf, host,
                           remote.read(host) if host else None, fit_cfg,
                           sorted(out.split()) if rc == 0 else [])


# ================================================================ the pre-checks (pure)


def fit_T_SC(snap: HandeyeSnapshot, fit: HandeyeFit):
    """(the cameras' T_SC the captures were taken with, or None, why)"""
    g = fit.geometry
    if g.T_SC is not None:
        return g.T_SC, ""
    if snap.fit_config_text is None:
        return None, f"the fit-time config {g.okvis_config} is not on the robot any more"
    if _sha256(snap.fit_config_text) != g.okvis_config_sha256:
        return None, (f"the fit-time config {g.okvis_config} changed since the captures (sha256 "
                      f"{_sha256(snap.fit_config_text)[:12]}, the bursts say {g.okvis_config_sha256[:12]})")
    return t_sc(snap.fit_config_text), ""


def precheck(snap: HandeyeSnapshot, fit: HandeyeFit, registry=None) -> list:
    """every reason not to deploy, [] when none (module text, "The checks")"""
    out = []
    try:
        ident = robots.identify(snap.hostname, snap.machine_id, registry)
        if not ident.machine_id_verified:
            out.append(f"robot identity: {ident.robot.bot} matched by host name only (its machine-id is not in "
                       "robots.yaml): not deploying a per-robot geometry on that")
        elif ident.robot.bot != fit.geometry.robot:
            out.append(f"robot identity: this is {ident.robot.bot} (machine-id), the fit is {fit.geometry.robot}'s")
    except robots.IdentityError as e:
        out.append(f"robot identity: {e}")
    out += fit.problems()
    ds = snap.deploy
    if not ds.config_text:
        out.append(f"no live OKVIS config to change (robot.env names {ds.calibration!r})")
        return out
    if snap.env_urdf:
        out.append(f"robot.env sets OKVIS_URDF={snap.env_urdf}, which overrides the config's urdf:; the deploy "
                   "switches the config's: remove OKVIS_URDF from robot.env first")
    if not snap.urdf:
        out.append("the live config names no URDF (dma: urdf:): OKVIS reads none, nothing to correct")
    elif snap.urdf_text is None:
        out.append(f"the URDF OKVIS reads ({snap.urdf_host}) is not on the robot")
    elif _md5(snap.urdf_text) != fit.base_md5:
        out.append(f"the URDF OKVIS reads now ({snap.urdf_host}, md5 {_md5(snap.urdf_text)[:8]}) is not the one the "
                   f"captures were matched with (md5 {str(fit.base_md5)[:8]}): fit again on this URDF")
    try:
        live_bs = t_bs(ds.config_text)
        if np.abs(live_bs - fit.geometry.T_BS).max() > T_TOL:
            out.append(f"hand-eye stale for this camera geometry: the live config's T_BS ({ds.calibration}) is not the "
                       f"one the captures were taken with ({fit.geometry.okvis_config}); run the hand-eye again")
        fsc, why = fit_T_SC(snap, fit)
        if fsc is None:
            out.append(f"cannot check the cameras' T_SC: {why}; give geometry.json its T_SC")
        else:
            live_sc = t_sc(ds.config_text)
            if len(live_sc) != len(fsc) or any(np.abs(a - b).max() > T_TOL for a, b in zip(live_sc, fsc)):
                out.append(f"hand-eye stale for this camera geometry: the live config's camera T_SC "
                           f"({ds.calibration}) is not the one the captures were taken with; run the hand-eye again")
    except ValueError as e:
        out.append(f"the live config: {e}")
    mk = marker(ds.config_text)
    if mk and mk.get("fit") == fit.fit_id:
        out.append(f"the live config already carries this fit ({fit.fit_id}, mode {mk.get('mode')}): its HEAD term "
                   "is never applied twice")
    if ds.restart.get("kind") == "none":
        out.append(f"no OKVIS restart path: {ds.restart.get('detail')}")
    return out


# ================================================================ the new files (pure)


@dataclass
class Build:
    mode: str
    model: str
    head_mm: object
    urdf_name: str
    urdf_text: str
    config_name: str
    config_text: str
    size: tuple
    marker: dict
    T_BS_old: list
    T_BS_new: list
    predict_mm: float               # max |written files' prediction - the fit's| over the fitted captures

    @property
    def urdf_container(self) -> str:
        return dp.container_path(self.urdf_name)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("urdf_text"), d.pop("config_text")
        d["head_mm"] = None if self.head_mm is None else np.asarray(self.head_mm).tolist()
        return d


def free_name(stem, ext, taken) -> str:
    taken = set(taken or ())
    for suf in [""] + [chr(c) for c in range(ord("b"), ord("z") + 1)]:
        n = f"{stem}{suf}{ext}"
        if n not in taken:
            return n
    raise ValueError(f"no free name for {stem}{ext}")


def predict_check(fit: HandeyeFit, base, change, head_mm, new_urdf_text, T_BS_new) -> float:
    """max |offset| difference (mm) over the fitted captures between the fit's prediction (urdffit.off_change: base +
    change, camera moved by HEAD) and the written files' (the new URDF parsed, T_BC' = T_BS' T_BS_fit^-1 T_BC)"""
    from armcal import urdffit
    from armcal.urdf import Urdf
    ch = dict(change)
    if head_mm is not None:
        ch["HEAD"] = {"dxyz": np.asarray(head_mm, float)}
    new = Urdf(new_urdf_text)
    move = np.asarray(T_BS_new, float) @ np.linalg.inv(fit.geometry.T_BS)
    worst = 0.0
    for side in ("Right", "Left"):
        obs = fit.selected(side)
        if not obs:
            continue
        want = urdffit.off_change(obs, base, ch)
        got = urdffit.offsets([replace(o, T_BC=move @ o.T_BC) for o in obs], new)
        worst = max(worst, float(np.abs(want - got).max()))
    return worst


def build(snap: HandeyeSnapshot, fit: HandeyeFit, mode="head", model=None, date=None, session="") -> Build:
    """the new URDF and config texts; ValueError when they cannot be made or do not reproduce the fit"""
    from armcal.urdf import Urdf, variant_text
    model, change, head = choose(fit, mode, model)
    date = date or time.strftime("%Y%m%d")
    robot = (snap.hostname or fit.geometry.robot).split(".")[0]
    live = snap.deploy.config_text
    base = Urdf(snap.urdf_text, snap.urdf_host)
    urdf_name = free_name(f"{robot}_mk1_handeye_{date}", ".urdf", snap.listing)
    stem = re.sub(r"_handeye\d{8}[a-z]?$", "", PurePosixPath(snap.deploy.calibration or "okvis.yaml").stem)
    config_name = free_name(f"{stem}_handeye{date}", ".yaml", set(snap.listing) | {urdf_name})
    T_old = t_bs(live)
    text = apply_head(live, head) if head is not None else live
    T_new = t_bs(text)
    text = set_urdf(text, dp.container_path(urdf_name))
    sc = t_sc(text)
    p = fit.report["models"][model]
    terms = ", ".join(f"{n} = {v[0]:+.3f} +- {v[1]:.3f}" for n, v in p["params"].items())
    prov = [f"deployed by calibtool.handeye_deploy, session {session or '-'}, {date}, mode {mode}",
            f"fit {fit.fit_id} ({fit.path}), model {model}: {terms} (mm, deg)",
            f"captures taken with {fit.geometry.okvis_config} (sha256 {fit.geometry.okvis_config_sha256[:12]}), "
            f"session {fit.geometry.session}",
            f"goes with the OKVIS config {config_name}, made from the live {snap.deploy.calibration} (sha256 "
            f"{_sha256(live)[:12]}); its T_BS sha {geom_sha(T_new)}" + (
                f" = the live T_BS (sha {geom_sha(T_old)}) moved by HEAD {np.round(head, 3).tolist()} mm"
                if head is not None else " = the live T_BS, unchanged (no HEAD term)")]
    urdf_text = variant_text(base, change, prov)
    mk = {"fit": fit.fit_id, "session": session, "date": date, "mode": mode, "model": model,
          "head_mm": None if head is None else [round(float(v), 6) for v in head],
          "T_BS": geom_sha(T_new), "T_SC": geom_sha(sc), "urdf": dp.container_path(urdf_name),
          "urdf_md5": _md5(urdf_text), "base_urdf": snap.urdf, "base_md5": _md5(snap.urdf_text),
          "from": f"{PurePosixPath(snap.deploy.calibration or '?').name} ({_sha256(live)[:12]})"}
    head_line = (f"T_BS translation + HEAD {np.round(head, 3).tolist()} mm (head frame; T_BS' = D(HEAD) T_BS, applied "
                 f"once), the T_BS before: {np.round(T_old[:3, 3], 9).tolist()} m" if head is not None else
                 "T_BS unchanged (model only: no HEAD term)")
    text = add_note(text, [f"hand-eye deploy {date} (calibtool.handeye_deploy, session {session or '-'}): from "
                           f"{mk['from']}; urdf: {snap.urdf} -> {mk['urdf']} (fit {fit.fit_id}, model {model})",
                           head_line, f"{MARKER} {json.dumps(mk, separators=(',', ':'))}"])
    if t_bs(text).tolist() != T_new.tolist() or dma_urdf(text) != mk["urdf"]:
        raise ValueError("the new config does not read back as written")
    worst = predict_check(fit, base, change, head, urdf_text, T_new)
    if not worst <= PREDICT_TOL_MM:
        raise ValueError(f"the written URDF and T_BS predict the fitted captures {worst:.2g} mm off the fit "
                         f"(limit {PREDICT_TOL_MM:g} mm): not deploying")
    return Build(mode, model, head, urdf_name, urdf_text, config_name, text, image_size(text), mk, T_old.tolist(),
                 T_new.tolist(), worst)


# ================================================================ the consumers


@dataclass(frozen=True)
class Consumer:
    name: str
    kind: str
    reads: str = ""
    t_bs: str = ""
    owner_action: str = ""
    why: str = ""


def load_consumers(path=CONSUMERS_YAML) -> list:
    import yaml
    raw = (yaml.safe_load(Path(path).read_text()) or {}).get("consumers") or []
    out = []
    for e in raw:
        if e.get("kind") not in KINDS:
            raise ValueError(f"{path}: consumer {e.get('name')!r}: kind {e.get('kind')!r} (known: {', '.join(KINDS)})")
        out.append(Consumer(str(e["name"]), e["kind"], *(str(e.get(k) or "") for k in
                                                         ("reads", "t_bs", "owner_action", "why"))))
    if sum(c.kind == "okvis" for c in out) != 1:
        raise ValueError(f"{path}: exactly one consumer of kind okvis")
    return out


def consumer_lines(consumers, b: Build, okvis_state: str, published: dict) -> list:
    out = []
    for c in consumers:
        if c.kind == "okvis":
            out.append(f"{c.name}: {okvis_state}")
        elif c.kind == "offline_file":
            p = published.get(c.name)
            out.append(f"{c.name}: NOT switched on the robot (reads {c.reads}); "
                       + (f"the per-robot URDF is published at {p}" if p else "the per-robot URDF is not published")
                       + (f"; its owner must: {c.owner_action}" if c.owner_action else "")
                       + (f"; T_BS: {c.t_bs}" if c.t_bs else ""))
        else:
            out.append(f"{c.name}: not updated ({c.why})")
    return out


# ================================================================ the transaction


def dry_run_remote(run, recorder=None) -> dp.Remote:
    """the dry run: reads go to the robot, every mutating command is RECORDED and not sent (Remote.text(): one per
    line; with a sessionctl.Recorder, also in its command log under [handeye])"""
    return dp.Remote(run, dry_run=True, log=recorder.hook("handeye") if recorder is not None else None)


@dataclass
class HandeyeResult:
    ok: bool
    dry_run: bool
    detail: str
    build: dict
    deploy: dict
    rollback: object = None         # the automatic rollback's result (dict), or None
    consumers: list = field(default_factory=list)
    left: object = None             # what the deploy left on the robot (a later rollback compares with it)

    def as_dict(self) -> dict:
        return asdict(self)

    def report(self) -> str:
        b = self.build
        L = [f"hand-eye deploy: {self.detail}",
             f"  mode {b.get('mode')}, model {b.get('model')}, HEAD {b.get('head_mm')} mm; written files predict the "
             f"fit to {b.get('predict_mm', float('nan')):.1e} mm",
             f"  URDF {b.get('urdf_name')}, config {b.get('config_name')} ({b.get('size')})",
             f"  rollback: bash {self.deploy.get('rollback_script')}  (R in the tool)"]
        if self.rollback is not None:
            L.append(f"  ROLLED BACK automatically: {self.rollback}")
        L += [f"  {ln}" for ln in self.consumers]
        return "\n".join(L)


class HandeyeDeployer:
    """remote: dp.Remote (dry_run_remote for a dry run); fit: HandeyeFit; snap: take_snapshot(); deploy_dir: the deploy
    folder ON THE ROBOT (backups, rollback.sh); publish_dir: a local folder for the offline consumers' URDF (None: not
    published)"""

    def __init__(self, remote: dp.Remote, fit: HandeyeFit, snap: HandeyeSnapshot, deploy_dir, mode="head", model=None,
                 session="", date=None, registry=None, consumers=None, publish_dir=None, clock=time.monotonic,
                 sleep=time.sleep, up_timeout_s=dp.UP_TIMEOUT_S):
        self.remote, self.fit, self.snap, self.mode, self.model = remote, fit, snap, mode, model
        self.session, self.date, self.registry = session, date, registry
        self.consumers = load_consumers() if consumers is None else list(consumers)
        self.publish_dir = publish_dir
        self.inner = dp.Deployer(remote, snap.deploy, deploy_dir, clock=clock, sleep=sleep, session=session,
                                 up_timeout_s=up_timeout_s)
        self.build = None

    @property
    def rollback_path(self):
        return self.inner.rollback_path

    def plan(self) -> dp.DeployPlan:
        b = self.build
        u = b.urdf_container
        return dp.DeployPlan(b.config_name, b.config_text, b.size,
                             extra_files=((f"{dp.HOST_CFG_DIR}/{b.urdf_name}", b.urdf_text),),
                             log_expect=(f"[Urdf] loaded '{u}'",),
                             log_fail=(f"Failed to load URDF '{u}'", f"[Urdf] failed to read '{u}'"))

    def deploy(self) -> HandeyeResult:
        d = dp.drift(self.snap.expected(), self.remote)
        if d:
            raise dp.DeployDrift("the robot changed since the tool read it; not deploying, ask the operator: "
                                 + "; ".join(d))
        problems = precheck(self.snap, self.fit, self.registry)
        if not problems:
            try:
                self.build = build(self.snap, self.fit, self.mode, self.model, self.date, self.session)
            except (ValueError, KeyError) as e:
                problems.append(str(e))
        if problems:
            raise dp.DeployRefused("not deploying the hand-eye: " + "; ".join(problems))
        b = self.build
        rb = None
        try:
            res = self.inner.deploy(self.plan())
        except dp.DeployFailed as e:
            rb = self._auto_rollback(f"the deploy failed: {e}")
            return HandeyeResult(False, False, f"deploy failed, rolled back: {e}", b.summary(), {}, rb,
                                 consumer_lines(self.consumers, b, "deploy FAILED and rolled back", {}))
        published = {}
        if res.dry_run:
            state = f"dry run: would switch to {b.config_name} + {b.urdf_name} (commands recorded, nothing sent)"
        elif res.ok:
            state = (f"switched to {b.config_name} + {b.urdf_name}; its log names the config ({res.sha12}) and "
                     f"\"[Urdf] loaded '{b.urdf_container}'\"")
            published = self._publish()
        else:
            rb = self._auto_rollback(f"OKVIS did not load the new files: {res.detail}")
            state = f"NOT loaded ({res.detail}); rolled back"
        left = [list(x) for x in (self.inner.left or [])] or None
        detail = res.detail if rb is None else f"{res.detail}; rolled back automatically"
        return HandeyeResult(bool(res.ok), bool(res.dry_run), detail, b.summary(), res.as_dict(), rb,
                             consumer_lines(self.consumers, b, state, published), None if rb else left)

    def _auto_rollback(self, reason):
        try:
            return self.inner.rollback(reason).as_dict()
        except dp.RollbackDrift as e:
            return {"ok": False, "drift": str(e)}

    def rollback(self, reason="the operator asked for it (R)", force=False, left=None):
        """R: robot.env as saved, OKVIS restarted, the old config's sha checked in its log (dp.Deployer.rollback);
        left: what the deploy left (HandeyeResult.left) when this is another process than the deploy's"""
        if left is not None and self.inner.left is None:
            self.inner.left = [tuple(x) for x in left]
        return self.inner.rollback(reason, force=force)

    def _publish(self) -> dict:
        out = {}
        if self.publish_dir is None:
            return out
        for c in self.consumers:
            if c.kind == "offline_file":
                p = Path(self.publish_dir) / self.build.urdf_name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(self.build.urdf_text)
                out[c.name] = str(p)
        return out


# ================================================================ the head tool's coupling


def stale(config_text, deployed: dict) -> list:
    """why the hand-eye deployed as `deployed` (a `# handeye:` record) does not fit this config: [] when it does"""
    out = []
    try:
        if geom_sha(t_bs(config_text)) != deployed.get("T_BS"):
            out.append("T_BS changed")
        if geom_sha(t_sc(config_text)) != deployed.get("T_SC"):
            out.append("camera T_SC changed")
    except ValueError as e:
        out.append(f"the config cannot be read ({e})")
    u = dma_urdf(config_text)
    if u != deployed.get("urdf"):
        out.append(f"its urdf: is {u}, not the hand-eye URDF {deployed.get('urdf')}")
    return out


def stale_line(new_text, live_text=None) -> str:
    """the head tool's deploy report line: "" unless a hand-eye is deployed (the new config's `# handeye:` record, else
    the live config's) and this config no longer fits it"""
    mk = marker(new_text) or marker(live_text)
    if not mk:
        return ""
    why = stale(new_text, mk)
    if not why:
        return ""
    return (f"hand-eye STALE after this deploy: {'; '.join(why)} (deployed {mk.get('date')}, fit {mk.get('fit')}); "
            "run the hand-eye calibration again")


# ================================================================ the post-check (pure)


def wrist_errors(obs, urdf, T_BS, T_BS_records) -> np.ndarray:
    """|offset| (mm) of each capture under the geometry (urdf, T_BS); the captures' own T_BC was T_BS_records T_SC"""
    from armcal import urdffit
    from armcal.urdf import Urdf
    move = np.asarray(T_BS, float) @ np.linalg.inv(np.asarray(T_BS_records, float))
    obs = [replace(o, T_BC=move @ o.T_BC) for o in obs]
    return np.linalg.norm(urdffit.offsets(obs, Urdf.of(urdf)), axis=1) if obs else np.zeros(0)


def postcheck_scores(records, old, new, T_BS_records) -> dict:
    """fresh read-B records (readb outputs, either arm) scored with the old and the new geometry, paired: old / new =
    (URDF, T_BS); T_BS_records: the T_BS the bursts were taken with (the burst stores it)"""
    from armcal import urdffit
    sides, poses, o_mm, n_mm = [], [], [], []
    for r in records:
        obs = urdffit.observations(r["res"], r["side"])
        o_mm.append(wrist_errors(obs, old[0], old[1], T_BS_records))
        n_mm.append(wrist_errors(obs, new[0], new[1], T_BS_records))
        sides += [o.side for o in obs]
        poses += [o.pose for o in obs]
    cat = lambda a: np.concatenate(a) if a else np.zeros(0)  # noqa: E731
    return {"old_mm": cat(o_mm), "new_mm": cat(n_mm), "side": np.array(sides), "pose": np.array(poses)}


@dataclass
class PostVerdict:
    decision: str                   # PASS | GATE_MISSED | NOT_BETTER | WORSE | NOT_ENOUGH_DATA
    passed: bool
    rollback: bool                  # advised: only when new is shown WORSE than old (as the head tool's post-check)
    reasons: list
    stats: dict

    def as_dict(self) -> dict:
        return asdict(self)


def postcheck_verdict(old_mm, new_mm, side=None, gate_mm=GATE_MM, alpha=ALPHA, min_n=MIN_POSTCHECK) -> PostVerdict:
    """pass = new better than old, paired and robust (one-sided Wilcoxon signed-rank on old - new per capture, p <
    alpha, and the median paired gain > 0), AND the new median within gate_mm on every arm. No single-capture
    threshold: at ~4-5 mm noise per capture one bad capture must not decide. Roll back is advised only when new is
    shown worse than old by the same test; a missed gate or an inconclusive test is reported for the operator."""
    from scipy.stats import wilcoxon
    old, new = np.asarray(old_mm, float), np.asarray(new_mm, float)
    side = np.asarray(side if side is not None else ["all"] * len(old))
    if len(old) != len(new) or len(side) != len(old):
        raise ValueError("old, new and side must pair capture by capture")
    ok = np.isfinite(old) & np.isfinite(new)
    old, new, side = old[ok], new[ok], side[ok]
    n = len(old)
    if n < min_n:
        return PostVerdict("NOT_ENOUGH_DATA", False, False, [f"{n} paired captures; need {min_n}"], {"n": n})
    d = old - new
    nz = d[d != 0]
    p_better = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) else 1.0
    p_worse = float(wilcoxon(nz, alternative="less").pvalue) if len(nz) else 1.0
    per = {s: {"n": int((side == s).sum()), "old_median": float(np.median(old[side == s])),
               "new_median": float(np.median(new[side == s]))} for s in sorted(set(side.tolist()))}
    stats = {"n": n, "median_gain_mm": float(np.median(d)), "better": int((d > 0).sum()), "p_better": p_better,
             "p_worse": p_worse, "old_median": float(np.median(old)), "new_median": float(np.median(new)),
             "per_side": per, "gate_mm": gate_mm}
    line = (f"{n} captures: wrist error median new {stats['new_median']:.1f} mm vs old {stats['old_median']:.1f} mm, "
            f"median gain {stats['median_gain_mm']:+.1f} mm, {stats['better']}/{n} better (Wilcoxon p {p_better:.3g})")
    better = stats["median_gain_mm"] > 0 and p_better < alpha
    misses = [f"{s} median {v['new_median']:.1f} mm > {gate_mm:g} mm" for s, v in per.items()
              if v["new_median"] > gate_mm]
    if better and not misses:
        return PostVerdict("PASS", True, False, [line, f"every arm within {gate_mm:g} mm"], stats)
    if better:
        return PostVerdict("GATE_MISSED", False, False, [line, "better than old, but " + "; ".join(misses)], stats)
    if stats["median_gain_mm"] < 0 and p_worse < alpha:
        return PostVerdict("WORSE", False, True, [line, f"new shown WORSE than old (p {p_worse:.3g}): roll back"],
                           stats)
    return PostVerdict("NOT_BETTER", False, False, [line, f"new not shown better than old (p >= {alpha:g})"], stats)


# ================================================================ CLI


def ssh_runner(host, run=None, default_timeout=60):
    """deploy.Remote's runner over ssh (the deploy run from a laptop / Moonstone with the whole repo, armcal included:
    the calib kit on the robot carries calibtool, scripts and config only); stdin passes through (Remote.write)"""
    import shlex
    import subprocess
    opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5"]

    def go(argv, input=None, timeout=None):
        cmd = ["ssh", *opts, host, shlex.join([str(a) for a in argv])]
        if run is not None:
            return run(cmd, input=input, timeout=timeout)
        try:
            r = subprocess.run(cmd, input=input, capture_output=True, text=True, timeout=timeout or default_timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 127, f"{type(e).__name__}: {e}"
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    return go



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m calibtool.handeye_deploy", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry", help="geometry.json from a session's bursts")
    g.add_argument("bursts", nargs="+")
    g.add_argument("--robot", required=True, help="the bot name (robots.yaml), e.g. P10")
    g.add_argument("--session", required=True)
    g.add_argument("--out", required=True)
    d = sub.add_parser("deploy", help="deploy a fit folder (on the robot, or from here with --host)")
    d.add_argument("--host", default=None, help="root@phantomNN: run the robot commands over ssh")
    d.add_argument("--fit", required=True, help="the fit folder (fit.json, geometry.json, readb-*.json)")
    d.add_argument("--deploy-dir", required=True, help="on the robot: backups, rollback.sh, the result")
    d.add_argument("--model-only", action="store_true", help="T_BS kept; the URDF only (default model M12)")
    d.add_argument("--model", default=None)
    d.add_argument("--session", default="")
    d.add_argument("--publish-dir", default=None, help="where the offline consumers' URDF goes")
    d.add_argument("--dry-run", action="store_true", help="record every robot command, send none")
    d.add_argument("--no-wait-key", action="store_true", help="do not offer R (roll back) after the deploy")
    a = ap.parse_args(argv)
    if a.cmd == "geometry":
        geo = geometry_from_bursts(a.bursts, a.robot, a.session)
        Path(a.out).write_text(json.dumps(geo.as_dict(), indent=1))
        print(f"wrote {a.out}: {geo.okvis_config} ({geo.okvis_config_sha256[:12]})")
        return 0
    fit = HandeyeFit.load(a.fit)
    run = ssh_runner(a.host) if a.host else dp.shell_runner()
    remote = dry_run_remote(run) if a.dry_run else dp.Remote(run)
    snap = take_snapshot(remote, fit)
    hd = HandeyeDeployer(remote, fit, snap, a.deploy_dir, "model_only" if a.model_only else "head", a.model,
                         a.session, publish_dir=a.publish_dir)
    try:
        res = hd.deploy()
    except (dp.DeployRefused, dp.DeployDrift) as e:
        print(f"NOT DEPLOYED: {e}")
        return 1
    print(res.report())
    if a.dry_run:
        print("\n# the commands a deploy would send:\n" + remote.text())
        return 0
    remote.write(f"{a.deploy_dir}/handeye_deploy.json", json.dumps(res.as_dict(), indent=1, default=str))
    if res.ok and not a.no_wait_key and sys.stdin.isatty():
        if input("R + Enter rolls back, Enter keeps it: ").strip().upper() == "R":
            rb = hd.rollback()
            print(f"rolled back: {rb.ok} {rb.notes}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
