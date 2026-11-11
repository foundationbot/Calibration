"""The robot URDF in memory: joints, forward kinematics of any chain, origin changes, and the corrected file's writer.

    T_root,link(q) = O_1 · Rot(a_1, q_1) · O_2 · Rot(a_2, q_2) · …

O the joints' fixed <origin> transforms (URDF rpy: R = Rz(y) Ry(p) Rx(r)) and a their axes in the child frame, exactly as
scripts/neckfk.py (the FK the head calibration uses), so the chain Torso -> Head equals NeckChain.T_TB. Fixed joints
contribute their origin only. Angles in rad, keyed by joint name; raw encoders, the URDF's signs.

A change (the stage-2 fit's parameters, as OKVIS2-X's urdfx/urdfvar.py on phantom14) is {joint: {...}} with
  dxyz           mm (3): added to the origin's xyz (in the parent frame);
  xyz            m (3): the origin's xyz replaced;
  pre            deg (3, rotation vector): R_new = Rot(pre) · R_old (a rotation in the parent frame);
  post_axis_deg  deg: R_new = R_old · Rot(axis, deg), a joint zero offset (as adding deg to the encoder).
Urdf.changed(change) applies it in memory (no file round trip); write_variant() edits only those <origin> lines of the
base file's text, adds a provenance comment, and checks the written file's FK against the in-memory model."""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from armcal import SCRIPTS  # noqa: F401  (scripts/ on sys.path)
import neckfk  # noqa: E402
from rigmath import rot_axis, transform  # noqa: E402

CHANGE_KEYS = {"dxyz", "xyz", "pre", "post_axis_deg"}
WRITE_FK_TOL_M = 1e-7          # the written file's origins print 9 decimals: their FK agrees to ~1e-9 m


@dataclass(frozen=True)
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray            # 4x4: parent link -> joint frame at q = 0
    axis: np.ndarray | None       # unit, in the joint (child) frame; None for a fixed joint

    @property
    def moves(self) -> bool:
        return self.axis is not None


def _parse_joint(j) -> Joint:
    o, ax = j.find("origin"), j.find("axis")
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    r, p, y = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    axis = None
    if j.get("type") != "fixed":
        a = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
        if np.linalg.norm(a) < 1e-12:
            raise ValueError(f"joint {j.get('name')!r} has a zero axis")
        axis = a / np.linalg.norm(a)
    return Joint(j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
                 transform(neckfk.rpy(r, p, y), xyz), axis)


class Urdf:
    """the joints of one URDF (text kept for the writer); immutable: changed() returns a new one"""

    def __init__(self, text: str, path=None, joints: dict | None = None):
        self.text, self.path = text, (Path(path) if path is not None else None)
        if joints is None:
            joints = {}
            for j in ET.fromstring(text.encode()).iter("joint"):
                jt = _parse_joint(j)
                joints[jt.name] = jt
        self.joints = joints
        self._by_child = {j.child: j for j in joints.values()}
        self._chains: dict = {}

    @classmethod
    def load(cls, path) -> "Urdf":
        return cls(Path(path).read_text(), path)

    @staticmethod
    def of(urdf) -> "Urdf":
        """a Urdf from a Urdf or a path"""
        return urdf if isinstance(urdf, Urdf) else Urdf.load(urdf)

    @property
    def md5(self) -> str:
        return hashlib.md5(self.text.encode()).hexdigest()

    def joint(self, name: str) -> Joint:
        if name not in self.joints:
            raise KeyError(f"joint {name!r} not in {self.path or 'the URDF'}")
        return self.joints[name]

    def chain(self, root: str, link: str) -> "Chain":
        key = (root, link)
        if key not in self._chains:
            self._chains[key] = Chain(self, root, link)
        return self._chains[key]

    def changed(self, change: dict) -> "Urdf":
        """a new Urdf with the change applied to its joints' origins (see the module doc); the text is the base's"""
        joints = dict(self.joints)
        for name, c in change.items():
            bad = set(c) - CHANGE_KEYS
            if bad:
                raise ValueError(f"{name}: unknown change keys {sorted(bad)} (known: {sorted(CHANGE_KEYS)})")
            j = self.joint(name)
            T = j.origin.copy()
            R, xyz = T[:3, :3], T[:3, 3].copy()
            if "pre" in c:
                R = _rotvec(np.radians(np.asarray(c["pre"], float))) @ R
            if "post_axis_deg" in c:
                if j.axis is None:
                    raise ValueError(f"{name} is fixed: it has no zero offset")
                R = R @ rot_axis(j.axis, np.radians(float(c["post_axis_deg"])))
            if "dxyz" in c:
                xyz = xyz + 1e-3 * np.asarray(c["dxyz"], float)
            if "xyz" in c:
                xyz = np.asarray(c["xyz"], float)
            joints[name] = replace(j, origin=transform(R, xyz))
        return Urdf(self.text, self.path, joints)


def _rotvec(w) -> np.ndarray:
    a = float(np.linalg.norm(w))
    return np.eye(3) if a < 1e-15 else rot_axis(np.asarray(w) / a, a)


class Chain:
    """the joints from link `root` down to link `link`; T(angles) -> 4x4 mapping `link` coordinates into `root`'s"""

    def __init__(self, urdf: Urdf, root: str, link: str):
        steps, at = [], link
        while at != root:
            j = urdf._by_child.get(at)
            if j is None:
                raise ValueError(f"link {link!r} is not below {root!r} in {urdf.path or 'the URDF'}")
            steps.append(j)
            at = j.parent
        self.steps = steps[::-1]
        self.joint_names = [j.name for j in self.steps if j.moves]

    def T(self, angles: dict) -> np.ndarray:
        T = np.eye(4)
        for j in self.steps:
            T = T @ j.origin
            if j.moves:
                if j.name not in angles:
                    raise KeyError(f"no angle for joint {j.name!r}")
                T = T @ transform(rot_axis(j.axis, angles[j.name]), np.zeros(3))
        return T


# ---------------------------------------------------------------------------------------------------------- writer
def rpy_of(R) -> np.ndarray:
    """URDF (r, p, y) with neckfk.rpy(r, p, y) == R"""
    from scipy.spatial.transform import Rotation
    e = Rotation.from_matrix(np.asarray(R, float)).as_euler("xyz")
    if not np.allclose(neckfk.rpy(*e), R, atol=1e-12):
        raise ValueError("rpy round trip failed")
    return e


def _origin_line_edit(text: str, j: Joint) -> str:
    m = re.search(r"(<joint\s+name=\"%s\"[^>]*>\s*<origin\s)([^/]*)(/>)" % re.escape(j.name), text)
    if not m:
        raise ValueError(f"joint {j.name!r}: no <origin .../> right after its <joint> tag to edit")
    xyz, rpy = j.origin[:3, 3], rpy_of(j.origin[:3, :3])
    new = "xyz=\"%.9f %.9f %.9f\" rpy=\"%.9f %.9f %.9f\" " % (*xyz, *rpy)
    return text[:m.start(2)] + new + text[m.end(2):]


def _comment(lines) -> str:
    body = "\n".join("     " + str(s).replace("--", "- -") for s in lines)
    return "<!-- hand-eye URDF (tools/calibration/armcal)\n" + body + " -->\n"


def write_variant(base: Urdf, change: dict, out, provenance: list[str]) -> Path:
    """the base file's text with the changed joints' <origin> rewritten and a provenance comment after the XML
    declaration; refuses to write a file whose own FK differs from the in-memory changed model"""
    text = variant_text(base, change, provenance)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def variant_text(base: Urdf, change: dict, provenance: list[str]) -> str:
    """write_variant's text, not written anywhere (the deploy sends it to the robot)"""
    if base.text is None:
        raise ValueError("the base URDF has no text to edit")
    new = base.changed(change)
    text = base.text
    for name in change:
        text = _origin_line_edit(text, new.joint(name))
    head = _comment([f"base: {base.path} (md5 {base.md5})",
                     f"changed origins: {', '.join(change)}"] + list(provenance))
    if text.lstrip().startswith("<?xml"):
        k = text.index("?>") + 2
        text = text[:k] + "\n" + head + text[k:].lstrip("\n")
    else:
        text = head + text
    written = Urdf(text)
    for name in base.joints:
        d = np.abs(written.joint(name).origin - new.joint(name).origin).max()
        if d > WRITE_FK_TOL_M:
            raise ValueError(f"the written {name} origin differs from the fit by {d:.2e}")
    return text
