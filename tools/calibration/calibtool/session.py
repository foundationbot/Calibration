"""The Session data model: the per-take corner files of one calibration session (step 15b's format) as one pairs table
and one corners table, in the session's take order.

Hold ids are global, made the way step 16 makes them: each take's pose indices (img_hold) offset by the running sum of
(max pose index + 1) over the takes before it, in take order. So a hold id of 16's split output names the same hold
here, and any pair of a step-16/45/66 file joins back to its take, pose and neck angles by img_t0 (index_of_t0).

pairs (dict of arrays, one row per stereo pair, in take order then file order):
  take (str), take_idx, hold (global), pose (img_hold within the take), t0 / t1 (left / right stamp, ns),
  neck_pitch_rad / neck_yaw_rad (measured, img_pitch / img_yaw; nan for a file without them),
  approach_pitch / approach_yaw (img_ap / img_ay, -1/0/+1; 0 without them), n_cam0 / n_cam1 (corners per camera)
corners (dict of arrays, one row per detected corner): pair (row of pairs), cam (0 = left = OKVIS cam0), tag, k, u, v
"""
from dataclasses import dataclass

import numpy as np

PAIR_IN = {"img_hold": "pose", "img_t0": "t0", "img_t1": "t1", "img_yaw": "neck_yaw_rad",
           "img_pitch": "neck_pitch_rad", "img_ay": "approach_yaw", "img_ap": "approach_pitch"}
DET_IN = {"det_img": "pair", "det_cam": "cam", "det_tag": "tag", "det_k": "k", "det_u": "u", "det_v": "v"}
INT_KEYS = {"img_hold", "img_t0", "img_t1", "img_ay", "img_ap", "det_img", "det_cam", "det_tag", "det_k"}
OPTIONAL = {"img_yaw": np.nan, "img_pitch": np.nan, "img_ay": 0, "img_ap": 0}     # step-15 files lack the neck keys


@dataclass(frozen=True)
class Hold:
    """one still hold (one neck pose of one take): its global id, take, pose index, pair rows, first left stamp"""
    hold: int
    take: str
    pose: int
    pairs: np.ndarray
    t0: int


class Session:
    def __init__(self, name, take_order, pairs, corners, hold_offset, has_neck):
        self.name = name
        self.take_order = list(take_order)
        self.pairs = pairs
        self.corners = corners
        self.hold_offset = hold_offset
        self.has_neck = has_neck          # {take: the corner file carried img_yaw/img_pitch}
        self._t0_index = {int(t): i for i, t in enumerate(pairs["t0"])}

    @classmethod
    def load(cls, stored):
        """a registry entry (sessions.StoredSession): its corner files in its take order"""
        return cls.from_files(list(zip(stored.take_order, stored.corner_files)), name=stored.name)

    @classmethod
    def from_files(cls, files, name=""):
        """[(take name, corner npz path or dict)] in take order"""
        cols = {v: [] for v in list(PAIR_IN.values()) + ["take", "take_idx", "hold", "n_cam0", "n_cam1"]}
        dets = {v: [] for v in DET_IN.values()}
        offsets, has_neck, n, offset = {}, {}, 0, 0
        for ti, (take, f) in enumerate(files):
            z = dict(np.load(f)) if not isinstance(f, dict) else f
            m = len(z["img_t0"])
            has_neck[take] = "img_yaw" in z and "img_pitch" in z
            for k, v in PAIR_IN.items():
                col = np.asarray(z[k]) if k in z else np.full(m, OPTIONAL[k])
                cols[v].append(col.astype(np.int64 if k in INT_KEYS else float))
            pose = np.asarray(z["img_hold"], np.int64)
            offsets[take] = offset
            cols["hold"].append(pose + offset)
            cols["take"].append(np.full(m, take, dtype=object))
            cols["take_idx"].append(np.full(m, ti, np.int64))
            img, cam = np.asarray(z["det_img"], np.int64), np.asarray(z["det_cam"], np.int64)
            cols["n_cam0"].append(np.bincount(img[cam == 0], minlength=m)[:m])
            cols["n_cam1"].append(np.bincount(img[cam == 1], minlength=m)[:m])
            for k, v in DET_IN.items():
                col = np.asarray(z[k], np.int64 if k in INT_KEYS else float)
                dets[v].append(col + n if k == "det_img" else col)
            offset += int(pose.max()) + 1 if m else 0          # step 16's rule
            n += m
        pairs = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
        pairs["take"] = pairs["take"].astype(str)
        corners = {k: np.concatenate(v) if v else np.zeros(0) for k, v in dets.items()}
        t0 = pairs["t0"]
        if len(np.unique(t0)) != len(t0):
            raise ValueError(f"session {name!r}: duplicate left stamps (img_t0) across its corner files: a take "
                             "listed twice, or two takes of the same recording")
        return cls(name, [t for t, _ in files], pairs, corners, offsets, has_neck)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs["t0"])

    def holds(self):
        """the holds in capture order (first left stamp), each once"""
        hold, t0 = self.pairs["hold"], self.pairs["t0"]
        ids = np.unique(hold)
        first = {int(h): int(t0[hold == h].min()) for h in ids}
        for h in sorted(first, key=lambda h: (first[h], h)):
            idx = np.flatnonzero(hold == h)
            yield Hold(hold=h, take=str(self.pairs["take"][idx[0]]), pose=int(self.pairs["pose"][idx[0]]), pairs=idx,
                       t0=first[h])

    def to_npz_dict(self, mask=None, hold="global"):
        """the pairs in mask (bool per pair; None: all) as a step-15b corner dict, pairs renumbered from 0 in table
        order: what step 16 (saved as npz), stereofit.Data, 45 and 66 read. hold="global": img_hold = the global hold
        id; "pose": the pose index within the take (a single take written this way is its corner file again)"""
        if hold not in ("global", "pose"):
            raise ValueError(f"hold={hold!r}: 'global' or 'pose'")
        keep = np.ones(self.n_pairs, bool) if mask is None else np.asarray(mask, bool)
        idx = np.flatnonzero(keep)
        remap = np.full(self.n_pairs, -1, np.int64)
        remap[idx] = np.arange(len(idx))
        out = {}
        for k, v in PAIR_IN.items():
            col = self.pairs["hold"] if (k == "img_hold" and hold == "global") else self.pairs[v]
            out[k] = col[idx].astype(np.int64 if k in INT_KEYS else float)
        d = keep[self.corners["pair"]]
        for k, v in DET_IN.items():
            col = self.corners[v][d]
            out[k] = remap[col] if k == "det_img" else col
        return out

    def index_of_t0(self, t0):
        """pair rows of left stamps t0 (KeyError naming the first stamp not in the session)"""
        try:
            return np.array([self._t0_index[int(t)] for t in np.atleast_1d(t0)], np.int64)
        except KeyError as e:
            raise KeyError(f"left stamp {e.args[0]} is not a pair of session {self.name!r}") from None
