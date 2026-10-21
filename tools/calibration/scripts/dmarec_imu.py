"""One IMU from DMA.streams' dma-recorder segments (zstd MCAP, protobuf /raw_imu, all IMUs by
index at 500 Hz), e.g. the Torso during a neck take, and how far the torso moved between held
poses. Reading needs mcap and mcap-protobuf-support (both in calib-py:2); the other helpers need
only numpy.

The recorder writes one segment a minute (named in UTC) and keeps it open while writing; an open
segment has no MCAP footer yet (the seeking `mcap` reader raises on it), so a segment counts as
closed only once it ends with the MCAP magic. One that still has no footer minutes after its last
write was never finished (a recorder restart): it is read without its footer, as far as it goes."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

MCAP_MAGIC = b"\x89MCAP0\r\n"


def _closed(path: Path) -> bool:
    with open(path, "rb") as f:
        f.seek(0, 2)
        if f.tell() < 2 * len(MCAP_MAGIC):
            return False
        f.seek(-len(MCAP_MAGIC), 2)
        return f.read() == MCAP_MAGIC


def segments_covering(folder: Path, t0_ns: int, t1_ns: int, seg_s: float = 61.0,
                      unfinished_s: float = 120.0, now_s: float | None = None):
    """-> (closed, still_open, unfinished): segments whose minute overlaps [t0, t1]; unfinished =
    no footer and not written for unfinished_s. A segment's file mtime is when it was last written
    (its last stamp, within ~0.01 s on phantom12): keep it with copy2."""
    now_s = time.time() if now_s is None else now_s
    closed, still_open, unfinished = [], [], []
    for p in sorted(Path(folder).glob("recording_*.mcap")):
        end = p.stat().st_mtime
        if end >= t0_ns / 1e9 and end - seg_s <= t1_ns / 1e9:
            if _closed(p):
                closed.append(p)
            else:
                (unfinished if now_s - end > unfinished_s else still_open).append(p)
    return closed, still_open, unfinished


def copy_segments(paths, dest: Path) -> list:
    """Copy segments out of the recorder's folder (7-day cleanup); copy2 keeps the mtime that
    segments_covering relies on."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return [Path(shutil.copy2(p, dest / Path(p).name)) for p in paths]


def read_raw_imu(paths, index: int, t0_ns: int | None = None, t1_ns: int | None = None):
    """-> (t int64 ns, gyro N x 3, accel N x 3, errors) of IMU `index` from /raw_imu, stamped and
    windowed by the MCAP publish_time (= the frame's own stamp, host realtime, the clock the record
    app's streams use). A segment without its footer is read front to back as far as it goes. A
    segment that fails to read is named in `errors` as (path, message); the frames read from it
    before the failure are kept. Frames without IMU `index` are counted there."""
    from mcap.reader import NonSeekingReader, make_reader
    from mcap_protobuf.decoder import DecoderFactory
    t, w, a, errors = [], [], [], []
    for p in paths:
        missing = 0
        try:
            with open(p, "rb") as f:
                r = (make_reader(f, decoder_factories=[DecoderFactory()]) if _closed(Path(p))
                     else NonSeekingReader(f, decoder_factories=[DecoderFactory()]))
                # file order: the default log-time order reads everything before yielding, so an
                # unfinished segment's EndOfFile would lose all of it (the samples are sorted below)
                for _, _, msg, pb in r.iter_decoded_messages(topics=["/raw_imu"], log_time_order=False):
                    if (t0_ns is not None and msg.publish_time < t0_ns) or (t1_ns is not None and msg.publish_time > t1_ns):
                        continue
                    if index >= len(pb.imus):
                        missing += 1
                        continue
                    m = pb.imus[index]
                    t.append(int(msg.publish_time))
                    w.append((m.gyro_x, m.gyro_y, m.gyro_z))
                    a.append((m.accel_x, m.accel_y, m.accel_z))
        except Exception as e:  # a truncated or foreign file must not end the quick check
            errors.append((str(p), f"{type(e).__name__}: {e}"))
        if missing:
            errors.append((str(p), f"{missing} frames without IMU {index}"))
    order = np.argsort(np.array(t, np.int64))
    return (np.array(t, np.int64)[order], np.array(w, float).reshape(-1, 3)[order],
            np.array(a, float).reshape(-1, 3)[order], errors)


def torso_motion(t, gyro, accel, holds, min_samples: int = 50, window: int = 9,
                 step_deg: float = 0.05) -> list:
    """Per held pose, how far the torso was from where it stood for most of the take. The holds
    are split into groups wherever two consecutive holds differ by more than step_deg in yaw or in
    tilt, and the largest group is the reference (a step anywhere, even mid-take, then leaves the
    smaller side off; a median over all holds would put a mid-take step half on each side):
    tilt_deg  angle of the hold's mean accelerometer direction to the reference group's median
              direction (gravity: no drift; a translation of the torso is invisible to this check);
    yaw_deg   rotation about the vertical: the gyro, less a bias interpolated between the holds'
              robust local means (median over `window` holds), integrated over the whole take (so a
              turn during a hold carries into the holds after it). A turn spread evenly over more
              than about window / 2 holds is taken for bias drift. The window comes from phantom12's
              torso gyro: its 3 s means wander ~4e-5 rad/s over 45 s (15 Oct segments). A hold is
              read two ways and the larger counts: chained to the reference through the steps at
              the group boundaries (each measured between neighbouring holds, so the integrated
              yaw's slow drift over the take cannot shrink a step), plus its offset from its own
              group's median; and directly against the reference group's median (the chain alone
              takes a slide spread over several moves, split at only some of them, for in-group
              drift). A turn that falls in a gap of the stream is lost: time steps are capped.
              Tilt needs neither: the accelerometer does not drift;
    turn_deg  rotation during the hold, about any axis: how far its mean rate departs from that
              local bias, times its length (the gyro alone cannot tell a slow turn inside one hold
              from bias, but its neighbours can);
    yaw_sigma_deg  how far yaw_deg can be off from the bias estimate alone: the robust scatter of
              the holds' mean rates about the vertical around their local bias (rate_scatter_rad_s,
              the same for every hold), times the hold's time from the reference group's centre. On
              a still torso the scatter is the gyro's (~4e-5 rad/s); a robot rocking on its gantry
              while the head is moved by hand showed 2.6e-4 to 6.5e-4 (16 Oct takes), and then a
              yaw of a degree can be the estimate's, not the torso's;
    in_reference  whether the hold is in the reference group (a reference group under half the holds
              means the torso shifted: a step or slide, not wander).
    The bias comes from means, never a median of samples: the gyro is quantised (0.53 mrad/s
    steps on phantom12) and a median sits on that grid, up to half a step off, which integrates to
    a degree over a take. Time steps are capped at 3 sample periods, so a gap adds no noise (the
    caller reports gaps). None for a hold with fewer than min_samples samples."""
    t = np.asarray(t, np.int64)
    g, a = np.asarray(gyro, float), np.asarray(accel, float)
    masks = [(t >= h.t0) & (t <= h.t1) for h in holds]
    use = [k for k, m in enumerate(masks) if m.sum() >= min_samples]
    out = [None] * len(holds)
    if not use:
        return out
    ts = (t - t[0]) / 1e9
    centres = np.array([((holds[k].t0 + holds[k].t1) / 2 - t[0]) / 1e9 for k in use])
    means = np.array([g[masks[k]].mean(axis=0) for k in use])
    hw = window // 2
    local = np.array([np.median(means[max(0, i - hw):i + hw + 1], axis=0) for i in range(len(use))])
    bias = np.stack([np.interp(ts, centres, local[:, c]) for c in range(3)], axis=1)
    dirs = np.array([a[masks[k]].mean(axis=0) / np.linalg.norm(a[masks[k]].mean(axis=0)) for k in use])
    up = np.median(dirs, axis=0)
    up /= np.linalg.norm(up)
    dt = np.diff(t, prepend=t[0]) / 1e9
    if len(dt) > 1:
        dt = np.minimum(dt, 3.0 * np.median(dt[1:]))
    yaw = np.degrees(np.cumsum(((g - bias) @ up) * dt))
    yaws = np.array([yaw[masks[k]].mean() for k in use])
    ang = lambda u, v: float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v)))
    dev = (means - local) @ up
    scatter = 1.4826 * float(np.median(np.abs(dev - np.median(dev))))
    group = np.zeros(len(use), int)
    for i in range(1, len(use)):
        # a yaw step must also clear 3x the yaw noise between the two holds (a rocking robot made
        # neighbours differ by ~0.17 deg over 5 s on 16 Oct and split every pair)
        yaw_step = max(step_deg, 3 * np.degrees(scatter * (centres[i] - centres[i - 1])))
        stepped = abs(yaws[i] - yaws[i - 1]) > yaw_step or ang(dirs[i], dirs[i - 1]) > step_deg
        group[i] = group[i - 1] + int(stepped)
    ref_g = np.argmax(np.bincount(group))
    off = np.zeros(group[-1] + 1)          # each group's yaw, chained through its boundary steps
    for i in range(1, len(use)):
        if group[i] != group[i - 1]:
            off[group[i]] = off[group[i - 1]] + yaws[i] - yaws[i - 1]
    med = np.array([np.median(yaws[group == j]) for j in range(group[-1] + 1)])
    rel = off[group] - off[ref_g] + yaws - med[group]
    absolute = yaws - med[ref_g]           # a slide over several moves, split at only some of them
    rel = np.where(np.abs(absolute) > np.abs(rel), absolute, rel)
    up_ref = np.median(dirs[group == ref_g], axis=0)
    up_ref /= np.linalg.norm(up_ref)
    t_ref = float(np.median(centres[group == ref_g]))
    for i, k in enumerate(use):
        out[k] = {"tilt_deg": ang(dirs[i], up_ref),
                  "yaw_deg": float(rel[i]),
                  "turn_deg": float(np.degrees(np.linalg.norm(means[i] - local[i]) * (holds[k].t1 - holds[k].t0) / 1e9)),
                  "yaw_sigma_deg": float(np.degrees(scatter * abs(centres[i] - t_ref))),
                  "rate_scatter_rad_s": scatter, "in_reference": bool(group[i] == ref_g)}
    return out
