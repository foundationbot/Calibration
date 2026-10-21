#!/usr/bin/env python3
"""Validate mk13_stereo.yaml's T_SC pair and T_BS against PHYSICS, not against config.

Four checks, in order of how much they can prove. Nothing here needs CAD, a URDF
camera link, or the calibration bags -- only the yaml, one live static IMU sample
and one live stereo frame pair.

  1. SELF-CONSISTENCY (offline).   The T_SC pair implies a relative stereo pose.
     Its baseline length and its rotation-axis decomposition must match what the
     stereo calibration reported. This is the part already covered by the
     held-out epipolar |dy|, so it should always pass; it is here to catch a
     hand-edit, a bad paste, or a convention flip (T_SC vs T_CS).

  2. GRAVITY vs THE SCENE (live, passive).   A static accelerometer measures the
     vertical in the IMU frame. R_SC maps that into the camera frame. Independently,
     triangulating the raw stereo pair and fitting planes recovers the vertical from
     the horizontal surfaces the cameras can see. The angle between the two is the
     PITCH/ROLL error of R_SC, measured absolutely. Uses the RAW images and the
     relative pose (the validated part), so there is no rectification frame to guess.

  3. BASELINE LEVEL (live, passive).   The stereo baseline is a physical near-
     horizontal bar. Its angle to the measured gravity must be ~90 deg. Independent
     of check 2 and sensitive to a different combination of R_SC's DOF.

  4. HEAD-FRAME PLAUSIBILITY (offline).   T_BC = T_BS @ T_SC is what a simulator,
     a URDF export or the planner reads. Two rigidly mounted cameras in a head MUST
     come out: baseline ~100% along the head's lateral axis, symmetric about the
     head centreline, no forward stagger, optical axes near dead ahead. Each of
     those is independent, so a single rotation that repairs all of them at once
     localises the fault to T_BS.

WHAT NONE OF THESE CAN DO. Gravity is blind to rotation ABOUT gravity, so a common
yaw error in both T_SC entries is indistinguishable from a yaw error in T_BS. Run
the script over two INDEPENDENT calibrations (--compare) to break the tie: a defect
that survives a recalibration lives in the factor they share, i.e. T_BS. The direct
measurement is check 5 in the docstring of measure_T_BS_from_neck_motion() below.

Host, IMU name/index, robot paths and the body-frame axes come from
config/rig.yaml; the lens model of each camera from the yaml's distortion_type.

Usage:
    python3 80_validate_extrinsics.py CALIB.yaml [--host H] [--compare OLD.yaml] [--rig RIG]
    python3 80_validate_extrinsics.py CALIB.yaml --offline        # checks 1 and 4 only
"""
import argparse, re, subprocess, sys
import numpy as np

import rigcfg

np.set_printoptions(precision=4, suppress=True)

# The Head/body frame convention (mk13: read off 20250206_phantom_mk1.urdf —
# NeckYaw axis (0,0,1), NeckPitch axis (0,1,0), LeftShoulderPitch at +y, so
# x forward, y LEFT, z UP). Set from rig.yaml robot.body_axes in main().
HEAD_X_FWD, HEAD_Y_LEFT, HEAD_Z_UP = 0, 1, 2
RIG = None


# --------------------------------------------------------------------------- yaml
def load_calib(path):
    """Pull the two T_SC blocks, T_BS and the intrinsics out of an OpenCV-YAML rig
    file. Hand-parsed because this file is %YAML:1.0 with OpenCV extensions that
    PyYAML refuses, and because we want to read exactly the bytes OKVIS reads."""
    text = open(path).read()
    nums = lambda s: [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)]
    tsc = [np.array(nums(m)[:16]).reshape(4, 4)
           for m in re.findall(r"T_SC:\s*\[([^\]]+)\]", text, re.S)]
    assert len(tsc) == 2, f"{path}: expected 2 T_SC blocks, found {len(tsc)}"
    tbs = re.search(r"T_BS:\s*\[([^\]]+)\]", text, re.S)
    assert tbs, f"{path}: no T_BS"
    K = [np.array(nums(m)[:2]) for m in re.findall(r"focal_length: \[([^\]]+)\]", text)]
    C = [np.array(nums(m)[:2]) for m in re.findall(r"principal_point: \[([^\]]+)\]", text)]
    Q = [int(x) for x in re.findall(r"dma_queue: (\d+)", text)]
    L = [rigcfg.Lens(x) for x in re.findall(r"distortion_type: (\w+)", text)]
    if len(L) != 2:
        L = [RIG.lens, RIG.lens]
    # a kb4t block (step 71 --okvis-kb4t) carries six numbers, k1..k4, p1, p2; the others four
    D = [np.array(nums(m)[:6 if lens.kind == "kb4t" else 4])
         for m, lens in zip(re.findall(r"distortion_coefficients: \[([^\]]+)\]", text), L)]
    return dict(T_SC=tsc, T_BS=np.array(nums(tbs.group(1))[:16]).reshape(4, 4),
                f=K, c=C, d=D, queue=Q, lens=L)


def axis_angle(R):
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    S = (R - R.T) / 2
    v = np.array([S[2, 1], S[0, 2], S[1, 0]])
    n = np.linalg.norm(v)
    return ang, (v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0]))


# ------------------------------------------------------------------ check 1
def check_self_consistency(cal):
    print("=" * 78)
    print("CHECK 1  T_SC pair self-consistency  (offline)")
    print("=" * 78)
    ok = True
    for i, T in enumerate(cal["T_SC"]):
        R = T[:3, :3]
        orth = np.abs(R @ R.T - np.eye(3)).max()
        det = np.linalg.det(R)
        flag = "" if (orth < 1e-9 and abs(det - 1) < 1e-9) else "   <-- NOT A ROTATION"
        ok &= not flag
        print(f"  cam{i}: det(R)={det:.9f}  max|RR^T-I|={orth:.2e}  "
              f"|t_SC|={1000*np.linalg.norm(T[:3,3]):7.1f} mm{flag}")
    T_rel = np.linalg.inv(cal["T_SC"][0]) @ cal["T_SC"][1]
    ang, ax = axis_angle(T_rel[:3, :3])
    z1 = T_rel[:3, :3] @ np.array([0, 0, 1.0])
    verg = np.degrees(np.arccos(np.clip(z1[2], -1, 1)))
    print(f"\n  relative stereo pose implied by the pair:")
    print(f"    baseline            {1000*np.linalg.norm(T_rel[:3,3]):.2f} mm   "
          f"direction in cam0 {T_rel[:3,3]/np.linalg.norm(T_rel[:3,3])}")
    print(f"    rotation            {ang:.4f} deg about {ax}")
    print(f"      -> {100*abs(ax[2]):.1f}% of it is ROLL about the optical axis")
    print(f"      -> true vergence between the optical axes is only {verg:.2f} deg")
    print("    NOTE: quoting the total rotation without decomposing the axis reads as a")
    print("          splay and implies far less image overlap than the rig actually has.")
    return ok


# ------------------------------------------------------------------ live capture
def read_gravity(host, samples_secs=8):
    """Static gravity in the IMU frame, via the sanctioned reader on the robot.

    Deliberately shells out to deployment/display_imu.py rather than decoding
    /raw_imu_actuals here: that ring is a versioned dma_common layout and
    hand-decoding it has produced plausible garbage before."""
    imu = RIG.imu
    cmd = (f"cd {RIG.robot['display_imu_dir']} && "
           f"timeout {samples_secs} python3 display_imu.py --rate 4 2>&1")
    out = subprocess.run(["ssh", host, cmd], capture_output=True, text=True).stdout
    blk = re.findall(
        r"IMU\[" + str(int(imu["index"])) + r"\] \(" + re.escape(imu["name"]) + r"\):\s*\n"
        r"\s*Acceleration \(m/s.\):\s*X=\s*(-?[\d.]+)\s*"
        r"Y=\s*(-?[\d.]+)\s*Z=\s*(-?[\d.]+)\s*\n\s*Angular Velocity \(rad/s\):\s*"
        r"X=\s*(-?[\d.]+)\s*Y=\s*(-?[\d.]+)\s*Z=\s*(-?[\d.]+)", out)
    if not blk:
        raise SystemExit(f"could not parse IMU[{imu['index']}] ({imu['name']}) samples "
                         "from display_imu.py:\n" + out[:1500])
    A = np.array([[float(v) for v in b[:3]] for b in blk])
    G = np.array([[float(v) for v in b[3:]] for b in blk])
    a, w = A.mean(0), np.linalg.norm(G.mean(0))
    print(f"  {imu['name']} IMU, {len(A)} samples: accel {a}  |a|={np.linalg.norm(a):.4f} m/s^2")
    print(f"    |omega| = {w:.5f} rad/s", end="")
    if w > 0.02:
        raise SystemExit("\n  ROBOT IS MOVING -- this test needs a static robot.")
    print("  (static: OK)")
    # An accelerometer at rest reads the reaction to gravity, i.e. it points UP.
    return a / np.linalg.norm(a)


def read_frames(host, queues):
    sys.path.insert(0, RIG.robot["vio_tuner_tools"])
    from vio_tuner.videoqueue import CameraQueueReader, decode_bgr
    out = {}
    for q in queues:
        f = CameraQueueReader(host, q).read()
        img = decode_bgr(f)
        if img is None:
            raise SystemExit(f"queue {q}: undecodable format {f.format_name}")
        print(f"  queue {q}: {f.width}x{f.height} {f.format_name} seq={f.frame_sequence}")
        out[q] = img
    return out


# ------------------------------------------------------------------ checks 2 + 3
def check_gravity(cal, up_S, frames):
    import cv2
    print("\n" + "=" * 78)
    print("CHECK 2/3  R_SC vs gravity, and baseline level  (live, passive)")
    print("=" * 78)
    T0, T1 = cal["T_SC"]
    K = [np.array([[cal["f"][i][0], 0, cal["c"][i][0]],
                   [0, cal["f"][i][1], cal["c"][i][1]], [0, 0, 1]]) for i in (0, 1)]
    T_rel = np.linalg.inv(T0) @ T1
    R01, t01 = T_rel[:3, :3], T_rel[:3, 3]
    up_C0 = T0[:3, :3].T @ up_S
    print(f"  gravity('up') in IMU frame     = {up_S}")
    print(f"  the same, mapped into cam0 by R_SC = {up_C0}")
    print(f"    => cam0 optical axis is {np.degrees(np.arcsin(np.clip(-up_C0[2],-1,1))):+.2f} "
          f"deg DOWN from horizontal")

    g0 = cv2.cvtColor(frames[cal["queue"][0]], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frames[cal["queue"][1]], cv2.COLOR_BGR2GRAY)
    # SIFT, not ORB/BRISK: the pair carries a ~21 deg relative roll and only a
    # rotation-normalising detector matches across it.
    sift = cv2.SIFT_create(nfeatures=20000)
    k0, f0 = sift.detectAndCompute(g0, None)
    k1, f1 = sift.detectAndCompute(g1, None)
    good = [m for m, n in cv2.BFMatcher().knnMatch(f0, f1, k=2) if m.distance < 0.75 * n.distance]
    print(f"  SIFT {len(k0)}/{len(k1)} kp -> {len(good)} matches")
    p0 = np.array([k0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    p1 = np.array([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    n0 = cal["lens"][0].undistort(p0, K[0], cal["d"][0]).reshape(-1, 2)
    n1 = cal["lens"][1].undistort(p1, K[1], cal["d"][1]).reshape(-1, 2)
    r0 = np.hstack([n0, np.ones((len(n0), 1))]); r0 /= np.linalg.norm(r0, axis=1, keepdims=True)
    r1 = np.hstack([n1, np.ones((len(n1), 1))]); r1 /= np.linalg.norm(r1, axis=1, keepdims=True)
    r1c = (R01 @ r1.T).T
    a = np.einsum('ij,ij->i', r0, r0); b = np.einsum('ij,ij->i', r0, r1c)
    c = np.einsum('ij,ij->i', r1c, r1c)
    d0 = r0 @ t01; d1 = r1c @ t01
    den = a * c - b * b; ok = np.abs(den) > 1e-9
    s = np.where(ok, (c * d0 - b * d1) / np.where(ok, den, 1), 0)
    u = np.where(ok, (b * d0 - a * d1) / np.where(ok, den, 1), 0)
    P0, P1 = r0 * s[:, None], t01 + r1c * u[:, None]
    gap = np.linalg.norm(P0 - P1, axis=1)
    X = (P0 + P1) / 2
    sel = ok & (s > 0) & (u > 0) & (gap < 0.02) & (X[:, 2] > 0.3) & (X[:, 2] < 8.0)
    X = X[sel]
    print(f"  triangulated {len(X)} points, depth p5/50/95 = "
          f"{np.percentile(X[:,2],[5,50,95]).round(2)} m, ray gap median "
          f"{1000*np.median(gap[sel]):.1f} mm")

    rem = np.ones(len(X), bool); horiz = []
    print("\n   pts    rms    dist   normal in cam0                 angle to IMU gravity")
    for _ in range(6):
        Y = X[rem]
        if len(Y) < 40: break
        rng = np.random.default_rng(1); best = None
        for _ in range(4000):
            i = rng.choice(len(Y), 3, replace=False); A = Y[i]
            nv = np.cross(A[1] - A[0], A[2] - A[0]); nn = np.linalg.norm(nv)
            if nn < 1e-9: continue
            nv /= nn; dd = -nv @ A[0]
            inl = np.abs(Y @ nv + dd) < 0.015
            if best is None or inl.sum() > best[0]: best = (inl.sum(), inl)
        if best is None or best[0] < 40: break
        cnt, inl = best
        Q = Y[inl]; ctr = Q.mean(0)
        nv = np.linalg.svd(Q - ctr)[2][2]; dd = -nv @ ctr
        if nv @ up_C0 < 0: nv, dd = -nv, -dd
        rms = 1000 * np.sqrt((np.abs(Q @ nv + dd) ** 2).mean())
        ang = np.degrees(np.arccos(np.clip(nv @ up_C0, -1, 1)))
        tag = "  <- horizontal" if ang < 20 else ""
        if ang < 20: horiz.append((cnt, nv))
        print(f"  {cnt:4d}  {rms:5.1f}mm {abs(dd):5.2f}m  {nv}  {ang:6.2f} deg{tag}")
        rem[np.where(rem)[0][inl]] = False

    verdict = None
    if horiz:
        W = np.array([c for c, _ in horiz], float); N = np.array([n for _, n in horiz])
        mean = (N * W[:, None]).sum(0); mean /= np.linalg.norm(mean)
        spread = [np.degrees(np.arccos(np.clip(n @ mean, -1, 1))) for n in N]
        verdict = np.degrees(np.arccos(np.clip(mean @ up_C0, -1, 1)))
        print(f"\n  {len(horiz)} horizontal surfaces, mutual spread {np.mean(spread):.2f} deg "
              f"(max {np.max(spread):.2f}) <- noise floor of this test")
        print(f"  >>> R_SC pitch/roll error = {verdict:.2f} deg  "
              f"({'PASS' if verdict < 3*max(np.mean(spread),1.0) else 'SUSPECT'})")
    else:
        print("\n  no horizontal surface found -- point the cameras at a floor.")

    bh = t01 / np.linalg.norm(t01)
    lvl = 90 - np.degrees(np.arccos(abs(bh @ up_C0)))
    print(f"  >>> stereo baseline is {lvl:.2f} deg off level (physically must be ~0)")
    return verdict


# ------------------------------------------------------------------ check 4
def Rz(t):
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4); T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def head_geometry(T_BS, T_SC, corr=None):
    corr = np.eye(4) if corr is None else corr
    A, B = corr @ T_BS @ T_SC[0], corr @ T_BS @ T_SC[1]
    bl = B[:3, 3] - A[:3, 3]
    z = (A[:3, 2] + B[:3, 2]) / 2
    return dict(pos=(A[:3, 3], B[:3, 3]), baseline=bl,
                lateral=bl[HEAD_Y_LEFT] / np.linalg.norm(bl),
                centre=(A[HEAD_Y_LEFT, 3] + B[HEAD_Y_LEFT, 3]) / 2,
                stagger=B[HEAD_X_FWD, 3] - A[HEAD_X_FWD, 3],
                azimuth=np.degrees(np.arctan2(z[HEAD_Y_LEFT], z[HEAD_X_FWD])),
                elevation=np.degrees(np.arcsin(-z[HEAD_Z_UP])))


def check_head_frame(cal, label="", up_S=None):
    print("\n" + "=" * 78)
    print(f"CHECK 4  head-frame plausibility of T_BC = T_BS @ T_SC   {label}")
    print("=" * 78)
    T_BS, T_SC = cal["T_BS"], cal["T_SC"]
    if up_S is not None:
        up_H = T_BS[:3, :3] @ up_S
        tilt = np.degrees(np.arccos(np.clip(up_H[HEAD_Z_UP], -1, 1)))
        print(f"  T_BS vs gravity: head tilt {tilt:.2f} deg from vertical "
              f"(neck pitch limit is 28 deg) -- {'plausible' if tilt < 45 else 'IMPOSSIBLE'}")
        print("    NOTE this is insensitive to a yaw error: a rotation about the head's own")
        print("    vertical does not change the z-component of 'up'. Gravity cannot audit yaw.\n")

    def show(tag, corr):
        g = head_geometry(T_BS, T_SC, corr)
        print(f"  {tag}")
        print(f"    cam0 (fwd,left,up) = {g['pos'][0]}   cam1 = {g['pos'][1]}")
        print(f"    baseline           = {g['baseline']}   {100*abs(g['lateral']):.1f}% lateral"
              f"   [want ~100%]")
        print(f"    centre off head centreline = {1000*g['centre']:+7.1f} mm   [want ~0]")
        print(f"    forward stagger between the two cameras = {1000*g['stagger']:+7.1f} mm   [want ~0]")
        print(f"    optical axis: azimuth {g['azimuth']:+.1f} deg LEFT [want ~0], "
              f"elevation {g['elevation']:+.1f} deg DOWN")

    show("AS CALIBRATED -- this is what a simulator / URDF export reads:", None)
    th = np.linspace(-np.pi, np.pi, 72001)
    cost = []
    for t in th:
        g = head_geometry(T_BS, T_SC, Rz(t))
        cost.append((1 + g['lateral']) ** 2 * 4 + (g['centre'] / 0.02) ** 2 * 0.02
                    + (g['stagger'] / 0.02) ** 2 * 0.02 + (g['azimuth'] / 30.0) ** 2 * 0.1)
    best = th[int(np.argmin(cost))]
    print()
    show(f"WITH A {np.degrees(best):+.2f} deg YAW CORRECTION ON T_BS:", Rz(best))
    print(f"\n  >>> yaw correction needed: {np.degrees(best):+.2f} deg about the head vertical.")
    print("      Baseline direction, left/right symmetry and optical azimuth are three")
    print("      INDEPENDENT quantities; one angle repairing all three is not a coincidence.")
    return np.degrees(best)


def measure_T_BS_from_neck_motion():
    """CHECK 5 -- the DIRECT determination of R_BS. Needs neck motion, so it is
    documented rather than run: driving/motion is operator-authorised on this rig.

    The two neck joint axes are known EXACTLY from the URDF and are head-fixed
    directions, so the gyro can be used to read them off in the IMU frame:

      NeckPitch axis is (0,1,0) in the Head frame -- it is the rotation axis, so
      it is invariant to the pitch angle itself. Command a slow +/-0.2 rad pitch
      sweep, take the dominant gyro axis w_S (normalise the mean of w over the
      sweep, or the principal eigenvector of sum w w^T). Then
                          R_BS @ w_S  must equal  (0, 1, 0).

      NeckYaw axis is (0,0,1) in the NeckYawLink frame, which equals the Head
      frame only at pitch = 0. So park the pitch at 0, sweep yaw +/-0.2 rad, and
                          R_BS @ w_S  must equal  (0, 0, 1).

    Two measured axes fix R_BS completely -- including the yaw that gravity cannot
    see -- with no CAD and no camera involved. Build it as
        y_hat = pitch axis, z_hat = yaw axis orthonormalised against y_hat,
        x_hat = y_hat x z_hat,  R_BS = [x_hat y_hat z_hat]^T   (rows in S).
    Sanity: the two measured axes should already be ~90 deg apart; how far off
    they are is the error bar on the result.

    The TRANSLATION r_BS is not observable from gyro alone. Either measure it with
    a rule (the NeckPitch board to each lens; |t_SC| says 167/179 mm, so a 5 mm
    rule is a 2-3% check), or exploit the fact that the Head origin lies ON the
    pitch axis: during a pure pitch sweep the camera swings on an arc whose radius
    is its perpendicular distance from that axis, which VIO or triangulation of a
    static scene recovers directly.
    """


def main():
    global RIG, HEAD_X_FWD, HEAD_Y_LEFT, HEAD_Z_UP
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("--host", default=None, help="default: rig robot.host")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--compare", help="a second, INDEPENDENT calibration of the same rig")
    rigcfg.add_arg(ap)
    a = ap.parse_args()
    RIG = rigcfg.load(a.rig)
    ax = RIG.robot["body_axes"]
    HEAD_X_FWD, HEAD_Y_LEFT, HEAD_Z_UP = int(ax["forward"]), int(ax["left"]), int(ax["up"])
    a.host = a.host or RIG.robot["host"]

    cal = load_calib(a.calib)
    print(f"calibration under test: {a.calib}")
    print(f"  OKVIS cam0 <- dma_queue {cal['queue'][0]},  cam1 <- dma_queue {cal['queue'][1]}\n")
    check_self_consistency(cal)

    up_S = None
    if not a.offline:
        print("\nlive capture from", a.host)
        up_S = read_gravity(a.host)
        frames = read_frames(a.host, cal["queue"])
        check_gravity(cal, up_S, frames)

    yaw_new = check_head_frame(cal, "(calibration under test)", up_S)

    if a.compare:
        old = load_calib(a.compare)
        yaw_old = check_head_frame(old, f"({a.compare})", up_S)
        print("\n" + "=" * 78)
        print("TIE-BREAK: is the fault in T_SC or in the shared T_BS?")
        print("=" * 78)
        print(f"  yaw correction wanted by {a.calib}:   {yaw_new:+.2f} deg")
        print(f"  yaw correction wanted by {a.compare}: {yaw_old:+.2f} deg")
        same = np.allclose(cal["T_BS"], old["T_BS"])
        print(f"  the two files share T_BS byte-for-byte: {same}")
        if same and abs(yaw_new - yaw_old) < 5:
            print("  => TWO INDEPENDENT camera-IMU fits demand the SAME correction. Independent")
            print("     fits do not reproduce one arbitrary angle, so the fault is in the factor")
            print("     they SHARE: T_BS. The T_SC pair is exonerated.")
        else:
            print("  => the corrections DIFFER, so the fault tracks the camera-IMU fit, not T_BS.")


if __name__ == "__main__":
    main()
