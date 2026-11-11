"""Synthetic captures for the armcal tests: a stored phantom14 pose of each arm (4 Nov 2026), the arm's meshes placed
there and moved by a known correction, splatted into a small depth image."""
import numpy as np

from armcal import REPO_URDF, mesh, meshfit, readb
from armcal.urdf import Urdf
from rigmath import nearest_rotation, rot_axis, transform

K, W, H = (234.18, 234.18, 321.06, 182.59), 640, 400     # phantom14's 1280x800 depth camera at half size
_T_BC = np.array([[0.006168, -0.491482, 0.870866, 0.151781], [-0.99981, -0.019131, -0.003715, -0.004274],
                  [0.018486, -0.870677, -0.491507, 0.040023], [0.0, 0.0, 0.0, 1.0]])
T_BC = transform(nearest_rotation(_T_BC[:3, :3]), _T_BC[:3, 3])
POSES = {   # base_w2 07-p09-10-0 and base_L 17-pL04-22-0: (arm joints, neck yaw, neck pitch), rad
    "Right": ({"RightShoulderPitch": -1.149872, "RightShoulderRoll": -0.347025, "RightShoulderYaw": 0.598868,
               "RightElbowPitch": -1.167933, "RightWristYaw": 1.000419, "RightWristRoll": 0.356826,
               "RightGripperYaw": -0.548542}, -0.002388, 0.179558),
    "Left": ({"LeftShoulderPitch": -1.131819, "LeftShoulderRoll": 0.198874, "LeftShoulderYaw": -0.542669,
              "LeftElbowPitch": -1.031988, "LeftWristYaw": -0.726068, "LeftWristRoll": -0.117381,
              "LeftGripperYaw": 1.117396}, -0.001334, 0.386536),
}
URDF = Urdf.load(REPO_URDF)
_ARMS = {}


def arm(side, voxel=0.001):
    if (side, voxel) not in _ARMS:
        _ARMS[side, voxel] = readb.arm_model(URDF, side, cfg=meshfit.with_config(meshfit.FAST, model_voxel=voxel))
    return _ARMS[side, voxel]


def pose(side, urdf=URDF):
    q, yaw, pitch = POSES[side]
    T_TC = urdf.chain("Torso", "Head").T({"NeckYaw": yaw, "NeckPitch": pitch}) @ T_BC
    return dict(q), yaw, pitch, T_TC


def rigid(deg, axis, t_mm):
    a = np.asarray(axis, float)
    return transform(rot_axis(a / np.linalg.norm(a), np.radians(deg)), 1e-3 * np.asarray(t_mm, float))


def depth_image(side, E=np.eye(4), noise=0.0005, seed=0, background=1.4, q=None, T_TC=None):
    """(H,W) depth of the arm's links moved by E (camera frame) in front of a flat background, splatted from the
    densely sampled meshes (0.5 mm voxels), with Gaussian noise"""
    q0, _, _, T0 = pose(side)
    q, T_TC = q or q0, T0 if T_TC is None else T_TC
    a = arm(side, 0.0005)
    P, _, _ = a.place(q, np.linalg.inv(T_TC), E)
    d, _ = mesh.render(P, np.eye(4), K, W, H)
    d = np.where(d > 0, d, background)
    return d + noise * np.random.default_rng(seed).standard_normal(d.shape)
