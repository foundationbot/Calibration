# phantom12 stereo calibration 3 (15 Oct 2026, evening): recommended for depth

Data: takes 2 + 2b (15:14, 15:18) and take 3 (16:12, near range and picture edges, live view),
bottom camera off, left/right fixed 3000 us / ISO 800, cameras in sync (+0.1..+0.5 ms).
Coverage of the merged tree: 95.0 / 95.6 %, reach 67.5 / 66.0 deg (gates 60 % / 60 deg).

**Recommended: `calib_stereo3noedge-camchain.yaml`** (equidistant, all takes, pairs with any
corner more than 450 px (~46 deg) from the image centre left out: 1103 of 1350 pairs).
- Kalibr reprojection sigma 0.29/0.26 (cam0), 0.27/0.26 (cam1) px; baseline 44.83 mm.
- Step 60: held-out median |dy| 0.204 px PASS. Step 65: +0.208 % PASS (0.44-2.15 m).
- Scale against distance on 722 pairs no fit used: +1.0 / +1.0 / +0.6 / +0.4 / -0.4 / -1.0 %
  at 0.4 / 0.6 / 0.8 / 1.05 / 1.4 / 2.1 m (4-6 mm at 0.4-0.85 m).
- Caveat: beyond ~46 deg off-axis the lens is extrapolated; higher-order distortion terms (k3, k4)
  are weakly constrained and differ between the two lenses.

Why the edge frames are left out: with them, the 4-parameter equidistant model cannot fit the
outer ring of this ~130 deg lens, and the fit compensates in the centre's stereo geometry.
Same 722 unused pairs, scale error %:

| fit | 0.4 | 0.6 | 0.8 | 1.05 | 1.4 | 2.1 m |
|---|---|---|---|---|---|---|
| calib 2 (takes 2+2b, equi) | +1.7 | +1.2 | +0.7 | +0.4 | -0.0 | -0.4 |
| calib 3 (all, equi) | +0.6 | +0.0 | -0.9 | -2.0 | -3.4 | -5.0 |
| calib 3r (all, radtan) | +0.7 | +0.6 | +1.0 | +0.9 | +0.7 | +0.7 |
| take 3 only (equi) | +0.5 | -0.4 | -1.6 | -3.2 | -5.2 | -7.3 |
| all minus close < 0.6 m (equi) | +1.0 | +0.3 | -0.8 | -2.1 | -3.6 | -5.6 |
| **all minus edges > 450 px (equi)** | **+1.0** | **+1.0** | **+0.6** | **+0.4** | **-0.4** | **-1.0** |

Fallback if the full field matters (e.g. SLAM features at the border): `calib_stereo3r` (radtan),
flat +0.7 % at every distance but a poorer lens fit (held-out reprojection 1.13 px vs 0.59).

Not done yet: the camera-IMU take (operator moves the head), then step 70 (template: the repo
`mk12_stereo_holevisor.yaml`, dma_queue 1,2) and step 80.
