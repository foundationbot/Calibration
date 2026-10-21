# phantom12 stereo calibration "Fit A" (20 Oct 2026): KB4T

The stereo calibration the user chose on 20 Oct: KB4T = equidistant fisheye k1..k4 plus tangential p1, p2
(`scripts/lensmodel.py`), both cameras and the stereo pose, fitted by the analysis code of 18-20 Oct
(`~/calib/phantom12/results/tbs-2026-10-17/lens/`, not yet in the repo: the productising plan
`docs/superpowers/plans/2026-10-20-calibration-kb4t-productise.md` brings it in).

- Data: the 19 Oct lens take (stand + neck grid, 0.5-1.6 m) and the 16-17 Oct neck takes P1, P2, A2, A3, even-ranked holds;
  1777 pairs; converged (115 evaluations).
- Test: held-out odd-ranked holds, no board beyond 1.3 m: 12 of 12 cells with >= 8 holds within +-0.5 % depth at
  0.4-1.3 m (worst: image centre 0.9-1.3 m, -0.45 % [-0.51, -0.38], 25 holds).
- Not validated: the image corners (a fit without the 1.6 m boards differs by up to 6 % at 1 m there), the bottom of
  the image (no cell with 8 holds; the fits agree there within 0.2-0.6 %), beyond 1.3 m, another day.
- `fit-a-kb4t.json`: `intr` per camera `[fx, fy, cx, cy, k1, k2, k3, k4, p1, p2]`; `C` = camera 1 in camera 0.
- OKVIS cannot load KB4T yet (its depth rectification knows equidistant and radial-tangential only).
