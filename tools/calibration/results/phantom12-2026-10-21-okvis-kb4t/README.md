# phantom12: OKVIS depth rectification with the KB4T lens (21 Oct 2026)

Plan: `docs/superpowers/plans/2026-10-21-okvis-kb4t-depth-rectification.md`. `make_fixture.py` / `.out`: Task 1 (the
formula and the C++ test's fixture).

## Candidate phantom12 config

**Needs an OKVIS image with the kb4t change (commits 7c86e75b, 0e44f1ec); an older image refuses this file: it logs
`unrecognized distortion type kb4t`, builds no rectified camera and stops.** Not deployed: nothing was copied to the
robot.

`phantom12_stereo_dense_thor-fita.yaml` (sha256 `ccba2f1c28f5...`) is the config phantom12 runs with three things
replaced:

- **Both cameras' lens:** Fit A (`../phantom12-2026-10-20-lens/fit-a-kb4t.json`, key `KB4T@all@800`), written as
  `distortion_type: kb4t` with `distortion_coefficients: [k1, k2, k3, k4, p1, p2]`. Depth uses the whole lens. VIO uses
  k1..k4 only, which moves pixels by up to 0.90 px within 350 px of the centre and 4.17 px at the edge (camera 0).
- **Camera 1's pose** `T_SC` = `T_SC0 · C`, where C is Fit A's stereo pose (baseline 44.43 mm; the deployed file has
  44.83 mm).
- **`T_BS`** (the IMU in the Head link), from step 83 with the hand-eye reports
  `../phantom12-2026-10-20-handeye/fita450-P{1,2}.json`: `T_BS = mean(X) · T_SC0^-1`. With it, camera 0 sits
  **0.495 deg / 13.99 mm** from where the deployed file puts it in the Head.

Everything else is the deployed file's, byte for byte. That includes camera 0's `T_SC`, `image_delay` and all tuning.
New comment lines at the top of the file say what changed and where it came from. The older header lines further down
are kept, and the new lines say those describe the values that were replaced.

**Source:** `/etc/phantomos/okvis/phantom12_stereo_dense_thor.yaml`, copied from phantom12 on 21 Oct with one read-only
`scp`. sha256 `7dd63554ba77b42314d0b26fe7aacbba36e7119fd6270748c436faa3a7e32046`, the 16 Oct transplant. It is
byte-identical to the repo copy `../phantom12-2026-10-15/imucam/deploy/phantom12_stereo_dense_thor.yaml`, so
`make_candidate.py` with no argument rebuilds the candidate from the repo alone. A rerun gives a new sha256 because the
header carries the run time, but the numbers come out the same.

**How it was made:** `make_candidate.py` (its log: `make_candidate.out`) uses the package's own text surgery. Step 71's
`okvis_text(kb4t=True)` calls step 70's `replace_tsc`, `repl_seq` and `replace_distortion_type`. `okvis_text` rewrites
`T_SC0` as `okviscfg` reads it, which is cleaned to the nearest rotation and differs by about 1e-16. The script puts
the source's `T_SC0` text back. Then it runs step 83 as a program.

```
cd tools/calibration/results/phantom12-2026-10-21-okvis-kb4t
uv run --with numpy --with scipy --with opencv-python-headless --with pyyaml python make_candidate.py \
    <copy of the deployed file> > make_candidate.out 2>&1
```

### Checks (all passed; the script writes nothing unless every one passes)

| check | result | limit |
|---|---|---|
| source sha256 | `7dd63554...`, equal to the repo copy | must start `7dd63554` |
| (a) `okviscfg` reads it as | `pinhole-kb4t`, 2 cameras x 10 numbers = Fit A's | largest relative difference 2.9e-16 (16-digit print), limit 1e-14 |
| (b) camera 0's `T_SC` | the source's text, byte for byte | identical |
| (c) `inv(T_SC0) · T_SC1 - C` | 3.9e-16 (2.9e-16 without `okviscfg`'s rotation cleaning) | 1e-12 |
| (d) `T_BS · T_SC0 - X` (`T_BC-fita.json`, the hand-eye mean) | 1.1e-16 (4.4e-16 without cleaning) | 1e-9 |
| (e) values OKVIS reads (OpenCV FileStorage, OKVIS's parser) | of 178, 136 identical, 42 changed, 4 added | changes only in the lens keys of both cameras, camera 1's `T_SC`, `T_BS` |
| (f) text lines | 14 replaced by 27 | changes only in header comments, the lens lines, camera 1's `T_SC`, `T_BS` |
| wrong source | the 9 Aug file: STOP, nothing written | stops |

How to read it:
- "largest entry" in (c) and (d) is the largest difference between two 4x4 matrices, entry by entry. Rotations have no
  unit; translations are in metres. Values near 1e-16 are the last digit a double holds.
- (e): the 42 changed values are 9 per camera (focal length 2, principal point 2, the first 4 coefficients, the type),
  12 in camera 1's `T_SC` and 12 in `T_BS`. In each 4x4 matrix the bottom row `0 0 0 1` is unchanged. The 4 added
  values are p1 and p2 for each camera.
- (f): the 13 new header lines (7 from step 83, 6 for the lens), 4 lens lines per camera, and 3 matrix rows each for
  camera 1's `T_SC` and for `T_BS`.

`candidate.diff` is the unified diff against the deployed file. It contains only these lines.

### Deploy check

This needs the user's yes, and nothing has been done. phantom12 would first run the CI image `thor-<sha>` of a commit
that has the kb4t change, through its PKO host-config (`images: { okvis2x: ... }`, platform infra). Then this file goes
beside the current one, `OKVIS_CALIBRATION` points at it, and `ds/okvis2x` gets a rollout restart. Follow
`deployment/RUNBOOK.md`. Switch the image first, with the current file still selected: an older image refuses this
file, while the current file works with both images.

In the pod log (Loki) at startup:

    [entrypoint] config: phantom12_stereo_dense_thor-fita.yaml ... (ccba2f1c28f5)
    kb4t: VIO uses k1..k4; p1, p2 are used by the stereo-network rectification only        (WARNING, once)
    KB4T (VIO: equidistant k1..k4) pinhole camera 0 ...   (and camera 1)
    KB4T rectification (stereo network): camera 0 p1=0.000852307 p2=-0.00111309, camera 1 p1=-0.000370916 p2=-7.26592e-05

**`KB4T rectification (stereo network)` is the line that proves depth uses the whole lens.** If it is missing, depth is
not running Fit A. `unrecognized distortion type kb4t` means the image does not have the change.

### Rollback

- The deployed file `/etc/phantomos/okvis/phantom12_stereo_dense_thor.yaml` (sha256 `7dd63554`) stays on the robot and
  is not modified. It keeps working with the current image, and an image with the kb4t change runs it as before (the plan:
  existing configs behave byte-identically).
- A deploy takes a fresh `robot.env` backup before switching (as `../phantom12-2026-10-15/imucam/deploy/install.sh`
  does). To roll back, restore that backup, which points `OKVIS_CALIBRATION` back at
  `/etc/okvis/phantom12_stereo_dense_thor.yaml`, then do a rollout restart of `ds/okvis2x`. If needed, the host-config
  image goes back to the current one.
- **`robot.env.bak-20261016-161136` is not this rollback.** It is the state from before the 16 Oct transplant and
  points at `okvis_stereo_dense_thor.yaml`, the 9 Aug file with another camera pair's calibration. Restoring it would
  go two steps back.

### Caveats

- **`T_SC0` (camera 0 in the IMU) is Kalibr's 15 Oct camera-IMU fit, made with the 15 Oct lens.** It is kept as the
  plan says. The camera in the Head (`T_BS · T_SC0 = X`) is exact by construction. The IMU in the Head (`T_BS`) and
  VIO's camera-IMU geometry inherit whatever Fit A's lens change would move in a camera-IMU refit. `image_delay`
  (-0.0056 s) is also the 15 Oct value.
- `T_BS` is IMPROVE, not REPLACE: 2 sigma is 5.6 / 5.1 mm (P1 / P2), against the 3 mm / 0.3 deg gate
  (`../phantom12-2026-10-20-handeye/FINDINGS.md`).
- Fit A is not validated in the image corners, at the bottom of the image or beyond 1.3 m
  (`../phantom12-2026-10-20-lens/README.md`).
- No OKVIS binary has loaded this file yet. The C++ kb4t path is covered by `kb4t_rectify_test` and this file by the
  Python checks above. The first real load is the deploy check.
