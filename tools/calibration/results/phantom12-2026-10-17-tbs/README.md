# phantom12 T_BS: powered-neck takes, 17 Oct 2026

Two takes with the neck held by its motors (P1, P2), checked with step 81's pixel-level hand-eye fit;
the 16 Oct hand-held takes (A2, A3) rerun with the same code as negative controls. The outcome and the
numbers: `FINDINGS.md`. How the takes were done (motors and neck through DMA.ethercat's own tools, no
walker): `powered/README.md`, `powered/tbs-grid.sh`.

| file | what |
|---|---|
| `FINDINGS.md` | the answer, the numbers per take, the cameras' disagreement, settled / not settled, the rule changes |
| `reports/tbs-neck-*-qc.json`, `-handeye.json`, `-gyro.json` | steps 75 / 81 / 82 reports, the committed code (Thor, 18 Oct) |
| `reports/run-tbs-final.log` | the runner's output for P1, P2 (vs P1), A3, A2, with the package's tests on Thor |
| `powered/` | the powered take's procedure and grid script |
| `analysis/dump_corners.py` | per hold, stereo pairs' detected corners and neck angles -> `<take>-corners.npz` (Thor) |
| `analysis/proto.py` | the four measurement models compared (mono, stereo-pose, stereo-points, pixels); `proto-*.txt` its output |
| `analysis/split.py`, `split-P1.txt` | the pose-level fit's residual split by Head axis, camera, lens edge, neck model |
| `analysis/split_half.py`, `split_half.txt` | each camera alone on the even and odd holds: the 5 mm between the cameras is a systematic (Thor) |
| `analysis/dump_frames.py` | step 81's per-image board poses (pose model) for the 16 Oct analysis scripts (Thor); the 16 Oct copies of `dump_frames.py` and `crosscheck.py` predate `_frames` returning `(frames, n_gap)` and need this one's unpacking |
| `analysis/gravity_fit2-P1.txt`, `attribution-P1.txt` | encoders vs head IMU gravity (0.09 deg rms), camera vs gravity per hold |

The analysis scripts run from `../phantom12-2026-10-16-tbs/analysis` (they import its `fusedfit` and
`gravity_fit`) or, those marked Thor, in `calib-py:2` with the package's scripts on `PYTHONPATH`.
