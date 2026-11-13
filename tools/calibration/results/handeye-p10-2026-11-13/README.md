# Second live hand-eye session: phantom10, 13 Nov 2026 (session 20261113-125417)

Run from `calib`'s hand-eye session (`calibtool/handeyeflow.py`).
- **Robot setup:** OKVIS depth-only (S2M2-L 1280) on that day's head calibration
  (`phantom10_stereo_kb4t_20261113_640_imu.yaml` scaled to 1280).
- **Plans:** `config/handeye/P10.json`, re-planned in MuJoCo from the 12:45 snapshot.
- **Captures:** every pose at the 1280 and 768 depth profiles (`profiles: "1280,768"`).

**Plain result.**
- **Right arm:** wave complete (24 captures at each profile).
- **Left arm:**
  - started after a motor-power brownout and the robot's reboot;
  - stopped once at pose pL08, when both head cameras dropped off USB for 35 s;
  - finished at 1280 only.
- **The phantom14-derived model (M12+head) missed the 8 mm deploy gate** (left 17 mm). Adding the **camera's rotation
  in the head** (M12+headR) fixes most of it, and also improves 11 Nov's data (3.9 / 3.6 mm).
- **Not deployed;** the deploy was dry-run on the robot with M12+headR (`fitR/`): the written files reproduce the fit to
  5e-7 mm.

## Accuracy (leave one arm pose out; mm at the wrist)

| Model | Right median / p90 | Left median / p90 | Unmodified URDF, R / L |
|---|---|---|---|
| M12+head | 9.1 / 12.8 | 17.1 / 22.3 | 32.3 / 32.4 |
| M12 | 8.8 / 12.5 | 16.9 / 22.0 | |
| **M12+headR** (now the default) | **5.7 / 10.5** | **11.2 / 14.8** | |
| M12+headRT (overfits: camera 43 mm sideways) | 5.6 / 8.8 | 8.1 / 15.1 | |
| 11 Nov data, M12+head | 7.0 / 11.5 | 5.1 / 8.4 | 45 / 38 |
| **11 Nov data, M12+headR** | **3.9 / 10.4** | **3.6 / 8.4** | |

How to read it:
- one row per model;
- lower is better;
- the user's aim is < 6 mm;
- 13 Nov's left arm had only 7 usable captures (12 clean, 5 dropped by the 15 mm outlier cut), so its numbers are weak.

## The camera-rotation finding

- **The pattern:** with M12+head, the sideways offset grew with neck yaw (about ±15 mm over ±12°). That is the
  signature of a camera rotation in the head, which the translation-only model cannot fit.
- **The fitted rotation (HEADR, rotation vector, deg):**

  | Day | HEADR |
  |---|---|
  | 11 Nov | −0.29 ± 0.27, +3.38 ± 0.45, +0.04 ± 0.36 |
  | 13 Nov | −2.68 ± 0.50, +2.08 ± 0.82, −1.47 ± 0.55 |

- **The +3.4° about y** is in both days' data: a tilt of the camera in the head relative to the URDF / T_BS.
- **The extra −2.4° (x) and −1.5° (z)** appeared with the 13 Nov head refit (another session's work), which changed the
  cameras' T_SC by about 1-2°. That refit's camera rotation looks 2-3° off; it matters for SLAM too.
- **The shoulder-roll zeros** agree between days: right −1.2 to −1.7°, left −0.6 to −1.2°.

## Hardware events during the session (since fixed by the user, 13 Nov)

| Time | Event |
|---|---|
| 13:07:45 | 48 V bus 30 V (min, RightKneePitch), all 29 drives 0x3120, robot limp; power lost by 13:19 (rebooted by the user) |
| 13:46:14-13:46:48 | both head camera boards off USB (2-3.1, 2-3.2; briefly on USB 2); port 1-1 flapping |

## Files

- `readb-{right,left}.json`: read B records.
- `fit.json`: M12+head, the menu's fit.
- `fit-headR.json`: the default after the change.
- `fit-1111a-headR.json`: 11 Nov's data refitted.
- `phantom10_mk1_handeye-20261113-headR.urdf`: the URDF terms; the camera term is stated in its header.
- `log-{right,left}.jsonl`: every command and its time.
- The bursts are on phantom10 in `/data/calib/handeye/20261113-125417/{right,left}/` and `right/r768/` (768 profile, not
  yet analysed).
