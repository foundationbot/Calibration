# Read B on a Thor CPU (phantom10, 11 Nov 2026 02:08, measured)

The robust matcher (`armcal.readb`, default since f07c632) on 4 stored phantom14 bursts (5 frames, 1280x800) copied
to phantom10, run by `scripts/handeye_match.sh` at nice 15, with OKVIS running beside it (load average about 6).

| Processes | Wall time, 4 captures | Per capture, match only |
|---|---|---|
| 1 | 20 s | 1.7-1.8 s |
| 4 | 6 s | 2.2-2.3 s |

How to read it:
- wall time includes loading and fusing the bursts and both matcher passes;
- "match only" is the matcher's own time per capture as `readb` prints it;
- the offsets equal Moonstone's on the same bursts;
- a 20-capture wave is about 30 s with 4 processes (an extrapolation from these rows).
