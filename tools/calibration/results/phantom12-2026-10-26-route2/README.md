# Route 2: the camera's rotation from the robot's own torso mesh (25 Oct 2026, reviewed): NOT USABLE

Plan: `docs/superpowers/plans/2026-10-25-tagfree-selfcal.md` (stereo-depth-eval branch), entries 25 Oct 23:25 (rules,
fixed before data) and 25 Oct 23:55 (result). Code `torso_fit.py`, tests `check_torso_fit.py` (Moonstone, with the
point-cloud code), output `route2.json` (sha256 9fa38fcec9aa…), log `route2.log`. Run 9 min 14 s on Moonstone cores 0-7.

**Question.** Can fitting the torso CAD (`MK1_1_Torso_Shell` + the two cover panels, MJCF Torso frame = #816 URDF
Torso frame) in the head depth measure the camera's rotation in the torso to 0.1 deg, as a tag-free replacement for the
plate's B1 rotation (0.188, 0.001, -0.015) deg?

**Answer: no.** By the rule fixed before data, no session passes:

| session | neck | torso in view | mesh check (median, E = I) | captures counted | median rotation vs B1 | profile | verdict |
|---|---|---|---|---|---|---|---|
| sc1025d | 28 deg | bottom ~100 rows, ~70k px | 0.96 mm (pass) | 13 / 13 | 3.05 deg off; per-capture std (2.9, 2.2, 1.0) deg | flat about torso x and z | FLAT |
| sc1025i | 23 deg | a sliver, ~11k px | 1.69 mm (pass) | 5 / 13 | 23.6 deg off | flat on all axes | FLAT |
| sc1025j | 18 deg | none | no torso pixels | 0 | - | - | M0 FAIL |

Do-nothing baseline (no correction): 0.189 deg from B1, so the fit is ~16x worse than doing nothing.

**Why.** The chain already places the torso CAD within about 1 mm of the depth (mesh check), so the mesh and frame are
right. But the chest is seen at a grazing angle in a thin band at the image bottom, and its curved shell lets the fit
trade rotation for translation: across sc1025d's captures ry correlates with tx at -1.00 and rx with ty at 0.98. Even on
perfect synthetic depth at sc1025d's camera pose the weakest direction of the fit is a rotation about torso x (normal
matrix eigenvalue ratio 1.2e-6), which is the axis B1's 0.19 deg lies on. Lowering the head further is not possible
(NeckPitch's range ends at ~27.5 deg).

**Review:** one Sonnet subagent recomputed every per-session number, flat flag and the verdict from route2.json; all
match; no alternative reading of the rule (flat on both sides, all ok captures, M0-passing sessions only) changes it.
The code had one earlier review (its findings fixed before this run: 5c94a5ab).
