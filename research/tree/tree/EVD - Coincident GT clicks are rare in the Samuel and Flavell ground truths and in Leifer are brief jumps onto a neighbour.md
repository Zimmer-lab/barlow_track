---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1aa84-abf1-7fe3-b2f2-75ba5ca8d667
status: open
priority: normal
dateCreated: 2026-11-05T08:03:03+01:00
dateModified: 2026-11-05T08:59:53+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Conclusion
Clicks whose nearest other-neuron click in the same volume is under 1.0 µm:
- Samuel 153: 0.04% of 120,923 clicks.
- Samuel 137: none.
- Samuel 184: none of 210,336 at the probably correct 0.45 µm spacing (0.17% at the NWB's 0.3 µm).
- Flavell: 0.01% of 231,268.
- Leifer: 0.68% of 119,019, which is 4 to 50 times the others.

The Leifer coincidences are not tight neighbours. The pairs involved are normally 2.6–5 µm apart (median over the recording), and their clicks meet in short runs: up to 9 volumes, 74 pairs. So one click jumps onto the neighbour, which is the case for excluding coincident clicks there. In the other recordings, 1 µm coincidences can cost an evaluation that tracks the clicks as objects at most about 0.04 points.

Samuel 184 at first showed 0.17% of clicks coincident and persistent close pairs (054/080 at a median of 1.28 µm). These come from its NWB grid_spacing of 0.3 µm in xy, which is probably wrong (issue i261105-1). In pixels its head matches 153 and 137, and at 0.45 µm (the spacing wbfm's Harvard converter writes) 184 has no clicks within 1 µm and a median nearest click of 3.01 µm, like the others.

## Method
- `gt_click_coincidence.py` (43203f6; the 0.45 µm rerun of 184 with `--spacing`, e2a899d).
- 184's spacing: in pixels, the click cloud's s.d. along the major / minor axis is 43.3 / 11.5 px (153: 46.9 / 12.0, 137: 42.8 / 12.9), and the nearest click in xy is 3.7 px (3.8, 4.3).
- Clicks from each NWB's final tracks via `_load_nwb_arrays`, converted to µm with the NWB imaging-plane grid_spacing.
- Per click: distance to the nearest other neuron's click in the same volume.
- Per neuron pair ever within 1.0 µm: volumes within, runs, longest run, and median and p10 distance over all shared volumes.
- Output: `~/.cache/barlow_scratch/train/gt_coincidence/` (coincidence_summary.json, `<name>_pairs.csv`, nearest_click_distance.png).

| Recording | Grid xyz (µm) | Clicks | Partner within 1.0 / 1.5 / 2.0 µm | Nearest other click median (p1) µm | Pairs within 1 µm |
|---|---|---|---|---|---|
| Samuel 153 | 0.45, 0.45, 1.75 | 120,923 | 0.04% / 0.26% / 4.5% | 3.22 (1.72) | 1 |
| Samuel 137 | 0.45, 0.45, 1.75 | 85,353 | 0 / 0.04% / 2.3% | 3.37 (1.86) | 0 |
| Samuel 184-004, NWB spacing | 0.3, 0.3, 1.75 | 210,336 | 0.17% / 7.9% / 36% | 2.20 (1.15) | 10 |
| Samuel 184-004 at 0.45 µm (probably correct) | 0.45, 0.45, 1.75 | 210,336 | 0 / 0.28% / 5.1% | 3.01 (1.66) | 0 |
| Flavell (images_for_charlie) | 0.3, 0.3, 0.3 | 231,268 | 0.01% / 2.7% / 47% | 2.03 (1.38) | 10 |
| Leifer re-export A2 | 0.3226, 0.3226, 1.5 | 119,019 | 0.68% / 1.3% / 2.8% | 4.52 (1.29) | 74 |

Top pairs:
- **Samuel 153:** neurons 003/014, 23 volumes in 9 runs (longest 5), median 2.62 µm apart.
- **Samuel 184:**
  - 054/080: 92 volumes in 86 runs (longest 3), median 1.28 µm (p10 1.07).
  - 004/022: 54 volumes, median 1.58 µm.
  - 018/055: 13 volumes, median 1.71 µm.
- **Leifer:**
  - 010/047: 69 volumes in 42 runs (longest 9), median 5.03 µm.
  - 050/070: 42 volumes, median 2.57 µm.
  - 002/046: 32 volumes, median 3.53 µm.
  - 016/025: 30 volumes, median 3.83 µm.
- **Flavell:** at most 4 volumes per pair.

## Caveats
- Distances depend on the NWB grid_spacing, and two of them are unverified.
  - Samuel 184's 0.3 µm xy is probably wrong (i261105-1; the 0.45 µm row is the likely one). It also set visnet_z_scale 5.833 instead of 3.889 in round_wk1_samuel_184 and round_bayes_wk3_samuel_184.
  - Flavell's isotropic 0.3 µm (z included) is unlikely to be the real z step.
- Click-only GTs have no masks, so merges (two neurons in one object) cannot be detected this way.
- The old Leifer NWB and Zimmer (mask-based GT) were not checked.
