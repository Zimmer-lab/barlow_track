---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a20cea-ed8d-7666-baa4-97beb9bf3707
status: done
priority: normal
dateCreated: 2026-11-24T10:40
dateModified: 2026-11-24T10:40
completedDate: 2026-11-24
run: barlow_track@e274815:barlow_track/scripts/paper_scripts/revision/t1_deformation_table.py
projects:
  - "[[barlowtrack]]"
---

## Description

Revision test T1 (paper repo `review/TEST_PLAN.md`), local CPU, GT positions only, 2026-11-24. Ran from the
pre-move copy at `scratch/revision_t1_t2/deformation_table.py` (c10ecab), with the same code as the committed file.

Inputs: GT positions of the 12 benchmark recordings (`gt_positions.GT_PATHS`, the GT sources of
round_shared_v1's records). z is scaled by each recording's trial-59 visnet_z_scale. Detections come from each
recording's segmentation (project, Flavell / Samuel NWB, Leifer StarDist project), every 10th volume. Per-frame
accuracy comes from round_shared_v1 `tracks_v2/` for trial 59, with 1 replicate on tuning recordings and 5 on
held-out ones. Volume rates are from paper Table 1; Samuel is 10 Hz (Susoy et al. 2021). 300 frame pairs per
lag, seed 0.

Outputs in `/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/revision_analysis/`:
- `t1_dataset_table.tsv`
- `t1_frames_<lab>.tsv` (per-frame d_lag1, d_novel, rot_lag1, acc_t59)
- `t1_deciles.tsv`, `t1_decile_summary.tsv`
- `t1_segments.tsv` (the 5 most deformed 20-frame segments per recording)
- `neuron_deformation_<lab>.npz`
- caches `cache/gt_positions_<lab>.npz`, `cache/detection_nn_<lab>.json`

A first run (same day) used fixed-frame lags (1/10/100) and the GT-only NN distance as the unit. Both were
replaced: Zimmer and Leifer annotate a subset of the neurons (GT 36-73% and 57% of detections), which inflates
the GT-only NN; and fixed-frame lags span 10 s on Samuel but 59 s on Flavell.
