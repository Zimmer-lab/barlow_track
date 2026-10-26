---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1776e-9588-7bb1-966b-dcbb16a0b2e0
status: done
priority: normal
dateCreated: 2026-10-23T15:20
dateModified: 2026-10-23T15:20
completedDate: 2026-10-23
run: barlow_track@95b93fb:barlow_track/scripts/scratch/paper_factorial_check.py
data:
  - /home/charles/.cache/barlow_scratch/runs/flavell_factorial_20261023_1520
  - /home/charles/.cache/barlow_scratch/runs/flavell_trained_factorial_20261023_1533
projects:
  - "[[barlowtrack]]"
---

## Description

Re-tracks the preprint's saved Flavell embeddings (analyzed_projects/flavell/{baseline,untrained}/2025_07_01trial_0) and scores them against the NWB GT, varying one thing at a time: A = v0.1.0 tracking code + the saved SVD features (u[:, :50]) + 100 label-propagation seeds (exact reproduction); B = HEAD code; C = HEAD + eval_accuracy's TruncatedSVD (U*S); D = C with 25 seeds. Local CPU, seed 0. Runs r261023-2158169 (untrained) and r261023-2165243 (trained). Also checked the crop normalization: barlow_track@95b93fb:barlow_track/scripts/scratch/normalization_check.py (30 frames, paper baseline checkpoint).

The script was run uncommitted on bc9ff96 and committed unchanged as 95b93fb, which the notebook entry of 2026-10-24 08:42 records.
