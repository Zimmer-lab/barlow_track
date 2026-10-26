---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a17931-f2a0-7b18-bc0a-bfd8b6e3ec57
status: done
priority: normal
dateCreated: 2026-10-26T15:18
dateModified: 2026-10-26T15:18
completedDate: 2026-10-26
run: barlow_track@0ad34c2:barlow_track/scripts/scratch/local_geometry_change.py
projects:
  - "[[barlowtrack]]"
---

## Description

For the GT positions of four datasets (zimmer_1128 project, Flavell, Leifer and Samuel NWBs), the std of log(distance at t + lag / distance at t) over neuron pairs within the ViSNet cutoff (0.245 x max volume dimension), binned by initial distance in nearest-neighbour (NN) units (0-1.5/1.5-2.5/2.5-4/4-6/6-10/10-cutoff), for real frame pairs (lags 1, 10, 100; 100 pairs) and for the two training views of one GT frame under the position pipeline's augmentation (global affine with scale jitter 0.1, per-point jitter 1 voxel), plus a grid of global elastic settings (914141c) matched to real lag 100 by RMS log mismatch over the bins. Results in NOTEBOOK.md 2026-10-26 (entries after 0ad34c2).
