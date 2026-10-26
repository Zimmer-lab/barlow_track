---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1776e-969a-7e8d-9ad8-6d3ac22fb50b
status: done
priority: normal
dateCreated: 2026-10-24T02:07
dateModified: 2026-10-24T02:07
completedDate: 2026-10-24
run: barlow_track@95b93fb:barlow_track/scripts/scratch/compare_gt_sources.py
projects:
  - "[[barlowtrack]]"
---

## Description

Compares the two Flavell GT versions, `_load_project_gt(flavell_data/projects/2025_07_01)` and `_load_project_gt(flavell_data.nwb)`, matched by segmentation id, and scores the paper's stored tracks against each. Companion checks: barlow_track@95b93fb:barlow_track/scripts/scratch/gt_mask_consistency.py (GT positions against the project's masks, 6 frames) and barlow_track@95b93fb:barlow_track/scripts/scratch/nwb_gt_coverage.py (NWB field alignment per lab). Uncommitted when run; committed unchanged as 95b93fb.
