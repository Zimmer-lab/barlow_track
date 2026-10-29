---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a18539-3373-7fc9-a8e7-3abcfd3cd9a8
status: done
priority: normal
dateCreated: 2026-10-29T02:14:40+01:00
dateModified: 2026-10-29T02:14:40+01:00
completedDate: 2026-10-29
run: barlow_track@230c297:barlow_track/scripts/scratch/svd_gt_matching_check/run.sh
projects:
  - "[[barlowtrack]]"
---

## Summary
Re-tracks cached Samuel 153 backbone embeddings of two networks (round_bayes_samuel trial 0, round_visnet_full trial 27) with eval_accuracy.py's opt-in `--svd whiten` (the SVD space production tracking uses, U instead of U·S) and `--gt_matching gt_valid`, against the defaults; label propagation, 25 seeds, CPU tracking.

## Method
Run r261028-3231600; outputs in ~/.cache/barlow_scratch/runs/svd_gt_matching_check/. Options added in 482dcfb after a code review (findings #2 and #4).
