---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a188f0-d6b9-7473-9f90-2793ada3237b
status: done
priority: normal
dateCreated: 2026-10-29T19:34:07+01:00
dateModified: 2026-10-29T19:34:07+01:00
completedDate: 2026-10-29
run: barlow_track@ef21c16:barlow_track/scripts/scratch/posonly_fast/
projects:
  - "[[barlowtrack]]"
---

## Description

posonly_fast.py trains one position-only cell locally (keypoints only, no volumes; an epoch in seconds to ~2 min) from base_<lab>.yaml plus key=value overrides and prints the frame-pair matching proxy per epoch (val_val_match_accuracy_auto: optimal assignment between 30 held-out frames, all detections included as distractors, GT-named neurons scored; seed noise about 0.04 at epoch 14). --eval adds full-video tracking accuracy with label propagation (eval_accuracy --mode trained). Batches 1-21 and long runs 6-15 (batchN.sh, longN.sh, queueN.sh), 2026-10-28 to 29, each logged in NOTEBOOK.md with its run id; outputs ~/.cache/barlow_scratch/train/posonly_fast/. Batch 20 and 21 use the same driver with --base round_fused_fix_zimmer/train_config.yaml for image-only and fused networks.
