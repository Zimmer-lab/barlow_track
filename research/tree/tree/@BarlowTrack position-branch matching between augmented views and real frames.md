---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a181fd-0c35-7b54-8dd8-bbeef8224b97
status: done
priority: normal
dateCreated: 2026-10-28T11:10:07+01:00
dateModified: 2026-10-28T11:10:07+01:00
completedDate: 2026-10-28
run: barlow_track@a3bab89:barlow_track/scripts/scratch/position_branch_matching.py
projects:
  - "[[barlowtrack]]"
---

## Description

For four fused (ViSNet, `add`) Flavell position-pipeline checkpoints (paper_flavell_use_position, _norot, _elastic, _legacyaug under ~/.cache/barlow_scratch/train/), each frame's detections (segmentation centroids, median 178; GT neurons median 139, identified by raw_segmentation_id, carried across frames by neuron name) are embedded with the position branch alone, norm_pos(encode_position(kpts)), keypoints normalized as in training, then standardized per frame (as the Barlow loss does) and matched by Hungarian assignment on cosine distance. Augmented: the two training views of 100 frames, keypoint half of VolumeCoordsDataset._augmented_view with each checkpoint's own global_augment / position_augment. Real: 100 frame pairs each at lags 1, 10, 100, 500, 1000, no augmentation. Score: fraction of GT neurons present in both sets assigned to themselves. Each checkpoint also with its position branch re-initialized (seed 0). Effective rank: participation ratio of one frame's descriptor spectrum. Output /home/charles/.cache/barlow_scratch/position_branch_matching/results.jsonl, run r261028-2940337.
