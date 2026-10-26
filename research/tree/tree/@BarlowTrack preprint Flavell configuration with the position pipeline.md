---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1776e-9786-7dfb-93e4-1227126be296
status: done
priority: normal
dateCreated: 2026-10-25T07:25
dateModified: 2026-10-25T07:25
completedDate: 2026-10-25
run: barlow_track@3013e57:barlow_track/scripts/scratch/paper_flavell_use_position/run.sh
data:
  - /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position
projects:
  - "[[barlowtrack]]"
---

## Description

Local single cell of round_flavell_paper_ablation: the preprint Flavell config with only `use_position: true` (global augmentation before cropping + ViSNet add fusion; emb 64, batchnorm projector, 11,900 steps like the preprint), code at bc9ff96, seed 0; scored with eval_accuracy.py --source project against the NWB GT at descriptor_stage auto (fused) and backbone. Run r261025-2444559, 3.5 h of training.
