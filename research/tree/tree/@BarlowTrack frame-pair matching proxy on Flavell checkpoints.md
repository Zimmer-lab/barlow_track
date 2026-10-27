---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a17ce6-a781-7f35-ba6f-b8208a3f68ca
status: done
priority: normal
dateCreated: 2026-10-27T11:27:33+01:00
dateModified: 2026-10-27T11:27:33+01:00
completedDate: 2026-10-27
run: barlow_track@db105ba:barlow_track/scripts/scratch/val_matching_on_checkpoints.py
data:
  - /home/charles/.cache/barlow_scratch/val_matching/proxy_30.jsonl
projects:
  - "[[barlowtrack]]"
---

## Description

utils/val_matching.py (db105ba) applied to 24 Flavell checkpoints with known NWB-GT accuracy: the 5 local training runs under ~/.cache/barlow_scratch/train and the 19 rescored round_flavell_paper_ablation and round_flavell_more_epochs trials. 30 frames evenly spaced over the 1600 (435 pairs), local Flavell mirror, project GT names, ~6 s per checkpoint on the GPU. Accuracies joined from eval_seed0.jsonl and rescore_nwb_gt.jsonl.
