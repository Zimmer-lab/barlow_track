---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1776e-9713-7127-bc05-4c057f35c935
status: done
priority: normal
dateCreated: 2026-10-24T07:30
dateModified: 2026-10-24T07:30
completedDate: 2026-10-24
run: barlow_track@95b93fb:barlow_track/scripts/scratch/retrain_paper_baseline_flavell/run.sh
data:
  - /home/charles/.cache/barlow_scratch/train/baseline_flavell_head_rerun
projects:
  - "[[barlowtrack]]"
---

## Description

The preprint's Flavell baseline config (TrainedBarlow/baseline_flavell/trial_0/train_config.yaml: emb 64, no position, 50 epochs, 300 frames, z 16; project_path repointed to flavell_data/projects/2025_07_01) trained with the code at bc9ff96 on the local Titan V, seed 0, 11,900 steps (23 h), then scored with eval_accuracy.py --source project against the NWB GT, 25 seeds. Runs r261024-2611790 (the first launch, r261024-2567328, failed for missing log/ and checkpoints/ folders); seed repeats 1 and 2 from the cached embedding with barlow_track@748aa70:barlow_track/scripts/scratch/retrain_paper_baseline_flavell/seeds.sh (r261025-2443993).
