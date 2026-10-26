---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1659f-a73f-71f3-a51d-62d1c7865355
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
completedDate: 2025-11-12
run: barlow_track@987eca2:barlow_track/scripts/paper_scripts/paper_run_parameter_sweeps.py
projects:
  - "[[barlowtrack]]"
---

## Description

Runs optimize_hyperparameters as direct grids on TrainedBarlow/training_data_sweep_{leifer,zimmer,flavell} (number of training volumes, Fig. 3i). Commented entries show the same driver ran the one-at-a-time augmentation sweeps (augmentation_sweep_leifer, inverse_augmentation_sweep_leifer) and the loss-term ablations (only_original_loss_leifer, only_new_loss_leifer) of Fig. S1e-f.

Outputs not rerun here. Cited at tag v0.1.0 (987eca2, 2025-10-29); code is identical to the last commit before the preprint (7a555d5), but the paper's runs may predate the tag, so the exact code version behind the figure is uncertain. The sweep templates live on the cluster under TrainedBarlow/<sweep>/hyperparameter_search_template.yaml, not in the repository.
