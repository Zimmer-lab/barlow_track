---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1659f-a666-7ee1-8e47-402e88dfee7e
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
completedDate: 2025-11-12
run: barlow_track@987eca2:barlow_track/scripts/multiproject_scripts/analyze_project_with_all_barlow_trials.py
projects:
  - "[[barlowtrack]]"
---

## Description

Collects validation loss (log/stats.json) and tracking accuracy for every trial of a hyperparameter search on one ground-truth project (utils/utils_trials.py build_final_dict) and scatters them. The single-project form of Fig. 3g; the per-lab, colour-coded version with fit lines in the paper is not in the repository.

Outputs not rerun here. Cited at tag v0.1.0 (987eca2, 2025-10-29); code is identical to the last commit before the preprint (7a555d5), but the paper's runs may predate the tag, so the exact code version behind the figure is uncertain.
