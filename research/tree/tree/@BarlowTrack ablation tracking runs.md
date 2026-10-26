---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1659f-a7ae-76f5-a111-e22a1e453ec7
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
completedDate: 2025-11-12
run: barlow_track@987eca2:barlow_track/scripts/paper_scripts/paper_make_projects_and_run_pipeline.py
data:
  - /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects/
projects:
  - "[[barlowtrack]]"
---

## Description

Turns each folder of trained networks into tracked projects (label propagation) per lab: untrained_<lab>, only_new_loss, only_original_loss, training_data_sweep_<lab>, augmentation_sweep_<lab>, inverse_augmentation_sweep(_only_original_loss)_<lab>, hyperparameter_sweep_leifer, loss_balance_sweep_<lab>. The tracking half of Fig. 3i, Fig. S1e-f and the loss-balance sweep of section 2.5 (the list at the tag is the last one run, loss_balance_sweep_{flavell,zimmer}; the rest are commented out).

Outputs not rerun here. Cited at tag v0.1.0 (987eca2, 2025-10-29); code is identical to the last commit before the preprint (7a555d5), but the paper's runs may predate the tag, so the exact code version behind the figure is uncertain.
