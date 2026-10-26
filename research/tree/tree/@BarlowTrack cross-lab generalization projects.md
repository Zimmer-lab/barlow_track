---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1659f-a6d1-7e8c-bbfa-38e53079b480
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
completedDate: 2025-11-12
run: barlow_track@987eca2:barlow_track/scripts/paper_scripts/paper_make_generalization_projects.py
data:
  - /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects/
projects:
  - "[[barlowtrack]]"
---

## Description

For every ground-truth project of each lab (3 Zimmer, 5 Flavell, 1 Leifer), creates a project tracked with each lab's baseline network (TrainedBarlow/baseline_<lab>) and label propagation, skipping the network's own training project. These are the train-lab by test-lab cells averaged in Fig. 3h. Produces analysed projects; accuracy and the heatmap were computed elsewhere.

Outputs not rerun here. Cited at tag v0.1.0 (987eca2, 2025-10-29); code is identical to the last commit before the preprint (7a555d5), but the paper's runs may predate the tag, so the exact code version behind the figure is uncertain.
