---
nodeTypeId: node_ygg_dataset
nodeInstanceId: 01a1659e-d266-7ac2-9e22-7357e7f0d019
status: done
priority: normal
dateCreated: 2026-10-22T23:57:49+02:00
dateModified: 2026-10-23T00:05:40+02:00
location: /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/samuel_data/153.nwb
n_animals: 1
n_recordings: 1
projects:
  - "[[barlowtrack]]"
completedDate: 2026-10-23
---

## Description

A Samuel lab recording (NWB, 1331 frames, 98 named neurons) used as a fourth ground-truth set in the revision benchmarks (tag label `samuel`). Not in the preprint. Presumably from the mating / multi-animal imaging of Susoy et al. 2021 (paper ref. 24), which the Discussion names as a harder setting; the repo does not say which experiment it is.

## Provenance

docs/accuracy_evaluation.md (path given as `.../barlow_track_paper/samuel_data/153.nwb`; the `barlow_track_paper` root is the one the paper scripts use) and AGENTS.md. Untrained-network weights `TrainedBarlow/untrained_samuel` exist from the paper period (paper_make_projects_and_run_pipeline.py), but no Samuel result is in the preprint.
