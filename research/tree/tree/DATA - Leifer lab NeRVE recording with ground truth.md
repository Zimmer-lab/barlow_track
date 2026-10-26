---
nodeTypeId: node_ygg_dataset
nodeInstanceId: 01a1659e-d1f8-7eba-90df-b9a8281e3adf
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2026-10-23T00:05:40+02:00
location: /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/Leifer_NeRVE_Worm1.nwb
n_animals: 1
n_recordings: 1
projects:
  - "[[barlowtrack]]"
completedDate: 2026-10-23
---

## Description

One freely moving *C. elegans* recording from Nguyen et al. 2016 (NeRVE, paper ref. 18): 0.32x0.32x1.5 µm/pixel, 6 volumes/s, 1536 volumes, 79 ground-truth objects from semi-automated annotation (paper Table 1). Z scanned with a triangle wave and reindexed; slice timing alternates between volumes.

The NWB carries the red volume (1536, 32, 632, 600) and GT xyz. The analysed project `leifer_data/projects/leifer_project_gt` (paper scripts) can no longer be re-embedded because its `1-segmentation/metadata.pickle` is gone; current evaluation uses `eval_accuracy.py --source nwb` and matches on `raw_neuron_ind_in_list` (docs/accuracy_evaluation.md). Leifer benchmark cells are the slowest (hours).

## Provenance

Paper Table 1, Methods 4.1; docs/accuracy_evaluation.md.
