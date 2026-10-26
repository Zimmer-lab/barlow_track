---
nodeTypeId: node_ygg_dataset
nodeInstanceId: 01a1659e-d17b-7b62-b826-8ca81a1db951
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2026-10-23T00:05:40+02:00
location: /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/
n_animals: 5
n_recordings: 5
projects:
  - "[[barlowtrack]]"
completedDate: 2026-10-23
---

## Description

Freely moving *C. elegans* from Atanas et al. 2023 (paper ref. 9), NeuroPAL background with a heterogeneously expressed pan-neuronal reference, 0.33x0.33x0.33 µm/pixel, 1.7 volumes/s, 130-180 objects (paper Table 1 lists 3200 volumes; docs/accuracy_evaluation.md has 1600 frames for the evaluated copy). Strong nonrigid motion correction (heading mostly fixed, rare 180° flips). Segmented with the Flavell lab's U-net pipeline; objects split across several segments were dropped from the ground truth.

The "ground truth" is not manual: it is the Flavell lab's own tracking algorithm (checked by them against a stochastic GFP), marked * in Table 1.

Locations: projects `flavell_data/projects/{2025_07_01, for_Charlie2-2022-03-16-03, forCharlie_2-2022-03-24-02, forCharlie_2-2022-03-30-01, forCharlie_2-2022-03-30-02}` (paper scripts at 987eca2); GT NWB `flavell_data/images_for_charlie/flavell_data.nwb` with 163 named neurons (docs/accuracy_evaluation.md). Needs `target_sz_z: 16` crops (AGENTS.md, Methods 4.3).

## Provenance

Paper Table 1, Methods 4.1; paths from barlow_track@987eca2:barlow_track/scripts/paper_scripts/paper_make_generalization_projects.py and docs/accuracy_evaluation.md.
