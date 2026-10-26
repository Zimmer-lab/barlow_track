---
nodeTypeId: node_ygg_dataset
nodeInstanceId: 01a1659e-d115-7544-893c-617d0401f2ca
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2026-10-23T00:05:40+02:00
location: /lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/
n_animals: 3
n_recordings: 3
projects:
  - "[[barlowtrack]]"
completedDate: 2026-10-23
---

## Description

Freely moving young adult *C. elegans* (strain ZIM2165, GCaMP7b with an evenly expressed nuclear reference), whole-head spinning-disk volumes at 0.35x0.35x1.5 µm/pixel, 3.5 volumes/s, 1667 volumes per recording, 60-130 ground-truth objects (paper Table 1). Segmentation by one pretrained StarDist model (≈10 partially annotated volumes); semi-automatic ground truth made with the tracklet GUI of Fieseler et al. 2025 (paper ref. 10). Only background subtraction and rigid z-slice alignment applied; 2 of 24 z-slices (flyback) dropped.

Projects (wbfm project format), as listed in barlow_track@987eca2:barlow_track/scripts/paper_scripts/paper_make_generalization_projects.py:
- `paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format` (the default GT for `eval_accuracy.py --lab zimmer`; 131 finished neurons, 1667 frames per docs/accuracy_evaluation.md)
- `paper_data/2022-11-23_worm11_updated_format`
- `paper_data/ZIM2165_Gcamp7b_worm1-2022-12-10_updated_format`

Benchmark tag label `zimmer_1128` in the sweep driver (AGENTS.md).

## Provenance

Paper Table 1 and Methods 4.1; paths from the paper scripts and docs/accuracy_evaluation.md (config lacks `exposure_time`; working copies patch it to 12). Recordings from Fieseler et al. 2025 (bioRxiv 2025.03).
