---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1659f-a5fc-70cd-a56e-800db9da6aba
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
completedDate: 2025-11-12
run: barlow_track@987eca2:barlow_track/utils/utils_paper.py
projects:
  - "[[barlowtrack]]"
---

## Description

Paper helpers: plot_hist_and_accuracy (histogram of track coverage with per-track accuracy, in the paper's Zimmer/Leifer/Flavell colours; Fig. 3d-f), calculate_track_metrics (matches BarlowTrack tracks to GT on raw_segmentation_id and computes accuracy), calculate_distance_diff and calculate_nearest_neighbor_distance (inter-volume displacement and nearest-neighbour distance; Fig. S1a-c, and plausibly Fig. 1d). The notebook that calls them and saves the panels is not in the repository.

Outputs not rerun here. Cited at tag v0.1.0 (987eca2, 2025-10-29); code is identical to the last commit before the preprint (7a555d5), but the paper's runs may predate the tag, so the exact code version behind the figure is uncertain.
