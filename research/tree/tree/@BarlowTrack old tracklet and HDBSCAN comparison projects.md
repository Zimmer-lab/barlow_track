---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1659f-a88c-7666-b4d5-5c7d02490f0e
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
completedDate: 2025-11-12
run: barlow_track@987eca2:barlow_track/scripts/paper_scripts/paper_run_old_analysis_methods.py
projects:
  - "[[barlowtrack]]"
---

## Description

Copies each lab's analysed baseline project into baseline_old_tracklets and baseline_hdbscan, so the same segmentation is re-tracked with the authors' earlier tracklet method and with HDBSCAN on the same BarlowTrack network (Fig. S1d, Table 2 HDBSCAN and Zimmer Tracklet rows). The companion paper_update_config_files.py at the same commit rewrites the configs (use_barlow_tracker False for tracklets; barlow_tracker.tracking_mode global for HDBSCAN).

Outputs not rerun here. Cited at tag v0.1.0 (987eca2, 2025-10-29); code is identical to the last commit before the preprint (7a555d5), but the paper's runs may predate the tag, so the exact code version behind the figure is uncertain.
