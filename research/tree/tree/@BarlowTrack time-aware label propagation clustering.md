---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1659f-a58e-76ec-bcf8-2442e86b0c0c
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
completedDate: 2025-11-12
run: barlow_track@987eca2:barlow_track/utils/utils_label_propagation.py
projects:
  - "[[barlowtrack]]"
---

## Description

Algorithm 1: nearest-neighbour graph in embedding space, clamped label propagation from seed time points, one label per time point, and alignment of the seeded labelings (with utils/utils_spectral_relabeling.py). Paper parameters: n_neighbors 20, n_layers 100, top_k 2, 100 seeds, confidence threshold 0.1. Used for the "Label Propagation" rows of Table 2 and the Baseline of Fig. S1d.

Outputs not rerun here. Cited at tag v0.1.0 (987eca2, 2025-10-29); code is identical to the last commit before the preprint (7a555d5), but the paper's runs may predate the tag, so the exact code version behind the figure is uncertain. The post-preprint PR02 plan (docs/pr_plan) lists edge-case bugs in this module (disconnected nodes, label 0, alignment with 0-1 labelings).
