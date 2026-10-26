---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a2-ea4c-764c-bd27-60a6acc22b25
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
confidence: low
completedDate: 2025-11-12
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Record which hyperparameters the per-dataset sweeps found important.

## Method

Methods 4.3: per-dataset sweeps of learning rate, projector_final, embedding_dim, target_sz_z, target_sz_xy and lambd_obj; chosen values 1e-4, 512, 64, 8, 64, 0.67.

## Results

No strong dependence on dimensionality, loss balance or learning rate; embedding dims below 64 were worse. Small crops performed poorly; Flavell needed 16 z-slices instead of 8 (proportional to its 0.33 µm z resolution).

## Conclusion

The crop must cover the neuron and its neighbourhood in physical units (z 8 at 1.5 µm, 16 at 0.33 µm); learning rate, loss balance and embedding size above 64 hardly matter.

## Caveats

As reported in the paper; not rerun. Stated in Methods without a figure or numbers.
