---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1659f-a526-7a0e-8e5c-d02cf9ae131a
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
completedDate: 2025-11-12
run: barlow_track@987eca2:barlow_track/utils/barlow.py
projects:
  - "[[barlowtrack]]"
---

## Description

BarlowTwins3d: the encoder, projector and loss of Fig. 2b and Eqs. 1-4. `loss = (1 - lambd_obj) * loss_original + lambd_obj * loss_transpose`, where loss_transpose is the object-space term (correlation across features between objects of one time point; one volume is one batch). Also holds the augmentation Transform and load_barlow_model.

Outputs not rerun here. Cited at tag v0.1.0 (987eca2, 2025-10-29); code is identical to the last commit before the preprint (7a555d5), but the paper's runs may predate the tag, so the exact code version behind the figure is uncertain.
