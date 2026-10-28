---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1821c-8994-73ea-a998-f016fe567724
status: open
priority: high
dateCreated: 2026-10-28T11:44:30+01:00
dateModified: 2026-10-28T11:44:30+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find whether a position-only network can learn neuron identity, and what has kept the ViSNet position branch from learning it ([[EVD - Trained position branches carry no more neuron identity than an untrained one on Flavell]]): the RBF scale problem (issue i261028-2), the neighbourhood size, or too little deformation between the training views. Also the first position-only baseline, which the paper revision needs.

## Method

Planned, staged for the user to launch: round_posonly_{zimmer,flavell,samuel} (barlow_track/scripts/scratch/, README in round_posonly_flavell). Per dataset a 2 x 4 x 2 grid, one seed, 16 jobs: visnet_rbf_rescale {false, true} x visnet_cutoff_nn {0 = absolute 0.35, 3, 6, 12 NN} x global_augment.elastic_scale {0, 1 = field matched to real lag-100 motion}. fusion position_only, keypoint-only global augmentation, 50 epochs, objective accuracy on the training project, frame-pair proxy each epoch. Code barlow_track@7641f83.

## Results

## Conclusion

## Caveats
