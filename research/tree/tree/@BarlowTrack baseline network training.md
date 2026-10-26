---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1659f-a4ba-7c6a-b6e1-e52bb1c29f4f
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
completedDate: 2025-11-12
run: barlow_track@987eca2:barlow_track/scripts/train_barlow_clusterer.py
data:
  - /lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/
projects:
  - "[[barlowtrack]]"
---

## Description

Trains a BarlowTrack encoder on one project's crops (`python barlow_track/scripts/train_barlow_clusterer.py -p train_config.yaml`; outputs resnet50.pth + args.pickle). The default config, barlow_track/barlow_project_template/train_config.yaml at the same commit, matches paper Methods 4.3 exactly: lr 1e-4, projector_final 512, embedding_dim 64, target_sz_z 8 (16 for Flavell), target_sz_xy 64, lambd_obj 0.67, 200 frames, 50 epochs, augmentations Affine/flip/Elastic p=1, Blur and Noise p=0.1. Produces the baseline networks (TrainedBarlow/baseline_{zimmer,flavell,leifer}) behind Table 2, Fig. 3 and S1.

Outputs not rerun here. Cited at tag v0.1.0 (987eca2, 2025-10-29); code is identical to the last commit before the preprint (7a555d5), but the paper's runs may predate the tag, so the exact code version behind the figure is uncertain.
