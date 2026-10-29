---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a188f0-d640-7331-a3c9-8a1831652294
status: done
priority: normal
dateCreated: 2026-10-29T19:34:07+01:00
dateModified: 2026-10-29T19:34:07+01:00
completedDate: 2026-10-29
run: barlow_track@fc206be:barlow_track/scripts/scratch/round_posonly2_zimmer/
projects:
  - "[[barlowtrack]]"
---

## Description

round_posonly2_{zimmer,flavell,samuel} (staged 2026-10-29, ecd3617 / 2443724 / fc206be, launched by the user the same day; README in round_posonly2_zimmer): the fast loop's best position-only network (RBF fix on, width 512, 600 frames, 60 epochs, elastic off, keypoints only) with a 2 x 2 x 2 grid, two seeds, 16 jobs per dataset: position_augment.jitter_std {1.0, auto} x visnet_cutoff_nn {0, 6} x position_augment.dropout_p {0, 0.02}. Objective accuracy on the trial's own recording; Flavell and Samuel scored against their NWB GT objects (objective_gt), Zimmer on the updated-format 2022-11-28 project. Records in config_files/round_posonly2_*/exp_results.jsonl; first records read 2026-10-29 evening (Flavell trials 0-5, Samuel trials 0-6, no Zimmer yet).
