---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a188f0-d5c0-720d-aaa0-eef4260c6938
status: done
priority: normal
dateCreated: 2026-10-29T19:34:07+01:00
dateModified: 2026-10-29T19:34:07+01:00
completedDate: 2026-10-29
run: barlow_track@6545296:barlow_track/scripts/scratch/round_posonly_zimmer/
projects:
  - "[[barlowtrack]]"
---

## Description

round_posonly_{zimmer,flavell,samuel} (staged at 7641f83, launched by the user 2026-10-28; README in round_posonly_flavell): position-only ViSNet (fusion position_only, keypoints only), 2 x 4 x 2 grid per dataset, one seed, 16 jobs: visnet_rbf_rescale {false, true} x visnet_cutoff_nn {0, 3, 6, 12} x global_augment.elastic_scale {0, 1}; width 64, 100 frames, object dropout 0.2, 50 epochs, objective accuracy on the trial's own training project (label propagation). Records in /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/round_posonly_*/exp_results.jsonl; Flavell scored against the NWB GT (round_posonly_flavell rescored with scratch/rescore_posonly_flavell_nwb.sh). round_posonly_samuel's rbf-on trials have no records yet (crashed evals, rescore queued in scratch/cpu_rescores/run.sh).
