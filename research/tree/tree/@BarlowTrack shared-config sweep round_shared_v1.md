---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1c054-48c8-7c2a-baef-b5d070c39c5e
status: done
priority: normal
dateCreated: 2026-11-06
dateModified: 2026-11-06
completedDate: 2026-11-06
run: barlow_track@5695edc:barlow_track/scripts/optimize_shared_config.py
run_start: 2026-11-06
run_end: 2026-11-08
data:
  - /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/round_shared_v1/
projects:
  - "[[barlowtrack]]"
---

## Description

One Ax search for a single training recipe shared by every recording (cluster job 7910480, run r261106-j7910480, dispatched 2026-11-06 08:49, ended 2026-11-08 10:55). Each of 60 trials trains one network with the same config and scores it on 6 tuning recordings (zimmer_1210, zimmer_1123, flavell_2022033002, flavell_2022032402, samuel_137, samuel_184), 360 GPU jobs, 4 trials at a time. Objective: lower quartile over recordings of (agree_mean, the GT-free label-propagation consistency, minus that recording's best wk2/wk3 agree_mean). Accuracy is recorded in every cell. Held out and never trained or scored: zimmer_1128, flavell 2025-07-01, flavell_2022031603, flavell_2022033001, samuel 153, leifer_stardist_strict. Search space: wk3's grid without fusion (add), num_frames up to 1000. Dispatcher optimize_shared_config.py (370f72c, 5695edc; the job may have run later code, the README does not say). Output: round_shared_v1/shared_results.jsonl, best_parameters.yaml, README.md.
