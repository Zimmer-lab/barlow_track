---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a17b4d-311c-73cc-8631-0ce07cd012fd
status: done
priority: normal
dateCreated: 2026-10-27T04:00:18+01:00
dateModified: 2026-10-27T04:00:18+01:00
completedDate: 2026-10-27
run: barlow_track@ee2a06a:barlow_track/scripts/scratch/rescore_flavell_nwb_gt.sh
data:
  - /home/charles/.cache/barlow_scratch/rescore
projects:
  - "[[barlowtrack]]"
---

## Description

rescore_flavell_nwb_gt.sh re-tracks round_flavell* sweep cells from each trial's cached embedding (emb_cache, copied locally) and scores them against the Flavell NWB GT the preprint used (local mirror of flavell_data.nwb), with eval_accuracy --source project --skip_embed, CPU tracking, labelprop seed 0. The sweeps' own records score against the project GT. Run r261026-1606883: round_flavell_paper_ablation trials 4-7 and 12-14, round_flavell_more_epochs trials 17-26, 28, 29. Results in rescore/<round>/rescore_nwb_gt.jsonl, logs in rescore/<round>/logs/.
