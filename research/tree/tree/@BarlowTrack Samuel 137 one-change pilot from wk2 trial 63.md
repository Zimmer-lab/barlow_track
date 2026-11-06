---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a19fd4-db64-7ac6-a1b6-a66ae98941db
status: done
priority: normal
dateCreated: 2026-11-03T06:14:49+01:00
dateModified: 2026-11-03T06:14:49+01:00
completedDate: 2026-11-03
run: barlow_track@2854f93:barlow_track/scripts/scratch/s137_pilot/
projects:
  - "[[barlowtrack]]"
---

## Description

run_pilot.py copies round_bayes_wk2_samuel_137 trial 63's train_config.yaml (sweep score 0.931), changes one thing per variant, trains locally on the Titan V (num_workers 4 for every variant) and scores each checkpoint exactly as the sweep objective does: resolve_objective on the round's template, NWB source with GT objects, 137.nwb GT, CPU embedding, label propagation with 25 seeds at seed 0. Variants: base, lambd0 (lambd_obj 0, feature-space term only), lambd1 (lambd_obj 1, object-space term only), reaug (reaugment_each_epoch true), each at training seeds 3464164355 (trial 63's own), 2 and 3. lp_seeds.py (32ce9cf) re-tracks the cached embeddings of base and base_seed2 at label propagation seeds 1 and 2, and at seed 0 with 100 seeds. Runs r261102-365990, r261102-2656039, r261103-1931191, r261103-732047 (reaug_seed3 rerun after its first attempt crashed in the validation plot, fixed in bc3c1bf), r261102-77194 (label propagation seeds). Outputs ~/.cache/barlow_scratch/train/s137_pilot/results.jsonl and lp_seeds.jsonl.
