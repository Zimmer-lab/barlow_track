---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1845a-8bdf-72fc-81cf-f6172715d3ba
status: done
priority: normal
dateCreated: 2026-10-28T22:11:28+01:00
dateModified: 2026-10-28T22:11:46+01:00
completedDate: 2026-10-28
run: barlow_track@d2c59c3:barlow_track/scripts/scratch/samuel_check/
projects:
  - "[[barlowtrack]]"
---

## Description

eval_accuracy --mode trained (or --mode image) --source nwb --nwb_objects gt, label propagation with 25 seeds tracked on CPU, scored with the current code, on Samuel 137, 153 and 184-004, for two networks: round_bayes_samuel trial 0 (revision, trained on the 153 project 2025_07_30_gt_traces) and the paper-era hyperparameter_search_harvard_153_flip trial 8 (trained on 153.nwb), plus the untrained image baseline. On 153 both networks were also tracked with the global clusterer (SVD, UMAP, HDBSCAN) from the same saved embeddings (--skip_embed --cluster global). stored_acc_samuel.py scores the 88 stored paper-era track sets of the 153 project (old WormTsneTracker, HDBSCAN windows, Aug 2025). The first queue (run_queue.sh, pid 3202498) was started by a subagent and not logged as a run; the follow-up is r261028-3286550 (run_queue_more.sh). Outputs ~/.cache/barlow_scratch/runs/samuel_check/results.jsonl and stored_acc.jsonl.
