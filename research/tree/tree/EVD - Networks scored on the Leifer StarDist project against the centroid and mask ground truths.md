---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1a5c0-7821-7264-95a5-664328dec207
status: in-progress
priority: normal
dateCreated: 2026-11-04T09:50:16+01:00
dateModified: 2026-11-04T09:50:16+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Conclusion
In progress. So far every network scores 0.935-0.941 against the mask GT and 0.937-0.948 against the centroid GT: the clicks the mask rule adds are harder (about 0.84 accuracy on them), so the centroid GT flatters scores by about one point. Training on the StarDist segmentation itself gains at most 0.3 points over the NeRVE-trained networks, within seed spread. Treated as a slightly harder benchmark than the NeRVE-object Leifer scores (about 0.96).

## Method
`eval_accuracy --mode trained --source project`, project = the StarDist project for both GTs (one embedding per network, so only the GT differs), label propagation 25 seeds at seed 0, embedding and tracking on CPU. Accuracy over GT cells; misses and mismatches in brackets.

Local sweep, `score_reseg.py` (d7b6519; runs r261103-77928 centroid GT, r261104-2000932 mask GT); results `~/.cache/barlow_scratch/train/leifer_reseg/scores.jsonl` and `scores_maskgt.jsonl`. Sweep variants retrain round_bayes_wk2_leifer_reexport trial 57 on the StarDist project (`run_pilot.py`), one change each.
| Network | Centroid GT (103,052 cells) | Mask GT (113,499 cells) |
|---|---|---|
| wk2 trial 57, trained on NeRVE (cluster) | 0.9448 (2873 / 2818) | 0.9351 (1258 / 6107) |
| trial 57 retrained locally on NeRVE | 0.9464 (2847 / 2679) | 0.9372 (1279 / 5847) |
| base: trial 57 config on StarDist | 0.9476 (2971 / 2430) | 0.9385 (1183 / 5800) |
| lambd0 (lambd_obj 0) | 0.9366 (3629 / 2908) | 0.9351 (966 / 6402) |
| lambd0_nf400 (lambd_obj 0, 400 frames) | 0.9416 (3672 / 2343) | 0.9405 (1024 / 5725) |
| lambd1, nf400, lambd1_nf400 | pending | pending |

Older networks, benchmark `run_trials_on_all_ground_truth.py --labs leifer_stardist leifer_stardist_maskgt` (9a1d725; r261104-4089020, r261104-4089023), results in `wbfm_tracker_baseline/leifer_resegment/benchmark/<round>.jsonl`: wk2 Leifer trials 57 74 65 50 63 37 76 14 9 24 (spread over the NeRVE ranking), wk1 Leifer trial 1, snap1 Leifer trial 0, wk2 Zimmer 1128 trial 0, wk2 Samuel 137 trial 63. Pending.

## Caveats
- One seed per network; seed spread on Samuel 137 was several points, so differences under about a point are noise (lambd0_nf400 vs base flips sign between the two GTs).
- Both GTs exclude merges and coincident clicks, so neither scores the hardest neighbours.
- The trial 9 centroid cell of the benchmark failed on a transient CIFS read error and needs one rerun.
