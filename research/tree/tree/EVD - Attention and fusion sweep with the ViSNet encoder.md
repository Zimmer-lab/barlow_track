---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-7e41-7a48-8b24-e1994767d18e
status: open
priority: normal
dateCreated: 2026-10-23T00:05:07+02:00
dateModified: 2026-10-23T00:05:07+02:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Settle the open position-encoder choices: attention, self-attention depth, fusion (concat vs add vs position only).

## Method

Bayesian or grid rounds with optimize_hyperparameters.py (`objective: accuracy`), benchmarked on all four GT sets with run_trials_on_all_ground_truth.py (`--no_fail_fast`, resume), including the position-only ablation (`use_position: true, fusion: position_only`).

## Results

## Conclusion

## Caveats

Records with the old add-fusion Leifer script are not comparable (docs/accuracy_evaluation.md).
