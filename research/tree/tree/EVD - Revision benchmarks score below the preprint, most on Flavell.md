---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-125d-7c24-af34-0575ef013139
status: done
priority: normal
dateCreated: 2026-10-19
dateModified: 2026-10-19
confidence: low
completedDate: 2026-10-19
outcome: ambiguous
projects:
  - "[[barlowtrack]]"
---

## Purpose

Compare revision-era trained networks with the preprint's accuracies.

## Method

(1) Trained image-only network (barlow_track_revisions/config_files/initial_position_embedding/trial_4, emb 1024, crop 4x64x64), label propagation with 25 seeds (docs/accuracy_evaluation.md). (2) Oct 2026 Bayesian sweep rounds scored by run_trials_on_all_ground_truth.py with `eval_accuracy.py --mode trained`, summarized in AGENTS.md.

## Results

(1) Zimmer 0.887, Flavell 0.791, Leifer 0.943, Samuel 0.827. (2) Best ViSNet trials: Zimmer .949, Leifer .961, Samuel .950; Flavell flat at .62-.73 across all architectures. Preprint: Zimmer 97.3%, Flavell 96.7%, Leifer 97.4%.

## Conclusion

No revision-era network has yet matched the preprint's accuracies: the best are 1-2 points lower on Zimmer and Leifer and about 25 points lower on Flavell; why is unresolved.

## Caveats

Not like for like: different architectures, embedding dims and crops (trial_4 uses z 4; Flavell needs z 16 per the paper), 25 vs 100 seeds, Flavell GT now read from an NWB with 163 named neurons rather than the project GT, Leifer evaluated from GT-centred crops, evaluation code changed since the preprint. The sweep numbers come from a git-ignored summary (AGENTS.md); the cluster records were not checked. Outcome ambiguous until the preprint's baseline networks are re-scored with the same pipeline.
