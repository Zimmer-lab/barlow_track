---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-117b-7e2c-bab5-d420a538d8c3
status: done
priority: normal
dateCreated: 2026-09-30
dateModified: 2026-09-30
confidence: medium
completedDate: 2026-09-30
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Check that the revision evaluation pipeline reproduces the preprint before comparing new architectures with it.

## Method

eval_accuracy.py, `--mode image`, the paper's untrained networks (TrainedBarlow/untrained_<lab>, emb 64, crop 8x64x64), label propagation with 25 seeds, compared with the accuracy of the paper's stored tracks (stored_acc.py). CPU-only machine, September 2026; written up in docs/accuracy_evaluation.md section 5.

## Results

Untrained image networks: Flavell 0.696 (paper stored 0.696, equal to 4 decimals), Zimmer 0.653 (stored 0.618), Leifer 0.892 (stored 0.872), Samuel 0.769 (not in the paper). Untrained position variants are much worse on Zimmer/Flavell (position-only 0.02/0.12).

## Conclusion

The revision pipeline reproduces the preprint's untrained Flavell accuracy exactly and is within 2-4 points on Zimmer and Leifer, so the untrained baseline of Fig. S1e stands.

## Caveats

From the committed write-up, not rerun here. 25 seeds vs the paper's 100; Leifer evaluated from NWB crops around GT positions, not the original segmentation.
