---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1776f-7c2f-7f00-a63c-ad800103da14
status: done
priority: normal
dateCreated: 2026-10-24T01:13
dateModified: 2026-10-26T09:59:36+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-10-26
---

## Purpose

Check that the current scoring reproduces the preprint's Flavell numbers when given the preprint's own embeddings, and which code changes since v0.1.0 move them.

## Method

Factorial on the saved trained (baseline) and untrained trial_0 embeddings, scored against the NWB GT (runs r261023-2165243 and r261023-2158169): A v0.1.0 code + saved SVD features + 100 seeds; B HEAD code; C HEAD + TruncatedSVD (U*S); D C with 25 seeds. Seed 0. Crop normalization checked separately on 30 frames with the paper baseline checkpoint.

## Results

Trained: stored paper tracks 0.9683 (five trials 0.963-0.968); A 0.9680, B 0.9668, C 0.9586, D 0.9584. Untrained: stored 0.6961; A 0.6994, B 0.7070, C 0.7249, D 0.7246. Saved features are u[:, :50] of the raw 64-d embedding (s1 = 2140 vs s2 = 11, so U*S is dominated by one direction). Crop normalization changed after the preprint, from percentiles (5, 100) to (5, 99.5) (d1ff47a, #136): (5, 100) reproduces the saved embedding (cosine median 1.000), (5, 99.5) gives 0.986.

## Conclusion

The harness reproduces the preprint's stored Flavell accuracies (0.968 trained, within 0.3 pt untrained). Label-propagation changes since v0.1.0 and the seed count (100 vs 25) are neutral; eval_accuracy's U*S SVD costs about 0.8 pt on trained features (+1.8 untrained), and preprint checkpoints scored with today's (5, 99.5) normalization get a train/eval mismatch.

## Caveats

One dataset (Flavell), one tracking seed per cell. Uses the preprint's segmentation detections, so it says nothing about the NWB input path.
