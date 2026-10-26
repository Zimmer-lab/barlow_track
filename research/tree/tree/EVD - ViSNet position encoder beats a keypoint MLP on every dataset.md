---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-133c-7c1c-aa88-9e789ed0f1d6
status: done
priority: normal
dateCreated: 2026-10-19
dateModified: 2026-10-19
confidence: low
completedDate: 2026-10-19
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Choose a position encoder for the next BarlowTrack version.

## Method

Oct 2026 Bayesian rounds comparing ViSNet and keypoint-MLP position encoders (with attention and fusion variants), scored on four GT sets including cross-dataset transfer (summary in AGENTS.md); untrained position variants in docs/accuracy_evaluation.md.

## Results

ViSNet wins on every dataset including transfer (best: Zimmer .949, Leifer .961, Samuel .950). Embedding dim 256/512/1024 barely matters. Concat fusion is boom-or-bust, add never collapses; attention and self-depth unresolved. Flavell flat .62-.73 for all. Untrained: position variants far below image-only on Zimmer/Flavell.

## Conclusion

Among position encoders ViSNet is the better choice everywhere, but no position-aware network has yet been shown to beat the preprint's image-only accuracies.

## Caveats

Git-ignored summary; cluster records not checked. Known NaN causes (ViSNet cutoff too large, duplicate-voxel keypoints, fixed in 60e4068) may have removed trials.
