---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a1b113-01f7-7bbf-b54c-3aa1a08e05fa
status: in-progress
priority: high
dateCreated: 2026-11-06T14:36:15+01:00
dateModified: 2026-11-06T14:36:15+01:00
projects:
  - "[[barlowtrack]]"
---

## Question

Which changes to `track_using_label_propagation_clusterer` (kNN graph, seed volumes, propagation, spectral sync, post-processing) raise tracking accuracy on saved sweep embeddings without retraining, within the current runtime and memory?

## Context

Plan, harness and variant list: barlow_track/scripts/scratch/labelprop_opt/PLAN.md (written 2026-11-06; the tracker code it describes is unchanged at ec7d6ed). The harness splits the tracker into stages that reproduce it exactly (100-150-volume smokes on all four scoring paths, and 0.9333 vs the recorded 0.9338 on the full S137 t63 video).

What is known so far:
- Re-tracking the same network spreads about 0.8 pt across RNG seeds. That spread comes almost entirely from the choice of the 25 seed volumes, not from the NNDescent graph (99.7% of edges shared across seeds).
- The errors differ by lab. On Samuel (GT objects only) they are swaps. On recordings with distractors (Zimmer, Flavell, Leifer), much of the error is track fragmentation: 9.7 pt on Leifer in the first 100 volumes, with a perfect-merge bound of +8.9 pt. The fragmentation sub-question follows from this one.
- Section 2 of the plan lists traps in the current code: k=20 is 19 real neighbours, the "symmetric" normalisation is uniform 1/(k+1), the sync discards the propagation probabilities, and `alpha`/`tau` are dead parameters.
- Earlier whitening results conflict: [[EVD - Whitened SVD costs label propagation 1.4 to 1.7 points on Samuel 153]] against +0.8 pt on the preprint Flavell net (notebook). The `whiten` variant retests it.

Decision rules (plan section 4): promote from the screen if the mean paired delta is at least +0.3 pt and the variant wins at least 6 of 8 pairs. Adopt after confirmation if the mean is at least +0.3 pt, p < 0.05, and no case's mean is below -0.5 pt. Any adopted change alters recorded accuracies and the GT-free proxy references.
