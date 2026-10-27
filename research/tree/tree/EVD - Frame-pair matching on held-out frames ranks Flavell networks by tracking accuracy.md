---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17ce6-a7f9-742b-81e2-24af5e8dd85a
status: done
priority: normal
dateCreated: 2026-10-27T11:27:33+01:00
dateModified: 2026-10-27T11:27:33+01:00
confidence: medium
completedDate: 2026-10-27
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find a per-epoch training metric that tracks Flavell tracking accuracy, after validation loss failed to: the elastic position run reached half the validation loss of the same cell without elastic and tracked 8 points worse.

## Method

For each checkpoint, crops of 30 frames evenly spaced over the video are embedded as in eval_accuracy --mode trained (same crops, normalization and descriptor stage); every object of one frame is assigned to one of another by linear_sum_assignment on cosine distance (untracked objects included as distractors); a pair scores the fraction of GT-named neurons present in both frames whose partner has the same name; the score is the mean over 435 pairs. 24 checkpoints, 27 network-stage pairs (auto and backbone where both were scored), against full-video NWB-GT accuracy (25 labelprop seeds).

## Results

Spearman 0.74 (p 1e-5), Pearson 0.88, n 27. Within the fused position-pipeline networks alone (accuracy 0.66-0.90) Spearman 0.57 (n 22). Legacy retrain 0.600 (accuracy 0.956); position networks 0.36-0.55; failed augmentation arms 0.25-0.40 (accuracy 0.49-0.72). Largest disagreements: emb-512 networks score higher on the proxy than emb-64 networks of equal accuracy (trial14 0.546 at 0.882, trial13 0.468 at 0.904).

## Conclusion

Matching held-out frames pairwise ranks Flavell networks by full-video tracking accuracy (Spearman 0.74 over 0.49-0.96), in 6 seconds per checkpoint instead of about 2 hours, and unlike the validation loss it does not depend on the training augmentation. It is coarse within a narrow band (0.57) and may favour larger embeddings.

## Caveats

Needs GT names, so it serves model selection and diagnosis, not users without ground truth. Training frames of these checkpoints were not excluded (about 19% overlap); in training (val_match_frames) they are. One dataset, one frame sample; noise from the 30-frame sample not measured. Pairwise matching ignores the temporal graph that label propagation uses, so absolute values (0.25-0.60) are far below tracking accuracy.
