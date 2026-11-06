---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b138-3750-7d57-a813-0a4111352d01
status: done
priority: high
dateCreated: 2026-11-06T15:16:53+01:00
dateModified: 2026-11-06T16:57:38+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
outcome: ambiguous
completedDate: 2026-11-06
---

## Purpose

Find out why fewer propagation layers can raise accuracy: do long walks spread wrong labels ("bleeding"), or do fewer layers help by leaving far objects unlabelled (abstaining)? Also measure how accuracy depends on hop distance and time distance from the seed volume. Asked by the user, 2026-11-06.

## Method

barlow_track/scripts/scratch/labelprop_opt/diag_layers_hops.py (barlow_track@5094974), run r261106-416617 (a first run, r261106-398256, was restarted to add the confidence metrics). Setup: base graph and seed list at RNG seed 0, and the first 5 seed volumes of each of the 4 dev cases. For each seed volume, labels are propagated one layer at a time, with hard clamping and with spreading at alpha 0.9 and 0.97. At L = 1 … 300 the top-1 label of every GT object outside the seed volume is scored. A label is correct if the object is the same GT neuron as the seed object owning the label, using the tracker's 1.1/C labelling rule. Accuracy is reported separately for objects whose neuron is present in the seed volume (these can be right) and absent from it (always wrong if labelled). An object's hop distance is the first layer at which it receives mass. Confidence: the ROC AUC of propagated mass (row sum), top-1 probability and hop distance for a correct label and for a present neuron, plus what dropping the lowest-scoring 5-50% of labels would remove. The user recalls that low mass did not predict correctness, and that dropping low-mass labels hurt the final accuracy.

## Results

Smoke run (S137 t50, 3 seed runs; the full run is pending):
- Per-run accuracy on present neurons is 0.652 at L=30, 0.629 at L=100 and 0.577 at L=300.
- Coverage is already 1.0 by L=20 (the largest hop distance is 11).
- Spreading at alpha 0.9 levels off at 0.649.
- 19% of GT objects belong to neurons absent from the seed volume. All of them get a label, and those labels are necessarily wrong. They make up 95% of the objects at 10 or more hops.
- Accuracy is flat in |t − t_seed|.
- Confidence (2 runs, L = 30): mass is far from 1 (median 0.008; walks rarely reach a seed volume in 100 steps). It predicts a correct label weakly (AUC 0.69) and flags absent-neuron objects better (AUC 0.83). Top-1 probability predicts correctness best (AUC 0.85): dropping its lowest 20% would remove 38% of the wrong labels and 5% of the correct ones. The effect on final accuracy is tested by `thr03` in the screen.

## Conclusion

Pending the full run. The smoke points to bleeding as the cause, not abstention. It also shows that absent-neuron objects get confident-looking wrong labels in every run; hop distance would identify them.

## Caveats

The smoke covers one weak network and 3 seed runs. These are per-run labels before the sync, which repairs much of this: S137 t50 tracks at 0.888 after the sync against about 0.65 per run. Samuel has no distractors.
