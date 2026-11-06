---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b194-71df-7db7-a734-61eae35cfb3b
status: done
priority: normal
dateCreated: 2026-11-06T16:57:38+01:00
dateModified: 2026-11-06T16:57:38+01:00
confidence: medium
completedDate: 2026-11-06
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find out why fewer propagation layers can raise accuracy: do long walks spread wrong labels, or do fewer layers help by leaving far objects unlabelled? Also measure whether propagated mass, top-1 probability or hop distance tells correct per-run labels from wrong ones. The user recalls, from building the tracker, that low mass did not predict correctness and that dropping low-mass labels hurt the final accuracy.

## Method

barlow_track/scripts/scratch/labelprop_opt/diag_layers_hops.py (barlow_track@5094974), run r261106-416617. Setup: the 4 dev cases (S137 t50, S184 t27, Z1210 t36, F0330-02 t61), the base graph and the first 5 seed volumes of the tracker's seed list at RNG seed 0. For each seed volume, labels are propagated one layer at a time with hard clamping (the tracker) and with spreading at alpha 0.9 and 0.97. At L = 1 … 300 the top-1 label of every GT object outside the seed volume is scored against the GT neuron of the seed object that owns the label, using the tracker's 1.1/C labelling rule. Objects whose neuron is present in the seed volume are scored separately from those whose neuron is absent. Confidence is measured as the ROC AUC of mass (row sum), top-1 probability and minus hop distance for a correct label, and by what dropping the lowest 20% of labels removes. CPU only.

## Results

Per-run accuracy on GT objects whose neuron is present in the seed volume, at L = 30 / 100 / 300 (mean of 5 runs):

| case | L=30 | L=100 | L=300 | share present | coverage |
|---|---|---|---|---|---|
| S137 t50 | .654 | .632 | .585 | .84 | 1.00 |
| S184 t27 | .839 | .821 | .789 | .98 | 1.00 |
| Z1210 t36 | .859 | .858 | .857 | .94 | .94 |
| F0330-02 t61 | .837 | .837 | .832 | .90 | 1.00 |

- Coverage is complete by L = 20 on every case except Zimmer. There, 6% of GT objects are never reached from a seed volume, because the kNN graph has pieces with no path to it; on Flavell it is about 0.3%.
- Objects whose neuron is absent from the seed volume are labelled, and therefore wrong, in 93-100% of cases.
- Spreading at alpha 0.9 levels off at its L = 30 value and does not fall.
- Accuracy does not depend on |t − t_seed|.

AUC for a correct per-run label (clamp, L = 30). Dropping the lowest 20% by top-1 probability would remove this share of the wrong labels and of the correct ones:

| case | mass | top-1 prob | −hops | wrong removed | correct removed |
|---|---|---|---|---|---|
| S137 | .64 | .81 | .60 | .38 | .08 |
| S184 | .64 | .90 | .61 | .66 | .11 |
| Z1210 | .61 | .75 | .59 | .50 | .15 |
| F0330 | .64 | .82 | .61 | .51 | .11 |

Mass and hops identify absent-neuron objects somewhat better than they identify correctness (AUC .63-.79).

## Conclusion

Propagating past about 30 layers spreads wrong labels on the two Samuel recordings (GT objects only): -2.2 and -1.8 pt per run at 100 layers, and -6.9 and -5.0 pt at 300. On Zimmer and Flavell, which have distractors, per-run accuracy is flat from 20 layers on. Fewer layers acts through less bleeding, not abstention, so its benefit should be Samuel-specific. Propagated mass is a weak predictor of a correct label on all four recordings (AUC .61-.64), as the user remembered. Top-1 probability is clearly better (AUC .75-.90).

## Caveats

These are per-run labels before the sync. The sync repairs much of the per-run error (S137 t50 tracks at .889), and in the partial batch 1 screen, dropping low-probability labels before the sync (`thr03`, -0.32 pt) or weighting votes by probability (`probw`, -0.28 pt) lowers final accuracy. Per-run predictability therefore does not imply a final-accuracy gain. One network per recording, 5 seed runs each, RNG seed 0. Leifer is held out and not measured.
