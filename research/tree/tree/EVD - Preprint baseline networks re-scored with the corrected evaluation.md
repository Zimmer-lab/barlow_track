---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-7c94-70a3-a257-348b9c0d4292
status: open
priority: high
dateCreated: 2026-10-23T00:05:06+02:00
dateModified: 2026-10-26T09:59:59+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Separate code changes from network changes in the gap between preprint and revision accuracies.

## Method

Run `eval_accuracy.py --mode trained` with the preprint's baseline networks (TrainedBarlow/baseline_{zimmer,flavell,leifer}, emb 64, crop 8x64x64, Flavell z 16), label propagation with a fixed random_state (docs/accuracy_evaluation.md step 2), on the Zimmer, Flavell, Leifer and Samuel GT; compare with Table 2 and with the stored preprint tracks (stored_acc.py).

Flavell is done: [[EVD - The preprint Flavell network scores 95 to 97 percent under the current evaluation]] (0.952 project path, 0.969 aligned NWB path). Zimmer, Leifer and Samuel remain. Seed count (25 vs 100) proved neutral on Flavell, so 25 seeds suffice.

## Results

## Conclusion

## Caveats

Leifer can only be scored from NWB crops (its segmentation metadata is gone), so its preprint number cannot be reproduced exactly. Check that the Zimmer GT path is not affected by the Flavell NWB neuron-order bug (issue i261026-1; Leifer and Samuel NWBs are aligned). Score preprint checkpoints aware of the crop normalization change ((5, 100) at training, (5, 99.5) now; about 0.7 pt on Flavell).
