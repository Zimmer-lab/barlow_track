---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a2-71df-7edb-8652-4f469fc5563c
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
confidence: medium
completedDate: 2025-11-12
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Check whether the self-supervised loss, available without ground truth, predicts tracking accuracy.

## Method

Fig. 3g: accuracy vs validation loss for each trained network of a hyperparameter search (lr, embedding dim, crop size), coloured by lab, with per-lab linear fits.

## Results

Negative slope in all three labs: networks at validation loss ≈0.0005 reach ≈0.95-0.98 accuracy, those near 0.002 drop to 0.6-0.9 (a few Leifer points near 0.4). Spread is large at intermediate loss.

## Conclusion

Across one hyperparameter search per lab, lower validation loss went with higher tracking accuracy, so the loss is a usable but noisy proxy within that search.

## Caveats

As reported in the paper; not rerun. Number of trials per lab not stated; no correlation statistic. Sweeps that varied crop size change what the loss measures. Later revision sweeps found the relation holds only within narrow regimes (see opposing Evidence).
