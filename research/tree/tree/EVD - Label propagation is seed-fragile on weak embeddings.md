---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a5-11f0-7c68-ae2f-505ffd10565b
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

Measure run-to-run variability of tracking on fixed embeddings.

## Method

Identical label-propagation runs on the same embeddings (docs/accuracy_evaluation.md section 5); UMAP-seed variance check with var_test.py on Samuel attention embeddings.

## Results

Samuel position-only embeddings: 0.001 vs 0.566 accuracy across identical runs, traced to an unseeded random.shuffle of seed times. UMAP-seed spread on fixed embeddings ≈±0.01. With seeding, two labelings of identical embeddings can still differ by <0.01.

## Conclusion

On weak embeddings label propagation can swing from failure to partial success between identical runs; on good embeddings run-to-run noise is about 0.01, so sub-0.01 differences are noise.

## Caveats

From the committed write-up, not rerun here. One dataset for the large swing. Seeding since added (80a2bbd, #144) but PR06 not ticked.
