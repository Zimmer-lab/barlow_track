---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a18539-33f2-7407-98bc-364afc22c1db
status: done
priority: normal
dateCreated: 2026-10-29T02:14:41+01:00
dateModified: 2026-10-29T02:14:41+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-10-29
---

## Conclusion
Tracking the same embeddings in the whitened SVD space that production tracking uses, instead of the scaled one the evaluation uses, lowers Samuel 153 accuracy from 0.951 to 0.937 (round_bayes_samuel trial 0) and from 0.948 to 0.931 (round_visnet_full trial 27). Production should set `svd_whiten: false` (opt-in key since 482dcfb) so that its accuracy matches what the evaluation reports. Matching tracks to ground truth with `gt_valid` instead of the default gives identical scores here, as expected when the segmentation holds only ground-truth neurons.

## Method
Run r261028-3231600: cached backbone embeddings, label propagation with 25 seeds, NWB GT objects.

## Caveats
One recording (Samuel 153) and oracle detections; not checked on Zimmer, Leifer or Flavell, where gt_valid could differ because segmentations hold non-GT objects. No seed repeats; the 1.4-1.7 point drop is consistent across both networks.
