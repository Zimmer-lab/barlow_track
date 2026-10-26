---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a165a2-177f-7413-8bb8-ee0a9cf6301f
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

Measure what the new object-space (OS) loss term, which decorrelates the embeddings of all objects in one time point, adds over the original Barlow Twins redundancy-reduction (RR) term.

## Method

Fig. S1e: networks trained with only the original RR loss ("Original loss term") or only the OS loss ("New loss term") vs the baseline mixture (lambd_obj 0.67), tracked with label propagation on the three benchmark labs. Section 2.5: a sweep over the balance lambd_obj (loss_balance_sweep_{zimmer,flavell} folders).

## Results

Both single-term networks sit at baseline accuracy on all three labs (Fig. S1e, all ≈0.95-0.97). The loss-balance sweep put all accuracies near baseline, i.e. sensitivity to the balance is effectively zero. Methods 4.3 likewise reports no strong dependence on the loss balance.

## Conclusion

On three benchmark labs, the OS term alone, the RR term alone and their mixture train equally accurate trackers: the new term can replace the original one without penalty but does not improve accuracy.

## Caveats

As reported in the paper; not rerun. The loss-balance sweep values and plot are not shown in the paper; the text still carries a placeholder ("a balance of ?? is optimal", co-author markup). Accuracy near the ceiling (≈97%) may hide differences; no harder dataset or reduced-data regime was tested.
