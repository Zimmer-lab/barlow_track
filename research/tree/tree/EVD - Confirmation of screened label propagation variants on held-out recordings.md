---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b113-4cff-7dde-892d-b7ad89e9cfe4
status: open
priority: high
dateCreated: 2026-11-06T14:36:34+01:00
dateModified: 2026-11-06T14:36:34+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Confirm the batch 1 winners, singly and in cross-stage combinations, on networks and recordings the screen did not see, before any change goes into the library tracker.

## Method

variants.py `BATCH2` (filled from batch 1's promoted variants and neighbouring values, plus speed winners) on CONFIRM_CASES × seeds 0, 1, 2 (15 pairs): S137 t63, S184 t32, Z1210 t52, Leifer re-export t57 (held-out lab) and Flavell 2022-03-24-02 t28 (held-out recording). Adopt per PLAN.md section 4. Then check the adopted recipe at 100 seeds on 2 cases.

## Results

Not started.

## Conclusion

Pending.

## Caveats

Report Leifer's fragment columns separately; it is the case where fragmentation matters most.
