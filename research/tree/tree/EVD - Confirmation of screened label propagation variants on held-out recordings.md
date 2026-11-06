---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b113-4cff-7dde-892d-b7ad89e9cfe4
status: in-progress
priority: high
dateCreated: 2026-11-06T14:36:34+01:00
dateModified: 2026-11-06T20:58:44+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Confirm the batch 1 winners, singly and in cross-stage combinations, on networks and recordings the screen did not see, before any change goes into the library tracker.

## Method

variants.py `BATCH2` (barlow_track@d8a1335): base; the six batch-1 promotions singly (svd100, seeds50, seeds100, seed_strat, temporal5, tsmooth05); neighbours svd150, temporal2, temporal10; combinations c_seeds (50 stratified seeds), c_all (svd100 + 50 stratified seeds + temporal5 + tsmooth05), c_all_s100, c_all_notemp, c_all_gh (+ gauss edges, Hungarian per run), c_all_l30 (+ 30 layers). Cases (CONFIRM_CASES) × seeds 0, 1, 2: S137 t63, S184 t32, Z1210 t52, Leifer re-export t57 (held-out lab, NWB every detection), Flavell 2022-03-24-02 t28 (held-out recording), and Leifer StarDist t57 (the same network on the StarDist re-segmentation, strict click GT; the paper's Leifer; added at 3756753 on the user's request). Runs r261106-595573, r261106-598057 (Leifer then Z1210 on the GPU, F0324 relaunched after an embedding-copy race), r261106-606174 (StarDist, CPU). Output ~/.cache/barlow_scratch/labelprop_opt/batch2_confirm.jsonl. Adopt per PLAN.md section 4 (mean ≥ +0.3 pt, paired t-test p < 0.05 over all pairs, no case mean below -0.5 pt). Then check the adopted recipe at 100 seeds on 2 cases.

## Results

Running since 2026-11-06 20:47.

## Conclusion

Pending.

## Caveats

Report the Leifer fragment columns separately; it is the case where fragmentation matters most. The StarDist case scores assigned strict clicks only (README of round_bayes_wk4_leifer_stardist_strict: multiply by 0.97826 for the all-kept headline).
