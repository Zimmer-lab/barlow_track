---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b113-4cff-7dde-892d-b7ad89e9cfe4
status: in-progress
priority: high
dateCreated: 2026-11-06T14:36:34+01:00
dateModified: 2026-11-07T01:11:46+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Confirm the batch 1 winners, singly and in cross-stage combinations, on networks and recordings the screen did not see, before any change goes into the library tracker.

## Method

variants.py `BATCH2` (barlow_track@d8a1335, chain variants b5e20d2): base; the six batch-1 promotions singly (svd100, seeds50, seeds100, seed_strat, temporal5, tsmooth05); neighbours svd150, temporal2, temporal10; combinations c_seeds (50 stratified seeds), c_all (svd100 + 50 stratified seeds + temporal5 + tsmooth05), c_all_s100, c_all_notemp, c_all_gh (+ gauss edges, Hungarian per run), c_all_l30 (+ 30 layers); chain voting chain05, chain1, c_all_chain, c_all_gh_chain (chain_w 0.5 in place of tsmooth). Cases (CONFIRM_CASES) × seeds 0, 1, 2: S137 t63, S184 t32, Z1210 t52, Leifer re-export t57 (held-out lab, NWB every detection), Flavell 2022-03-24-02 t28 (held-out recording), and Leifer StarDist t57 (the same network on the StarDist re-segmentation, strict click GT; the paper's Leifer; added at 3756753 on the user's request). Runs r261106-595573, r261106-598057, r261106-606174, r261106-673582, r261106-677223 (superseded schedulers; their records are kept, the sweep is resumable) and r261107-743860 (v4: everything left on the GPU in two lanes; F0324 had run about 1.5 h per variant on the CPU). Output ~/.cache/barlow_scratch/labelprop_opt/batch2_confirm.jsonl, analysed with analyze.py --per_case. Adopt per PLAN.md section 4 (mean ≥ +0.3 pt, paired t-test p < 0.05 over all pairs, no case mean below -0.5 pt). Then check the adopted recipe at 100 seeds on 2 cases.

## Results

Partial, 2026-11-07 01:10 (221 records; complete: S137 and S184 all seeds, Leifer re-export seeds 0-2 without the chain variants, StarDist seed 0, Z1210 seed 0 without chain variants; F0324 4-8 of 20 variants per seed). Paired gain over base in pt (per-case means):

| variant | mean | wins | Leifer NWB | StarDist s0 | S137 | S184 | Z1210 s0 |
|---|---|---|---|---|---|---|---|
| c_all_gh | +1.24 | 11/11 | +0.49 | +0.23 | +1.02 | +2.62 | +0.99 |
| c_all_notemp | +1.09 | 11/11 | +0.38 | +0.27 | +1.03 | +2.16 | +1.04 |
| c_all_l30 | +1.04 | 11/11 | +0.41 | +0.22 | +0.95 | +2.03 | +1.03 |
| c_all | +0.96 | 11/11 | +0.40 | +0.25 | +0.77 | +1.92 | +1.03 |
| c_all_s100 | +0.93 | 11/11 | +0.30 | +0.38 | +0.74 | +1.88 | +1.04 |
| svd150 | +0.60 | 7/11 | -0.11 | +0.42 | +0.80 | +1.42 | -0.19 |
| svd100 | +0.55 | 9/11 | +0.01 | +0.37 | +0.66 | +1.28 | -0.11 |
| tsmooth05 | +0.39 | 11/11 | +0.32 | +0.12 | +0.27 | +0.70 | +0.31 |
| c_all_gh_chain | (+1.66, 7/7, no Leifer NWB / Z1210 yet) | | | +0.23 | +1.08 | +2.71 | |

Base: Leifer NWB 0.8789, StarDist 0.9371, S137 0.9346, S184 0.9306, Z1210 0.9408. On StarDist every variant but seed_strat beats base at seed 0, but by at most 0.42 pt (svd150 0.9414, c_all_gh 0.9394). Chain voting on top of c_all_gh adds under 0.1 pt where both exist.

## Conclusion

Pending (Flavell 0324, Z1210 seeds 1-2, StarDist seeds 1-2 still running). So far every c_all combination wins every pair; c_all_gh is ahead at +1.2 pt. The gains are largest on Samuel and smallest on the two Leifer versions (+0.5 NWB, +0.2 StarDist).

## Caveats

Report the Leifer fragment columns separately; it is the case where fragmentation matters most. The StarDist case scores assigned strict clicks only (README of round_bayes_wk4_leifer_stardist_strict: multiply by 0.97826 for the all-kept headline). Partial results weight cases unevenly (S137 and S184 have all seeds).
