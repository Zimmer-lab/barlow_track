---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b113-4c10-703a-a4c0-6f4f54037cc3
status: done
priority: high
dateCreated: 2026-11-06T14:36:34+01:00
dateModified: 2026-11-06T20:43:42+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-11-06
---

## Purpose

Tell apart the three causes of fragmentation on full-length recordings with distractors, and see which batch 1 variants reduce each one.

## Method

From the batch 1 records (432, seeds 0-2) of [[EVD - One-change screen of label propagation variants on four development recordings]] on Z1210 t36 and F0330-02 t61. For the `base` rows, read `err_frag_split`, `err_frag_distractor`, `err_frag_other` and `acc_merge_bound - acc_fast`. For each variant, the paired deltas of those columns over 3 seeds per case (`analyze.py batch1_dev.jsonl --metric err_frag_split`, and the same for err_frag_distractor; lower is better). Seed variants (`seeds50`, `seeds100`, `seed_complete`, `seed_strat`) test cause 1; `temporal*` and `reprop*` test cause 2; `framenorm`, `thr03`, `runw` and `drop20` test cause 3. Run ids are in the screen's Method.

## Results

Base, % of GT points (mean of 3 seeds):

| case | fragment | split | distractor | other | merge bound − acc_fast |
|---|---|---|---|---|---|
| Z1210 | 3.05 | 0.77 | 2.11 | 0.16 | 2.39 |
| F0330 | 1.26 | 0.25 | 1.01 | 0.01 | 1.10 |
| S137, S184 | 0 | 0 | 0 | 0 | ≤ 0.11 |

Paired deltas (pt), Z1210 split / distractor, F0330 split / distractor: seeds100 -0.06 / -0.00, -0.00 / -0.18; seed_strat +0.04 / -0.03, -0.14 / -0.18; seed_complete -0.01 / -0.04, +0.24 / -0.23; temporal1 -0.24 / +0.24, -0.12 / -0.10; temporal5 -0.07 / +0.06, -0.03 / -0.19; reprop09_w4 -0.01 / -0.47, -0.20 / +0.07 (with -4.2 pt accuracy on Z1210); k10 -0.27 / -0.05, +0.15 / -0.29; sym -0.25 / +0.11; framenorm, thr03, runw, drop20 all within ±0.1. No variant moves either kind of fragment by more than 0.5 pt.

## Conclusion

Fragmentation costs 3.1 pt on Z1210 and 1.3 pt on F0330, and about 70-80% of it is GT neurons whose objects sit in a track dominated by distractor detections, not GT tracks split in two. None of the 35 tracker settings in batch 1 reduces either kind by more than 0.5 pt without losing more accuracy elsewhere, so fragmentation on these recordings is set by the embeddings (distractors that look like neurons), not by seeding, temporal edges or post-processing of the tracker.

## Caveats

The split is by GT-optimal matching (`acc_fast`), not the official 'nandist' matching. The "distractor" class uses a GT share below 0.5 of a label's objects, which also counts GT neurons missing from the GT file as distractors. One network per lab. Leifer, where fragmentation is largest, is held out until confirmation; the Samuel cases have no distractors (NWB GT objects only), hence no fragments. Worth retrying with a tracker change aimed at distractors directly (e.g. a per-label distractor score), which batch 1 did not contain.
