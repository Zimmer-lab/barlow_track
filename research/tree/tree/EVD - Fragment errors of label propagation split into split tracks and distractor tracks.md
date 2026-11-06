---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b113-4c10-703a-a4c0-6f4f54037cc3
status: in-progress
priority: high
dateCreated: 2026-11-06T14:36:34+01:00
dateModified: 2026-11-06T14:36:34+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Tell apart the three causes of fragmentation on full-length recordings with distractors, and see which batch 1 variants reduce each one.

## Method

From the batch 1 records of [[EVD - One-change screen of label propagation variants on four development recordings]] on Z1210 t36 and F0330-02 t61. For the `base` rows, read `err_frag_split`, `err_frag_distractor`, `err_frag_other` and `acc_merge_bound - acc_fast`. For each variant, read the paired deltas of those columns: `analyze.py batch1_dev.jsonl --metric err_frag_split --cases zimmer flavell`, and the same for err_frag_distractor (lower is better). Seed variants (`seeds50`, `seeds100`, `seed_complete`, `seed_strat`) test cause 1; `temporal*` and `reprop*` test cause 2; `framenorm`, `thr03`, `runw` and `drop20` test cause 3.

## Results

Pending batch 1.

## Conclusion

Pending.

## Caveats

The split is by GT-optimal matching (`acc_fast`), not the official 'nandist' matching. The "distractor" class uses a GT share below 0.5 of a label's objects, which also counts GT neurons missing from the GT file as distractors. Leifer, where fragmentation is largest, is held out until confirmation.
