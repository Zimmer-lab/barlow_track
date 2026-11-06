---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1b2e3-9ed1-72e8-a698-7c22b98fed12
status: in-progress
priority: high
dateCreated: 2026-11-06T23:03:44+01:00
dateModified: 2026-11-06T23:03:44+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

The fragment errors are long swaps with a neighbouring nucleus ([[EVD - Fragmented neurons on Zimmer swap with a neighbouring nucleus for long stretches]]). Test whether letting chains of adjacent-volume links vote on an object's label, before the final per-volume Hungarian, fixes them or other errors.

## Method

`lp_core.chain_vote` / `chain_w` (barlow_track@7b4f1df, labelprop_opt): chains are connected components of adjacent-volume mutual nearest neighbours in the embedding (the links `temporal_smooth` uses); each object's consensus row gets `chain_w` times the mean consensus row of its chain added. `diag_chain.py`, run r261106-647605, on the cached base consensus at seed 0 (from diag_merge.py) of S137 t50, Z1210 t36 and F0330 t61: chain statistics against GT, then official accuracy for chain_w {0, 0.25, 0.5, 1, 2, 5, 20} × temporal_smooth {0, 0.5}.

## Results

| case | links joining one GT neuron | chain length of an object, q10 / median / q90 | best | chain_w 0.5 | chain_w 5 |
|---|---|---|---|---|---|
| Z1210 | 99.8% | 10 / 159 / 728 | +1.66 pt (chain_w 1) | +1.61 | +1.49 |
| S137 | 90.3% | 1 / 7 / 26 | +0.61 (chain_w 1) | +0.52 | -0.07 |
| F0330 | 97.6% | 2 / 15 / 105 | +0.09 (0.25-1) | +0.09 | -0.61 |

On Z1210 the gain is unlabelled objects (1.60 → 0.14% of GT points at chain_w 1); swaps and fragments do not fall (fragments 3.14 → 3.20%). With temporal_smooth 0.5 the two overlap: Z1210 +1.69 at best, S137 at most +0.45 (they cancel at chain_w ≥ 1), F0330 worse with both.

## Conclusion

Pending the confirmation batch (chain05, chain1, c_all_chain, c_all_gh_chain in batch 2, b5e20d2). At seed 0 on the dev networks, chain voting with weight 0.5-1 adds 1.6 pt on Z1210, by labelling objects the tracker had left unlabelled, and 0.5 pt on S137, but nothing on F0330. It does not fix the long swaps with neighbouring nuclei, which stay as frequent.

## Caveats

One seed, dev networks only, cached consensus (no propagation rerun), so the weight is tuned on these three; the confirmation batch is the test. Gains in labelled objects can include distractors that take a neuron's label where the GT has no click; the official accuracy does not count those.
