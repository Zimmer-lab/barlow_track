---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17770-1f70-745f-a76d-601322f9b596
status: in-progress
priority: high
dateCreated: 2026-10-26T09:59:59+01:00
dateModified: 2026-10-26T09:59:59+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find which of the revision design changes cost or recover Flavell accuracy, starting from the preprint configuration.

## Method

round_flavell_paper_ablation (barlow_track@54690ba:barlow_track/scripts/scratch/round_flavell_paper_ablation/, fixed in d8cbddd so the legacy cells get no global augmentation): use_position {false, true} x embedding_dim {64, 512} x projector_norm {batchnorm, layernorm}, 50 epochs, objective accuracy, --repetitions 2 (16 cluster jobs; trial_0-7 seed 43, replicates seed 921677937). Launched by the user on 2026-10-25, run r261025-j7666018 (first jobs 7666018-7666027). The objective scores against the project GT: rescore every cell against the NWB GT from emb_cache, per the README. Local cells already done: (false, 64, bn) 0.957; (true, 64, bn) 0.795 fused / 0.756 backbone.

## Results

## Conclusion

## Caveats

use_position switches augmentation and fusion together; separating them needs a visual-only fusion mode (a code change, deferred).
