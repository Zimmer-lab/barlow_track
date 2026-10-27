---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17b4d-6877-73b1-8480-874f8cd2be8b
status: done
priority: normal
dateCreated: 2026-10-27T04:00:33+01:00
dateModified: 2026-10-27T04:00:33+01:00
confidence: medium
completedDate: 2026-10-27
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find whether round_flavell_more_epochs trial27 (0.927 against the NWB GT, the number behind the 2.8-point gap to the preprint) is typical of its configuration or the best of several seeds.

## Method

round_flavell_more_epochs trials 17-29 share one configuration (position pipeline, ViSNet add fusion, emb 512, layernorm, crop 16x64, 75 epochs x 300 frames) and differ only in training seed and use_attention (no attention: 17, 27, 28, 29; attention: 18-26). The 12 other than trial27 were rescored against the NWB GT from their cached embeddings, labelprop seed 0 (r261026-1606883); trial27 from r261024-2431943. Labelprop seed spread on fixed embeddings, for comparison: trial27 0.921-0.927 over seeds 0-2 (r261024-2472680).

## Results

NWB GT (project GT): no attention 0.875 (0.836), 0.927 (0.880), 0.872 (0.836), 0.828 (0.797); attention 0.886, 0.886, 0.815, 0.865, 0.802, 0.857, 0.818, 0.876, 0.861. All 13: range 0.802-0.927, median 0.865, mean 0.859; without attention mean 0.876 (n 4), with attention 0.852 (n 9). NWB minus project GT: +2.9 to +4.1 points (mean 3.5). In round_flavell_paper_ablation the second training seed beat the first by 5-12 points in all three completed pairs.

## Conclusion

Across 13 training seeds of one position-pipeline configuration, Flavell accuracy runs from 0.80 to 0.93 (median 0.865), about 25 times the labelprop seed spread. trial27 is the best of the 13, so the typical position-pipeline network trails the preprint network (0.952) by about 9 points, not 2.8. The training seed matters more than any architecture choice swept so far.

## Caveats

The legacy pipeline's training-seed spread is not yet known: two networks (preprint checkpoint 0.952, local retrain 0.957) differ in code version as well as seed; the legacy cells of round_flavell_paper_ablation will give eight more. The attention difference (2.4 points) is inside the seed spread at these n. One labelprop seed per network. Trials 0-16 of the round (crop 16x32 or z 8) were not rescored.
