---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17932-6621-7398-8121-321e354ed94c
status: done
priority: high
dateCreated: 2026-10-26T14:11
dateModified: 2026-10-26T22:24:34+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-10-26
---

## Purpose

Separate the position pipeline's data path from position fusion: train the use_position cell of the preprint Flavell config with the position branch cut out of the descriptor, so the image backbone alone carries the Barlow loss under the same global augmentation.

## Method

[[@BarlowTrack preprint Flavell configuration in the position pipeline with image features only]]: same as [[@BarlowTrack preprint Flavell configuration with the position pipeline]] (use_position true, emb 64, batchnorm projector, 11,900 steps, seed 0), with BarlowWithPosition.fused_descriptors replaced by the backbone output. Scored with eval_accuracy.py --source project against the NWB GT at descriptor_stage backbone, 25 label-propagation seeds. Run r261026-1561729.

What would answer it, against the legacy pipeline's 0.952-0.957 and the fused position cell's training-seed range (NWB GT): 0.93 or above means fusion costs the accuracy; 0.88 or below means the data path itself does; in between needs a second seed.

## Results

0.8186 backbone (misses 13,673, mismatches 28,280 of 231,268). The same config fused, three training seeds, NWB GT: 0.769 (seed 43, cluster), 0.795 (seed 0, local; its own backbone 0.756), 0.848 (seed 921677937, cluster); the cluster cells rescored from round_flavell_paper_ablation trials 4 and 12 (r261026-1606883). The legacy pipeline with the same config: 0.957 (seed 0). Validation loss falls then rises (0.040 at epoch 4, 0.064 at epoch 49) as in the fused runs (0.036 -> 0.054), against a final train loss of 1.3e-4; the legacy pipeline's validation loss falls throughout, 0.0044 -> 0.0009, with train loss 2.8e-5. Fused effective rank 24 -> 34, no collapse.

## Conclusion

With position removed from training, the position pipeline's image features track Flavell at 0.819, inside the fused cell's training-seed range (0.77-0.85) and 14 points below the legacy pipeline with the same config and seed. Position fusion is not what costs the accuracy; the data path is. Fusion does weaken the fused network's own backbone (0.756 vs 0.819, one seed each) but the fused output makes up most of it.

## Caveats

One training seed for the image-only arm; the fused arm's spread across three seeds is 8 points. Validation losses of the two pipelines are not strictly comparable (different view generation, dropout of objects in the position pipeline), but the trends (rising vs falling) differ in every run. Flavell only.
