---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a188f2-04eb-79b8-9aa8-73c9f5c949b5
status: done
priority: normal
dateCreated: 2026-10-29T19:35:24+01:00
dateModified: 2026-10-29T19:35:24+01:00
confidence: low
completedDate: 2026-10-29
outcome: ambiguous
projects:
  - "[[barlowtrack]]"
---

## Purpose

Decide between a fixed ViSNet neighbour cutoff (0.35 in normalised volume units, visnet_cutoff_nn 0) and one rescaled per frame to N median nearest-neighbour distances (visnet_cutoff_nn N, scale-invariant), for the final position branch.

## Method

Full-video accuracy (label propagation): round_posonly_zimmer (old base: width 64, 100 frames, dropout 0.2, one seed), round_posonly2_samuel (NWB GT objects, seed 43), and local long15 (r261029-3865613; Zimmer at the posonly2 base with jitter auto and dropout 0.02, 60 epochs, seed 43). Epoch-14 proxy: batches 17 (r261029-3755323) and 18 (r261029-3914302), seeds 43 and 44.

## Results

Full video, cutoff_nn 0 vs 6: Zimmer old base 0.825 vs 0.868; Zimmer new base 0.961 vs 0.943; Samuel jitter 1.0 0.706 vs 0.727, jitter auto 0.904 vs 0.936; Flavell (round_posonly2, jitter 1.0) 0.980 vs 0.975. Proxy, cutoff_nn 6 against 0: about -0.05 on Zimmer, Samuel and Flavell on both seeds.

## Conclusion

Which cutoff is better depends on the dataset and the rest of the recipe (one seed each on full video): the scale-normalised cutoff wins on Samuel by 0.02-0.03 and on Zimmer's old base, loses on Zimmer's current base by 0.02 and ties on Flavell. The epoch-14 proxy penalises cutoff_nn 6 everywhere, against the full video on Samuel, so the proxy cannot decide this setting.

## Caveats

Single seeds on full video; the Zimmer comparisons differ in base. Keep both values in sweeps until the second seeds of round_posonly2 are in.
