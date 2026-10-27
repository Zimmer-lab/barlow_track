---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17932-66a4-7b50-a536-a58ca1587314
status: done
priority: high
dateCreated: 2026-10-26T18:11:48+01:00
dateModified: 2026-10-27T10:31:15+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
outcome: failed
completedDate: 2026-10-27
---

## Purpose

Test whether non-rigid augmentation between the two training views, matched to real Flavell motion, recovers the position pipeline's Flavell accuracy. After [[EVD - Image features alone in the position pipeline]] the data path, not fusion, holds the accuracy, and non-rigid deformation between views is the largest thing it lacks compared to real frames.

## Method

The use_position cell of the preprint Flavell config (emb 64, batchnorm projector, 11,900 steps, seed 0) with global_augment elastic_std (1.1, 4.1, 4.1), elastic_length 41 and position jitter_std 0.3 instead of 1.0 (914141c; the match in [[EVD - The position augmentation deforms mid-range neighbour distances less than real motion does]]), local Flavell mirror; barlow_track@01e6be3:barlow_track/scripts/scratch/paper_flavell_use_position_elastic/run.sh. The first attempt (r261026-1722467, num_workers 6) crashed on BadZipFile because forked workers share the zarr ZipStore handle; rerun serially (num_workers 0) as r261026-1730274. Scored with eval_accuracy.py --source project against the NWB GT, 25 labelprop seeds, seed 0, at descriptor_stage auto (fused) and backbone (image features only). Comparator: the same cell without the elastic field, same training seed (paper_flavell_use_position, r261025 local run).

What would answer it: fused 0.93 or above means non-rigid deformation is most of what the data path lacks; 0.86-0.93 needs a second seed; 0.85 or below means it is not.

## Results

Fused 0.7175 (misses 19,750, mismatches 45,590 of 231,268); backbone 0.5982 (misses 29,092, mismatches 63,832). Same cell without elastic, seed 0: fused 0.7953, backbone 0.7559; its other two training seeds 0.769 and 0.848 fused. Validation loss bottoms at 0.018 (epoch 8; without elastic 0.036) and oscillates 0.022-0.043 afterwards, ending at 0.035.

## Conclusion

Flavell-matched elastic deformation between views does not recover the position pipeline: it lowers Flavell accuracy by 7.8 points fused and 15.8 points in the image features alone, against the same cell and training seed without it, and the fused score is below all three seeds of that cell. Together with [[EVD - Preprint per-crop augmentation in the position pipeline]] (fused 0.657, backbone 0.487), every geometric augmentation added to the position pipeline so far has made its image features worse, whereas the legacy pipeline trains on per-crop elastic deformation and reaches 0.957.

## Caveats

One training seed and one labelprop seed; training seed moves this cell by up to 8 points, but the drop in the image features (15.8 points) is larger than that. The lower validation loss is measured on deformed views of held-out frames and does not carry over to tracking. Elastic and jitter_std (1.0 to 0.3) changed together. Up to ~50 crops per run at the volume's y edge (centroid y about 119.5 in a 120-wide volume) became zero placeholders, in this and earlier position-pipeline runs.
