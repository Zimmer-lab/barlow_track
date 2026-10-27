---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a17cd8-2cf4-7abe-8829-a1639ac83031
status: in-progress
priority: high
dateCreated: 2026-10-27T11:11:44+01:00
dateModified: 2026-10-27T18:00:34+01:00
confidence: low
projects:
  - "[[barlowtrack]]"
---

## Purpose

Test whether the whole-volume rotation, which on Flavell drops a tenth of the neurons per view and puts 18% of crops off-centre ([[EVD - Crop geometry of the position pipeline on long narrow volumes]]), is what costs the position pipeline its Flavell accuracy. ViSNet position features are rotation-invariant, so the rotation only serves the image features.

## Method

Two arms of the use_position cell of the preprint Flavell config (emb 64, batchnorm projector, ViSNet add fusion, 11,900 steps, seed 0, position jitter 1.0, dropout 0.2; local Flavell mirror), against paper_flavell_use_position (same seed: fused 0.795, backbone 0.756; other seeds 0.769, 0.848). norot: global max_degrees_z 0 and p_flip 0, scale and translation kept (barlow_track@44c0e11:barlow_track/scripts/scratch/paper_flavell_use_position_norot/run.sh). croprot: as norot plus crop_photometric.crop_degrees_z 180, a rotation about each crop's own centre and z axis as in the legacy pipeline (paper_flavell_use_position_croprot/run.sh). Scored with eval_accuracy.py --source project against the NWB GT, 25 labelprop seeds, seed 0, descriptor_stage auto and backbone.

What would answer it: an arm at 0.90 or above fused (beyond 18 of 20 position-pipeline networks) means the rotation geometry is most of the cost; 0.80-0.90 needs a second seed; at or below the comparator means it is not. norot above croprot would mean the image features are better off without rotation invariance on Flavell, whose volumes are aligned to the worm.

## Results

norot (training r261027-2062947, evals from cached embeddings in r261027-2180262 via barlow_track@b39f823:barlow_track/scripts/scratch/paper_flavell_use_position_norot/eval_then_croprot.sh): fused 0.787, backbone 0.780 against the NWB GT (comparator 0.795 / 0.756). Per epoch, val loss bottomed at epoch 10 (0.0061) and rose to 0.0110 at epoch 49 while the frame-pair matching proxy on 30 held-out frames rose 0.390 -> 0.474 fused and 0.361 -> 0.469 backbone (Spearman of val loss with the proxy across epochs +0.80).

croprot (r261027-2190583): running; epoch-0 proxy 0.463 fused / 0.391 backbone against norot's 0.390 / 0.361.

## Conclusion

Partial (norot only). Turning off the global rotation and flip leaves fused accuracy where it was (0.787 against 0.795) and moves the image features by 2.4 points (0.780 against 0.756), inside the 5-12 point training-seed spread, so the whole-volume rotation and its loss of edge neurons are not what costs the position pipeline its Flavell accuracy.

## Caveats

One training seed per arm. The comparator differs from norot in rotation and flip together. Scale and translation of the global affine are still on in both arms.
