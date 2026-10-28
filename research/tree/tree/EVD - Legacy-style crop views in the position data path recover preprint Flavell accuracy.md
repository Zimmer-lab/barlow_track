---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a181ae-d863-786f-abc2-1f6ced0badbf
status: done
priority: high
dateCreated: 2026-10-28T09:44:41+01:00
dateModified: 2026-10-28T10:42:15+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-10-28
---

## Purpose

Bisect the 14-point Flavell gap between the position pipeline's image features (0.819) and the legacy crop pipeline (0.956). Keep the position pipeline's data path (VolumeCoordsDataModule, crops from the full volume) and make every view-generation step legacy-like, so the gap lands either in the global geometric augmentation and crop transforms (score near 0.95) or in the rest of the data path (near 0.80).

## Method

[[@BarlowTrack preprint Flavell configuration in the position pipeline matched to the legacy crop views]]: image features only, preprint per-crop view transforms, no global volume affine, no crop-centre jitter or dropout, views re-drawn each epoch; seed 0. The epoch-12 checkpoint (of 50) scored with eval_accuracy.py --source project against the NWB GT at descriptor_stage backbone, 25 label-propagation seeds, at the default truncated crop centres and at rounded ones (as it trains). Runs r261028-838680 (training, stopped at epoch 18), r261028-3899580 (eval).

## Results

Epoch 12: 0.9375 truncated (misses 4,044, mismatches 10,399 of 231,268), 0.9498 rounded. Against, NWB GT: image-only position pipeline 0.819 ([[EVD - Image features alone in the position pipeline]], seed 0, 50 epochs, truncated); fused position cells 0.77-0.90 over 16 training runs; legacy pipeline 0.944-0.957. Training curves look like the legacy pipeline's: val loss 0.0043 at epoch 0 (legacy 0.0044; position runs 0.036-0.045) and falling through epoch 13 (0.0014), where every position run's rose after a few epochs; frame-pair proxy 0.48 -> 0.58 by epoch 13, above every earlier position run at any epoch (legacy final checkpoint 0.600).

## Conclusion

On Flavell, the position pipeline's data path itself (crops from the full volume at metadata centroids, its frame selection and training loop) is not what costs accuracy: with legacy-style views it reaches 0.95 at rounded centres after 12 of 50 epochs, within a point of the legacy pipeline and 13 points above the same data path with the revision's view generation. The cost lies in how the revision pipeline makes its views: geometric augmentation that reaches the crops (global volume affine, crop-centre jitter, object dropout) and/or the photometric-only per-crop transform in place of the preprint's view 1 / view 2 transforms.

## Caveats

Closed on the epoch-12 checkpoint by decision (2026-10-28): the result already sits at the legacy level, so the remaining ~20 h of local training would not change the bisection. The run was stopped at epoch 18 and never scored at 50 epochs; its checkpoints/checkpoint.pth (epoch 18) is kept under /home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_legacy_matched/, but training does not resume from it, so a 50-epoch number means a fresh ~25 h run (run.sh). The `visual_only` / keypoints-only cell of round_flavell_legacy_crop_aug replicates this arm at 50 epochs with 3 seeds and supersedes this as the full-length number. One training seed, Flavell only. The arm changes several things at once relative to the image-only position run (global affine off, jitter and dropout off, legacy per-crop transforms, re-drawn views), so which of them matters is open; the round separates the global augmentation reaching the crops from the per-crop transforms and tests position fusion on top. The model was loaded with another run's args.pickle (same architecture; training writes its own only at the end). Rounded-centre eval adds 1.2 points here, 1.9 on the fused comparator.
