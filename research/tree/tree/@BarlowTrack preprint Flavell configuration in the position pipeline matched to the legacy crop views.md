---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a181ae-9a99-76e6-92b0-364466e04fe8
status: done
priority: normal
dateCreated: 2026-10-28T09:44:26+01:00
dateModified: 2026-10-28T09:44:26+01:00
completedDate: 2026-10-28
run: barlow_track@b3f816b:barlow_track/scripts/scratch/paper_flavell_position_pipeline_legacy_matched/run.sh
projects:
  - "[[barlowtrack]]"
---

## Description

The preprint Flavell config through VolumeCoordsDataModule (crops from the full volume at metadata centroids, frame selection and splits of the position pipeline), made as close to the legacy crop pipeline as it goes (scratch/train_position_pipeline_legacy_matched.py): image features only (fused_descriptors returns the backbone output), each view's crop stack gets the preprint's per-crop transform (barlow.Transform view 1, view 2), no global volume affine (p_global_affine 0), no crop-centre jitter or object dropout, views re-drawn each epoch (reaugment_each_epoch). emb 64, batchnorm projector, 50 epochs x 300 frames, seed 0, local mirror of the Flavell project. Training run r261028-838680 (started by r261027-808962, code at barlow_track@61894ab); the epoch-12 checkpoint was scored by r261028-3899580 (barlow_track@2683e5c eval_partial.sh) with the reaug cell's args.pickle (same architecture keys).
