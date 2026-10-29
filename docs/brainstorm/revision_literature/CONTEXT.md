# Context for literature agents (barlow_track revision, 2026-10-29)

BarlowTrack: self-supervised tracking of neurons in whole-brain 3D fluorescence videos of
C. elegans (Zimmer lab; also Flavell, Leifer, Samuel lab data). Pipeline: per-frame
segmentation -> per-object 3D crops (8x64x64 voxels, z x y; Flavell 16 z) + object centroids
(keypoints) -> encoder (3D CNN image backbone and/or a point-set position encoder) ->
Barlow Twins loss between two augmented views of the same frame (feature-space term, plus a
"transpose" object-space term weighted lambd_obj that decorrelates objects within a batch)
-> embeddings -> label propagation (or HDBSCAN) across ~1600 frames -> tracks, scored
against ground truth. A paper is in revision (preprint exists); the network is trained
anew per recording (self-supervised; cross-dataset transfer does not matter).

Current state (local + cluster, full-video accuracy, label propagation):
- Image-only 0.89-0.98 depending on recording; fused image+position up to 0.97-0.99.
- Position-only network (ViSNet point-set encoder on centroids only, trained with keypoint
  jitter/dropout augmentation): 0.96 Zimmer, 0.98 Flavell, 0.94 Samuel. Key fixes found:
  ViSNet's radial basis functions were mis-scaled (fix decisive); keypoint jitter sized
  from the segmentation's own centroid noise (estimated GT-free from consecutive frames via
  affine ICP) +0.2 on Samuel; a scale-normalized cutoff (cutoff = N median nearest-neighbour
  distances) helps on Samuel, hurts on Zimmer.
- Tried and NOT helpful for position-only: synthetic segmentation errors (spurious/merge/
  split detections), keypoint bending, per-axis scale jitter, plane cuts, real frame pairs
  as views, global scale jitter; elastic deformation neither helps nor hurts (4x slower).
- Transpose loss (lambd_obj): hurts position-only slightly; in image-only networks early
  results suggest much faster training (same proxy in ~4x fewer epochs). The preprint found
  final accuracy insensitive to it.
- Flavell image features: preprint per-crop augmentation vs newer global (whole-volume)
  augmentation pipeline differ by ~3 points.

Revision constraints (the user's words): changes must fit a paper revision. NO dramatic
method changes (e.g. no cycle-consistency training, no new tracking paradigm). Fine:
different deformations/augmentations, different encoders, drop-in loss variants, eval or
clustering tweaks. No augmentation fitted to the ground truth (GT-free dataset statistics
are fine). Efficiency matters (hundreds of training jobs; GPU budget is tight, CPU plentiful).

Output rules: write Markdown to the file named in your task. For each idea: one-line
summary; citation (authors, year, venue, DOI/arXiv link; say if code is public);
why it might matter HERE (tie to a result above); how it would plug into this code
(augmentation / encoder / loss / eval); rough cost (local hours, GPU need); a fit-for-
revision rating (high / medium / low) with the reason. Rank ideas by expected value for the
revision. Be honest: say when an idea is speculative or when evidence is weak. Do not
invent citations: every paper you cite must be one you actually found (web or Zotero);
give the link. Do not modify any repository, the ygg research tree, or Zotero.
