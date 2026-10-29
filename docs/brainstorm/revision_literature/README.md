# Revision literature brainstorm (2026-10-29)

Four literature agents searched the web and the Zotero library, scoped to changes that fit a paper
revision. They were told not to suggest dramatic changes such as cycle consistency; deformations,
encoders, drop-in losses and tracking-stage tweaks were all in scope. The prompt they shared is
[CONTEXT.md](CONTEXT.md). Their full reports:

- [augmentations.md](augmentations.md): deformations and augmentations (12 ideas, plus why bending and
  segmentation errors did not help)
- [encoders.md](encoders.md): point-set encoders, image backbone, fusion (14 ideas)
- [losses_tracking.md](losses_tracking.md): drop-in losses and the tracking stage (13 ideas)
- [competitors.md](competitors.md): published trackers a reviewer would expect, with code availability

This is a brainstorm document, not the research tree. Only ideas the user picks become planned
Evidence notes. The reports were written by agents and are only partly checked:
- I checked the claims about our own code (marked below).
- I did not check most citations one by one. The agents were told to cite only papers they found and
  to give links, but verify before citing in the paper.
- Example: the agents disagree on ASCENT's venue and authors (ICCV 2025, Han & Lu, vs bioRxiv 2025,
  Han, Pritchard & Lu).
- "TWT" and "eat-it" in competitors.md were placeholder names from my prompt, not trackers the user
  named.

## Checked in our code

- **Keypoints reach ViSNet in voxel units, so z is squashed** (encoders.md, idea 1; confirmed).
  - `superglue.normalize_keypoints` divides all three axes by one scalar.
  - On Zimmer, z is 1.5 µm per plane against 0.325 µm per xy pixel, so z distances are compressed
    4.6x. The other labs' project `physical_units` are "CHANGE ME" placeholders.
  - Every ViSNet network so far, fused and position-only, saw this metric. It may explain why
    `cutoff_nn` helps on Samuel and hurts on Zimmer.
  - Fix: opt-in `visnet_z_scale` (d9cfd1a). Tested by fast-loop batch 22 (r261029-4126504).
- **`run_label_propagation(alpha=0.95)` ignores `alpha`** (losses_tracking.md; confirmed). The
  propagation is plain clamped diffusion (Zhu et al. 2003), so the methods text should describe it
  that way.
- **The kNN graph used for label propagation includes edges between objects in the same frame**
  (losses_tracking.md idea 4; not yet checked).

## Shortlist for the revision (my ranking, by value per unit cost)

### Experiments that need little or no GPU

1. **Tracking-stage ablation on cached embeddings** (losses_tracking.md idea 5; CPU only).
   - Compare HDBSCAN, Hungarian matching to one template frame (fDNC-style), serial frame-to-frame
     Hungarian (ASCENT-style), label propagation from one seed frame, and multi-seed consensus.
   - Plot accuracy against distance from the seed frame.
   - This gives the label-propagation benefit (goal c) a table and a mechanism.
   - Add Poisson learning (idea 3) and dropping same-frame edges (idea 4) as extra label-propagation
     rows.
2. **The transpose term as the sample-contrastive dual of Barlow** (losses_tracking.md idea 1).
   - Garrido et al. (ICLR 2023) show the two criteria are equivalent up to normalization. That
     predicts the same optimum but different optimization speed, which matches the preprint (final
     accuracy insensitive) and batch 20 (faster training).
   - Evidence needed: effective rank of the embeddings per epoch against `lambd_obj`.
   - Untested hypothesis from the agent: the speed-up grows with embedding dim / objects per frame.
     A cheap sweep axis would test it.
3. **Keypoint metric and graph** (encoders.md ideas 1 and 3).
   - `visnet_z_scale` crossed with an explicit kNN graph (large cutoff plus `visnet_neighbors=k`).
     Both are config-level.
   - If one setting wins on every dataset, the paper gets a single rule instead of a per-dataset
     cutoff.
4. **Jitter-strength sweep** (augmentations.md idea 7): 0.5x to 4x the GT-free noise estimate, for a
   figure-ready "best near 1x the data's own noise" claim. No new code.

### Small code changes, one GPU round each

5. **More than two views per frame** (losses_tracking.md idea 2).
   - Agrawal/Ghosh et al. (ICLR 2024) report Barlow Twins on CIFAR-10 going from 86.4% with 2 views
     to 92.7% with 8, and up to 4x fewer unique samples.
   - This is the best lead for data efficiency at 30-100 frames (goal d).
6. **Cheap smooth deformation fields per crop** (augmentations.md idea 2): evaluate a smooth random
   field and its Jacobian at each centroid instead of warping the whole volume. It could make elastic
   deformation nearly free; elastic is now 4x slower and makes no difference to accuracy.
7. **Real deformations from unlabeled nearby frames** (augmentations.md idea 1): fit smooth fields
   between real frames without GT and apply them to a single frame, so correspondences stay exact. It
   can be precomputed on CPU.
8. **Anisotropy-aware image backbone** (encoders.md idea 9): in-plane kernels and pooling first, and a
   wider head. Test on Flavell first, where image features are the weak point.

### Comparisons with other methods (competitors.md)

9. **Must compare:** ASCENT (self-supervised, trained per recording, public code, reports 95.9% on
   our exact Leifer recording); fDNC (the wbfm wrapper exists, and an agent is running it now);
   ZephIR (needs at least one annotated frame; report it as "uses k GT frames").
10. **Cheap floor:** CPD or Hungarian matching on centroids.
11. **Leifer literature table:** the published numbers are on our exact recording, though each paper
    uses its own metric.

### Writing only

- Bending and segmentation-error null results: cite Atanas et al. 2025. They found custom
  worm-bend, contrast and blur augmentations gave about the same performance as standard ones, and
  fDNC never ablated its own augmentations.
- The invariance argument: ViSNet only sees local neighbourhoods, so only deformation within one
  cutoff radius changes its input (augmentations.md Part 1).
- Position ASCENT carefully: its recipe overlaps BarlowTrack's (two augmented views, image plus
  position, trained per recording) and it does not cite BarlowTrack.
