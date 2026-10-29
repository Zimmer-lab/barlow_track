# round_posonly2_{zimmer,flavell,samuel} and round_fused_fix_zimmer

Staged 2026-10-29 from the local fast loop (`scratch/posonly_fast`, batches 1-14 and the
long runs, all in research/tree/NOTEBOOK.md). On zimmer_1128 the local position-only
network now tracks the full video at 0.932 (long10_drop0), above trial 27's image
backbone alone (0.923) and below trial 27 fused (0.968). What got it there, each checked
over seeds (single-seed gaps under ~0.04 in the epoch-14 proxy are noise):

| change | effect |
|---|---|
| ViSNet RBF fix (`visnet_rbf_rescale`) | +0.24 full video on the cluster (round_posonly_zimmer trials 8/9 vs 0/1) |
| 600 frames instead of 100 | proxy 0.40 -> 0.55 |
| no object dropout (0.2 -> 0) | full video 0.888 -> 0.932 |
| ViSNet width 128 -> 512 | proxy +0.05 per doubling |
| elastic deformation off | no loss, 4x faster |
| vector norm, cutoff 0.5, 5 layers, lr 3e-4 | nothing beyond seed noise |

## round_posonly2_{zimmer,flavell,samuel} (16 jobs each)

Baseline: the settings above, keypoints only (no volumes loaded), 60 epochs, width 512, no
dropout, isotropic position jitter 1.0, elastic off. No augmentation setting is fitted to a
dataset's GT (the user, 2026-10-29: the point is to generalize to unseen datasets).

Grid (revised 2026-10-29 after batches 14-16 and round_posonly_zimmer):
jitter_std {1.0, auto} x visnet_cutoff_nn {0, 6} x dropout_p {0, 0.02}, two seeds.

- `jitter_std: auto` sizes the per-axis jitter from the segmentation alone (consecutive-frame
  centroid residuals after an affine ICP; Zimmer ~(0.30, 0.91, 0.94), Flavell ~(0.80, 0.87,
  0.85), Samuel ~(0.10, 0.30, 0.30) voxels). Local proxy: +0.05 over 1.0 on both seeds;
  jitter 0 costs 0.09. GT-matched (0.2, 1, 1) was +0.07, used only as a diagnostic.
- `visnet_cutoff_nn: 6`: round_posonly_zimmer's best trial (0.868 vs 0.825 at 0, elastic 0).
- `dropout_p: 0.02`: local full video 0.945 vs 0.932 at 0 (one seed each, long13 / long10).

Dropped from the first draft (local proxy, batch 14 / 15 / 16, seed 43 against base 0.719):
synthetic segmentation errors (spurious 0.2 0.666, merge 0.05 0.700, split 0.05 0.713, all
with anisotropic jitter 0.691 vs that jitter alone 0.785), keypoint bend 0.707, per-axis
scale 0.695, plane cut 0.664, global scale jitter 0 / 0.2 0.686 / 0.659, and GT-free real
frame pairs (pair_p 0.5 0.580). All stay in the code, off by default.

## round_fused_fix_zimmer (8 jobs)

Trial 27's config with that position branch (width 512, no dropout, isotropic jitter 1.0,
600 frames, 40 epochs). Grid: visnet_rbf_rescale {false, true} x fusion {concat, add}, two seeds.
`false` is the control: the blind ViSNet every fused network so far had. Afterwards score
both descriptor stages and cross-dataset (trial 27's fused head cost 11 points on Samuel).

## Launch (cluster)

The cluster checkout (`/lisc/.../wbfm/code/barlow_track`) is at d0f45b2: it lacks dd0e82e
(the NaN-centroid fix every round_posonly_samuel eval crashed on), the vector norm and the
segmentation-error augmentations. Push from the workstation and pull there first.

```bash
B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track && git pull
for R in round_posonly2_zimmer round_posonly2_flavell round_posonly2_samuel round_fused_fix_zimmer; do
  mkdir -p $B/$R && cp barlow_track/scripts/scratch/$R/*.yaml $B/$R/
  # Background each dispatcher, or the rounds run one after another
  nohup python barlow_track/scripts/optimize_hyperparameters.py -p $B/$R/hyperparameter_search_template.yaml \
      --direct_parameter_sweep --repetitions 2 > $B/$R/dispatcher.log 2>&1 &
done
```

Then rescore round_posonly_samuel's crashed trials (their checkpoints are on disk) with
`python barlow_track/scripts/multiproject_scripts/run_trials_on_all_ground_truth.py --trial_parent_dir $B/round_posonly_samuel --trials 0 1 2 --labs samuel` (add the others as they crash).
