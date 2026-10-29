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

Baseline: the settings above (Zimmer on the updated-format 2022-11-28 copy, the benchmark's
zimmer_1128 project), keypoints only (no volumes loaded), 60 epochs, width 512, no
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
`false` is the control: the blind ViSNet every fused network so far had. It stays at trial
27's position settings (jitter 1.0, cutoff_nn 0) so the RBF fix is the only change; the
position-only rounds test the rest. Trains on the updated-format copy of 2022-11-28 (the
same recording as trial 27's original-format project), so its in-job record is the
benchmark's zimmer_1128 cell. Afterwards score both descriptor stages and cross-dataset
(trial 27's fused head cost 4-21 points on Samuel).

## In-job objective ground truth

round_posonly2_flavell and round_posonly2_samuel set `objective_source: nwb` and
`objective_gt: <the recording's NWB>` (new template keys): the in-job score is then the
benchmark's 'flavell' / 'samuel' cell (GT objects). Before, the objective scored the project
GT under that cell's tag: for Flavell that is the Flavell lab's tracker output
(round_posonly_flavell's 0.870 / 0.881), and the benchmark then skipped the real cell.

## Launch (cluster)

The cluster checkout (`/lisc/.../wbfm/code/barlow_track`) is at 7d9970b; the rounds need
the objective_gt keys and the staged yamls from later commits. Push from the workstation
and pull there first.

```bash
B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track && git pull
for R in round_posonly2_zimmer round_posonly2_flavell round_posonly2_samuel round_fused_fix_zimmer; do
  mkdir -p $B/$R && cp barlow_track/scripts/scratch/$R/*.yaml $B/$R/
  # Background each dispatcher, or the rounds run one after another
  nohup python barlow_track/scripts/optimize_hyperparameters.py -p $B/$R/hyperparameter_search_template.yaml \
      --direct_parameter_sweep --repetitions 2 > $B/$R/dispatcher.log 2>&1 &
done
# CPU rescores (stale Flavell 03-31-01 cells, round_posonly_samuel, round_posonly_flavell vs
# the NWB GT); they wait by themselves until the round_posonly_* jobs have left the queue
nohup bash barlow_track/scripts/scratch/cpu_rescores/run.sh > $B/cpu_rescores.log 2>&1 &
```

If a dispatcher dies, relaunch the same command with `--skip_completed_grid_jobs` added;
without it, finished grid jobs are trained again into new trial_N folders.
