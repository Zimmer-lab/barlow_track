# round_wk1_* and round_wk1f30_* (weekend 2026-10-31)

Per-recording comparison of candidate recipes for the final architecture: one network
trained on each GT recording (the user, 2026-10-29), scored by full-video tracking accuracy
on that recording in a CPU-only job (`objective: accuracy`, `objective_cpu_job: true`).

## What is compared

Grid on every recording: `fusion` {visual_only (image only), add (image + RBF-fixed ViSNet)} x
`lambd_obj` {0 (feature loss only), 0.67 (default), 1 (transpose loss only; the user,
2026-10-30)}.

From the local fast loop (research/tree/NOTEBOOK.md, batches 20-23):
- `visnet_z_scale` = the recording's z / xy voxel size, from GT-free metadata (NWB grid_spacing;
  Zimmer 1.5 / 0.325): Zimmer 4.615, Samuel 3.889, samuel_184 5.833 (its NWB says xy 0.3 um,
  the other Samuel recordings 0.45; check), Leifer 4.65, Flavell 1.0. Position-only, full video:
  Zimmer 0.922 -> 0.967, Samuel 0.741 -> 0.983. Untested in fused networks before this round.
- `lambd_obj` 0.67 vs 0 in image-only networks: +0.14 / +0.05 (Zimmer, epoch-14 proxy), +0.16 to
  +0.18 (Samuel, 30 and 100 frames, epoch 29), both seeds; no gain in fused networks (-0.01 / -0.02).
- `jitter_std: auto` (GT-free keypoint jitter): +0.04 on Samuel, 0 on Flavell.

Base (`base_train_config.yaml`): round_fused_fix_zimmer's config (trial 27 + the RBF-fixed position
branch, width 512, no dropout) with 30 epochs, 100 frames, `jitter_std: auto`, elastic off.
round_fused_fix_zimmer's own results (rbf fix x concat / add) were not back when this was staged.

## Rounds and job counts (GPU training jobs; evals are CPU jobs)

| round | recordings | frames | repetitions | jobs |
|---|---|---|---|---|
| round_wk1_<label> | all 12 | 100 | 2 on zimmer_1128, samuel, flavell_2022033001; 1 elsewhere | 90 |
| round_wk1f30_<label> | zimmer_1128, samuel, flavell_2022033001 | 30 | 2 | 36 |

The 12 recordings: zimmer_1123 / 1128 / 1210, flavell + 4 forCharlie_2 recordings, samuel (153),
samuel_137, samuel_184, leifer.
- Leifer trains on the wbfm project of the re-export (NeRVE's masks; `barlow_track_revisions/
  wbfm_tracker_baseline/leifer_reexport/`) and is scored on the old Leifer NWB (GT objects), as the
  benchmark's 'leifer' cell. Its eval is slow (the old file's chunking): objective_minutes 720.
- samuel_137 / 184 train on projects built from their NWBs (same folder; centroid fix applied).

## Launch (cluster)

Push from the workstation and pull on the cluster first (the rounds need `visnet_z_scale`,
d9cfd1a, and the new projects).

```bash
B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track && git pull
S=barlow_track/scripts/scratch
for R in $(cd $S && ls -d round_wk1_* round_wk1f30_* | grep -v _src); do
  case $R in
    round_wk1f30_*|round_wk1_zimmer_1128|round_wk1_samuel|round_wk1_flavell_2022033001) REP=2 ;;
    *) REP=1 ;;
  esac
  mkdir -p $B/$R && cp $S/$R/*.yaml $B/$R/
  # Background each dispatcher, or the rounds run one after another; each caps itself at
  # 10 parallel GPU jobs, so with 15 dispatchers Slurm holds the rest pending
  nohup python barlow_track/scripts/optimize_hyperparameters.py -p $B/$R/hyperparameter_search_template.yaml \
      --direct_parameter_sweep --repetitions $REP > $B/$R/dispatcher.log 2>&1 &
done
```

If a dispatcher dies, relaunch the same command with `--skip_completed_grid_jobs` added.
To favour the main question if the GPU queue is short: launch round_wk1_* first and the
round_wk1f30_* dispatchers later.

Regenerate the round directories with:

```bash
python -m barlow_track.scripts.scratch.make_per_recording_rounds --name round_wk1 \
  --base barlow_track/scripts/scratch/round_wk1_src/base_train_config.yaml \
  --grid barlow_track/scripts/scratch/round_wk1_src/grid_main.yaml \
  --out barlow_track/scripts/scratch --z_scale_from_voxels --preflight
python -m barlow_track.scripts.scratch.make_per_recording_rounds --name round_wk1f30 \
  --base barlow_track/scripts/scratch/round_wk1_src/base_train_config_f30.yaml \
  --grid barlow_track/scripts/scratch/round_wk1_src/grid_f30.yaml \
  --out barlow_track/scripts/scratch --z_scale_from_voxels --labels zimmer_1128 samuel flavell_2022033001
```
