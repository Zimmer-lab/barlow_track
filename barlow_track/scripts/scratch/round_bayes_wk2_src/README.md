# round_bayes_wk2_* (weekend 2026-10-31: one wide Bayesian search per recording)

The user, 2026-10-30: rather than detailed grids, one large Bayesian search that informs the next
3-day run (Tue-Thu, user travelling) and then the next weekend. Architecture should come out
shared across recordings; lr, training length, frames and augmentation may differ per recording.

Ax (Sobol for the first 26 trials, then GPEI) maximizes full-video tracking accuracy on the
trial's own recording (`objective: accuracy`), scored in a CPU-only job (`objective_cpu_job`).
Four recordings that span the range: zimmer_1128, samuel_137 (among the hardest),
flavell_2022033001 and leifer_reexport (both scored with every detection).

## Search space (13 parameters, grid.yaml)

- Architecture: `fusion` {visual_only, add}, `self_layers` {0, 1, 2}.
- Training: `lambd_obj` [0, 1], `lr` [1e-5, 3e-4] log, `epochs` [2, 30] log, `num_frames`
  [30, 500] log (training speed and data efficiency, goal d).
- Augmentation, all GT-free: global rotation `max_degrees_z` [0, 180], `scale_jitter` [0, 0.25],
  `elastic_scale` [0, 1.5] (field std (1, 4, 4) voxels, length 40), per-crop rotation
  `crop_degrees_z` [0, 180], crop noise `p_noise` [0, 0.5], keypoint `dropout_p` [0, 0.3],
  synthetic `spurious_p` detections [0, 0.3].
- Left out: real frame pairs (`pair_p`: lowered the proxy in every cell, filter tuned on GT),
  keypoint-only geometry (needs `apply_to_volume: false`), embedding width.
- Fixed at round_wk1's base: jitter auto, visnet_z_scale = voxel aspect, width 512, ViSNet.

No epoch snapshots: Ax sees only the final checkpoint, and `epochs` is a parameter instead.

## Cost

A 30-epoch, 100-frame job took 0.1-0.15 h (Samuel), 0.3-0.5 h (Zimmer, Leifer), 0.5-0.9 h
(Flavell) with 14 workers (epoch_times_from_runs.py). The log-uniform frames x epochs average
is ~0.6 of that, so ~1 GPU-h per trial summed over the 4 recordings: `--total_budget 120`
per recording is ~120-150 GPU-h, plus the CPU evals (`objective_minutes` 1440).

## Launch (cluster)

```bash
B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track && git pull
S=barlow_track/scripts/scratch
for R in $(cd $S && ls -d round_bayes_wk2_* | grep -v _src); do
  mkdir -p $B/$R && cp $S/$R/*.yaml $B/$R/
  nohup python barlow_track/scripts/optimize_hyperparameters.py -p $B/$R/hyperparameter_search_template.yaml \
      --total_budget 120 > $B/$R/dispatcher.log 2>&1 &
done
```

No `--direct_parameter_sweep`. A dispatcher that dies is restarted with the same command: it
re-attaches the trials already on disk (scored ones, and trained-but-unscored ones whose eval
it re-runs; trials killed mid-training are dropped). `--total_budget` counts the NEW trials of
one invocation, so on a restart pass 120 minus the `trial_*` folders already there.

Regenerate with:

```bash
python -m barlow_track.scripts.scratch.make_per_recording_rounds --name round_bayes_wk2 \
  --base barlow_track/scripts/scratch/round_bayes_wk2_src/base_train_config.yaml \
  --grid barlow_track/scripts/scratch/round_bayes_wk2_src/grid.yaml --out barlow_track/scripts/scratch \
  --z_scale_from_voxels --labels zimmer_1128 samuel_137 flavell_2022033001 leifer_reexport
```
