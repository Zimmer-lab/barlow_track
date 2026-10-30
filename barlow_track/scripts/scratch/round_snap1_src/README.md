# round_snap1_* (accuracy vs training length)

How many epochs do the networks need? Each trial keeps its weights after epochs 1, 3, 10 and
20 (`epoch_snapshots`), and its CPU eval job scores the final (epoch 30) checkpoint, then each
snapshot (`objective_snapshots: true`), on that recording's full video, as round_wk1 scores its
final checkpoints. Snapshot records are tagged `<round>_trial<N>_<label>_epNNN`; the Ax
objective stays the final checkpoint.

Grid: `fusion` {visual_only, add} x `lambd_obj` {0, 0.67}, so the curves also ask whether the
transpose loss reaches its accuracy sooner (goal d: faster training counts). Base: round_wk1's
(100 frames, jitter auto, `visnet_z_scale` = voxel aspect). 4 recordings that span the range:
zimmer_1128, samuel_137 (among the hardest), flavell_2022033001 (every detection), leifer_reexport
(every detection, GT by segmentation id). 16 GPU training jobs, 1 seed (round_wk1 has the
other seeds of the epoch-30 cells).

Resources: `train_gpu_job_cpus: 16` (14 DataLoader workers, 3e695bb / aab23eb); each CPU eval job
runs ~5 full-video evals, so `objective_minutes: 2880` (local 16-thread evals: Samuel 153
34 min, Zimmer 158 min; Leifer longer). A job killed mid-way keeps what it recorded: rerunning
the dispatcher with `--skip_completed_grid_jobs` after it ends re-runs only unscored work
(snapshots already recorded are skipped).

Needs the code at 0dce172 or later on the cluster (epoch_snapshots, objective_snapshots,
train_gpu_job_cpus, the NWB-label eval fix).

## Launch (cluster)

```bash
B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track && git pull
S=barlow_track/scripts/scratch
for R in $(cd $S && ls -d round_snap1_* | grep -v _src); do
  mkdir -p $B/$R && cp $S/$R/*.yaml $B/$R/
  nohup python barlow_track/scripts/optimize_hyperparameters.py -p $B/$R/hyperparameter_search_template.yaml \
      --direct_parameter_sweep > $B/$R/dispatcher.log 2>&1 &
done
```

Regenerate with:

```bash
python -m barlow_track.scripts.scratch.make_per_recording_rounds --name round_snap1 \
  --base barlow_track/scripts/scratch/round_snap1_src/base_train_config.yaml \
  --grid barlow_track/scripts/scratch/round_snap1_src/grid.yaml --out barlow_track/scripts/scratch \
  --z_scale_from_voxels --labels zimmer_1128 samuel_137 flavell_2022033001 leifer_reexport
```
