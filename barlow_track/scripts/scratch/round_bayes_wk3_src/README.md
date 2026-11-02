# round_bayes_wk3_* (Tue-Fri 2026-11-03..06: round_bayes_wk2's search on the other recordings)

The user, 2026-11-02 (travelling until Friday): run the same wide search on the remaining GT
recordings to see whether any is an outlier; Zimmer 1128 (0.98), Samuel 137 (0.93, the hard one)
and Flavell 0330-01 (0.985) need no more exploration. Grid and base config are round_bayes_wk2's
unchanged (only the wandb name differs), so results are directly comparable: see
`../round_bayes_wk2_src/README.md` for the search space and costs.

Recordings: zimmer_1123, zimmer_1210, flavell (2025_07_01, scored on flavell_data.nwb),
flavell_2022031603, flavell_2022032402, flavell_2022033002, samuel (153), samuel_184. Flavell
cells score every detection (objective_nwb_objects all), as in wk2.

Launch everything (new searches, plus score-only resumes of the rounds whose dispatchers died on
2026-11-01) with one command on the cluster, after `git pull`:

```bash
cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track && git pull
bash barlow_track/scripts/scratch/round_bayes_wk3_src/launch_tue_fri.sh
```

Each dispatcher is its own 1-CPU Slurm job (`sbatch_dispatcher.sbatch`, 7 days), log in
`<round>/dispatcher_<jobid>.log`. Check within a few minutes that the first lines say
`Submitted trial ...`; if Slurm refuses sbatch from inside a job, the error is there.
wk3 rounds run at most 6 GPU trials each (`--num_parallel_jobs 6`, 48 for all eight).

Regenerate with:

```bash
python -m barlow_track.scripts.scratch.make_per_recording_rounds --name round_bayes_wk3 \
  --base barlow_track/scripts/scratch/round_bayes_wk3_src/base_train_config.yaml \
  --grid barlow_track/scripts/scratch/round_bayes_wk3_src/grid.yaml --out barlow_track/scripts/scratch \
  --z_scale_from_voxels --preflight --labels zimmer_1123 zimmer_1210 flavell flavell_2022031603 \
  flavell_2022032402 flavell_2022033002 samuel samuel_184
```
