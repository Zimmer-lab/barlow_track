Retrain of the preprint flavell baseline (TrainedBarlow/baseline_flavell/trial_0) with
the current code, scored with the eval_accuracy project path (as E2, where the preprint
checkpoint scores 0.952 against the NWB GT).

train_config.yaml is the preprint config with project_path pointed at the barlow_track_paper
copy, project_dir pointed at the local run folder, weight_decay as a float and seed: 0.
run.sh trains with WANDB_MODE=offline and then evaluates. Copied from
~/.cache/barlow_scratch/train/baseline_flavell_head_rerun/ (create log/ and checkpoints/ first).

Question: does the old design reach ~0.95 with the current code (gap is the new design) or
~0.915 like round_flavell_more_epochs trial27 (gap is in the pipeline)?
