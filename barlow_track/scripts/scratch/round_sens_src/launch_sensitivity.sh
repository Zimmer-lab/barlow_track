#!/usr/bin/env bash
# Sensitivity tests around the frozen round_shared_v1 trial 59: stage and launch the per-recording rounds.
# Uses the standard machinery only: make_per_recording_rounds.py writes round_sens_<R>_<label>/ (train_config.yaml +
# hyperparameter_search_template.yaml), optimize_hyperparameters.py runs each as a direct grid or one-at-a-time sweep
# (objective: accuracy, evaluated in the training GPU job), as the round_wk1_* rounds were.
#
#   cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track && git pull
#   S=barlow_track/scripts/scratch/round_sens_src
#   bash $S/launch_sensitivity.sh stage               # writes the round directories, submits nothing
#   bash $S/launch_sensitivity.sh launch core         # one dispatcher job per (round, recording)
#   bash $S/launch_sensitivity.sh launch ablations
#   SKIP=1 bash $S/launch_sensitivity.sh launch core  # resume after a dispatcher died (--skip_completed_grid_jobs)
#   bash $S/launch_sensitivity.sh launch loss_volume oat   # or name rounds
#
# Rounds (jobs per recording): loss_volume 18 combinations x 3 repetitions = 54, oat 31, attention2 1, crop_depth 2, only_affine / only_elastic /
# only_noise / aug_none / matched_steps 1 each. core = loss_volume + oat + matched_steps + attention2 + crop_depth; ablations = the four
# augmentation combinations. On the 6 tuning recordings: 54x6 + 31x6 + 6 + 6 + 12 = 534 core jobs, 24 ablation jobs.
# LABELS="..." overrides the recordings (e.g. add held-out ones); NUM_PARALLEL caps simultaneous GPU jobs per dispatcher.
set -euo pipefail
B=${B:-/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files}
CODE=${CODE:-/lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track}
S=$CODE/barlow_track/scripts/scratch/round_sens_src
LABELS=${LABELS:-"zimmer_1210 zimmer_1123 flavell_2022033002 flavell_2022032402 samuel_137 samuel_184"}
NUM_PARALLEL=${NUM_PARALLEL:-6}

# round name | grid file | sweep flag | repetitions
declare -A GRID=([loss_volume]=grid_loss_volume.yaml [oat]=grid_one_at_a_time.yaml [matched_steps]=grid_matched_steps.yaml
                 [only_affine]=grid_only_affine.yaml [only_elastic]=grid_only_elastic.yaml [only_noise]=grid_only_noise.yaml
                 [aug_none]=grid_aug_none.yaml [attention2]=grid_attention2.yaml [crop_depth]=grid_crop_depth.yaml)
declare -A FLAG=([loss_volume]=--direct_parameter_sweep [oat]=--one_at_a_time_sweep [matched_steps]=--direct_parameter_sweep
                 [only_affine]=--direct_parameter_sweep [only_elastic]=--direct_parameter_sweep [only_noise]=--direct_parameter_sweep
                 [aug_none]=--direct_parameter_sweep [attention2]=--direct_parameter_sweep [crop_depth]=--direct_parameter_sweep)
declare -A REP=([loss_volume]=3 [oat]=1 [matched_steps]=1 [only_affine]=1 [only_elastic]=1 [only_noise]=1 [aug_none]=1 [attention2]=1 [crop_depth]=1)

cmd=${1:?stage or launch}; shift || true
rounds=("$@")
[ ${#rounds[@]} -eq 0 ] && rounds=(core ablations)
expanded=()
for r in "${rounds[@]}"; do
  case $r in
    core) expanded+=(loss_volume oat matched_steps attention2 crop_depth) ;;
    ablations) expanded+=(only_affine only_elastic only_noise aug_none) ;;
    *) expanded+=("$r") ;;
  esac
done

cd "$CODE"
for R in "${expanded[@]}"; do
  [ -n "${GRID[$R]:-}" ] || { echo "unknown round $R"; exit 1; }
  if [ "$cmd" = stage ]; then
    PYTHONPATH=$CODE ${PY:-python} -m barlow_track.scripts.scratch.make_per_recording_rounds --name round_sens_$R --base $S/base_train_config.yaml \
        --grid $S/${GRID[$R]} --out "$B" --labels $LABELS --z_scale_from_voxels
  elif [ "$cmd" = launch ]; then
    for L in $LABELS; do
      D=$B/round_sens_${R}_$L
      [ -f $D/hyperparameter_search_template.yaml ] || { echo "$D is not staged (run stage first)"; exit 1; }
      BARLOW_DISPATCHER=barlow_track/scripts/optimize_hyperparameters.py sbatch --job-name=sens_${R}_$L \
          --output=$D/dispatcher_%j.log barlow_track/scripts/sbatch_dispatcher.sbatch \
          -p $D/hyperparameter_search_template.yaml ${FLAG[$R]} --repetitions ${REP[$R]} \
          --num_parallel_jobs $NUM_PARALLEL ${SKIP:+--skip_completed_grid_jobs}
    done
  else
    echo "usage: $0 stage|launch [rounds]"; exit 1
  fi
done
