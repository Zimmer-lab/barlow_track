#!/bin/bash
# CPU-only rescoring (embedding and label propagation), one Slurm job per (checkpoint, dataset)
# cell, started only once the position-only test rounds' jobs (round_posonly_{zimmer,flavell,
# samuel}, job ids from their *_submission.sh files) have left the queue: the CPU partition is
# busy until then (the user, 2026-10-29).
#   1. Flavell 2022-03-31-01 NWB cells recorded before the neuron-order fix (nwb_arrays None),
#      which the benchmark skipped as done: round_visnet_full t27, round_bayes_samuel t0,
#      round_2x2 t6 (auto stage) -> --rerun_completed.
#   2. round_posonly_samuel: every trial with a checkpoint on the benchmark's 'samuel' cell
#      (153 NWB, GT objects); the in-job evals crashed on the NaN-centroid bug (fixed in dd0e82e).
#   3. round_posonly_flavell: every trial with a checkpoint on the benchmark's 'flavell' cell
#      (NWB GT objects). Their in-job records carry that tag but scored the project GT (the
#      Flavell lab's tracker output). Records are keyed on (tag, settings), so that record
#      stays as a separate cell (best_of_table drops it); --rerun_completed is only a safeguard.
# Run from the barlow_track checkout on the cluster (after pulling), in the background; then
# rerun barlow_track/scripts/best_of_table.py.
set -u
cd /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track
B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
WAIT_IDS=$(ls $B/round_posonly_{zimmer,flavell,samuel}/*_submission.sh 2>/dev/null | xargs -n1 basename | cut -d_ -f1 | sort -u)
while true; do
  # A failed squeue must not read as an empty queue
  queued=$(squeue -u $USER -h -o %A) || { echo "$(date +%H:%M) squeue failed; retrying"; sleep 600; continue; }
  left=$(comm -12 <(echo "$queued" | sort -u) <(echo "$WAIT_IDS"))
  [ -z "$left" ] && break
  echo "$(date +%H:%M) waiting on $(echo $left | wc -w) round_posonly jobs"; sleep 600
done
DRV="python barlow_track/scripts/multiproject_scripts/run_trials_on_all_ground_truth.py \
    --device cpu --track_device cpu --slurm --slurm_time 12:00:00 --slurm_cpus 16 --slurm_mem 48G --no_fail_fast"
for R in round_visnet_full:27 round_bayes_samuel:0 round_2x2:6; do
  $DRV --trial_parent_dir $B/${R%%:*} --trials ${R##*:} --labs flavell --rerun_completed &
done
$DRV --trial_parent_dir $B/round_posonly_samuel --all --labs samuel --rerun_completed &
$DRV --trial_parent_dir $B/round_posonly_flavell --all --labs flavell --rerun_completed &
wait
echo "cpu_rescores done"
