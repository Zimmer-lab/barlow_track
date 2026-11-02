#!/usr/bin/env bash
# Tue-Fri cluster runs (user travelling 2026-11-03..06). Run on the cluster login node from
# the code checkout after `git pull`. Every dispatcher is its own 1-CPU Slurm job
# (sbatch_dispatcher.sbatch), so login-node restarts no longer kill them.
#
#   bash barlow_track/scripts/scratch/round_bayes_wk3_src/launch_tue_fri.sh
#
# 1. round_bayes_wk3_<label>: round_bayes_wk2's search on the 8 other GT recordings.
# 2. Score-only resumes of the rounds whose dispatchers died on 2026-11-01 04:42:
#    round_bayes_wk2_* (Ax mode needs a budget >= 1, so each adds one new trial) and the
#    round_wk1* grids (--skip_completed_grid_jobs: trained trials only get their CPU eval;
#    the few never-trained grid cells train).
set -eu
B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
S=barlow_track/scripts/scratch
DISPATCH=barlow_track/scripts/sbatch_dispatcher.sbatch

dispatch() {  # round, dispatcher args...
  local R=$1; shift
  sbatch --job-name=disp_$R --output=$B/$R/dispatcher_%j.log $DISPATCH \
      -p $B/$R/hyperparameter_search_template.yaml "$@"
}

# 1. New searches
for L in zimmer_1123 zimmer_1210 flavell flavell_2022031603 flavell_2022032402 flavell_2022033002 \
         samuel samuel_184; do
  R=round_bayes_wk3_$L
  mkdir -p $B/$R && cp $S/$R/*.yaml $B/$R/
  dispatch $R --total_budget 120 --num_parallel_jobs 6
done

# 2a. Weekend searches: score the trained-but-unscored trials
# (leifer_reexport left out: its segmentation needs work first, see leifer_deep_dive/HANDOFF.md)
for L in zimmer_1128 samuel_137 flavell_2022033001; do
  dispatch round_bayes_wk2_$L --total_budget 1
done

# 2b. round_wk1 grids (fusion x lambd_obj): score what trained, finish missing cells
for R in $(cd $B && ls -d round_wk1_* round_wk1f30_* | grep -v leifer); do
  case $R in
    round_wk1f30_*|round_wk1_zimmer_1128|round_wk1_samuel|round_wk1_flavell_2022033001) REP=2 ;;
    *) REP=1 ;;
  esac
  dispatch $R --direct_parameter_sweep --repetitions $REP --skip_completed_grid_jobs
done

squeue -u $USER -n "$(squeue -u $USER -h -o %j | grep '^disp_' | sort -u | paste -sd,)" -o '%.10i %.40j %.8T %.10M'
