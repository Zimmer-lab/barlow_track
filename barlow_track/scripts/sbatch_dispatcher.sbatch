#!/usr/bin/env bash

# Runs a sweep dispatcher (optimize_hyperparameters.py) as its own CPU-only Slurm job,
# so a login-node restart does not kill it. All arguments go to the dispatcher:
#
#   sbatch --output=$B/$R/dispatcher_%j.log barlow_track/scripts/sbatch_dispatcher.sbatch \
#       -p $B/$R/hyperparameter_search_template.yaml --total_budget 120
#
# Restarting after the job ends works as from the login node (see the round README).

#SBATCH --job-name=barlow_dispatch
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

pwd; hostname; date
echo "Dispatcher job $SLURM_JOB_ID: $*"

# The trial jobs this submits must not inherit this allocation's settings (e.g. memory
# or CPU binding conflicting with their own #SBATCH lines), so drop them first.
for v in $(compgen -e | grep -E '^(SLURM|SBATCH|SRUN)_'); do unset "$v"; done

BASE_CODE_PATH="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track"
cd "$BASE_CODE_PATH" || exit 1
python -u barlow_track/scripts/optimize_hyperparameters.py "$@"
