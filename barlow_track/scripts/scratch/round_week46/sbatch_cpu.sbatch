#!/usr/bin/env bash
# CPU-only job running one python script of the repo; all arguments are the script and its arguments.
#SBATCH --job-name=barlow_cpu
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
pwd; hostname; date
cd "${CODE:-/lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track}" || exit 1
python -u "$@"
