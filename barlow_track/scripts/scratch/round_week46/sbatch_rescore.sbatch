#!/usr/bin/env bash
# One array task = one shard of rescore_round.py (re-track cached embeddings, save tracks). Arguments go to it.
#SBATCH --job-name=rescore
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a30|t4|l4
#SBATCH --no-requeue
pwd; hostname; date
cd "${CODE:-/lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track}" || exit 1
python -u barlow_track/scripts/scratch/gt_free_proxy/rescore_round.py "$@" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}"
