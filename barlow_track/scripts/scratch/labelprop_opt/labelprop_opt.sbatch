#!/usr/bin/env bash
# sweep_labelprop.py on the cluster: one array task per (case, seed), one GPU each. NOT submitted by
# the agent that wrote it; review first. Each task writes its own JSON lines file (no concurrent
# appends on the network file system); analyze.py reads them all.
#
#   BATCH=batch1 SET=dev SEEDS="0 1" sbatch --array=0-7 labelprop_opt.sbatch          # 4 cases x 2 seeds
#   BATCH=batch1 SET=confirm SEEDS="0 1 2" VARIANTS="base k10 layers30" sbatch --array=0-14 labelprop_opt.sbatch
#   python analyze.py $OUT_DIR/batch1_dev_*.jsonl
#
#SBATCH --job-name=labelprop_opt
#SBATCH --time=0-06:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a30|l4
#SBATCH --output=labelprop_opt_%A_%a.log
#SBATCH --no-requeue

set -euo pipefail
pwd; hostname; date
CODE=${CODE:-/lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track}
HERE=${CODE}/barlow_track/scripts/scratch/labelprop_opt
OUT_DIR=${OUT_DIR:-/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/labelprop_opt}
BATCH=${BATCH:-batch1}; SET=${SET:-dev}; SEEDS=(${SEEDS:-0 1}); VARIANTS=(${VARIANTS:-})
source "$HERE/cases.sh"
if [ "$SET" = dev ]; then CASES=("${DEV_CASES[@]}"); else CASES=("${CONFIRM_CASES[@]}"); fi
i=${SLURM_ARRAY_TASK_ID:-0}
case=${CASES[$((i / ${#SEEDS[@]}))]}
seed=${SEEDS[$((i % ${#SEEDS[@]}))]}

WORK=${TMPDIR:-/tmp}/labelprop_opt_${SLURM_JOB_ID:-local}_$i   # local disk: embedding copy, GT cache
mkdir -p "$WORK" "$OUT_DIR"
export NUMBA_CACHE_DIR=$WORK/numba MPLCONFIGDIR=$WORK/mpl TQDM_DISABLE=1 PYTHONPATH=${CODE}:${PYTHONPATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
VARG=(); [ ${#VARIANTS[@]} -gt 0 ] && VARG=(--variants "${VARIANTS[@]}")
tag=$(echo "$case" | tr ':' '_')
python -u -W ignore "$HERE/sweep_labelprop.py" --work "$WORK" --out "$OUT_DIR/${BATCH}_${SET}_${tag}_s${seed}.jsonl" \
    --batch "$BATCH" "${VARG[@]}" --seeds "$seed" --device cuda --case "$case"
date
