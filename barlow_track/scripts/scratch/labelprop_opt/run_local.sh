#!/bin/bash
# Local runner for sweep_labelprop.py: one process, cases in sequence, label propagation on the GPU.
# Resumable (re-running skips finished (case, variant, seed) triples).
#
#   bash run_local.sh batch1 dev "0 1"          # batch-1 screen: 4 cases x 2 seeds x 28 variants
#   bash run_local.sh batch1 confirm "0 1 2" base k10 layers30 ...   # confirmation of chosen variants
#   bash run_local.sh speed dev "0"             # speed/equivalence check (base, local, batch5, layers)
#
# Env: WORK (scratch dir, default ~/.cache/barlow_scratch/labelprop_opt), DEVICE (default cuda),
#      THREADS (CPU threads for NNDescent / sync / scoring, default 16)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/cases.sh"
BATCH=${1:?batch}; SET=${2:?dev|confirm}; SEEDS=${3:-"0 1"}; shift 3 || true
VARIANTS=("$@")
WORK=${WORK:-$HOME/.cache/barlow_scratch/labelprop_opt}
DEVICE=${DEVICE:-cuda}
THREADS=${THREADS:-16}
PY=${PY:-/home/charles/anaconda3/envs/wbfm/bin/python}
mkdir -p "$WORK"
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$WORK/numba} MPLCONFIGDIR=${MPLCONFIGDIR:-$WORK/mpl} TQDM_DISABLE=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMBA_NUM_THREADS=$THREADS
if [ "$SET" = dev ]; then CASES=("${DEV_CASES[@]}"); else CASES=("${CONFIRM_CASES[@]}"); fi
VARG=(); [ ${#VARIANTS[@]} -gt 0 ] && VARG=(--variants "${VARIANTS[@]}")
for c in "${CASES[@]}"; do
  # one process per case: frees GPU / host memory between recordings
  $PY -u "$HERE/sweep_labelprop.py" --work "$WORK" --out "$WORK/${BATCH}_${SET}.jsonl" --batch "$BATCH" \
      "${VARG[@]}" --seeds $SEEDS --device "$DEVICE" --case "$c" || echo "case $c failed ($?)"
done
$PY "$HERE/analyze.py" "$WORK/${BATCH}_${SET}.jsonl"
