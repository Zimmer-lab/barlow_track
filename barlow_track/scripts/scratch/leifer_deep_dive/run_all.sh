#!/usr/bin/env bash
# The Leifer deep dive, one step at a time with a per-process memory cap (ulimit -v). Running
# three label propagations and the 16-worker geometry pass at once pushed the user slice past
# systemd-oomd's pressure limit twice (2026-11-02), which killed the terminal scope with them.
#   bash barlow_track/scripts/scratch/leifer_deep_dive/run_all.sh <trial emb npz> <out dir>
set -u
EMB=$1
OUT=$2
HERE=barlow_track/scripts/scratch/leifer_deep_dive
PY=/home/charles/anaconda3/envs/wbfm/bin/python
export CUDA_VISIBLE_DEVICES= NUMBA_CACHE_DIR=$HOME/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=$HOME/.cache/barlow_scratch/mpl
mkdir -p "$OUT"

step() {  # name, cap in GB, command...
  local name=$1 cap=$2; shift 2
  echo "== $(date +%T) $name (cap ${cap} GB)"
  ( ulimit -v $((cap * 1024 * 1024)); "$@" ) > "$OUT/$name.log" 2>&1
  echo "   exit $? at $(date +%T)"
}

step blank 16 env OMP_NUM_THREADS=4 $PY -u $HERE/blank_volumes.py
step masks 48 env OMP_NUM_THREADS=1 $PY -u $HERE/mask_geometry.py --out "$OUT/masks.npz" --workers 4
for s in all nerve gt; do
  step "decompose_$s" 90 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    $PY -u $HERE/decompose_errors.py --emb "$EMB" --subset $s --out "$OUT"
done
echo "== done $(date +%T)"
