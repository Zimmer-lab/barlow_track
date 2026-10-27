#!/bin/bash
# Track cached Flavell embeddings after per-frame centring (eval --center_per_volume), CPU.
# Comparator use_position cell (0.795 uncentred) and the legacy baseline (0.956) as control.
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/charles/anaconda3/envs/wbfm/bin/python
T=~/.cache/barlow_scratch/train
D=~/.cache/barlow_scratch/centered_check
L=~/.cache/barlow_scratch/local_data/flavell
mkdir -p $D/emb
for NAME in paper_flavell_use_position:paper_flavell_use_position_auto baseline_flavell_head_rerun:baseline_flavell_head_rerun; do
  NET=${NAME%%:*}; TAG=${NAME##*:}
  $PY -u barlow_track/scripts/scratch/center_cached_embeddings.py $T/$NET/emb/emb_flavell_trained_$TAG.npz \
      $D/emb/emb_flavell_trained_centered_${TAG}_centered.npz || exit 1
  $PY -u barlow_track/scripts/eval_accuracy.py --lab flavell --source project --project $L/2025_07_01/project_config.yaml \
      --gt $L/flavell_data.nwb --mode trained --weights $T/$NET/resnet50.pth --device cpu --track_device cpu \
      --center_per_volume --skip_embed --emb_dir $D/emb --tag ${TAG}_centered --results_jsonl $D/results.jsonl
done
