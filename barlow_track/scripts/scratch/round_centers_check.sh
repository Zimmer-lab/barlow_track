#!/bin/bash
# Does the half-voxel crop-centre offset at eval cost the position networks accuracy? Training
# crops at rounded centroids; the project-path eval truncates them by default (the legacy
# pipeline truncates in both). Re-embeds the comparator paper_flavell_use_position (auto,
# 0.795 truncated) with --round_crop_centers, then hands the GPU to the legacy-matched
# training once EVAL_PID (an eval holding GPU memory) ends, and tracks the rounded embedding
# on CPU from cache (eval_accuracy keeps ~4 GB of CUDA memory through label propagation).
# Usage: round_centers_check.sh EVAL_PID
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl WANDB_MODE=offline OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/charles/anaconda3/envs/wbfm/bin/python
D=/home/charles/.cache/barlow_scratch/train/paper_flavell_use_position
O=/home/charles/.cache/barlow_scratch/round_centers_check
L=/home/charles/.cache/barlow_scratch/local_data/flavell
M=barlow_track/scripts/scratch/paper_flavell_position_pipeline_legacy_matched
mkdir -p $O/emb
eval_args() {
    echo --lab flavell --source project --project $L/2025_07_01/project_config.yaml --gt $L/flavell_data.nwb \
        --mode trained --weights $D/resnet50.pth --track_device cpu --emb_dir $O/emb --round_crop_centers \
        --tag paper_flavell_use_position_auto_round --descriptor_stage auto --results_jsonl $O/results.jsonl "$@"
}
emb_done() { ls $O/emb/emb_*.npz 2>/dev/null | grep -v '\.tmp\.npz$' | head -1; }

$PY -u barlow_track/scripts/eval_accuracy.py $(eval_args --device cuda) > $O/embed.log 2>&1 &
EMB=$!
until [ -n "$(emb_done)" ]; do kill -0 $EMB 2>/dev/null || { echo "embed died $(date)"; exit 1; }; sleep 20; done
kill $EMB; wait $EMB 2>/dev/null
echo "embedded $(emb_done) $(date)"

while kill -0 $1 2>/dev/null; do sleep 30; done
MD=/home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_legacy_matched
mkdir -p $MD
nohup bash $M/run.sh > $MD/run.log 2>&1 &
echo "legacy_matched pid $! $(date)"

$PY -u barlow_track/scripts/eval_accuracy.py $(eval_args --device cpu --skip_embed) > $O/track.log 2>&1
echo "TRACK_DONE $(date)"
