#!/bin/bash
# Hand the GPU from the norot evals to the croprot training. eval_accuracy.py keeps its
# CUDA memory (~3 GB) through CPU label propagation, and croprot training needs ~10 GB of
# the 12, so: let run.sh embed the auto stage, stop it once the embedding is cached, embed
# the backbone stage the same way, start croprot, then track both stages from the cached
# embeddings on CPU (--skip_embed --device cpu; identical embeddings, so identical scores).
# Usage: eval_then_croprot.sh RUN_SH_PID EVAL_PID
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl WANDB_MODE=offline OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/charles/anaconda3/envs/wbfm/bin/python
D=/home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_norot
C=/home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_croprot
L=/home/charles/.cache/barlow_scratch/local_data/flavell
T=paper_flavell_use_position_norot
eval_args() {  # STAGE [extra args]
    local STAGE=$1; shift
    echo --lab flavell --source project --project $L/2025_07_01/project_config.yaml --gt $L/flavell_data.nwb \
        --mode trained --weights $D/resnet50.pth --track_device cpu --emb_dir $D/emb \
        --tag ${T}_$STAGE --descriptor_stage $STAGE --results_jsonl $D/eval_seed0.jsonl "$@"
}
wait_for() { until [ -f "$1" ]; do sleep 20; done; }

wait_for $D/emb/emb_flavell_trained_${T}_auto.npz
kill $1; kill $2
echo "auto embedded $(date)"

$PY -u barlow_track/scripts/eval_accuracy.py $(eval_args backbone --device cuda) > $D/embed_backbone.log 2>&1 &
EMB=$!
wait_for $D/emb/emb_flavell_trained_stagebackbone_${T}_backbone.npz
kill $EMB; wait $EMB 2>/dev/null
echo "backbone embedded $(date)"

nohup bash $C/run.sh > $C/run.log 2>&1 &
echo "croprot pid $! $(date)"

PIDS=""
for STAGE in auto backbone; do
    $PY -u barlow_track/scripts/eval_accuracy.py $(eval_args $STAGE --device cpu --skip_embed) > $D/eval_$STAGE.log 2>&1 &
    PIDS="$PIDS $!"
done
wait $PIDS
echo "EVALS_DONE $(date)"
