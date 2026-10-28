#!/bin/bash
# Early look at the legacy-matched run (about 25 h): once its per-epoch checkpoint has EPOCH
# epochs, copy it out and score its backbone stage on Flavell (NWB GT) at truncated (the
# default, as r261026-1561729's image-only 0.819 was scored) and rounded crop centres (as it
# trains). Training writes args.pickle only at the end, so the model is built from the reaug
# cell's args.pickle: same architecture keys; only augmentation and project_dir differ, which
# loading ignores. Embeds on the GPU beside the training (~4 GB each, one at a time; training
# holds ~6 GB of 12), stops each embed once its cache is written, tracks on CPU.
# Usage: eval_partial.sh EPOCH
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl WANDB_MODE=offline OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/charles/anaconda3/envs/wbfm/bin/python
D=/home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_legacy_matched
P=$D/partial_ep$1
L=/home/charles/.cache/barlow_scratch/local_data/flavell
T=paper_flavell_position_pipeline_legacy_matched_ep$1_backbone
ckpt_epoch() { $PY -c "import torch; print(torch.load('$D/checkpoints/checkpoint.pth', map_location='cpu')['epoch'])" 2>/dev/null || echo 0; }
until [ "$(ckpt_epoch)" -ge "$1" ]; do sleep 300; done
mkdir -p $P/emb_trunc $P/emb_round
$PY -c "
import torch
s = torch.load('$D/checkpoints/checkpoint.pth', map_location='cpu')
print('checkpoint epoch', s['epoch'])
torch.save(s['model'], '$P/resnet50.pth')" || exit 1
cp /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_reaug/args.pickle $P/args.pickle
cp $D/train_config.yaml $P/
eval_args() {  # NAME EMB_DIR [extra args]
    local NAME=$1 E=$2; shift 2
    echo --lab flavell --source project --project $L/2025_07_01/project_config.yaml --gt $L/flavell_data.nwb \
        --mode trained --weights $P/resnet50.pth --track_device cpu --emb_dir $E --descriptor_stage backbone \
        --tag ${T}_$NAME --results_jsonl $D/eval_partial.jsonl "$@"
}
emb_done() { ls $1/emb_*.npz 2>/dev/null | grep -v '\.tmp\.npz$' | head -1; }
embed() {  # NAME EMB_DIR [extra args]
    $PY -u barlow_track/scripts/eval_accuracy.py $(eval_args "$@" --device cuda) > $P/embed_$1.log 2>&1 &
    local EMB=$!
    until [ -n "$(emb_done $2)" ]; do kill -0 $EMB 2>/dev/null || { echo "embed $1 died $(date)"; exit 1; }; sleep 20; done
    kill $EMB; wait $EMB 2>/dev/null
    echo "embedded $1 $(date)"
}
embed trunc $P/emb_trunc
embed round $P/emb_round --round_crop_centers
$PY -u barlow_track/scripts/eval_accuracy.py $(eval_args trunc $P/emb_trunc --device cpu --skip_embed) > $P/track_trunc.log 2>&1 &
A=$!
$PY -u barlow_track/scripts/eval_accuracy.py $(eval_args round $P/emb_round --round_crop_centers --device cpu --skip_embed) > $P/track_round.log 2>&1 &
B=$!
wait $A $B
echo "TRACK_DONE $(date)"
