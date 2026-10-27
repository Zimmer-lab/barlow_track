#!/bin/bash
# The reaug training crashed in the epoch-31 validation plot (1-object view, fixed in the
# commit that adds this script) before saving resnet50.pth. Evaluate the last per-epoch
# checkpoint (32 epochs done) from its own folder, as run.sh would have evaluated the final one.
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl WANDB_MODE=offline OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/charles/anaconda3/envs/wbfm/bin/python
D=/home/charles/.cache/barlow_scratch/train/paper_flavell_use_position_reaug
P=$D/partial_ep32
L=/home/charles/.cache/barlow_scratch/local_data/flavell
mkdir -p $P
[ -f $P/resnet50.pth ] || $PY -c "import torch; torch.save(torch.load('$D/checkpoints/checkpoint-1.pth', map_location='cpu')['model'], '$P/resnet50.pth')" || exit 1
cp -n $D/args.pickle $D/train_config.yaml $P/
for STAGE in auto backbone; do
$PY -u barlow_track/scripts/eval_accuracy.py --lab flavell --source project --project $L/2025_07_01/project_config.yaml --gt $L/flavell_data.nwb --mode trained --weights $P/resnet50.pth --device cuda --track_device cpu --emb_dir $P/emb --tag paper_flavell_use_position_reaug_ep32_$STAGE --descriptor_stage $STAGE --results_jsonl $D/eval_seed0.jsonl
done
