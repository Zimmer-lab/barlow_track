#!/bin/bash
# Position-only pilot on Flavell: can the ViSNet branch learn neuron identity when it is all the network has?
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl WANDB_MODE=offline OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/charles/anaconda3/envs/wbfm/bin/python
D=/home/charles/.cache/barlow_scratch/train/flavell_position_only_pilot
mkdir -p $D && cp barlow_track/scripts/scratch/flavell_position_only_pilot/train_config.yaml $D/
$PY -u barlow_track/scripts/train_barlow_clusterer.py -p $D/train_config.yaml || exit 1
$PY -u barlow_track/scripts/scratch/position_branch_matching.py --weights $D/resnet50.pth --out $D/position_matching.jsonl
$PY -u barlow_track/scripts/eval_accuracy.py --lab flavell --source project --project /home/charles/.cache/barlow_scratch/local_data/flavell/2025_07_01/project_config.yaml --gt /home/charles/.cache/barlow_scratch/local_data/flavell/flavell_data.nwb --mode trained --weights $D/resnet50.pth --device cuda --track_device cpu --emb_dir $D/emb --tag flavell_position_only_pilot --round_crop_centers --results_jsonl $D/eval_seed0.jsonl
