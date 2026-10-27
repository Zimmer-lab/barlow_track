#!/bin/bash
# Position data path matched to the legacy crop pipeline (see train_config.yaml's header), then backbone eval.
# Reads the local mirror of the flavell project and GT NWB.
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl WANDB_MODE=offline OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/charles/anaconda3/envs/wbfm/bin/python
S=barlow_track/scripts/scratch/paper_flavell_position_pipeline_legacy_matched
D=/home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_legacy_matched
L=/home/charles/.cache/barlow_scratch/local_data/flavell
mkdir -p $D/log $D/checkpoints
cp $S/train_config.yaml $D/train_config.yaml
$PY -u barlow_track/scripts/scratch/train_position_pipeline_legacy_matched.py -p $D/train_config.yaml || exit 1
$PY -u barlow_track/scripts/eval_accuracy.py --lab flavell --source project --project $L/2025_07_01/project_config.yaml --gt $L/flavell_data.nwb --mode trained --weights $D/resnet50.pth --device cuda --track_device cpu --emb_dir $D/emb --tag paper_flavell_position_pipeline_legacy_matched_backbone --descriptor_stage backbone --results_jsonl $D/eval_seed0.jsonl
