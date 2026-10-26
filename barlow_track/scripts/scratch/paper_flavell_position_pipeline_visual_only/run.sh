#!/bin/bash
# Same as paper_flavell_use_position, but the position branch is cut out of the descriptor (scratch/train_position_pipeline_visual_only.py).
# Reads a local mirror of the flavell project and GT NWB (rsync from barlow_track_paper/flavell_data);
# reading them over /lisc died twice on EIO.
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl WANDB_MODE=offline OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/charles/anaconda3/envs/wbfm/bin/python
$PY -u barlow_track/scripts/scratch/train_position_pipeline_visual_only.py -p /home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_visual_only/train_config.yaml || exit 1
$PY -u barlow_track/scripts/eval_accuracy.py --lab flavell --source project --project /home/charles/.cache/barlow_scratch/local_data/flavell/2025_07_01/project_config.yaml --gt /home/charles/.cache/barlow_scratch/local_data/flavell/flavell_data.nwb --mode trained --weights /home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_visual_only/resnet50.pth --device cuda --track_device cpu --emb_dir /home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_visual_only/emb --tag paper_flavell_position_pipeline_visual_only_backbone --descriptor_stage backbone --results_jsonl /home/charles/.cache/barlow_scratch/train/paper_flavell_position_pipeline_visual_only/eval_seed0.jsonl
