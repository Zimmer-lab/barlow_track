#!/bin/bash
# Retrain the preprint flavell baseline config with HEAD code, then score it like E2.
cd /home/charles/Current_work/repos/barlow_track
while kill -0 2523206 2>/dev/null || kill -0 2523207 2>/dev/null; do sleep 60; done
export NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl WANDB_MODE=offline OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/charles/anaconda3/envs/wbfm/bin/python
$PY -u barlow_track/scripts/train_barlow_clusterer.py -p /home/charles/.cache/barlow_scratch/train/baseline_flavell_head_rerun/train_config.yaml || exit 1
$PY -u barlow_track/scripts/eval_accuracy.py --lab flavell --source project --project /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/projects/2025_07_01/project_config.yaml   --gt /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/images_for_charlie/flavell_data.nwb --mode trained --weights /home/charles/.cache/barlow_scratch/train/baseline_flavell_head_rerun/resnet50.pth --device cuda --track_device cpu   --emb_dir /home/charles/.cache/barlow_scratch/train/baseline_flavell_head_rerun/emb --tag baseline_flavell_head_rerun --results_jsonl /home/charles/.cache/barlow_scratch/train/baseline_flavell_head_rerun/eval_seed0.jsonl
