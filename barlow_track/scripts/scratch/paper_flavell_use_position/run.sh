#!/bin/bash
# Preprint flavell config with use_position on (global augmentation + ViSNet add fusion), local single cell of round_flavell_paper_ablation.
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl WANDB_MODE=offline OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/charles/anaconda3/envs/wbfm/bin/python
$PY -u barlow_track/scripts/train_barlow_clusterer.py -p /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position/train_config.yaml || exit 1
for STAGE in auto backbone; do
$PY -u barlow_track/scripts/eval_accuracy.py --lab flavell --source project --project /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/projects/2025_07_01/project_config.yaml --gt /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/images_for_charlie/flavell_data.nwb --mode trained --weights /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position/resnet50.pth --device cuda --track_device cpu --emb_dir /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position/emb --tag paper_flavell_use_position_$STAGE --descriptor_stage $STAGE --results_jsonl /home/charles/.cache/barlow_scratch/train/paper_flavell_use_position/eval_seed0.jsonl
done
