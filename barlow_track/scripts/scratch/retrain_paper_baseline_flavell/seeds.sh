#!/bin/bash
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for S in 1 2; do
/home/charles/anaconda3/envs/wbfm/bin/python -u barlow_track/scripts/eval_accuracy.py --lab flavell --source project --project /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/projects/2025_07_01/project_config.yaml --gt /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/images_for_charlie/flavell_data.nwb --mode trained --weights /home/charles/.cache/barlow_scratch/train/baseline_flavell_head_rerun/resnet50.pth --skip_embed --track_device cpu --emb_dir /home/charles/.cache/barlow_scratch/train/baseline_flavell_head_rerun/emb --tag baseline_flavell_head_rerun --seed $S --results_jsonl /home/charles/.cache/barlow_scratch/train/baseline_flavell_head_rerun/eval_seeds.jsonl
done
