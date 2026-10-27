#!/bin/bash
# --nwb_objects gt vs all on the first 60 Flavell frames, legacy baseline checkpoint, CPU.
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/charles/anaconda3/envs/wbfm/bin/python
D=~/.cache/barlow_scratch/nwb_objects_check
W=~/.cache/barlow_scratch/train/baseline_flavell_head_rerun/resnet50.pth
N=~/.cache/barlow_scratch/local_data/flavell/flavell_data.nwb
for OBJ in gt all; do
$PY -u barlow_track/scripts/eval_accuracy.py --lab flavell --source nwb --nwb $N --gt $N --mode trained --weights $W \
  --device cpu --max_frames ${MAX_FRAMES:-60} --nwb_objects $OBJ --emb_dir $D/emb --tag nwbobj_${OBJ}_${MAX_FRAMES:-60} --results_jsonl $D/results.jsonl
done
