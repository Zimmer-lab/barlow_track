#!/bin/bash
# Sequential Samuel-lab accuracy checks (one GPU eval at a time; tracking on CPU).
# Outputs: $HOME/.cache/barlow_scratch/runs/samuel_check/{results.jsonl,emb/,<tag>.log}
# Usage: bash run_queue.sh <job> [<job> ...]   (jobs defined below)
set -u
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=$HOME/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=$HOME/.cache/barlow_scratch/mpl
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
PY=/home/charles/anaconda3/envs/wbfm/bin/python
O=$HOME/.cache/barlow_scratch/runs/samuel_check
S=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/samuel_data
BAYES0=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/round_bayes_samuel/trial_0/resnet50.pth
PAPER8=/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/hyperparameter_search_harvard_153_flip/trial_8/resnet50.pth
COMMON="--lab samuel --cluster labelprop --num_seeds 25 --device cuda --track_device cpu --results_jsonl $O/results.jsonl --emb_dir $O/emb"

run() {  # tag, args...
  local tag=$1; shift
  echo "=== $(date +%T) start $tag"
  $PY -u barlow_track/scripts/eval_accuracy.py $COMMON --tag "$tag" "$@" > "$O/$tag.log" 2>&1
  echo "=== $(date +%T) end $tag rc=$? $(grep -o '"accuracy": [0-9.]*' "$O/$tag.log" | tail -1)"
}

for job in "$@"; do
  case $job in
    bayes0_153_nwball) run sc_bayes0_153_nwball --source nwb --nwb $S/153.nwb --gt $S/153.nwb --nwb_objects all \
        --mode trained --weights $BAYES0 --descriptor_stage auto ;;
    bayes0_153_project) run sc_bayes0_153_project --source project --project $S/projects/153/project_config.yaml \
        --gt $S/projects/153/project_config.yaml --mode trained --weights $BAYES0 --descriptor_stage auto ;;
    paper8_153_nwbgt) run sc_paper8_153_nwbgt --source nwb --nwb $S/153.nwb --gt $S/153.nwb --nwb_objects gt \
        --mode trained --weights $PAPER8 --descriptor_stage auto ;;
    paper8_153_nwbgt_global) run sc_paper8_153_nwbgt --source nwb --nwb $S/153.nwb --gt $S/153.nwb --nwb_objects gt \
        --mode trained --weights $PAPER8 --descriptor_stage auto --skip_embed --cluster global ;;
    bayes0_153_nwbgt_global) run sc_bayes0_153_nwbgt --source nwb --nwb $S/153.nwb --gt $S/153.nwb --nwb_objects gt \
        --mode trained --weights $BAYES0 --descriptor_stage auto --skip_embed --cluster global ;;
    bayes0_137_nwbgt) run sc_bayes0_137_nwbgt --source nwb --nwb $S/137.nwb --gt $S/137.nwb --nwb_objects gt \
        --mode trained --weights $BAYES0 --descriptor_stage auto ;;
    bayes0_184_nwbgt) run sc_bayes0_184_nwbgt --source nwb --nwb $S/thresholded_segmentation/184-004.nwb \
        --gt $S/thresholded_segmentation/184-004.nwb --nwb_objects gt --mode trained --weights $BAYES0 --descriptor_stage auto ;;
    image_153_nwbgt) run sc_image_153_nwbgt --source nwb --nwb $S/153.nwb --gt $S/153.nwb --nwb_objects gt --mode image ;;
    image_137_nwbgt) run sc_image_137_nwbgt --source nwb --nwb $S/137.nwb --gt $S/137.nwb --nwb_objects gt --mode image ;;
    *) echo "unknown job $job" ;;
  esac
done
echo "=== $(date +%T) queue done"
