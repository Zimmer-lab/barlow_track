#!/bin/bash
# Follow-up Samuel cells: paper-era network and image baseline on the other recordings,
# so every recording has {round_bayes_samuel trial 0, paper-era trial 8, image}.
# Waits for a running queue (arg 1: its PID, or 0) so only one GPU eval runs at a time.
# Same outputs and settings as run_queue.sh.
set -u
cd /home/charles/Current_work/repos/barlow_track
export NUMBA_CACHE_DIR=$HOME/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=$HOME/.cache/barlow_scratch/mpl
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
PY=/home/charles/anaconda3/envs/wbfm/bin/python
O=$HOME/.cache/barlow_scratch/runs/samuel_check
S=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/samuel_data
PAPER8=/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/hyperparameter_search_harvard_153_flip/trial_8/resnet50.pth
COMMON="--lab samuel --cluster labelprop --num_seeds 25 --device cuda --track_device cpu --results_jsonl $O/results.jsonl --emb_dir $O/emb"
N184=$S/thresholded_segmentation/184-004.nwb  # 184-004.nwb in samuel_data has no volumes

run() {  # tag, args...
  local tag=$1; shift
  echo "=== $(date +%T) start $tag"
  $PY -u barlow_track/scripts/eval_accuracy.py $COMMON --tag "$tag" "$@" > "$O/$tag.log" 2>&1
  echo "=== $(date +%T) end $tag rc=$? $(grep -o '"accuracy": [0-9.]*' "$O/$tag.log" | tail -1)"
}

wait_pid=${1:-0}; shift
while [ "$wait_pid" != 0 ] && kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done

for job in "$@"; do
  case $job in
    paper8_137_nwbgt) run sc_paper8_137_nwbgt --source nwb --nwb $S/137.nwb --gt $S/137.nwb --nwb_objects gt \
        --mode trained --weights $PAPER8 --descriptor_stage auto ;;
    paper8_184_nwbgt) run sc_paper8_184_nwbgt --source nwb --nwb $N184 --gt $N184 --nwb_objects gt \
        --mode trained --weights $PAPER8 --descriptor_stage auto ;;
    image_184_nwbgt) run sc_image_184_nwbgt --source nwb --nwb $N184 --gt $N184 --nwb_objects gt --mode image ;;
    *) echo "unknown job $job" ;;
  esac
done
echo "=== $(date +%T) queue done"
