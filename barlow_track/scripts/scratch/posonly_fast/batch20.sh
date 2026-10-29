#!/bin/bash
# Batch 20: does the transpose (object-space) loss help image networks, where it was meant to?
# Batch 19 found lambd_obj 0 best for position-only networks at 30-300 frames. Here the
# round_fused_fix_zimmer config (trial 27 + the RBF-fixed position branch) at 100 frames and
# 15 epochs (trial 27 trained on 100 frames), image-only (fusion visual_only) and fused
# (fusion add), lambd_obj 0 / 0.67, seeds 43 and 44; the per-epoch proxy also shows speed.
# For the user's goal (d), 2026-10-29. NWORKERS workers (default 2; crops use GPU memory).
cd /home/charles/Current_work/repos/barlow_track
D=$HOME/.cache/barlow_scratch/train/posonly_fast
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
BASE=barlow_track/scripts/scratch/round_fused_fix_zimmer/train_config.yaml
B="num_frames=100 epochs=15 wandb_name=null"
cells() {
  for s in 43 44; do
    for f in visual_only add; do
      for l in 0.0 0.67; do
        echo "b20_${f}_lobj${l}_s$s fusion=$f lambd_obj=$l"
      done
    done
  done
}
run_cell() {
  name=$1; shift
  [ -f $D/$name/resnet50.pth ] && { echo "$name already done"; return; }
  seed=${name##*_s}
  $P --base $BASE $name $B "$@" seed=$seed > $D/$name.cell.log 2>&1
  echo "$name exit $? $(grep -o 'epoch  14  proxy [0-9.]*' $D/$name.cell.log | tail -1)"
}
export -f run_cell; export P B D BASE
cells | xargs -P ${NWORKERS:-2} -L 1 bash -c 'run_cell "$@"' _
echo "batch20 done"
