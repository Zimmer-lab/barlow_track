#!/bin/bash
# Batch 19: does the transpose (object-space) loss term help where accuracy is not saturated?
# lambd_obj 0 (feature-space Barlow only) / 0.67 (default) / 1 (transpose only) crossed with
# 30 / 100 / 300 training frames (data efficiency), position-only on Zimmer from the
# batch-14 base, seeds 43 and 44. The per-epoch proxy curve also shows training speed
# (epochs to a given proxy). For the user's goal (d), 2026-10-29; the fused version needs
# the cluster. Starts after queue18; NWORKERS workers (default 3).
cd /home/charles/Current_work/repos/barlow_track
D=$HOME/.cache/barlow_scratch/train/posonly_fast
until grep -q "queue18 done" $D/queue18.log 2>/dev/null; do sleep 60; done
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="visnet_hidden=256 global_augment.elastic_scale=0.0 position_augment.dropout_p=0.0"
cells() {
  for s in 43 44; do
    for f in 30 100 300; do
      for l in 0.0 0.67 1.0; do
        echo "b19_f${f}_lobj${l}_s$s num_frames=$f lambd_obj=$l"
      done
    done
  done
}
run_cell() {
  name=$1; shift
  [ -f $D/$name/resnet50.pth ] && { echo "$name already done"; return; }
  seed=${name##*_s}
  $P $name $B "$@" seed=$seed > $D/$name.cell.log 2>&1
  echo "$name exit $? $(grep -o 'epoch  14  proxy [0-9.]*' $D/$name.cell.log | tail -1)"
}
export -f run_cell; export P B D
cells | xargs -P ${NWORKERS:-3} -L 1 bash -c 'run_cell "$@"' _
echo "queue19 done"
