#!/bin/bash
# Batch 18: do the Zimmer fast-loop settings transfer to Samuel and Flavell? From the batch-14
# base (hidden 256, 300 frames, elastic 0, dropout 0), on each lab's own recording and proxy:
# base, jitter_std auto (Zimmer +0.05 on both seeds; the GT-free estimate is 3x smaller than
# 1.0 on Samuel), visnet_cutoff_nn 6 (Zimmer proxy -0.05) and dropout 0.02. Seeds 43 and 44.
# NWORKERS parallel GPU workers (default 3; GNU xargs takes SIGUSR1 / SIGUSR2 to add / remove
# one); starts once long15's two runs have finished training (their evals then hold the GPU
# only while embedding). Cells with a resnet50.pth are skipped.
cd /home/charles/Current_work/repos/barlow_track
D=$HOME/.cache/barlow_scratch/train/posonly_fast
until [ -f $D/long15_cnn0/resnet50.pth ] && [ -f $D/long15_cnn6/resnet50.pth ]; do sleep 60; done
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="num_frames=300 visnet_hidden=256 global_augment.elastic_scale=0.0 position_augment.dropout_p=0.0"
cells() {
  for s in 43 44; do
    for lab in samuel flavell; do
      echo "$lab b18_${lab}_base_s$s"
      echo "$lab b18_${lab}_jitauto_s$s position_augment.jitter_std=auto"
      echo "$lab b18_${lab}_cnn6_s$s visnet_cutoff_nn=6"
      echo "$lab b18_${lab}_drop0.02_s$s position_augment.dropout_p=0.02"
    done
  done
}
run_cell() {
  lab=$1; name=$2; shift 2
  [ -f $D/$name/resnet50.pth ] && { echo "$name already done"; return; }
  seed=${name##*_s}
  $P --lab $lab $name $B "$@" seed=$seed > $D/$name.cell.log 2>&1
  echo "$name exit $? $(grep -o 'epoch  14  proxy [0-9.]*' $D/$name.cell.log | tail -1)"
}
export -f run_cell; export P B D
cells | xargs -P ${NWORKERS:-3} -L 1 bash -c 'run_cell "$@"' _
echo "queue18 done"
