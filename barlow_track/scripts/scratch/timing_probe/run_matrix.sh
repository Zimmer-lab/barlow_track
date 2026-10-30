#!/bin/bash
# Timing matrix: fused (add) and keypoints-only networks on each recording, GPU vs 16 CPU
# threads, 20 frames x 1 epoch, plus each recording's objective eval on 16 CPU threads.
# Three lanes on disjoint cores: GPU training (cores 48-55), CPU fused training + eval
# (0-15), CPU position-only training (16-31).
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/timing_probe/timing_probe.py"
LABELS="samuel zimmer_1128 samuel_137 flavell_2022033001 leifer_reexport"
lane_gpu() { for l in $LABELS; do for v in add posonly; do
  $P --label $l --variant $v --device cuda --threads 8 --first_core 48; echo "gpu $l $v exit $?"; done; done; }
lane_cpu_add() { for l in $LABELS; do
  $P --label $l --variant add --device cpu --threads 16 --first_core 0 --eval; echo "cpu $l add+eval exit $?"; done; }
lane_cpu_pos() { for l in $LABELS; do
  $P --label $l --variant posonly --device cpu --threads 16 --first_core 16; echo "cpu $l posonly exit $?"; done; }
lane_gpu & lane_cpu_add & lane_cpu_pos & wait
echo "timing matrix done"
