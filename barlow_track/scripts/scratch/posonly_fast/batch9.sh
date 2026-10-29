#!/bin/bash
# Batch 9: weaker augmentation with elastic off (batch 8: every stronger setting hurt the
# proxy, dropout most; elastic 0 matched or beat 1.0 and is 4x faster). From frames 300, hidden
# 128, cutoff 0.35, seeds 43 and 44, 15 epochs. The proxy matches GT objects only, so it
# cannot see what dropout buys against real missed detections: winners go to a full eval.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="visnet_cutoff_nn=0 num_frames=300 visnet_hidden=128 global_augment.elastic_scale=0.0"
( for s in 43 44; do
    $P b9_drop0.1_s$s $B position_augment.dropout_p=0.1 seed=$s
    $P b9_drop0_s$s $B position_augment.dropout_p=0.0 seed=$s
    $P b9_jit0.5_s$s $B position_augment.jitter_std=0.5 seed=$s
  done ) &
( for s in 43 44; do
    $P b9_frames600_s$s $B num_frames=600 seed=$s
  done ) &
wait
echo "batch9 done"
