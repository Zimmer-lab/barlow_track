#!/bin/bash
# Batch 8: augmentation strength, the axis batches 1-7 left untested (batch 1's elastic 1.0 vs 0
# was inside the ~0.04 seed noise). From frames 300, hidden 128, no vecnorm, cutoff 0.35 (mean
# 0.554 over three seeds at epoch 14), 15 epochs, seeds 43 and 44 per cell. Two streams.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="visnet_cutoff_nn=0 num_frames=300 visnet_hidden=128"
( for s in 43 44; do
    $P b8_el0_s$s $B global_augment.elastic_scale=0.0 seed=$s
    $P b8_el2_s$s $B global_augment.elastic_scale=2.0 seed=$s
  done ) &
( for s in 43 44; do
    $P b8_jit2_s$s $B global_augment.elastic_scale=1.0 position_augment.jitter_std=2.0 seed=$s
    $P b8_drop0.4_s$s $B global_augment.elastic_scale=1.0 position_augment.dropout_p=0.4 seed=$s
  done ) &
wait
echo "batch8 done"
