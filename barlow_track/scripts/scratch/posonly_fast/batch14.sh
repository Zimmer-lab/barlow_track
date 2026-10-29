#!/bin/bash
# Batch 14: segmentation-error augmentations (d05dec1) and real-noise-matched jitter, from the
# dropout-0 base at hidden 256 (frames 300, elastic 0, cutoff 0.35; 0.719/0.645/0.686 at epoch
# 14 over seeds 43-45). Real Zimmer frames: ~20% detections without a GT name, centroid noise
# (0.17, 1.05, 1.11) voxels, against the isotropic jitter_std 1.0. Seeds 43 and 44, two streams.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="visnet_cutoff_nn=0 num_frames=300 visnet_hidden=256 global_augment.elastic_scale=0.0 position_augment.dropout_p=0.0"
( for s in 43 44; do
    $P b14_jitz_s$s $B "position_augment.jitter_std=[0.2, 1.0, 1.0]" seed=$s
    $P b14_spur0.2_s$s $B position_augment.spurious_p=0.2 seed=$s
    $P b14_merge0.05_s$s $B position_augment.merge_p=0.05 seed=$s
  done ) &
( for s in 43 44; do
    $P b14_split0.05_s$s $B position_augment.split_p=0.05 seed=$s
    $P b14_all_s$s $B "position_augment.jitter_std=[0.2, 1.0, 1.0]" position_augment.spurious_p=0.2 \
        position_augment.merge_p=0.03 position_augment.split_p=0.03 seed=$s
  done ) &
wait
echo "batch14 done"
