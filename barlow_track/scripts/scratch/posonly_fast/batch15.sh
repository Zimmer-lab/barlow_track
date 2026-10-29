#!/bin/bash
# Batch 15: keypoint geometry augmentations (62f6943) and GT-free jitter, from batch 14's base
# (hidden 256, frames 300, elastic 0, dropout 0; 0.719 at epoch 14, seed 43). Batch 14 found
# jitter (0.2, 1, 1) +0.07 but it is matched to GT-track noise, so here the GT-free 'auto'
# estimate (Zimmer ~(0.30, 0.91, 0.94)) stands in for it; jitter 0 and global scale jitter
# 0 / 0.2 bound how much positional noise matters at all. bend_max bends the in-plane
# principal axis (worm posture), axis_scale_jitter stretches axes independently (compression,
# z-spacing), cut_p drops one end of the field of view. Seeds 43 and 44, one stream after
# batch 14 (the GPU is shared with long13/long14's evals).
cd /home/charles/Current_work/repos/barlow_track
D=$HOME/.cache/barlow_scratch/train/posonly_fast
until grep -q "batch14 done" $D/batch14.log 2>/dev/null; do sleep 60; done
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="visnet_cutoff_nn=0 num_frames=300 visnet_hidden=256 global_augment.elastic_scale=0.0 position_augment.dropout_p=0.0"
for s in 43 44; do
  $P b15_jitauto_s$s $B position_augment.jitter_std=auto seed=$s
  $P b15_jit0_s$s $B position_augment.jitter_std=0.0 seed=$s
  $P b15_axscale0.1_s$s $B position_augment.axis_scale_jitter=0.1 seed=$s
  $P b15_bend0.3_s$s $B position_augment.bend_max=0.3 seed=$s
  $P b15_cut0.5_s$s $B position_augment.cut_p=0.5 position_augment.cut_max=0.15 seed=$s
  $P b15_gscale0_s$s $B global_augment.scale_jitter=0.0 seed=$s
  $P b15_gscale0.2_s$s $B global_augment.scale_jitter=0.2 seed=$s
done
echo "batch15 done"
