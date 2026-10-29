#!/bin/bash
# Batch 17: visnet_cutoff_nn on the local base. round_posonly_zimmer (RBF fix on, elastic 0)
# scored cutoff_nn 6 0.868 against 0.825 at 0 (the fixed 0.35 cutoff every local batch uses),
# 12 0.801, 3 0.691. Same base as batches 14-16, seeds 43 and 44, cutoff_nn 6 and 12, and 6
# with the GT-free jitter; one worker after queue15_16 (the GPU is shared during the day).
cd /home/charles/Current_work/repos/barlow_track
D=$HOME/.cache/barlow_scratch/train/posonly_fast
until grep -q "queue15_16 done" $D/queue15_16b.log 2>/dev/null; do sleep 60; done
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="num_frames=300 visnet_hidden=256 global_augment.elastic_scale=0.0 position_augment.dropout_p=0.0"
for s in 43 44; do
  $P b17_cnn6_s$s $B visnet_cutoff_nn=6 seed=$s
  $P b17_cnn12_s$s $B visnet_cutoff_nn=12 seed=$s
  $P b17_cnn6_jitauto_s$s $B visnet_cutoff_nn=6 position_augment.jitter_std=auto seed=$s
done
echo "batch17 done"
