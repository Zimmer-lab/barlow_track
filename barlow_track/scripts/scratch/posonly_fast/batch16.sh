#!/bin/bash
# Batch 16: real frame pairs (position_augment pair_p, 2676174): the second view is frame
# t+1 with identities from GT-free non-rigid matching (Zimmer lag 1: ~42 matches per frame
# at 0.96 precision, scratch/pair_pseudolabels.py), unmatched objects id -1. Real non-rigid
# motion instead of synthetic warps, which the epoch-14 proxy (real frame pairs scored
# against GT) should reward if synthetic augmentation is the gap. Same base as batches 14-15,
# seeds 43 and 44, one stream after batch 15.
cd /home/charles/Current_work/repos/barlow_track
D=$HOME/.cache/barlow_scratch/train/posonly_fast
until grep -q "batch15 done" $D/batch15.log 2>/dev/null; do sleep 60; done
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="visnet_cutoff_nn=0 num_frames=300 visnet_hidden=256 global_augment.elastic_scale=0.0 position_augment.dropout_p=0.0"
for s in 43 44; do
  $P b16_pair0.5_s$s $B position_augment.pair_p=0.5 seed=$s
  $P b16_pair1.0_s$s $B position_augment.pair_p=1.0 seed=$s
  $P b16_pair0.5_lag3_s$s $B position_augment.pair_p=0.5 position_augment.pair_lag=3 seed=$s
  $P b16_pair0.5_r1.5_s$s $B position_augment.pair_p=0.5 position_augment.pair_resid=1.5 seed=$s
done
echo "batch16 done"
