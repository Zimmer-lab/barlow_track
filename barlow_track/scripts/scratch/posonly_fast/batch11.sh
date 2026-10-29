#!/bin/bash
# Batch 11: re-test capacity and steps on the new base (dropout 0, elastic 0: proxy 0.643 over
# two seeds), since a harder task under strong augmentation may have masked them. Frames 300,
# cutoff 0.35, 15 epochs, seeds 43 and 44. Runs beside long10's training.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="visnet_cutoff_nn=0 num_frames=300 visnet_hidden=128 global_augment.elastic_scale=0.0 position_augment.dropout_p=0.0"
for s in 43 44; do
  $P b11_h256_s$s $B visnet_hidden=256 seed=$s
  $P b11_l5vn_s$s $B visnet_layers=5 visnet_vecnorm=max_min seed=$s
  $P b11_lr3e-4_s$s $B lr=3.0e-4 seed=$s
  $P b11_frames1000_s$s $B num_frames=1000 seed=$s
done
echo "batch11 done"
