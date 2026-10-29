#!/bin/bash
# Batch 12: width on the dropout-0 base (batch 11: hidden 256 0.719/0.645 vs 128 0.657/0.629,
# at the noise edge). Third seed for 128 and 256, and hidden 512. Frames 300, 15 epochs.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="visnet_cutoff_nn=0 num_frames=300 global_augment.elastic_scale=0.0 position_augment.dropout_p=0.0"
$P b12_h128_s45 $B visnet_hidden=128 seed=45
$P b12_h256_s45 $B visnet_hidden=256 seed=45
for s in 43 44; do $P b12_h512_s$s $B visnet_hidden=512 seed=$s; done
echo "batch12 done"
