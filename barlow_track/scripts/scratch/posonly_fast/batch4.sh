#!/bin/bash
# Batch 4: scale and combine batch 3's winners (15 epochs each, from batch 1's best:
# cutoff_nn 0, elastic 1.0). num_frames 300 gave 0.521 vs 0.400 at 100 frames; visnet_hidden
# 128 0.451; lr 3e-4 0.436. More frames also means more steps per epoch, and held-out proxy
# frames sit closer in time to training frames.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
BEST="visnet_cutoff_nn=0 global_augment.elastic_scale=1.0"
$P b4_frames300_h128 $BEST num_frames=300 visnet_hidden=128
$P b4_frames300_h128_lr3e-4 $BEST num_frames=300 visnet_hidden=128 lr=3.0e-4
$P b4_frames600 $BEST num_frames=600
$P b4_frames1000 $BEST num_frames=1000
echo "batch4 done"
