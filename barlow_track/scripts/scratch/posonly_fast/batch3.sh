#!/bin/bash
# Batch 3: what lifts the position-only plateau (b2_nn0_el1.0_ep60 levels off at proxy ~0.50
# by epoch 50). One change per cell from batch 1's best (cutoff_nn 0, elastic 1.0; 0.400 at
# epoch 14), 15 epochs each. reaugment_each_epoch tests issue i261027-1 (the same view pairs
# every epoch); lr 3e-4 may only look better early.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
BEST="visnet_cutoff_nn=0 global_augment.elastic_scale=1.0"
$P b3_reaug $BEST reaugment_each_epoch=true
$P b3_frames300 $BEST num_frames=300
$P b3_hidden128 $BEST visnet_hidden=128
$P b3_layers5 $BEST visnet_layers=5
$P b3_lr3e-4 $BEST lr=3.0e-4
$P b3_cutoff0.25 $BEST visnet_cutoff=0.25
echo "batch3 done"
