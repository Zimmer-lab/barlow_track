#!/bin/bash
# Batch 6: around batch 5's best (vecnorm max_min, cutoff 0.5, frames 300, hidden 128: 0.589
# at epoch 14), 15 epochs each. A seed repeat sizes the noise between cells; cutoff 0.75 and
# 64 neighbours ask whether the larger neighbourhood helps further (at cutoff 0.5 most nodes
# hit the 32-neighbour cap); layers 5 stacks batch 5's other gain.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
BEST="visnet_cutoff_nn=0 global_augment.elastic_scale=1.0 num_frames=300 visnet_hidden=128 visnet_vecnorm=max_min visnet_cutoff=0.5"
$P b6_seed44 $BEST seed=44
$P b6_cutoff0.75 $BEST visnet_cutoff=0.75
$P b6_nbr64 $BEST visnet_neighbors=64
$P b6_layers5 $BEST visnet_layers=5
echo "batch6 done"
