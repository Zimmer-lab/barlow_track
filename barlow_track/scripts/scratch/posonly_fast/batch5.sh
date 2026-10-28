#!/bin/bash
# Batch 5: ViSNet vector norm (visnet_vecnorm=max_min, dfa4a92). Every RBF-fix run so far
# trained with vector features growing ~100x per layer on dense frames, hidden by the output
# LayerNorm; 5 layers gave a NaN loss. From batch 4's cheapest strong cell (frames 300,
# hidden 128: 0.548 at epoch 14), 15 epochs each. A larger cutoff means more neighbours,
# which made the blow-up worse, so it is retried only with the norm.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
BEST="visnet_cutoff_nn=0 global_augment.elastic_scale=1.0 num_frames=300 visnet_hidden=128"
$P b5_vecnorm $BEST visnet_vecnorm=max_min
$P b5_vecnorm_layers5 $BEST visnet_vecnorm=max_min visnet_layers=5
$P b5_vecnorm_cutoff0.5 $BEST visnet_vecnorm=max_min visnet_cutoff=0.5
echo "batch5 done"
