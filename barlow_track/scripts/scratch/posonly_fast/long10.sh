#!/bin/bash
# Two 60-epoch runs with full-video zimmer_1128 evals, differing only in object dropout
# (batch 9: dropout 0 lifts the epoch-14 proxy by 0.08 over 0.2). The proxy matches GT objects
# in both frames; the eval tracks the project segmentation, where real detections go missing,
# so it can show whether dropout earns its keep. Elastic off (batch 8), frames 600, hidden 128,
# cutoff 0.35, no vecnorm. Compare with long6: 0.893.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="visnet_cutoff_nn=0 num_frames=600 visnet_hidden=128 global_augment.elastic_scale=0.0 epochs=60"
$P long10_drop0 $B position_augment.dropout_p=0.0 --eval &
$P long10_drop0.2 $B position_augment.dropout_p=0.2 --eval &
wait
echo "long10 done"
