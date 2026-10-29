#!/bin/bash
# Small object dropout (the user's suggestion: 1-5%), same as long10 otherwise (elastic off,
# frames 600, hidden 128, cutoff 0.35, 60 epochs, full-video zimmer_1128 eval), to bracket
# long10's dropout 0 and 0.2 on the measure that sees missed detections.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="visnet_cutoff_nn=0 num_frames=600 visnet_hidden=128 global_augment.elastic_scale=0.0 epochs=60"
$P long13_drop0.02 $B position_augment.dropout_p=0.02 --eval &
$P long13_drop0.05 $B position_augment.dropout_p=0.05 --eval &
wait
echo "long13 done"
