#!/bin/bash
# Batch 1: round_posonly's grid with the RBF fix, locally on Zimmer in minutes (15 epochs,
# keypoints_only), plus the unfixed encoder as reference. Proxy only; the best cells get --eval.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
$P b1_rbf_off visnet_rbf_rescale=false
for nn in 0 3 6 12; do
    for el in 0.0 1.0; do
        $P b1_nn${nn}_el${el} visnet_cutoff_nn=$nn global_augment.elastic_scale=$el
    done
done
echo "batch1 done"
