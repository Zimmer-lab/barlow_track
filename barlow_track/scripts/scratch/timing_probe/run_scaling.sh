#!/bin/bash
# Scaling follow-up to run_matrix.sh: (1) fused GPU training with 4 DataLoader workers vs 0
# (Samuel, the one fused recording that fits next to the other GPU job); (2) position-only
# CPU training at 4 / 8 / 32 threads (16 is in the matrix), Zimmer and Leifer. Sequential.
cd /home/charles/Current_work/repos/barlow_track
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/timing_probe/timing_probe.py"
for w in 0 4; do
  $P --label samuel --variant add --device cuda --threads 8 --first_core 40 --set num_workers=$w; echo "gpu samuel add workers $w exit $?"
done
for l in zimmer_1128 leifer_reexport; do for n in 4 8 32; do
  $P --label $l --variant posonly --device cpu --threads $n --first_core 16; echo "cpu $l posonly threads $n exit $?"
done; done
echo "scaling done"
