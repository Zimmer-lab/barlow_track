#!/bin/bash
# Batch 22: keypoints in a less anisotropic metric before ViSNet (visnet_z_scale, d9cfd1a).
# Centroids reach ViSNet in voxel units scaled by one scalar, so z is compressed by the voxel
# aspect (Zimmer 1.5 / 0.325 = 4.6; the other labs' project physical units are placeholders).
# Values: Zimmer the physical 4.6 and the GT-free nearest-neighbour spacing ratio (~1.55, xy
# vs z NN offsets) plus 3; Samuel NN ratio ~2.9 and 5; Flavell 2 and 4. Batch-18 base with
# jitter auto (the best GT-free jitter), seeds 43 / 44, epoch-14 proxy. The proxy misjudged
# cutoff_nn on Samuel, so winners get a full-video check. From the encoder literature search
# (2026-10-29). NWORKERS workers (default 1: batch 20 / 21 share the GPU).
cd /home/charles/Current_work/repos/barlow_track
D=$HOME/.cache/barlow_scratch/train/posonly_fast
P="/home/charles/anaconda3/envs/wbfm/bin/python barlow_track/scripts/scratch/posonly_fast/posonly_fast.py"
B="num_frames=300 visnet_hidden=256 global_augment.elastic_scale=0.0 position_augment.dropout_p=0.0 position_augment.jitter_std=auto"
cells() {
  for s in 43 44; do
    for zs in 1.0 1.55 3.0 4.6; do echo "zimmer b22_zimmer_zs${zs}_s$s visnet_z_scale=$zs"; done
    for zs in 1.0 2.9 5.0; do echo "samuel b22_samuel_zs${zs}_s$s visnet_z_scale=$zs"; done
    for zs in 1.0 2.0 4.0; do echo "flavell b22_flavell_zs${zs}_s$s visnet_z_scale=$zs"; done
  done
}
run_cell() {
  lab=$1; name=$2; shift 2
  [ -f $D/$name/resnet50.pth ] && { echo "$name already done"; return; }
  seed=${name##*_s}
  $P --lab $lab $name $B "$@" seed=$seed > $D/$name.cell.log 2>&1
  echo "$name exit $? $(grep -o 'epoch  14  proxy [0-9.]*' $D/$name.cell.log | tail -1)"
}
export -f run_cell; export P B D
cells | xargs -P ${NWORKERS:-1} -L 1 bash -c 'run_cell "$@"' _
echo "batch22 done"
