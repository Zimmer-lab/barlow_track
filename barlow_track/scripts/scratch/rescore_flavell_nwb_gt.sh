#!/bin/bash
# Rescore round_flavell* sweep cells (scored against the project GT, ~4 pt low) against the
# flavell NWB GT behind the preprint's 96.7%, from each trial's cached embedding: CPU only.
# The embedding and checkpoint are copied locally first (reading over /lisc dies on EIO),
# project and GT come from the local mirror; results go to a local jsonl, not the round dir.
#
# Usage: rescore_flavell_nwb_gt.sh ROUND PARALLEL TRIAL [TRIAL ...]
#   e.g. rescore_flavell_nwb_gt.sh round_flavell_paper_ablation 4 4 5 6 7 12 13 14
ROUND=$1; PAR=$2; shift 2
SRC=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/$ROUND
LOCAL=~/.cache/barlow_scratch/local_data/flavell
OUT=~/.cache/barlow_scratch/rescore/$ROUND
mkdir -p $OUT/emb_cache $OUT/logs
cd /home/charles/Current_work/repos/barlow_track
# GPU hidden: --device cpu still opens a CUDA context (~430 MB each) next to a training run
export CUDA_VISIBLE_DEVICES= NUMBA_CACHE_DIR=~/.cache/barlow_scratch/numba_cache MPLCONFIGDIR=~/.cache/barlow_scratch/mpl OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export ROUND SRC LOCAL OUT
one() {
  N=$1; T=${ROUND}_trial${N}_flavell
  mkdir -p $OUT/trial_$N
  rsync -a $SRC/emb_cache/emb_flavell_trained_$T.npz $OUT/emb_cache/ && \
  rsync -a $SRC/trial_$N/resnet50.pth $SRC/trial_$N/args.pickle $SRC/trial_$N/train_config.yaml $OUT/trial_$N/ || { echo "copy failed for trial $N"; return 1; }
  /home/charles/anaconda3/envs/wbfm/bin/python -u barlow_track/scripts/eval_accuracy.py --lab flavell --source project \
      --project $LOCAL/2025_07_01/project_config.yaml --gt $LOCAL/flavell_data.nwb \
      --mode trained --weights $OUT/trial_$N/resnet50.pth --skip_embed --emb_dir $OUT/emb_cache --tag $T \
      --device cpu --track_device cpu --results_jsonl $OUT/rescore_nwb_gt.jsonl > $OUT/logs/trial_$N.log 2>&1
  echo "trial $N exit $?"
}
export -f one
printf '%s\n' "$@" | xargs -P $PAR -I{} bash -c 'one {}'
echo RESCORE_DONE
