#!/bin/bash
# eval_accuracy's opt-in --gt_matching gt_valid and --svd whiten vs the defaults, from the
# cached Samuel embeddings of the backbone rescore (no re-embedding; tracking on the CPU).
cd /home/charles/Current_work/repos/barlow_track
PY=/home/charles/anaconda3/envs/wbfm/bin/python
R=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
SRC=/home/charles/.cache/barlow_scratch/runs/backbone_rescore_off_flavell
OUT=/home/charles/.cache/barlow_scratch/runs/svd_gt_matching_check
mkdir -p $OUT
for cell in "round_bayes_samuel 0 emb_round_bayes_samuel_backbone" "round_visnet_full 27 emb_round_visnet_full_backbone"; do
    set -- $cell
    for opt in "--gt_matching nandist --svd scaled" "--gt_matching gt_valid" "--svd whiten"; do
        $PY -u barlow_track/scripts/eval_accuracy.py --lab samuel --source nwb --mode trained --device cpu \
            --track_device cpu --cluster labelprop --num_seeds 25 --descriptor_stage backbone --skip_embed \
            --tag ${1}_trial${2}_samuel --results_jsonl $OUT/results.jsonl --emb_dir $SRC/$3 \
            --weights $R/$1/trial_$2/resnet50.pth \
            --nwb /lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/samuel_data/153.nwb $opt
    done
done
echo "svd_gt_matching_check done"
