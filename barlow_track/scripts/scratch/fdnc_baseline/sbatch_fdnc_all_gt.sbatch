#!/usr/bin/env bash
# fDNC competitor baseline on the nine GT recordings, CPU only, one array task per
# (recording, orientation). NOT submitted by the agent that wrote it; review first.
#
#   sbatch --array=0-35 sbatch_fdnc_all_gt.sbatch          # 9 recordings x 4 orientations
#   OUT_DIR=/some/dir sbatch --array=0-35 sbatch_fdnc_all_gt.sbatch
#
#SBATCH --job-name=fdnc_baseline
#SBATCH --time=0-06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=fdnc_%A_%a.log

set -euo pipefail
pwd; hostname; date

CODE=${CODE:-/lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track}
FDNC=${FDNC:-/lisc/data/scratch/neurobiology/zimmer/fieseler/github_repos/fDNC_Neuron_ID}
OUT_DIR=${OUT_DIR:-/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/fdnc_baseline}
SCRIPT=${CODE}/barlow_track/scripts/scratch/fdnc_baseline/fdnc_track_and_score.py

# dataset label : source (labels are run_trials_on_all_ground_truth.DATASETS keys)
DATASETS=(zimmer_1123:project zimmer_1128:project zimmer_1210:project
          flavell:nwb flavell_2022031603:nwb flavell_2022032402:nwb flavell_2022033001:nwb
          flavell_2022033002:nwb samuel:nwb)
ORIENTS=(default swap mirror swap_mirror)

i=${SLURM_ARRAY_TASK_ID:-0}
entry=${DATASETS[$((i / ${#ORIENTS[@]}))]}
orient=${ORIENTS[$((i % ${#ORIENTS[@]}))]}
ds=${entry%%:*}; src=${entry##*:}

export CUDA_VISIBLE_DEVICES="" TQDM_DISABLE=1
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_$SLURM_JOB_ID MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_$SLURM_JOB_ID
export PYTHONPATH=${FDNC}:${CODE}:${PYTHONPATH:-}
mkdir -p "$OUT_DIR"

# Template frames: the wbfm pipeline template (10) plus four spread over the video;
# the script also scores their GT-free multi-template vote.
python -W ignore "$SCRIPT" --dataset "$ds" --source "$src" --orient "$orient" --rescale ref \
    --templates 10 --n_spread 4 --threads "${SLURM_CPUS_PER_TASK:-4}" --out_dir "$OUT_DIR"
date
