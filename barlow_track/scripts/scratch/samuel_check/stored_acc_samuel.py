"""Accuracy of stored paper-era Samuel tracks (df_barlow_tracks.h5) vs a GT project.

Same recipe as barlow_track/scripts/stored_acc.py (rename_columns_using_matching on
raw_segmentation_id, then calculate_accuracy). Read-only on /lisc.

Usage: python stored_acc_samuel.py <gt project_config.yaml> <out.jsonl> <df_barlow_tracks.h5> [...]
"""
import json
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
from barlow_track.utils.utils_ground_truth import (
    calculate_accuracy, pad_with_nan_rows, align_gt_pred_time_index)


def main(gt_cfg, out, preds):
    g = ProjectData.load_final_project_data(gt_cfg, allow_hybrid_loading=True, verbose=0)
    df_gt0 = g.get_final_tracks_only_finished_neurons()[0]
    if df_gt0 is None or df_gt0.empty:
        df_gt0 = g.final_tracks
    for pred_h5 in preds:
        try:
            df_pred = pd.read_hdf(pred_h5)
            df_gt, df_pred = align_gt_pred_time_index(df_gt0, df_pred)
            max_len = max(len(df_pred), len(df_gt))
            df_pred = pad_with_nan_rows(df_pred, max_len)
            df_gt = pad_with_nan_rows(df_gt, max_len)
            df_r, _, _, _ = rename_columns_using_matching(
                df_gt, df_pred, column='raw_segmentation_id', try_to_fix_inf=True)
            if 'unmatched_neuron' in df_r.columns.get_level_values(0):
                df_r = df_r.drop(columns='unmatched_neuron')
            cgt = df_gt.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
            cpr = df_r.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
            s = calculate_accuracy(cgt, cpr)
            n_pred_cols = len(df_pred.columns.get_level_values(0).unique())
            n_det = int(np.isfinite(df_pred.loc[:, (slice(None), 'raw_segmentation_id')].values.astype(float)).sum())
            rec = dict(pred=pred_h5, gt=gt_cfg, accuracy=float(s['accuracy']),
                       misses=int(s['total_misses']), mismatches=int(s['total_mismatches']),
                       total=int(s['total_ground_truth']), n_frames=int(len(df_pred)),
                       n_pred_tracks=n_pred_cols, n_pred_assignments=n_det)
        except Exception as e:
            rec = dict(pred=pred_h5, error=repr(e))
        print(json.dumps(rec), flush=True)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
