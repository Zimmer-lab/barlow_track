"""Competitor baseline: the original wbfm tracker on a GT recording, from a NEW project.

The existing barlow_track_paper projects are left alone. For a record label of
run_trials_on_all_ground_truth.DATASETS (NWB sources only):
  1. builds a new wbfm project from the label's NWB with
     wbfm/scripts/0-create_new_project_from_nwb.py (copy_nwb_file=True: False DELETES
     the source NWB), which unpacks videos and masks to zarr and recalculates the
     segmentation metadata from the masks;
  2. runs the wbfm tracker's steps with default configs, as the snakemake rules up to
     combine_tracking_and_tracklets do (2a build_frame_objects, 2b match_adjacent_volumes
     (SuperGlue), 2c postprocess_matches_to_tracklets, 3a track_time_independent
     (SuperGlue against a template), 3b match_tracklets_and_tracks_using_neuron_initialization),
     skipping a step whose output exists; CUDA is used only if visible;
  3. scores 3-tracking/postprocessing/combined_3d_tracks.h5 against the NWB GT with the
     scoring block of eval_accuracy.run_eval, matching on raw_segmentation_id (the mask
     labels, which the new project keeps from the NWB).

Results are appended to <out_root>/wbfm_tracker_results.jsonl.

  python run_wbfm_tracker.py --dataset samuel
"""
import argparse
import glob
import importlib
import json
import os
import sys
import time

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'barlow_track', 'scripts', 'multiproject_scripts'))

OUT_ROOT = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/wbfm_tracker_baseline'

STEPS = [
    ('2a-build_frame_objects', '2-training_data/raw/frame_dat.pickle'),
    ('2b-match_adjacent_volumes', '2-training_data/raw/match_dat.pickle'),
    ('2c-postprocess_matches_to_tracklets', '2-training_data/all_tracklets.pickle'),
    ('3a-track_time_independent', '3-tracking/postprocessing/df_tracks_postprocessed.h5'),
    ('3b-match_tracklets_and_tracks_using_neuron_initialization',
     '3-tracking/postprocessing/combined_3d_tracks.h5'),
]


def run_wbfm_script(name, config_updates):
    module = importlib.import_module(f"wbfm.scripts.{name}")
    module.ex.run(config_updates=config_updates)


def find_project(parent):
    found = sorted(glob.glob(os.path.join(parent, '*', 'project_config.yaml')))
    if len(found) > 1:
        raise RuntimeError(f"several projects under {parent}: {found}")
    return found[0] if found else None


def create_project(nwb, parent):
    os.makedirs(parent, exist_ok=True)
    # copy_nwb_file must stay True: build_project_structure_from_nwb_file removes the source otherwise
    run_wbfm_script('0-create_new_project_from_nwb',
                    dict(nwb_file=nwb, project_dir=parent, copy_nwb_file=True, unpack_nwb=True))
    cfg = find_project(parent)
    if cfg is None:
        raise RuntimeError(f"project creation left no project_config.yaml under {parent}")
    return cfg


def score(project_cfg, nwb):
    from wbfm.utils.projects.finished_project_data import ProjectData
    from wbfm.utils.projects.utils_redo_steps import add_metadata_to_df_raw_ind
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays, _ensure_ind_level
    from barlow_track.utils.utils_ground_truth import (pad_with_nan_rows, align_gt_pred_time_index,
                                                       match_columns_to_gt, calculate_accuracy)
    project_dir = os.path.dirname(project_cfg)
    df_pred = pd.read_hdf(os.path.join(project_dir, STEPS[-1][1]))
    match_col = 'raw_segmentation_id'
    if match_col not in df_pred.columns.get_level_values(1):
        project = ProjectData.load_final_project_data(project_cfg, verbose=0)
        df_pred = add_metadata_to_df_raw_ind(df_pred, project.segmentation_metadata)

    src = ProjectData.load_final_project_data(nwb, allow_hybrid_loading=True, verbose=0)
    df_gt = _load_nwb_arrays(src)['df_gt']
    gt_filter = 'full final_tracks'
    try:
        df_fin, _ = src.get_final_tracks_only_finished_neurons()
    except Exception:
        df_fin = None
    if df_fin is not None and not df_fin.empty:
        df_gt, gt_filter = _ensure_ind_level(df_fin), 'finished neurons'

    out = dict(gt_filter=gt_filter, n_pred_columns=int(df_pred.columns.get_level_values(0).nunique()))
    df_gt, df_pred = align_gt_pred_time_index(df_gt, df_pred)
    max_len = max(len(df_gt), len(df_pred))
    df_pred, df_gt = pad_with_nan_rows(df_pred, max_len), pad_with_nan_rows(df_gt, max_len)
    for gt_matching in ('nandist', 'gt_valid'):
        df_pred_r = match_columns_to_gt(df_gt, df_pred, match_col, gt_matching=gt_matching)
        if 'unmatched_neuron' in df_pred_r.columns.get_level_values(0):
            df_pred_r = df_pred_r.drop(columns='unmatched_neuron')
        col_gt = df_gt.loc[:, (slice(None), match_col)].droplevel(1, axis=1)
        col_pr = df_pred_r.loc[:, (slice(None), match_col)].droplevel(1, axis=1)
        stats = calculate_accuracy(col_gt, col_pr)
        key = 'accuracy' if gt_matching == 'nandist' else 'accuracy_gt_valid_matching'
        out[key] = float(stats['accuracy'])
        if gt_matching == 'nandist':
            out.update(n_gt=int(col_gt.shape[1]), n_gt_assigned=int(len(set(col_pr.columns) & set(col_gt.columns))),
                       **{k: float(stats[k]) for k in ('misses', 'mismatches') if k in stats})
    return out


def main():
    from run_trials_on_all_ground_truth import DATASETS
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='record label with an NWB source, e.g. samuel')
    ap.add_argument('--out_root', default=OUT_ROOT)
    ap.add_argument('--create_only', action='store_true', help='build the project, run no tracker step')
    ap.add_argument('--score_only', action='store_true')
    args = ap.parse_args()

    entry = DATASETS[args.dataset]
    nwb = entry.get('nwb')
    if entry.get('source') != 'nwb' or not nwb:
        raise ValueError(f"{args.dataset} has no NWB source; the Zimmer projects already ran the wbfm tracker")
    parent = os.path.join(args.out_root, args.dataset)
    rec = dict(tag=f"wbfm_tracker_{args.dataset}", dataset=args.dataset, nwb=nwb,
               cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), step_seconds={})

    t0 = time.time()
    project_cfg = find_project(parent)
    if project_cfg is None:
        project_cfg = create_project(nwb, parent)
        rec['step_seconds']['create'] = round(time.time() - t0)
    rec['project'] = project_cfg
    print(f"project: {project_cfg}", flush=True)
    if args.create_only:
        return

    project_dir = os.path.dirname(project_cfg)
    if not args.score_only:
        for name, output in STEPS:
            if os.path.exists(os.path.join(project_dir, output)):
                print(f"{name}: output exists, skipping", flush=True)
                continue
            t1 = time.time()
            print(f"== {name}", flush=True)
            run_wbfm_script(name, dict(project_path=project_cfg))
            rec['step_seconds'][name] = round(time.time() - t1)
            print(f"{name}: {rec['step_seconds'][name]} s", flush=True)

    rec.update(score(project_cfg, nwb))
    rec['total_seconds'] = round(time.time() - t0)
    with open(os.path.join(args.out_root, 'wbfm_tracker_results.jsonl'), 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == '__main__':
    main()
