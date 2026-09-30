import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
from barlow_track.utils.utils_ground_truth import calculate_accuracy, pad_with_nan_rows

pairs = {
    'zimmer': ('/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format/project_config.yaml',
               '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects/zimmer/untrained/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_formattrial_0/3-tracking/barlow_tracker/df_barlow_tracks.h5'),
    'flavell': ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/images_for_charlie/flavell_data.nwb',
                '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects/flavell/untrained/2025_07_01trial_0/3-tracking/barlow_tracker/df_barlow_tracks.h5'),
    'leifer': ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/Leifer_NeRVE_Worm1.nwb',
               '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects/leifer/untrained/leifer_project_gttrial_0/3-tracking/barlow_tracker/df_barlow_tracks.h5'),
}
for lab, (gt_cfg, pred_h5) in pairs.items():
    print(f'loading {lab} GT...', flush=True)
    g = ProjectData.load_final_project_data(gt_cfg, allow_hybrid_loading=True, verbose=0)
    df_gt = g.get_final_tracks_only_finished_neurons()[0]
    if df_gt is None or df_gt.empty:
        df_gt = g.final_tracks
    print(f'loading {lab} pred...', flush=True)
    df_pred = pd.read_hdf(pred_h5)
    max_len = max(len(df_pred), len(df_gt))
    df_pred = pad_with_nan_rows(df_pred, max_len)
    df_gt = pad_with_nan_rows(df_gt, max_len)
    df_r, _, _, _ = rename_columns_using_matching(df_gt, df_pred, column='raw_segmentation_id')
    cgt = df_gt.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
    cpr = df_r.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
    s = calculate_accuracy(cgt, cpr)
    print(f'{lab}: STORED untrained accuracy = {float(s["accuracy"]):.4f} '
          f'(miss={s["misses"]}, mismatch={s["mismatches"]}, total={s["total_ground_truth"]})', flush=True)
