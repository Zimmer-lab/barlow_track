"""Variance test: re-track SAVED samuel-attention embeddings 3x (UMAP stochastic)."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import defaultdict
from wbfm.utils.projects.finished_project_data import ProjectData
from barlow_track.utils.utils_tracking import WormClusterTracker
from barlow_track.utils.utils_ground_truth import calculate_accuracy, pad_with_nan_rows
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
from sklearn.decomposition import TruncatedSVD

d = np.load('/tmp/claude/emb_samuel_attention.npz', allow_pickle=True)
X, time_to_lin, lin_to_t_seg = d['X'], d['time_to_lin'].item(), d['lin_to_t_seg'].item()
print('X', X.shape, flush=True)
Xs = TruncatedSVD(n_components=50).fit_transform(X)

g = ProjectData.load_final_project_data(
    '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/samuel_data/153.nwb',
    verbose=0)
df_gt = g.final_tracks
proj = ProjectData.load_final_project_data(
    '/tmp/claude/eval_projects/samuel/project_config.yaml',
    allow_hybrid_loading=True, verbose=0)
print('gt', None if df_gt is None else df_gt.shape, flush=True)

for seed in [0, 1, 2]:
    tr = WormClusterTracker(Xs, dict(time_to_lin), linear_ind_to_t_and_seg_id=lin_to_t_seg,
                            opt_umap=dict(n_components=10, n_neighbors=10, random_state=seed))
    df_pred = tr.track_using_global_clusterer()
    from wbfm.utils.projects.utils_redo_steps import add_metadata_to_df_raw_ind
    df_pred = add_metadata_to_df_raw_ind(df_pred, proj.segmentation_metadata)
    max_len = max(len(df_gt), len(df_pred))
    df_r, _, _, _ = rename_columns_using_matching(
        pad_with_nan_rows(df_gt, max_len), pad_with_nan_rows(df_pred, max_len),
        column='raw_segmentation_id')
    cgt = pad_with_nan_rows(df_gt, max_len).loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
    cpr = df_r.loc[:, (slice(None), 'raw_segmentation_id')].droplevel(1, axis=1)
    s = calculate_accuracy(cgt, cpr)
    print(f'seed {seed}: accuracy={float(s["accuracy"]):.4f} df={df_pred.shape}', flush=True)
