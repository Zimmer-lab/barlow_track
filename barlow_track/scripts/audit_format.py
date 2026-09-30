"""Audit: identical loading + GT formatting across labs; seg-id overlap pred vs GT."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from wbfm.utils.projects.finished_project_data import ProjectData

COPIES = {
    'zimmer': '/tmp/claude/eval_projects/zimmer/project_config.yaml',
    'flavell': '/tmp/claude/eval_projects/flavell/project_config.yaml',
    'samuel': '/tmp/claude/eval_projects/samuel/project_config.yaml',
}
GTS = {
    'zimmer': '/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data/ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format/project_config.yaml',
    'flavell': '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/flavell_data/images_for_charlie/flavell_data.nwb',
    'samuel': '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/samuel_data/153.nwb',
}

for lab in ['zimmer', 'flavell', 'samuel']:
    p = ProjectData.load_final_project_data(COPIES[lab], allow_hybrid_loading=True, verbose=0)
    g = ProjectData.load_final_project_data(GTS[lab], allow_hybrid_loading=True, verbose=0)
    df_gt = g.get_final_tracks_only_finished_neurons()[0]
    if df_gt is None or df_gt.empty:
        df_gt = g.final_tracks
    print(f"[{lab}] proj frames={p.num_frames} red={tuple(p.red_data.shape)} "
          f"| gt shape={df_gt.shape} levels={df_gt.columns.names} "
          f"n_gt_neurons={len(df_gt.columns.get_level_values(0).unique())}", flush=True)

    # seg ids from MY crop path (same functions as exp) vs GT seg ids
    from barlow_track.utils.data_loading import get_bbox_data_for_volume_with_label
    from barlow_track.utils.volume_data import get_centroids_for_volume
    import pandas as pd
    zxy, segs = get_centroids_for_volume(p, 0)
    crops_d, seg2name, _ = get_bbox_data_for_volume_with_label(
        p, 0, target_sz=np.array([8, 64, 64]), include_untracked=True)
    gt_segs = set(pd.unique(df_gt.loc[:, (slice(None), 'raw_segmentation_id')].values.ravel().astype(float)))
    gt_segs.discard(np.nan)
    my_segs = set(segs.tolist())
    print(f"  frame0: {len(crops_d)} crops, {len(my_segs)} seg ids; "
          f"overlap with GT seg ids: {len(my_segs & gt_segs)}/{len(my_segs)}", flush=True)
