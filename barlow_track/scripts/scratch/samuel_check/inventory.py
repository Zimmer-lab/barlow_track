"""Inventory of Samuel-lab projects: frames, detections per frame, GT neurons.

Read-only on /lisc (ProjectData loading writes log files there, unavoidable).
Usage: python inventory.py <project_config.yaml> [...]
"""
import sys

import numpy as np


def summarize(cfg):
    from wbfm.utils.projects.finished_project_data import ProjectData
    print(f'===== {cfg}', flush=True)
    p = ProjectData.load_final_project_data(cfg, allow_hybrid_loading=True, verbose=0)
    print('num_frames', p.num_frames)
    try:
        print('red_data', getattr(p.red_data, 'shape', None))
    except Exception as e:
        print('red_data err', e)
    seg_meta = getattr(p, 'segmentation_metadata', None)
    try:
        n_det = []
        for t in [0, 100, 500, 1000, 1300]:
            if t < p.num_frames:
                n_det.append((t, len(seg_meta.get_all_volumes(t))))
        print('detections per frame (t, n):', n_det)
    except Exception as e:
        try:
            md = seg_meta.segmentation_metadata
            print('metadata keys', len(md), [(t, len(md[t])) for t in list(md)[:3]])
        except Exception as e2:
            print('seg metadata err', repr(e), repr(e2))
    try:
        df = p.get_final_tracks_only_finished_neurons()[0]
        if df is None or df.empty:
            df = p.final_tracks
        names = df.columns.get_level_values(0).unique()
        lvl1 = sorted(df.columns.get_level_values(1).unique())
        print('final_tracks', df.shape, 'neurons', len(names), 'fields', lvl1)
        col = 'raw_segmentation_id' if 'raw_segmentation_id' in lvl1 else lvl1[0]
        ids = df.loc[:, (slice(None), col)].values
        fin = np.isfinite(ids.astype(float))
        print(f'GT finite {col}: total', int(fin.sum()), 'median/frame', np.median(fin.sum(1)),
              'frames>0', int((fin.sum(1) > 0).sum()))
        # do GT seg ids coincide with project segmentation ids?
        for t in [100, 500]:
            if t >= p.num_frames:
                continue
            try:
                proj_ids = set(int(v) for v in seg_meta.get_all_volumes(t).index)
            except Exception:
                proj_ids = None
            gt_ids = set(ids[t][np.isfinite(ids[t].astype(float))].astype(int).tolist())
            if proj_ids is not None:
                print(f't={t}: GT ids {len(gt_ids)}, project ids {len(proj_ids)}, overlap {len(gt_ids & proj_ids)}')
    except Exception as e:
        print('final_tracks err', repr(e))


if __name__ == '__main__':
    for c in sys.argv[1:]:
        try:
            summarize(c)
        except Exception as e:
            print('FAILED', c, repr(e))
