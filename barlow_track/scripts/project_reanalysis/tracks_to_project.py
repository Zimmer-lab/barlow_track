"""Make a wbfm project copy whose final tracks are a tracking run's predictions, matched to the GT neurons.

For the Leifer traces comparison (GT-tracked vs tracker-tracked traces on the same segmentation): a run's saved
tracks (eval_accuracy.py --save_tracks, key pred_all: one column per predicted track, values = mask labels)
are matched to the GT neurons of a gt_tracks_from_clicks.py <name>_status.npz (assigned clicks only), and each
matched track is written under its GT neuron's name at EVERY volume it holds an object, including volumes the
GT has no click for: these are the traces a user of the tracker would get. Unmatched tracks are dropped.

The copy is made with wbfm make_project_like (preprocessing, segmentation and training_data linked to the
source project); the tracks h5 goes to 3-tracking/manual_annotation/<out_name>, is set as final_3d_tracks_df
and copied to 3-tracking/postprocessing/combined_3d_tracks.h5, so `snakemake traces` plans only the trace
rules (as for the GT-tracked projects).

    (--tracks GT writes the status file's assigned clicks instead: the GT-tracked copy of that GT variant.)

    python .../tracks_to_project.py --source_project .../stardist/project_config.yaml --suffix track_base \
        --tracks base_centroidgt.h5 --status .../variants/sd_strict_status.npz [--matching gt_valid] [--write]
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd


def _git_rev():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.check_output(['git', '-C', here, 'rev-parse', '--short', 'HEAD'], text=True).strip()
        dirty = subprocess.call(['git', '-C', here, 'diff', '--quiet', 'HEAD', '--', os.path.abspath(__file__)])
        return rev + (' (script modified)' if dirty else '')
    except Exception:
        return None


def matched_labels(pa, status, label, names, codes, matching):
    """(T, N) mask labels of the predicted track matched to each GT neuron (NaN where it holds none)."""
    from barlow_track.utils.utils_ground_truth import match_columns_to_gt
    col = 'raw_segmentation_id'
    T = len(status)
    pa = pa.reindex(range(T))  # volumes without objects have no tracker row
    pa_mi = pd.DataFrame(pa.values.astype(float),
                         columns=pd.MultiIndex.from_product([list(map(str, pa.columns)), [col]]))
    G = np.where(status == codes['assigned'], label, np.nan)
    gt_mi = pd.DataFrame(G, columns=pd.MultiIndex.from_product([names, [col]]))
    r = match_columns_to_gt(gt_mi, pa_mi, col, gt_matching=matching)
    if 'unmatched_neuron' in r.columns.get_level_values(0):
        r = r.drop(columns='unmatched_neuron')
    P = r.loc[:, (slice(None), col)].droplevel(1, axis=1).reindex(columns=names).values.astype(float)
    return P, G


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_project', required=True, help='project_config.yaml with the segmentation the run used')
    ap.add_argument('--suffix', required=True, help='suffix of the new project (wbfm make_project_like)')
    ap.add_argument('--target_directory', default=None, help='parent of the new project; default the source parent')
    ap.add_argument('--tracks', required=True,
                    help="eval_accuracy.py --save_tracks h5, or GT to write the status file's assigned clicks "
                         "(the GT-tracked copy of the same GT variant)")
    ap.add_argument('--status', required=True, help='gt_tracks_from_clicks.py _status.npz, GT neurons to match to')
    ap.add_argument('--matching', default='gt_valid')
    ap.add_argument('--out_name', default=None, help='default 3-tracking/manual_annotation/tracker_<suffix>.h5')
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from wbfm.utils.projects.finished_project_data import ProjectData
    from wbfm.utils.projects.project_config_classes import make_project_like
    from wbfm.utils.projects.utils_redo_steps import add_metadata_to_df_raw_ind

    with np.load(args.status) as f:
        st = {k: f[k] for k in f.files}
    codes = json.loads(str(st['codes']))
    names = [str(n) for n in st['names']]
    if args.tracks == 'GT':
        G = np.where(st['status'] == codes['assigned'], st['label'], np.nan).astype(float)
        P = G.copy()
    else:
        pa = pd.read_hdf(args.tracks, 'pred_all')
        P, G = matched_labels(pa, st['status'], st['label'], names, codes, args.matching)
    T, N = P.shape
    assigned = np.isfinite(G)
    summary = dict(neurons=N, matched_neurons=int(np.isfinite(P).any(0).sum()),
                   frames_with_track=int(np.isfinite(P).sum()), gt_assigned=int(assigned.sum()),
                   correct=int((assigned & (P == G)).sum()),
                   track_where_gt_has_no_assigned_click=int((np.isfinite(P) & ~assigned).sum()))
    summary['accuracy_assigned'] = round(summary['correct'] / max(summary['gt_assigned'], 1), 4)
    print(json.dumps(summary, indent=1), flush=True)

    src = ProjectData.load_final_project_data(args.source_project, allow_hybrid_loading=True, verbose=0)
    md = src.segmentation_metadata
    raw_ind = np.full((T, N), np.nan)
    not_in_md = 0
    for t in range(T):
        js = np.flatnonzero(np.isfinite(P[t]))
        if not len(js):
            continue
        lab2row = {int(v): i for i, v in enumerate(md.segmentation_metadata[t].index)}
        for j in js:
            i = lab2row.get(int(P[t, j]))
            if i is None:
                not_in_md += 1
            else:
                raw_ind[t, j] = i
    summary['labels_not_in_metadata'] = not_in_md
    if not_in_md:
        raise SystemExit(f'{not_in_md} predicted labels are not in the source metadata: wrong segmentation?')
    if not args.write:
        return

    target = args.target_directory or os.path.dirname(os.path.dirname(os.path.abspath(args.source_project)))
    new_cfg = make_project_like(args.source_project, target_directory=target, target_suffix=args.suffix,
                                steps_to_keep=['preprocessing', 'segmentation', 'training_data'])
    new_cfg = str(new_cfg)
    if not new_cfg.endswith('.yaml'):
        new_cfg = os.path.join(new_cfg, 'project_config.yaml')
    project_dir = os.path.dirname(new_cfg)
    out_name = args.out_name or f'3-tracking/manual_annotation/tracker_{args.suffix}.h5'
    out = os.path.join(project_dir, out_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = pd.DataFrame(raw_ind, columns=pd.MultiIndex.from_product([names, ['raw_neuron_ind_in_list']]))
    lik = pd.DataFrame(np.where(np.isfinite(raw_ind), 1.0, np.nan),
                       columns=pd.MultiIndex.from_product([names, ['likelihood']]))
    df = add_metadata_to_df_raw_ind(pd.concat([df, lik], axis=1), md).sort_index(axis=1)
    df.to_hdf(out, key='df_with_missing', mode='w')
    post = os.path.join(project_dir, '3-tracking', 'postprocessing', 'combined_3d_tracks.h5')
    os.makedirs(os.path.dirname(post), exist_ok=True)
    shutil.copyfile(out, post)
    prj = ProjectData.load_final_project_data(new_cfg, allow_hybrid_loading=True, verbose=0)
    cfg = prj.project_config.get_tracking_config()
    cfg.config['final_3d_tracks_df'] = out_name
    cfg.update_self_on_disk()
    prov = dict(created=datetime.datetime.now().isoformat(timespec='seconds'),
                script='barlow_track/scripts/project_reanalysis/tracks_to_project.py', barlow_track_git=_git_rev(),
                source_project=os.path.abspath(args.source_project), tracks=args.tracks if args.tracks == 'GT' else os.path.abspath(args.tracks),
                status=os.path.abspath(args.status), matching=args.matching, output=out,
                combined_3d_tracks=post, summary=summary,
                rule=('predicted tracks matched to the GT neurons of the status file on assigned clicks; each '
                      'matched track written at every volume it holds an object; unmatched tracks dropped'))
    with open(os.path.splitext(out)[0] + '_provenance.json', 'w') as f:
        json.dump(prov, f, indent=1)
    print('wrote', out, df.shape, 'and', post, '; final_3d_tracks_df ->', out_name, 'in', new_cfg)


if __name__ == '__main__':
    main()
