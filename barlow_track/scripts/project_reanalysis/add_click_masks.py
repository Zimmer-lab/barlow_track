"""Give a wbfm project a copy of another project's segmentation plus a mask around every GT click it missed.

The "segmentation plus click masks" control for the Leifer StarDist project, the counterpart of the NeRVE
re-export's pseudo masks: every click whose fate in --status (gt_tracks_from_clicks.py <name>_status.npz) is
a segmentation miss (no mask at the click, or a volume without objects) gets a new mask: an ellipsoid
centred on the click voxel with the median source object's extent per axis (sqrt(5) x its voxel s.d., the
semi-axes of a solid ellipsoid with that covariance), filling only background voxels; voxels two new masks
both claim go to the nearer click. New labels start above the volume's largest label (all stay below 1000).
The metadata gets a row per new mask (volume, centroid, total and mean red intensity, bbox), like the
source rows. Masks are copied voxel for voxel otherwise; the project's segmentation step is pointed at the
copy (as drop_mask_labels.py does). This adds objects exactly where the GT is, so it tests the tracker with
a segmentation that misses nothing, not a segmentation anyone could make without the GT.

    python .../add_click_masks.py --source_project .../stardist/project_config.yaml \
        --project .../stardist_clickmasks/project_config.yaml --status .../x_status.npz \
        --gt_nwb .../Leifer_NeRVE_Worm1_reexport_A2.nwb [--write]
"""
import argparse
import datetime
import json
import os
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_project', required=True)
    ap.add_argument('--project', required=True, help='project_config.yaml that gets the copy')
    ap.add_argument('--status', required=True, help='gt_tracks_from_clicks.py _status.npz on the source project')
    ap.add_argument('--gt_nwb', required=True, help='NWB with the clicks (final_tracks x/y/z)')
    ap.add_argument('--sample_volumes', type=int, default=100, help='volumes sampled for the median object shape')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    import zarr
    from ruamel.yaml import YAML
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays

    src = ProjectData.load_final_project_data(args.source_project, allow_hybrid_loading=True, verbose=0)
    seg_cfg = src.project_config.get_segmentation_config()
    z = zarr.open(str(seg_cfg.resolve_relative_path_from_config('output_masks')), mode='r')
    with open(str(seg_cfg.resolve_relative_path_from_config('output_metadata')), 'rb') as f:
        md = pickle.load(f)
    st = np.load(args.status)
    codes = json.loads(str(st['codes']))
    miss_codes = [v for k, v in codes.items() if k.startswith('missed_no_mask') or k.startswith('missed_empty')]
    nwb = ProjectData.load_final_project_data(args.gt_nwb, allow_hybrid_loading=True, verbose=0)
    a = _load_nwb_arrays(nwb)
    if list(a['neurons']) != list(st['names']):
        raise SystemExit('neuron order differs between the NWB and the status file')
    C = np.stack([a['gz'], a['gx'], a['gy']], 2)
    T = z.shape[0]
    shape = np.array(z.shape[1:])

    # Median object extent per axis (voxels): sqrt(5) x the voxel s.d. of sampled source objects
    rng = np.random.default_rng(0)
    sds = []
    for t in rng.choice(T, size=min(args.sample_volumes, T), replace=False):
        v = np.asarray(z[t])
        for lab in np.unique(v)[1:]:
            idx = np.argwhere(v == lab)
            if len(idx) > 5:
                sds.append(idx.std(0))
    semi = np.sqrt(5.0) * np.median(np.array(sds), axis=0)
    r = np.ceil(semi).astype(int)
    offs = np.stack(np.meshgrid(*[np.arange(-k, k + 1) for k in r], indexing='ij'), -1).reshape(-1, 3)
    offs = offs[((offs / semi) ** 2).sum(1) <= 1.0]

    dst_dir = os.path.dirname(os.path.abspath(args.project))
    dst_masks = os.path.join(dst_dir, '1-segmentation', 'masks.zarr')
    if args.write:
        if os.path.exists(dst_masks):
            raise SystemExit(f'{dst_masks} exists; remove it first')
        out = zarr.open(dst_masks, mode='w', shape=z.shape, chunks=z.chunks, dtype=z.dtype,
                        compressor=z.compressor, fill_value=z.fill_value)

    def one(t):
        v = np.asarray(z[t]).copy()
        js = np.flatnonzero(np.isin(st['status'][t], miss_codes) & np.isfinite(C[t]).all(1))
        rows = []
        if len(js):
            red = np.asarray(src.red_data[t], dtype=float)
            base = int(v.max())
            centre = np.round(C[t, js]).astype(int).clip(0, shape - 1)
            claim = np.full(v.shape, -1, np.int32)
            dist = np.full(v.shape, np.inf)
            for k, c in enumerate(centre):
                p = c + offs
                ok = (p >= 0).all(1) & (p < shape).all(1)
                p = p[ok]
                free = v[p[:, 0], p[:, 1], p[:, 2]] == 0
                p = p[free]
                d = (((p - C[t, js[k]]) / semi) ** 2).sum(1)
                better = d < dist[p[:, 0], p[:, 1], p[:, 2]]
                p, d = p[better], d[better]
                claim[p[:, 0], p[:, 1], p[:, 2]] = k
                dist[p[:, 0], p[:, 1], p[:, 2]] = d
            for k in range(len(js)):
                idx = np.argwhere(claim == k)
                if not len(idx):
                    continue
                lab = base + 1 + k
                v[idx[:, 0], idx[:, 1], idx[:, 2]] = lab
                vals = red[idx[:, 0], idx[:, 1], idx[:, 2]]
                lo, hi = idx.min(0), idx.max(0) + 1
                rows.append(dict(label=lab, neuron_volume=int(len(idx)), centroids=tuple(idx.mean(0).astype(float)),
                                 total_brightness=np.uint64(vals.sum()), bbox=tuple(int(x) for x in (*lo, *hi)),
                                 intensity_mean=float(vals.mean()), neuron=str(st['names'][js[k]])))
        if args.write:
            out[t] = v
        return t, rows, int(v.max())

    new_rows, max_lab = {}, 0
    with ThreadPoolExecutor(args.workers) as ex:
        for t, rows, mx in ex.map(one, range(T)):
            new_rows[t] = rows
            max_lab = max(max_lab, mx)
    n_new = sum(len(r) for r in new_rows.values())
    vols = [r['neuron_volume'] for rr in new_rows.values() for r in rr]
    summary = dict(volumes=T, clicks_given_masks=int(np.isin(st['status'], miss_codes).sum()), masks_added=n_new,
                   added_mask_voxels_pct_10_50_90=np.percentile(vols, [10, 50, 90]).tolist() if vols else None,
                   semi_axes_vox_zxy=np.round(semi, 2).tolist(), max_label=max_lab)
    print(json.dumps(summary, indent=1))
    if max_lab >= 1000:
        raise SystemExit('a label reached 1000; wbfm reindexing would fail')
    if not args.write:
        return

    md_new = {}
    for t, df in md.items():
        rows = new_rows.get(t, [])
        if rows:
            add = pd.DataFrame([{k: r[k] for k in ('neuron_volume', 'centroids', 'total_brightness', 'label', 'bbox',
                                                    'intensity_mean')} for r in rows])
            add.index = add['label'].values
            df = pd.concat([df, add.astype({'neuron_volume': 'int64', 'label': 'int64', 'total_brightness': 'uint64'})])
        md_new[t] = df
    with open(os.path.join(dst_dir, '1-segmentation', 'metadata.pickle'), 'wb') as f:
        pickle.dump(md_new, f)
    yaml = YAML()
    cfg_path = os.path.join(dst_dir, '1-segmentation', 'segment_config.yaml')
    with open(cfg_path) as f:
        cfg = yaml.load(f)
    cfg['output_masks'] = '1-segmentation/masks.zarr'
    cfg['output_metadata'] = '1-segmentation/metadata.pickle'
    cfg['segmentation_fname'] = dst_masks
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f)
    with open(args.project) as f:
        pcfg = yaml.load(f)
    pcfg['subfolder_configs']['segmentation'] = '1-segmentation/segment_config.yaml'
    with open(args.project, 'w') as f:
        yaml.dump(pcfg, f)
    prov = dict(created=datetime.datetime.now().isoformat(timespec='seconds'),
                script='barlow_track/scripts/project_reanalysis/add_click_masks.py', barlow_track_git=_git_rev(),
                source_project=os.path.abspath(args.source_project), status=os.path.abspath(args.status),
                gt_nwb=os.path.abspath(args.gt_nwb), output_masks=dst_masks, summary=summary,
                added=[dict(t=int(t), **{k: (list(v) if isinstance(v, tuple) else (int(v) if k != 'intensity_mean' and
                                                                                  k != 'neuron' else v))
                                         for k, v in r.items()}) for t, rr in new_rows.items() for r in rr])
    with open(os.path.join(dst_dir, '1-segmentation', 'add_click_masks_provenance.json'), 'w') as f:
        json.dump(prov, f, indent=1, default=float)
    print('wrote', dst_masks, 'and pointed', args.project, 'at 1-segmentation/segment_config.yaml')


if __name__ == '__main__':
    main()
