"""Give a wbfm project its own copy of another project's segmentation without mask labels >= --min_label.

wbfm reindexes masks with a 1000-entry lookup table and treats higher labels as an upstream error. The Leifer
NeRVE re-export (export_leifer_nwb.py) adds a pseudo mask, label 1000 + GT index, for every GT click NeRVE
missed (a watershed around the click, not a detection); those labels are not in the segmentation metadata.
This writes a NeRVE-only copy: every label >= --min_label set to background, every other voxel unchanged, the
metadata copied as is (it must hold no label >= --min_label), and the project's segmentation step pointed at
the copy (its own 1-segmentation/segment_config.yaml, relative paths).

    python .../drop_mask_labels.py --source_project .../Leifer_NeRVE_Worm1_reexport_A2/project_config.yaml \
        --project .../Leifer_NeRVE_Worm1_reexport_A2_nervegt/project_config.yaml --min_label 1000 [--write]
"""
import argparse
import datetime
import json
import os
import pickle
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np


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
    ap.add_argument('--source_project', required=True, help='project_config.yaml whose segmentation is copied')
    ap.add_argument('--project', required=True, help='project_config.yaml that gets the copy')
    ap.add_argument('--min_label', type=int, default=1000, help='labels >= this become background')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--write', action='store_true', help='write masks, metadata and configs (default: dry run)')
    args = ap.parse_args()

    import zarr
    from ruamel.yaml import YAML
    from wbfm.utils.projects.finished_project_data import ProjectData

    src = ProjectData.load_final_project_data(args.source_project, allow_hybrid_loading=True, verbose=0)
    seg_cfg = src.project_config.get_segmentation_config()
    src_masks = seg_cfg.resolve_relative_path_from_config('output_masks')
    src_meta = seg_cfg.resolve_relative_path_from_config('output_metadata')
    z = zarr.open(str(src_masks), mode='r')
    T = z.shape[0]

    # The metadata must not reference a label that is about to disappear
    md = src.segmentation_metadata
    meta_max = 0
    for t in range(T):
        n = len(src.get_centroids_as_numpy(t))
        if n:
            meta_max = max(meta_max, max(int(md.i_in_array_to_mask_index(t, i)) for i in range(n)))
    if meta_max >= args.min_label:
        raise SystemExit(f'segmentation metadata holds label {meta_max} >= {args.min_label}: dropping it would '
                         f'leave metadata without masks; relabel instead')

    dst_dir = os.path.dirname(os.path.abspath(args.project))
    dst_masks = os.path.join(dst_dir, '1-segmentation', 'masks.zarr')
    if args.write:
        if os.path.exists(dst_masks):
            raise SystemExit(f'{dst_masks} exists; remove it first')
        out = zarr.open(dst_masks, mode='w', shape=z.shape, chunks=z.chunks, dtype=z.dtype,
                        compressor=z.compressor, fill_value=z.fill_value)

    def one(t):
        v = np.asarray(z[t])
        drop = v >= args.min_label
        n_lab = len(np.unique(v[drop]))
        n_vox = int(drop.sum())
        if args.write:
            v = v.copy()
            v[drop] = 0
            out[t] = v
        return t, n_lab, n_vox, int(v.max())

    labels_dropped, voxels_dropped, vols_with = 0, 0, 0
    with ThreadPoolExecutor(args.workers) as ex:
        for t, n_lab, n_vox, mx in ex.map(one, range(T)):
            labels_dropped += n_lab
            voxels_dropped += n_vox
            vols_with += n_lab > 0
            if args.write:
                assert mx < args.min_label, (t, mx)
    summary = dict(volumes=T, volumes_with_dropped_labels=vols_with, labels_dropped=labels_dropped,
                   voxels_dropped=voxels_dropped, max_metadata_label=meta_max)
    print(json.dumps(summary, indent=1))
    if not args.write:
        return

    shutil.copyfile(str(src_meta), os.path.join(dst_dir, '1-segmentation', 'metadata.pickle'))
    with open(os.path.join(dst_dir, '1-segmentation', 'metadata.pickle'), 'rb') as f:
        pickle.load(f)  # readable

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
                script='barlow_track/scripts/project_reanalysis/drop_mask_labels.py', barlow_track_git=_git_rev(),
                source_project=os.path.abspath(args.source_project), source_masks=str(src_masks),
                source_metadata=str(src_meta), min_label=args.min_label,
                rule=f'labels >= {args.min_label} set to 0, every other voxel unchanged; metadata copied as is',
                output_masks=dst_masks, summary=summary)
    with open(os.path.join(dst_dir, '1-segmentation', 'drop_mask_labels_provenance.json'), 'w') as f:
        json.dump(prov, f, indent=1)
    print('wrote', dst_masks, 'and pointed', args.project, 'at 1-segmentation/segment_config.yaml')


if __name__ == '__main__':
    main()
