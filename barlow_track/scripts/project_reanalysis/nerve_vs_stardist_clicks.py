"""Which Leifer clicks does a network get wrong on NeRVE's masks but right on StarDist's, and why?

The same manual clicks are assigned to masks of two segmentations of one recording (gt_tracks_from_clicks.py
status files: StarDist strict and NeRVE-only). For one network tracked on each (saved pred_all, matched to the GT
neurons with gt_valid), every click kept in both is classed as correct on both / StarDist only / NeRVE only / neither.
Per click, the two objects it is assigned to are described from the masks themselves:

  vox_sd / vox_nv          object size in voxels
  nn_um_sd / nn_um_nv      distance from the object's centroid to the nearest other object of its segmentation
  nv_in_sd                 NeRVE masks covering >= --min_frac of the StarDist object (2+ = NeRVE splits it)
  sd_in_nv                 StarDist objects covering >= --min_frac of the NeRVE mask (2+ = NeRVE merges them)
  nv_mask_has_other_click  the NeRVE mask also holds another neuron's click (a NeRVE merge seen by the GT)

Writes <out>/clicks.csv and summary.json (class counts and per-class medians / fractions).

    python .../nerve_vs_stardist_clicks.py --sd_project ... --sd_status ... --sd_tracks ... \
        --nv_project ... --nv_status ... --nv_tracks ... --out DIR
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


def _load_status(path):
    z = np.load(path)
    return json.loads(str(z['codes'])), z['status'], z['label'], [str(n) for n in z['names']]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('sd', 'nv'):
        ap.add_argument(f'--{k}_project', required=True)
        ap.add_argument(f'--{k}_status', required=True)
        ap.add_argument(f'--{k}_tracks', required=True)
    ap.add_argument('--min_frac', type=float, default=0.2)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_volumes', type=int, default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    from wbfm.utils.projects.finished_project_data import ProjectData
    from barlow_track.scripts.project_reanalysis.tracks_to_project import matched_labels

    S = {}
    for k in ('sd', 'nv'):
        codes, st, lab, names = _load_status(getattr(args, f'{k}_status'))
        P, G = matched_labels(pd.read_hdf(getattr(args, f'{k}_tracks'), 'pred_all'), st, lab, names, codes, 'gt_valid')
        prj = ProjectData.load_final_project_data(getattr(args, f'{k}_project'), allow_hybrid_loading=True, verbose=0)
        S[k] = dict(codes=codes, st=st, lab=lab, names=names, P=P, G=G, prj=prj, seg=prj.raw_segmentation,
                    md=prj.segmentation_metadata,
                    kept=(st == codes['assigned']) | np.isin(st, [v for c, v in codes.items() if c.startswith('missed')]))
    if S['sd']['names'] != S['nv']['names']:
        raise SystemExit('neuron order differs between the two status files')
    puc = S['sd']['prj'].physical_unit_conversion
    um = np.array([puc.zimmer_um_per_pixel_z, puc.zimmer_fluroscence_um_per_pixel_xy,
                   puc.zimmer_fluroscence_um_per_pixel_xy])
    names = S['sd']['names']
    T = S['sd']['st'].shape[0] if args.max_volumes is None else args.max_volumes
    both_assigned = (S['sd']['st'] == S['sd']['codes']['assigned']) & (S['nv']['st'] == S['nv']['codes']['assigned'])
    lock = threading.Lock()

    def one(t):
        js = np.flatnonzero(both_assigned[t])
        if not len(js):
            return []
        with lock:
            vs, vn = np.asarray(S['sd']['seg'][t]), np.asarray(S['nv']['seg'][t])
        cent = {}
        for k in ('sd', 'nv'):
            m = S[k]['md'].segmentation_metadata[t]
            cent[k] = (np.array(m.index, int), np.stack([np.asarray(c, float) for c in m['centroids']]) * um)
        # one overlap table per volume: voxel counts of every (StarDist label, NeRVE label) pair
        size_s, size_n = np.bincount(vs.ravel()), np.bincount(vn.ravel())
        nz = (vs > 0) | (vn > 0)
        key, cnt = np.unique(vs[nz].astype(np.int64) * 65536 + vn[nz], return_counts=True)
        pa, pb = key // 65536, key % 65536
        out = []
        for j in js:
            ls, ln = int(S['sd']['G'][t, j]), int(S['nv']['G'][t, j])
            row = dict(t=int(t), neuron=names[j], sd_correct=bool(S['sd']['P'][t, j] == ls),
                       nv_correct=bool(S['nv']['P'][t, j] == ln),
                       vox_sd=int(size_s[ls]) if ls < len(size_s) else 0,
                       vox_nv=int(size_n[ln]) if ln < len(size_n) else 0)
            for k, l in (('sd', ls), ('nv', ln)):
                labs, C = cent[k]
                i = np.flatnonzero(labs == l)
                if len(i) and len(labs) > 1:
                    d = np.linalg.norm(C - C[i[0]], axis=1)
                    d[i[0]] = np.inf
                    row[f'nn_um_{k}'] = float(d.min())
            sel = (pa == ls) & (pb > 0)
            row['nv_in_sd'] = int(sum(c / size_n[b] >= args.min_frac for b, c in zip(pb[sel], cnt[sel])))
            sel = (pb == ln) & (pa > 0)
            row['sd_in_nv'] = int(sum(c / size_s[a] >= args.min_frac for a, c in zip(pa[sel], cnt[sel])))
            others = np.delete(S['nv']['lab'][t], j)[np.delete(S['nv']['st'][t], j) == S['nv']['codes']['assigned']]
            row['nv_mask_has_other_click'] = bool(np.isin(ln, others))
            out.append(row)
        return out

    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(one, range(T))):
            rows.extend(r)
            if i % 200 == 0:
                print(f'volume {i}: {len(rows)} clicks', flush=True)
    df = pd.DataFrame(rows)
    df['cls'] = np.select([df.sd_correct & df.nv_correct, df.sd_correct & ~df.nv_correct,
                           ~df.sd_correct & df.nv_correct], ['both', 'sd_only', 'nv_only'], 'neither')
    df.to_csv(os.path.join(args.out, 'clicks.csv'), index=False)
    summ = dict(clicks_assigned_in_both=int(len(df)), counts=df.cls.value_counts().to_dict(),
                kept_sd=int(S['sd']['kept'][:T].sum()), kept_nv=int(S['nv']['kept'][:T].sum()), per_class={})
    for c, g in df.groupby('cls'):
        summ['per_class'][c] = dict(
            n=int(len(g)), vox_sd_median=float(g.vox_sd.median()), vox_nv_median=float(g.vox_nv.median()),
            nn_um_sd_median=float(g.nn_um_sd.median()), nn_um_nv_median=float(g.nn_um_nv.median()),
            frac_nerve_splits_sd_object=float((g.nv_in_sd >= 2).mean()),
            frac_nerve_mask_spans_2plus_sd=float((g.sd_in_nv >= 2).mean()),
            frac_nerve_mask_has_other_click=float(g.nv_mask_has_other_click.mean()),
            frac_nerve_mask_smaller_than_half_sd=float((g.vox_nv < 0.5 * g.vox_sd).mean()))
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
