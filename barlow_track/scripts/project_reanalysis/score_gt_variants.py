"""Score one tracking run's saved tracks against several GT variants of the same clicks.

Label propagation never sees the GT, so a run's predicted tracks (eval_accuracy.py --save_tracks, key
pred_all: one column per predicted track, values = mask labels) can be scored offline against any GT built
from the same segmentation. Each variant is a gt_tracks_from_clicks.py <name>_status.npz: per click its fate
(STATUS codes) and, for assigned clicks, its mask label.

Per variant: kept clicks (assigned + missed by the segmentation; coincident and merged clicks excluded),
then predicted tracks matched to GT neurons on the assigned clicks (`gt_valid`: agreement on frames with
GT; `nandist`: wbfm's matching, which also penalises a track holding objects where the GT is NaN), and

  accuracy_assigned   correct / assigned clicks (what eval_accuracy reports)
  accuracy_all_kept   correct / kept clicks: every click the segmentation missed counts as an error
  errors split into   segmentation misses, tracker misses (assigned click, no track holds the neuron's
                      object at t), mismatches (the neuron's track holds another object)

    python .../score_gt_variants.py --tracks base_centroidgt.h5 --variants strict=.../x_status.npz ... \
        --out <dir> [--name base_seed0]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--variants', nargs='+', required=True, help='NAME=status.npz')
    ap.add_argument('--matching', nargs='+', default=['gt_valid', 'nandist'])
    ap.add_argument('--split_label_at', type=int, default=None,
                    help='also report accuracy on assigned clicks with GT label below / at or above this '
                         '(e.g. 1000: NeRVE masks vs pseudo masks built from the clicks)')
    ap.add_argument('--name', default=None, help='label for this tracking run in the output')
    ap.add_argument('--out', required=True, help='directory; appends to scores.jsonl')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from barlow_track.utils.utils_ground_truth import match_columns_to_gt
    pa = pd.read_hdf(args.tracks, 'pred_all')
    PA = pa.values.astype(float)
    col = 'raw_segmentation_id'
    pa_mi = pd.DataFrame(PA, columns=pd.MultiIndex.from_product([list(map(str, pa.columns)), [col]]))
    rows = []
    for spec in args.variants:
        vname, path = spec.split('=', 1)
        z = np.load(path)
        codes = json.loads(str(z['codes']))
        status, label, names = z['status'], z['label'], list(z['names'])
        if len(pa) != len(status):
            # Volumes without objects get no tracker row (e.g. the NeRVE project's last 13): align by time index
            idx = np.asarray(pa.index)
            if not np.issubdtype(idx.dtype, np.integer) or idx.min() < 0 or idx.max() >= len(status):
                raise SystemExit(f'{vname}: GT has {len(status)} volumes, the tracks {len(pa)} with index '
                                 f'{idx[:3]}...{idx[-3:]}')
            print(f'{vname}: tracks cover {len(pa)} of {len(status)} volumes; the rest count as no track', flush=True)
            pa_v = pa.reindex(range(len(status)))
            pa_mi_v = pd.DataFrame(pa_v.values.astype(float),
                                   columns=pd.MultiIndex.from_product([list(map(str, pa.columns)), [col]]))
        else:
            pa_mi_v = pa_mi
        assigned = status == codes['assigned']
        missed = np.isin(status, [v for k, v in codes.items() if k.startswith('missed')])
        excluded = np.isin(status, [v for k, v in codes.items() if k.startswith('excluded')])
        kept = assigned | missed
        G = np.where(assigned, label, np.nan)
        gt_mi = pd.DataFrame(G, columns=pd.MultiIndex.from_product([names, [col]]))
        for how in args.matching:
            r = match_columns_to_gt(gt_mi, pa_mi_v, col, gt_matching=how)
            if 'unmatched_neuron' in r.columns.get_level_values(0):
                r = r.drop(columns='unmatched_neuron')
            P = r.loc[:, (slice(None), col)].droplevel(1, axis=1).reindex(columns=names).values.astype(float)
            correct = assigned & (P == G)
            trk_miss = assigned & ~np.isfinite(P)
            mism = assigned & np.isfinite(P) & (P != G)
            rec = dict(run=args.name or os.path.basename(args.tracks), variant=vname, matching=how,
                       status_file=os.path.abspath(path), tracks=os.path.abspath(args.tracks),
                       clicks=int((status != codes['no_click']).sum()), kept=int(kept.sum()),
                       assigned=int(assigned.sum()), segmentation_misses=int(missed.sum()),
                       excluded_coincident=int((status == codes['excluded_coincident']).sum()),
                       excluded_merge=int((status == codes['excluded_merge']).sum()),
                       tracker_misses=int(trk_miss.sum()), mismatches=int(mism.sum()),
                       accuracy_assigned=round(float(correct.sum() / max(assigned.sum(), 1)), 4),
                       accuracy_all_kept=round(float(correct.sum() / max(kept.sum(), 1)), 4),
                       **{f'missed_{k[len("missed_"):]}': int((status == v).sum())
                          for k, v in codes.items() if k.startswith('missed')})
            if args.split_label_at is not None:
                for part, sel in (('below', assigned & (G < args.split_label_at)),
                                  ('at_or_above', assigned & (G >= args.split_label_at))):
                    rec[f'label_{part}_{args.split_label_at}'] = dict(
                        assigned=int(sel.sum()), correct=int((sel & (P == G)).sum()),
                        tracker_misses=int((sel & ~np.isfinite(P)).sum()),
                        mismatches=int((sel & np.isfinite(P) & (P != G)).sum()),
                        accuracy=round(float((sel & (P == G)).sum() / max(sel.sum(), 1)), 4))
            rows.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k not in ('status_file', 'tracks')}), flush=True)
    with open(os.path.join(args.out, 'scores.jsonl'), 'a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    df = pd.DataFrame(rows)
    print(df[['run', 'variant', 'matching', 'kept', 'assigned', 'segmentation_misses', 'tracker_misses', 'mismatches',
              'accuracy_assigned', 'accuracy_all_kept']].to_string(index=False))


if __name__ == '__main__':
    main()
