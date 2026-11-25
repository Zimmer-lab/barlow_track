"""R1.M2: paired fDNC / BarlowTrack table on the 12 round_shared_v1 trial-59 benchmark recordings.

Reads the fDNC records written by fdnc_track_and_score.py (one out_dir, four axis conventions per
recording) and the BarlowTrack trial-59 records (exp_results_v2.jsonl, labelprop v2), then per recording:

1. picks fDNC's axis convention GT-free: the convention whose 5-template vote has the highest mean fDNC
   match probability (the mean over the five single-template runs is reported as a cross-check; the
   GT-best convention is reported only to show what the GT-free rule costs, never used);
2. pairs it with BarlowTrack: the trial-59 network (tag round_shared_v1_trial59_<lab>), plus mean and
   range over the independent trial-59 trainings r0-r4 where they exist;
3. verifies the pairing:
   - frames: same n_frames;
   - detections: fDNC's per-frame detection ids (<dataset>_detections.npz) equal the ids BarlowTrack
     embedded (the trial-59 embedding cache's lin_to_t_seg), frame by frame;
   - reference identities and denominators: the GT matrix fDNC was scored against (<dataset>_scored_gt.h5)
     equals the 'gt' matrix saved with the BarlowTrack tracks (tracks_v2), cell by cell; same totals;
   - scoring: BarlowTrack's recorded accuracy is recomputed from its saved gt / pred_matched with the
     same calculate_accuracy.
4. Leifer: assigned-click score (the recorded metric) and all-kept-click score, correct / (assigned +
   clicks with no mask + clicks in volumes without objects) from sd_strict_status.npz; merges and
   coincident clicks stay out (decision 2026-11).

Run:
  /home/charles/anaconda3/envs/wbfm/bin/python fdnc_paired_table.py --fdnc_dir ~/.cache/barlow_scratch/fdnc/r1m2_trial59
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

ROUND = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/round_shared_v1'
BT_JSONL = os.path.join(ROUND, 'exp_results_v2.jsonl')
BT_TRACKS = os.path.join(ROUND, 'tracks_v2')
LEIFER_STATUS = ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/wbfm_tracker_baseline/'
                 'leifer_resegment/Leifer_NeRVE_Worm1_reexport_A2_stardist/3-tracking/manual_annotation/variants/'
                 'sd_strict_status.npz')
ORIENTS = ('default', 'swap', 'mirror', 'swap_mirror')
RECORDINGS = [  # (dataset label, lab, detections)
    ('flavell', 'Flavell', 'all NWB detections'),
    ('flavell_2022031603', 'Flavell', 'all NWB detections'),
    ('flavell_2022032402', 'Flavell', 'all NWB detections'),
    ('flavell_2022033001', 'Flavell', 'all NWB detections'),
    ('flavell_2022033002', 'Flavell', 'all NWB detections'),
    ('zimmer_1123', 'Zimmer', 'all segmented objects'),
    ('zimmer_1128', 'Zimmer', 'all segmented objects'),
    ('zimmer_1210', 'Zimmer', 'all segmented objects'),
    ('samuel', 'Samuel', 'GT objects (oracle)'),
    ('samuel_137', 'Samuel', 'GT objects (oracle)'),
    ('samuel_184', 'Samuel', 'GT objects (oracle)'),
    ('leifer_stardist_strict', 'Leifer', 'all StarDist objects'),
]
NAMES = {'flavell': 'Flavell 2022-03-31-01', 'samuel': 'Samuel 153', 'leifer_stardist_strict': 'Leifer worm 1'}


def display_name(ds):
    if ds in NAMES:
        return NAMES[ds]
    lab, rest = ds.split('_', 1)
    if lab == 'flavell':
        return f"Flavell {rest[:4]}-{rest[4:6]}-{rest[6:8]}-{rest[8:]}"
    return f"{lab.capitalize()} {rest}"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_by_tag(rows):
    out = {}
    for r in rows:
        out[r['tag']] = r
    return out


def choose_convention(fd, ds):
    """GT-free choice of the axis convention, plus diagnostics."""
    rows = {}
    for o in ORIENTS:
        multi = [r for r in fd.values() if r['dataset'] == ds and r['orient'] == o and r['variant'] == 'multi']
        singles = [r for r in fd.values() if r['dataset'] == ds and r['orient'] == o
                   and r['variant'].startswith('single')]
        if not multi:
            continue
        rows[o] = dict(multi=multi[0], singles=singles,
                       vote_prob=multi[0]['mean_match_prob'],
                       single_prob=float(np.mean([s['mean_match_prob'] for s in singles])) if singles else np.nan)
    if len(rows) < 4:
        return None, rows
    chosen = max(rows, key=lambda o: rows[o]['vote_prob'])
    return chosen, rows


def bt_records(ds):
    reps = defaultdict(list)
    for r in read_jsonl(BT_JSONL):
        m = re.match(r'round_shared_v1_trial59(r\d)?_(.+)$', r.get('tag') or '')
        if m and m.group(2) == ds and r.get('labelprop') == 'v2':
            reps[m.group(1) or 'r0'].append(r)
    for k, v in reps.items():  # duplicate rows of one replicate must agree (a rescore appended twice)
        accs = {round(x['accuracy'], 12) for x in v}
        if len(accs) != 1:
            raise RuntimeError(f"{ds} {k}: duplicate records disagree {accs}")
    return {k: v[-1] for k, v in sorted(reps.items())}


def frame_ids_from_embedding(emb_path):
    d = np.load(emb_path, allow_pickle=True)
    lin = d['lin_to_t_seg'].item()
    per = defaultdict(list)
    for _, (t, raw_ind, seg) in lin.items():
        per[int(t)].append(raw_ind)
    return per, int(d['n_frames'])


def frame_ids_from_fdnc(det_path):
    d = np.load(det_path)
    per = defaultdict(list)
    for t, i in zip(d['t'], d['ids']):
        per[int(t)].append(i)
    return per


def compare_detections(emb_path, det_path):
    """Per-frame multiset comparison of detection ids. BarlowTrack skips frames with < 2 detections."""
    bt, _ = frame_ids_from_embedding(emb_path)
    fd = frame_ids_from_fdnc(det_path)
    frames = sorted(set(bt) | set(fd))
    n_diff_frames = n_skipped_small = 0
    n_bt_nan = 0
    for t in frames:
        a = np.asarray(bt.get(t, []), dtype=float)
        b = np.asarray(fd.get(t, []), dtype=float)
        n_bt_nan += int(np.isnan(a).sum())
        if t not in bt and len(b) < 2:
            n_skipped_small += 1
            continue
        if not np.array_equal(np.sort(a), np.sort(b)):
            n_diff_frames += 1
    n_bt = sum(len(v) for v in bt.values())
    n_fd = sum(len(v) for t, v in fd.items() if t in bt or len(v) >= 2)
    return dict(n_frames_compared=len(frames), n_frames_differ=n_diff_frames,
                n_frames_lt2_skipped_by_bt=n_skipped_small, n_detections_bt=n_bt, n_detections_fdnc=n_fd,
                n_bt_ids_nan=n_bt_nan, identical=(n_diff_frames == 0 and n_bt == n_fd and n_bt_nan == 0))


def compare_gt(fdnc_gt_path, bt_tracks_path):
    a = pd.read_hdf(fdnc_gt_path, 'gt')
    b = pd.read_hdf(bt_tracks_path, 'gt')
    out = dict(fdnc_shape=list(a.shape), bt_shape=list(b.shape),
               fdnc_n_valid=int(a.notna().sum().sum()), bt_n_valid=int(b.notna().sum().sum()))
    same_cols = set(map(str, a.columns)) == set(map(str, b.columns))
    a.columns = a.columns.map(str)
    b.columns = b.columns.map(str)
    # align_gt_pred_time_index trims the GT to the prediction's frames; rows one side lacks must hold no GT
    # entry on the other (an all-NaN row scores nothing), so compare on the union of frames
    idx = a.index.union(b.index)
    extra_a = a.index.difference(b.index)
    extra_b = b.index.difference(a.index)
    out.update(rows_only_fdnc=len(extra_a), rows_only_bt=len(extra_b),
               gt_entries_in_unshared_rows=int(a.loc[extra_a].notna().sum().sum() + b.loc[extra_b].notna().sum().sum()))
    a, b = a.reindex(idx), b.reindex(idx)
    same_index = True
    out.update(same_columns=same_cols)
    if same_cols:
        b = b[a.columns]
        av, bv = a.values.astype(float), b.values.astype(float)
        out['n_cells_differ'] = int((~((av == bv) | (np.isnan(av) & np.isnan(bv)))).sum())
    else:
        out['n_cells_differ'] = None
    out['identical'] = bool(same_cols and same_index and out['n_cells_differ'] == 0)
    return out


def recompute_bt(bt_tracks_path):
    from barlow_track.utils.utils_ground_truth import calculate_accuracy
    g = pd.read_hdf(bt_tracks_path, 'gt')
    p = pd.read_hdf(bt_tracks_path, 'pred_matched')
    s = calculate_accuracy(g, p)
    return dict(accuracy=float(s['accuracy']), total=int(s['total_ground_truth']),
                misses=int(s['total_misses']), mismatches=int(s['total_mismatches']))


def leifer_all_kept_denominator(fdnc_gt_path=None):
    d = np.load(LEIFER_STATUS, allow_pickle=True)
    codes = json.loads(str(d['codes']))
    st = d['status']
    n = {k: int((st == v).sum()) for k, v in codes.items()}
    denom = n['assigned'] + n['missed_no_mask'] + n['missed_empty_volume'] + n.get('missed_label_not_in_metadata', 0)
    check = None
    if fdnc_gt_path:  # the scored GT's valid cells are exactly the 'assigned' clicks
        g = pd.read_hdf(fdnc_gt_path, 'gt')
        names = [str(x) for x in d['names']]
        g.columns = g.columns.map(str)
        g = g.reindex(columns=names)
        check = bool(np.array_equal(g.notna().values, st[:len(g)] == codes['assigned'])) and len(g) == len(st)
    return denom, n, check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fdnc_dir', default=os.path.expanduser('~/.cache/barlow_scratch/fdnc/r1m2_trial59'))
    args = ap.parse_args()
    fd = latest_by_tag(read_jsonl(os.path.join(args.fdnc_dir, 'fdnc_results.jsonl')))
    rows, details = [], {}
    for ds, lab, dets in RECORDINGS:
        det = dict(dataset=ds, lab=lab, detections=dets)
        chosen, conv = choose_convention(fd, ds)
        bt = bt_records(ds)
        r0 = bt.get('r0')
        det['bt_replicates'] = {k: v['accuracy'] for k, v in bt.items()}
        if chosen is None or r0 is None:
            det['missing'] = dict(fdnc_conventions=sorted(conv), bt_r0=r0 is not None)
            details[ds] = det
            rows.append(dict(recording=display_name(ds), lab=lab, detections=dets, fdnc=np.nan,
                             bt=r0['accuracy'] if r0 else np.nan, status='NOT SCORED'))
            continue
        m = conv[chosen]['multi']
        gt_best = max(conv, key=lambda o: conv[o]['multi']['accuracy'])
        single_choice = max(conv, key=lambda o: conv[o]['single_prob'])
        det['conventions'] = {o: dict(vote_accuracy=c['multi']['accuracy'], vote_mean_prob=c['vote_prob'],
                                      singles_mean_prob=c['single_prob']) for o, c in conv.items()}
        det.update(chosen=chosen, chosen_by_singles_prob=single_choice, gt_best=gt_best,
                   fdnc_tag=m['tag'], fdnc_record=m)
        # pairing checks
        bt_tracks = os.path.join(BT_TRACKS, f"round_shared_v1_trial59_{ds}.h5")
        fd_gt = os.path.join(args.fdnc_dir, f"{ds}_scored_gt.h5")
        fd_det = os.path.join(args.fdnc_dir, f"{ds}_detections.npz")
        checks = dict(n_frames=dict(fdnc=m['n_frames'], bt=r0['n_frames'], equal=m['n_frames'] == r0['n_frames']),
                      total=dict(fdnc=m['total'], bt=r0['total'], equal=m['total'] == r0['total']),
                      gt_filter=m.get('gt_filter'))
        checks['gt_matrix'] = compare_gt(fd_gt, bt_tracks)
        checks['detections'] = compare_detections(r0['emb_path'], fd_det)
        rec = recompute_bt(bt_tracks)
        checks['bt_rescore'] = dict(rec, recorded=r0['accuracy'],
                                    equal=abs(rec['accuracy'] - r0['accuracy']) < 1e-12 and rec['total'] == r0['total'])
        checks['all_identical'] = bool(checks['n_frames']['equal'] and checks['total']['equal']
                                       and checks['gt_matrix']['identical'] and checks['detections']['identical']
                                       and checks['bt_rescore']['equal'])
        det['checks'] = checks
        accs = np.array([v['accuracy'] for v in bt.values()])
        row = dict(recording=display_name(ds), lab=lab, detections=dets, frames=m['n_frames'],
                   gt_neurons=m['n_gt_neurons'], gt_entries=m['total'],
                   fdnc_convention=chosen, fdnc=m['accuracy'], fdnc_misses=m['misses'],
                   fdnc_mismatches=m['mismatches'],
                   bt=r0['accuracy'], bt_misses=r0['misses'], bt_mismatches=r0['mismatches'],
                   bt_n_reps=len(accs), bt_mean=float(accs.mean()), bt_min=float(accs.min()), bt_max=float(accs.max()),
                   reused='rescored' if m.get('reused_from') else 'new',
                   status='paired' if checks['all_identical'] else 'CHECK FAILED')
        if ds == 'leifer_stardist_strict':
            denom, n, ok = leifer_all_kept_denominator(fd_gt)
            row.update(fdnc_all_kept=(m['total'] - m['misses'] - m['mismatches']) / denom,
                       bt_all_kept=(r0['total'] - r0['misses'] - r0['mismatches']) / denom,
                       bt_all_kept_mean=float(np.mean([(v['total'] - v['misses'] - v['mismatches']) / denom
                                                       for v in bt.values()])),
                       all_kept_denominator=denom)
            det['leifer_clicks'] = dict(n, all_kept_denominator=denom, scored_gt_cells_equal_assigned_clicks=ok)
        rows.append(row)
        details[ds] = det
    df = pd.DataFrame(rows)
    paired = df[df.status == 'paired']
    eps = 0.0005  # a tie: equal to three decimals
    diff = paired.bt - paired.fdnc
    summary = dict(n_paired=int(len(paired)), bt_higher=int((diff > eps).sum()), fdnc_higher=int((diff < -eps).sum()),
                   ties=int((diff.abs() <= eps).sum()),
                   bt_higher_rep_min=int((paired.bt_min - paired.fdnc > eps).sum()),
                   bt_higher_rep_mean=int((paired.bt_mean - paired.fdnc > eps).sum()),
                   median_gap=float(diff.median()) if len(diff) else None,
                   min_gap=float(diff.min()) if len(diff) else None, max_gap=float(diff.max()) if len(diff) else None)
    df.to_csv(os.path.join(args.fdnc_dir, 'paired_table.csv'), index=False)
    with open(os.path.join(args.fdnc_dir, 'paired_table_details.json'), 'w') as f:
        json.dump(dict(summary=summary, recordings=details), f, indent=1, default=str)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print(df[['recording', 'detections', 'frames', 'gt_entries', 'fdnc_convention', 'fdnc', 'bt', 'bt_n_reps',
              'bt_mean', 'bt_min', 'bt_max', 'reused', 'status']].to_string(float_format=lambda x: f"{x:.4f}"))
    if 'fdnc_all_kept' in df:
        print(df.loc[df.fdnc_all_kept.notna(), ['recording', 'fdnc', 'fdnc_all_kept', 'bt', 'bt_all_kept',
                                                 'bt_all_kept_mean', 'all_kept_denominator']].to_string())
    print(json.dumps(summary, indent=1))
    for ds, d in details.items():
        c = d.get('checks', {})
        print(ds, 'chosen', d.get('chosen'), '| singles-prob choice', d.get('chosen_by_singles_prob'),
              '| GT-best', d.get('gt_best'), '| gt_matrix', c.get('gt_matrix', {}).get('identical'),
              '| detections', c.get('detections'), '| bt_rescore', c.get('bt_rescore', {}).get('equal'),
              '| gt_filter', c.get('gt_filter'))


if __name__ == '__main__':
    main()
