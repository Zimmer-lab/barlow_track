"""Re-track the scored cells of a round from their cached embeddings with another tracker setting, saving tracks.

Every cell (latest record per tag) of <round_dir>/exp_results.jsonl is scored again with its recorded
evaluation settings, except the label propagation recipe / clustering given here. Nothing is trained and,
as long as <round_dir>/emb_cache holds the cell's embedding, nothing is embedded. Records go to a SEPARATE
jsonl (default <round_dir>/exp_results_<name>.jsonl, name = v2 | legacy | global), so the round's own
records and its resume logic are untouched; the predicted tracks go to <round_dir>/tracks_<name>/<tag>.h5
(keys gt, pred_matched, pred_all: eval_accuracy --save_tracks), the input of cross_network_proxy.py.

    python barlow_track/scripts/scratch/gt_free_proxy/rescore_round.py --round_dir <round> --labelprop v2 \
        --shard 0 --num_shards 8          # one Slurm array task; finished cells are skipped on a rerun
    ... --cluster global --trials 59      # HDBSCAN (global clustering) on trial 59's networks only
    ... --list                            # print the cells and exit
"""
import argparse
import json
import os
import re
import sys
import time
import traceback

# settings of the recorded evaluation that are passed on unchanged
_KEEP = ('lab', 'source', 'seed', 'descriptor_stage', 'center_per_volume', 'l2_per_volume', 'crop_window',
         'svd', 'gt_matching')


def latest_records(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get('tag'):
                out[rec['tag']] = rec
    return out


def cell_kwargs(rec):
    """evaluate_trained_checkpoint keywords reproducing a record's evaluation (tracker settings excluded)."""
    kw = {k: rec[k] for k in _KEEP if rec.get(k) is not None}
    kw['round_crop_centers'] = rec.get('crop_center') == 'round'
    for k in ('nwb_objects', 'nwb_gt_mapping'):
        if rec.get(k) is not None:
            kw[k] = rec[k]
    return kw


def select_cells(records, trials=None, labs=None):
    cells = []
    for tag in sorted(records):
        rec = records[tag]
        if rec.get('accuracy') is None or rec.get('max_frames') or not rec.get('weights'):
            continue
        m = re.search(r'_trial(\d+)(r\d+)?_', tag)
        if trials and (m is None or int(m.group(1)) not in trials):
            continue
        if labs and rec.get('lab') not in labs:
            continue
        cells.append(rec)
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--round_dir', required=True)
    ap.add_argument('--labelprop', choices=['legacy', 'v2'], default='v2')
    ap.add_argument('--cluster', choices=['labelprop', 'global'], default='labelprop')
    ap.add_argument('--trials', nargs='+', type=int, default=None, help='only these trial numbers')
    ap.add_argument('--labs', nargs='+', default=None, help='only these recordings')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--track_device', default='cuda')
    ap.add_argument('--out_jsonl', default=None)
    ap.add_argument('--tracks_dir', default=None)
    ap.add_argument('--emb_dir', default=None)
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    rd = os.path.abspath(args.round_dir)
    name = 'global' if args.cluster == 'global' else args.labelprop
    out_jsonl = args.out_jsonl or os.path.join(rd, f'exp_results_{name}.jsonl')
    tracks_dir = args.tracks_dir or os.path.join(rd, f'tracks_{name}')
    emb_dir = args.emb_dir or os.path.join(rd, 'emb_cache')
    cells = select_cells(latest_records(os.path.join(rd, 'exp_results.jsonl')), args.trials, args.labs)
    mine = [c for i, c in enumerate(cells) if i % args.num_shards == args.shard]
    print(f"{len(cells)} cells in {rd}; shard {args.shard}/{args.num_shards}: {len(mine)}; -> {out_jsonl}", flush=True)
    if args.list:
        for c in mine:
            print(c['tag'], c['lab'], f"{c['accuracy']:.4f}")
        return
    os.makedirs(tracks_dir, exist_ok=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *['..'] * 4)))  # repo root
    from barlow_track.scripts.eval_accuracy import evaluate_trained_checkpoint
    n_fail = 0
    for c in mine:
        tag = c['tag']
        tracks = os.path.join(tracks_dir, f'{tag}.h5')
        done = latest_records(out_jsonl).get(tag)
        if done is not None and done.get('accuracy') is not None and os.path.exists(tracks):
            print(f"{tag}: done ({done['accuracy']:.4f})", flush=True)
            continue
        if not os.path.exists(c.get('emb_path') or ''):
            print(f"{tag}: WARNING no cached embedding at {c.get('emb_path')}; it will be embedded again", flush=True)
        t0 = time.time()
        try:
            rec = evaluate_trained_checkpoint(
                weights=c['weights'], project=c.get('project'), gt=c.get('gt'), tag=tag,
                results_jsonl=out_jsonl, emb_dir=emb_dir, cluster=args.cluster, labelprop=args.labelprop,
                track_device=args.track_device, save_tracks=tracks,
                extra_record=dict(rescore_of=os.path.join(rd, 'exp_results.jsonl'),
                                  recorded_accuracy=c['accuracy'], recorded_labelprop=c.get('labelprop') or 'legacy'),
                **cell_kwargs(c))
            print(f"{tag}: {name} accuracy {rec.get('accuracy'):.4f} (recorded {c['accuracy']:.4f}) "
                  f"in {time.time() - t0:.0f}s", flush=True)
        except Exception:
            n_fail += 1
            print(f"{tag}: FAILED\n{traceback.format_exc()}", flush=True)
    print(f"shard {args.shard} finished, {n_fail} failed of {len(mine)}", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
