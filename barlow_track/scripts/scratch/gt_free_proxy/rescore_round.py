"""Re-track the scored cells of a round from their cached embeddings with another tracker setting, saving tracks.

Every cell (latest record per tag) of <round_dir>/exp_results.jsonl is scored again with its recorded
evaluation settings, except the label propagation recipe / clustering given here. Nothing is trained and,
as long as <round_dir>/emb_cache holds the cell's embedding, nothing is embedded. Records go to a SEPARATE
jsonl (default <round_dir>/exp_results_<name>.jsonl, name = v2 | legacy | global), so the round's own
records and its resume logic are untouched; the predicted tracks go to <round_dir>/tracks_<name>/<tag>.h5
(keys gt, pred_matched, pred_all: eval_accuracy --save_tracks), the input of cross_network_proxy.py.

Slurm goes through the sweeps' own machinery: --slurm N makes this script a dispatcher (run it with
sbatch_dispatcher.sbatch and BARLOW_DISPATCHER, as optimize_hyperparameters.py / optimize_shared_config.py are)
that submits N shard jobs with submitit, using optimize_hyperparameters' job parameters (train_job_params:
one GPU; cpu_eval_job_params with --track_device cpu), waits for them and, with --then_proxy, runs
cross_network_proxy.py on the saved tracks.

    BARLOW_DISPATCHER=barlow_track/scripts/scratch/gt_free_proxy/rescore_round.py sbatch --output=<log> \
        barlow_track/scripts/sbatch_dispatcher.sbatch --round_dir <round> --labelprop v2 --slurm 8 --then_proxy
    ... --cluster global --trials 59 --track_device cpu --slurm 1   # HDBSCAN on trial 59's networks only
    python .../rescore_round.py --round_dir <round> --list          # print the cells and exit
    python .../rescore_round.py --round_dir <round> --shard 0 --num_shards 8   # one shard, in this process

Finished cells are skipped on a rerun.
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
import zlib

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


def paths(a):
    rd = os.path.abspath(a.round_dir)
    name = 'global' if a.cluster == 'global' else a.labelprop
    return (rd, name, a.out_jsonl or os.path.join(rd, f'exp_results_{name}.jsonl'),
            a.tracks_dir or os.path.join(rd, f'tracks_{name}'), a.emb_dir or os.path.join(rd, 'emb_cache'))


def run_shard(a, shard):
    """Score the cells of one shard in this process; returns the number of failed cells."""
    rd, name, out_jsonl, tracks_dir, emb_dir = paths(a)
    cells = select_cells(latest_records(os.path.join(rd, 'exp_results.jsonl')), a.trials, a.labs)
    # by a hash of the tag, not the position in the list: shards start at different times and the round's
    # jsonl can grow meanwhile (held-out cells), which shifts positions and left cells in no shard
    mine = [c for c in cells if zlib.crc32(c['tag'].encode()) % a.num_shards == shard]
    print(f"{len(cells)} cells in {rd}; shard {shard}/{a.num_shards}: {len(mine)}; -> {out_jsonl}", flush=True)
    os.makedirs(tracks_dir, exist_ok=True)
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
                results_jsonl=out_jsonl, emb_dir=emb_dir, cluster=a.cluster, labelprop=a.labelprop,
                track_device=a.track_device, save_tracks=tracks,
                extra_record=dict(rescore_of=os.path.join(rd, 'exp_results.jsonl'),
                                  recorded_accuracy=c['accuracy'], recorded_labelprop=c.get('labelprop') or 'legacy'),
                **cell_kwargs(c))
            print(f"{tag}: {name} accuracy {rec.get('accuracy'):.4f} (recorded {c['accuracy']:.4f}) "
                  f"in {time.time() - t0:.0f}s", flush=True)
        except Exception:
            n_fail += 1
            print(f"{tag}: FAILED\n{traceback.format_exc()}", flush=True)
    print(f"shard {shard} finished, {n_fail} failed of {len(mine)}", flush=True)
    return n_fail


def dispatch(a):
    """Submit the shards as Slurm jobs through submitit with the sweeps' job parameters, and wait."""
    from submitit import AutoExecutor
    from barlow_track.scripts import optimize_hyperparameters as oh
    rd, name, out_jsonl, tracks_dir, _ = paths(a)
    a.num_shards = a.slurm
    executor = AutoExecutor(folder=os.path.join(rd, 'slurm'), cluster='slurm')
    job_name = f'rescore_{name}_{os.path.basename(rd)}'
    if a.track_device == 'cpu':
        executor.update_parameters(**oh.cpu_eval_job_params({}, a.minutes, job_name))
    else:
        executor.update_parameters(timeout_min=a.minutes + 60, **oh.train_job_params({}, job_name))
    for v in oh._DISPATCH_THREAD_VARS:
        os.environ.pop(v, None)
    jobs = {s: executor.submit(run_shard, a, s) for s in range(a.slurm)}
    for v in oh._DISPATCH_THREAD_VARS:
        os.environ[v] = '1'
    print(f"Submitted {len(jobs)} shard jobs ({oh.slurm_duration(a.minutes + 60)} each): "
          f"{[j.job_id for j in jobs.values()]}", flush=True)
    n_fail, t0 = 0, time.time()
    while jobs:
        for s, job in list(jobs.items()):
            if not job.done():
                continue
            try:
                n_fail += int(job.result())
            except Exception as e:
                n_fail += 1
                print(f"shard {s} (job {job.job_id}): job error {e}", flush=True)
            del jobs[s]
        if jobs:
            print(f"t={time.time() - t0:.0f}s: {len(jobs)} shards running, "
                  f"{len(latest_records(out_jsonl))} cells recorded", flush=True)
            time.sleep(600)
    print(f"all shards finished; {n_fail} failed cells or jobs; {len(latest_records(out_jsonl))} cells recorded",
          flush=True)
    if a.then_proxy:
        import subprocess
        proxy = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cross_network_proxy.py')
        subprocess.run([sys.executable, '-u', proxy, '--tracks_dir', tracks_dir, '--results', out_jsonl,
                        '--out', os.path.join(rd, f'cross_network_proxy_{name}.jsonl')], check=False)
    return n_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--round_dir', required=True)
    ap.add_argument('--labelprop', choices=['legacy', 'v2'], default='v2')
    ap.add_argument('--cluster', choices=['labelprop', 'global'], default='labelprop')
    ap.add_argument('--trials', nargs='+', type=int, default=None, help='only these trial numbers')
    ap.add_argument('--labs', nargs='+', default=None, help='only these recordings')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--slurm', type=int, default=0, help='dispatcher mode: submit this many shard jobs and wait')
    ap.add_argument('--minutes', type=int, default=1440, help='--slurm: walltime budget of a shard job')
    ap.add_argument('--then_proxy', action='store_true', help='--slurm: run cross_network_proxy.py afterwards')
    ap.add_argument('--track_device', default='cuda')
    ap.add_argument('--out_jsonl', default=None)
    ap.add_argument('--tracks_dir', default=None)
    ap.add_argument('--emb_dir', default=None)
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *['..'] * 4)))  # repo root
    if a.list:
        for c in select_cells(latest_records(os.path.join(paths(a)[0], 'exp_results.jsonl')), a.trials, a.labs):
            print(c['tag'], c['lab'], f"{c['accuracy']:.4f}")
        return
    sys.exit(1 if (dispatch(a) if a.slurm else run_shard(a, a.shard)) else 0)


if __name__ == '__main__':
    main()
