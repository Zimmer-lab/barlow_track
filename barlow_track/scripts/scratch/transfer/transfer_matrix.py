"""Cross-recording transfer of the frozen recipe: each recording's trial-59 network scored on every other recording.

Base numbers only, no sweep (user decision 2026-11-09). Source networks are round_shared_v1/trial_59/<label>/
(replicate 0, trained on <label> alone); a cell scores source network S on target recording T with exactly the
evaluation settings of T's own trial-59 cell (its record in <round>/exp_results.jsonl: project or NWB, GT,
objects, crop window and centres), except:

  * the crop size is S's (the backbone's head is sized for it; a Flavell network crops 16 z planes everywhere),
  * the ViSNet z_scale is T's (--z_scale target, default: the z / xy voxel size is a property of the data, not a
    weight; T's value is the one T's own network trained with, so Flavell's hand-set 1.0 applies to Flavell);
    --z_scale source keeps S's (tags get a _zsrc suffix, so both can sit in one jsonl),
  * the label propagation recipe is --labelprop (default v2).

The diagonal (S = T) is not recomputed: --report takes it from <round>/exp_results_<labelprop>.jsonl (v2) or the
round's own records (legacy). Records go to <round>/exp_results_transfer_<labelprop>.jsonl, embeddings to
<round>/emb_cache_transfer/. Tags: transfer_t59_<S>_on_<T>[_zsrc]. Finished cells are skipped on a rerun.

Slurm as in rescore_round.py: --slurm N makes this a dispatcher (run it with sbatch_dispatcher.sbatch and
BARLOW_DISPATCHER) that submits N shard jobs through submitit with optimize_hyperparameters' GPU job parameters.

    BARLOW_DISPATCHER=barlow_track/scripts/scratch/transfer/transfer_matrix.py sbatch --output=<log> \
        barlow_track/scripts/sbatch_dispatcher.sbatch --round_dir <round_shared_v1> --slurm 8
    python .../transfer_matrix.py --round_dir <round> --list      # the cells, and each one's crop and z_scale
    python .../transfer_matrix.py --round_dir <round> --report    # accuracy matrix (rows: network, columns: data)
"""
import argparse
import os
import re
import sys
import time
import traceback
import zlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *['..'] * 4)))  # repo root

from barlow_track.scripts.scratch.gt_free_proxy.rescore_round import cell_kwargs, latest_records  # noqa: E402


def source_networks(round_dir, trial):
    """{label: checkpoint} of replicate 0 of every recording the trial was trained on."""
    td = os.path.join(round_dir, f'trial_{trial}')
    out = {}
    for d in sorted(os.listdir(td)):
        ckpt = os.path.join(td, d, 'resnet50.pth')
        if not re.search(r'_rep\d+$', d) and os.path.isfile(ckpt):
            out[d] = ckpt
    return out


def own_cell(records, round_name, trial, label):
    return records.get(f'{round_name}_trial{trial}_{label}')


def _z_scale(rec):
    v = (rec.get('train_config') or {}).get('visnet_z_scale')
    return None if v is None else float(v)


def _crop(rec):
    c = rec.get('train_config') or {}
    return [c.get('target_sz_z'), c.get('target_sz_xy'), c.get('target_sz_xy')]


def cells(a):
    """[(tag, source label, target label, weights, target record, z_scale, source crop)] off the diagonal."""
    rd = os.path.abspath(a.round_dir)
    name = os.path.basename(rd)
    records = latest_records(os.path.join(rd, 'exp_results.jsonl'))
    nets = source_networks(rd, a.trial)
    labels = sorted(l for l in nets if own_cell(records, name, a.trial, l) is not None)
    missing = sorted(set(nets) - set(labels))
    if missing:
        print(f"WARNING: no own record for {missing}; left out as sources and targets", flush=True)
    out = []
    for s in labels:
        if a.sources and s not in a.sources:
            continue
        src = own_cell(records, name, a.trial, s)
        for t in labels:
            if t == s or (a.targets and t not in a.targets):
                continue
            tgt = own_cell(records, name, a.trial, t)
            z = _z_scale(tgt) if a.z_scale == 'target' else _z_scale(src)
            tag = f'transfer_t{a.trial}_{s}_on_{t}' + ('_zsrc' if a.z_scale == 'source' else '')
            out.append((tag, s, t, nets[s], tgt, z, _crop(src)))
    return out


def paths(a):
    rd = os.path.abspath(a.round_dir)
    return (rd, a.out_jsonl or os.path.join(rd, f'exp_results_transfer_{a.labelprop}.jsonl'),
            a.emb_dir or os.path.join(rd, 'emb_cache_transfer'))


def run_shard(a, shard):
    rd, out_jsonl, emb_dir = paths(a)
    todo = cells(a)
    mine = [c for c in todo if zlib.crc32(c[0].encode()) % a.num_shards == shard]
    print(f"{len(todo)} transfer cells; shard {shard}/{a.num_shards}: {len(mine)}; -> {out_jsonl}", flush=True)
    os.makedirs(emb_dir, exist_ok=True)
    from barlow_track.scripts.eval_accuracy import evaluate_trained_checkpoint
    n_fail = 0
    for tag, s, t, weights, tgt, z, _ in mine:
        done = latest_records(out_jsonl).get(tag)
        if done is not None and done.get('accuracy') is not None:
            print(f"{tag}: done ({done['accuracy']:.4f})", flush=True)
            continue
        t0 = time.time()
        try:
            rec = evaluate_trained_checkpoint(
                weights=weights, project=tgt.get('project'), gt=tgt.get('gt'), tag=tag,
                results_jsonl=out_jsonl, emb_dir=emb_dir, labelprop=a.labelprop, track_device=a.track_device,
                visnet_z_scale=z if z is not None else 1.0,
                extra_record=dict(transfer_source=s, transfer_target=t, transfer_z_scale_policy=a.z_scale,
                                  target_own_tag=tgt['tag']),
                **cell_kwargs(tgt))
            print(f"{tag}: accuracy {rec.get('accuracy'):.4f} in {time.time() - t0:.0f}s", flush=True)
        except Exception:
            n_fail += 1
            print(f"{tag}: FAILED\n{traceback.format_exc()}", flush=True)
    print(f"shard {shard} finished, {n_fail} failed of {len(mine)}", flush=True)
    return n_fail


def dispatch(a):
    from submitit import AutoExecutor
    from barlow_track.scripts import optimize_hyperparameters as oh
    rd, out_jsonl, _ = paths(a)
    a.num_shards = a.slurm
    executor = AutoExecutor(folder=os.path.join(rd, 'slurm'), cluster='slurm')
    executor.update_parameters(timeout_min=a.minutes + 60,
                               **oh.train_job_params({}, f'transfer_{os.path.basename(rd)}'))
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
    print(f"all shards finished; {n_fail} failed cells or jobs", flush=True)
    report(a)
    return n_fail


def report(a):
    """Accuracy matrix: rows the network (trained on), columns the recording scored; diagonal = own cell."""
    rd, out_jsonl, _ = paths(a)
    name = os.path.basename(rd)
    diag_path = os.path.join(rd, 'exp_results.jsonl' if a.labelprop == 'legacy' else f'exp_results_{a.labelprop}.jsonl')
    diag, off = latest_records(diag_path), latest_records(out_jsonl)
    labels = sorted(source_networks(rd, a.trial))
    suffix = '_zsrc' if a.z_scale == 'source' else ''
    acc = {}
    for s in labels:
        for t in labels:
            r = own_cell(diag, name, a.trial, t) if s == t else off.get(f'transfer_t{a.trial}_{s}_on_{t}{suffix}')
            if r is not None and r.get('accuracy') is not None:
                acc[s, t] = float(r['accuracy'])
    short = {l: l.replace('flavell_', 'fl_').replace('samuel_', 'sa_').replace('zimmer_', 'zi_')
             .replace('leifer_stardist_strict', 'leifer') for l in labels}
    print(f"{a.labelprop}, z_scale {a.z_scale}; rows: trained on, columns: scored on; {len(acc)} cells")
    print(f"{'':>14}" + ''.join(f"{short[t]:>9}" for t in labels))
    for s in labels:
        print(f"{short[s]:>14}" + ''.join(f"{acc[s, t]:9.4f}" if (s, t) in acc else f"{'-':>9}" for t in labels))
    lab = lambda l: l.split('_')[0]
    print("\nper recording scored: own network, mean drop of other networks of the same lab / of other labs")
    for t in labels:
        own = acc.get((t, t))
        same = [acc[s, t] for s in labels if s != t and lab(s) == lab(t) and (s, t) in acc]
        other = [acc[s, t] for s in labels if lab(s) != lab(t) and (s, t) in acc]
        f = lambda v: f"{(sum(v) / len(v) - own) * 100:+6.2f} pt (n={len(v)})" if v and own is not None else '     -'
        print(f"{t:>24}  own {own if own is None else round(own, 4)}  same lab {f(same)}  other labs {f(other)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--round_dir', required=True)
    ap.add_argument('--trial', type=int, default=59)
    ap.add_argument('--sources', nargs='+', default=None, help='only networks trained on these recordings')
    ap.add_argument('--targets', nargs='+', default=None, help='only these recordings scored')
    ap.add_argument('--z_scale', choices=['target', 'source'], default='target')
    ap.add_argument('--labelprop', choices=['legacy', 'v2'], default='v2')
    ap.add_argument('--track_device', default='cuda')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--slurm', type=int, default=0, help='dispatcher mode: submit this many shard jobs and wait')
    ap.add_argument('--minutes', type=int, default=1440, help='--slurm: walltime budget of a shard job')
    ap.add_argument('--out_jsonl', default=None)
    ap.add_argument('--emb_dir', default=None)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.list:
        for tag, s, t, w, tgt, z, crop in cells(a):
            print(f"{tag}  crop {crop}  z_scale {z}  ({tgt.get('source')}: {tgt.get('project') or tgt.get('gt')})")
        return
    if a.report:
        report(a)
        return
    sys.exit(1 if (dispatch(a) if a.slurm else run_shard(a, a.shard)) else 0)


if __name__ == '__main__':
    main()
