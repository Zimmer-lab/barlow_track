"""Bayesian search (Ax, as optimize_hyperparameters.py) over label-propagation tracker settings on saved
embeddings, with persistent per-case workers fed through a file queue.

Objective: mean over the dev cases of the mean accuracy over RNG seeds (default 0, 1), scored exactly as
sweep_labelprop.py scores a variant (official accuracy after the sweep's matching). Every (trial, case,
seed) is one job, so several workers can serve one case. Everything is resumable: the Ax state is saved
after every change, jobs/claimed/results are files, and full records are appended to records.jsonl (the
sweep_labelprop schema, variant 'ax<trial>', so analyze.py reads them too).

Layout under --exp_dir:
    ax_state.json   jobs/<trial>__<label>__s<seed>.json   claimed/...   results/...   records.jsonl

    # driver (one)
    python ax_labelprop.py driver --exp_dir $WORK/ax/round1 --total_trials 40 --parallel 3
    # workers (one or more per dev case; a worker claims only its case's jobs)
    CUDA_VISIBLE_DEVICES= python ax_labelprop.py worker --exp_dir $WORK/ax/round1 --work $WORK \
        --case round_bayes_wk2_samuel_137:50:samuel_137 --device cpu

Selection happens on the dev cases only; confirm the winner against base on the held-out CONFIRM_CASES
with sweep_labelprop.py (PLAN.md section 4).
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV_CASES = [
    'round_bayes_wk2_samuel_137:50:samuel_137',
    'round_bayes_wk3_samuel_184:27:samuel_184',
    'round_bayes_wk3_zimmer_1210:36:zimmer_1210',
    'round_bayes_wk3_flavell_2022033002:61:flavell_2022033002',
]

# Search space: the knobs with signal in the batch-1 screen (PLAN.md), lp_core.DEFAULTS names.
PARAMETERS = [
    dict(name='svd_dim', type='choice', values=[50, 100, 150, 200], value_type='int', is_ordered=True),
    dict(name='svd_mode', type='choice', values=['scaled', 'whiten'], value_type='str', is_ordered=False),
    dict(name='graph', type='choice', values=['knn', 'mutual'], value_type='str', is_ordered=False),
    dict(name='k', type='range', bounds=[8, 30], value_type='int'),
    dict(name='edge_weight', type='choice', values=['uniform', 'gauss'], value_type='str', is_ordered=False),
    dict(name='num_layers', type='choice', values=[20, 30, 50, 100], value_type='int', is_ordered=True),
    dict(name='per_run_unique', type='choice', values=['greedy', 'hungarian'], value_type='str', is_ordered=False),
    dict(name='num_seeds', type='choice', values=[25, 50, 100], value_type='int', is_ordered=True),
    dict(name='seed_select', type='choice', values=['random', 'stratified'], value_type='str', is_ordered=False),
    dict(name='temporal_smooth', type='choice', values=[0.0, 0.5, 1.0], value_type='float', is_ordered=True),
    # round 2: adjacent-volume mutual-NN edges (batch 1: temporal5 +1.4 pt on Z1210, +0.46 mean)
    dict(name='temporal_w', type='choice', values=[0.0, 1.0, 2.0, 5.0, 10.0], value_type='float', is_ordered=True),
]
BASE = dict(svd_dim=50, svd_mode='scaled', graph='knn', k=20, edge_weight='uniform', num_layers=100,
            per_run_unique='greedy', num_seeds=25, seed_select='random', temporal_smooth=0.0, temporal_w=0.0)
# Attached first (round 2; round 1 at 5e40a5b attached base, svd100, whiten, mutual, seeds100, layers30,
# hungrun_gauss and a combo, which --import_records brings in): the batch-1 promotions not yet tried and
# their combination (batch 1 final, 12 pairs: svd100, seeds100, temporal5, seeds50, seed_strat, tsmooth05).
START = [
    ('base', {}),
    ('temporal5', dict(temporal_w=5.0)),
    ('seed_strat', dict(seed_select='stratified')),
    ('tsmooth05', dict(temporal_smooth=0.5)),
    ('c_all', dict(svd_dim=100, num_seeds=50, seed_select='stratified', temporal_w=5.0, temporal_smooth=0.5)),
    ('c_all_gh', dict(svd_dim=100, num_seeds=50, seed_select='stratified', temporal_w=5.0, temporal_smooth=0.5,
                      edge_weight='gauss', per_run_unique='hungarian')),
]


def _write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def job_name(trial, label, seed):
    return f'{trial:04d}__{label}__s{seed}.json'


def dirs(exp_dir):
    d = {k: os.path.join(exp_dir, k) for k in ('jobs', 'claimed', 'results')}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


# ---------------------------------------------------------------- worker

def worker(args):
    import torch
    import sweep_labelprop as sw
    import lp_core
    from barlow_track.utils.utils_tracking import WormClusterTracker
    from barlow_track.utils.utils_consistency import labelprop_consistency
    d = dirs(args.exp_dir)
    rnd, trial, label = args.case.split(':')[:3]
    tag, weights, emb, cfg, ekw = sw.case_settings(rnd, int(trial), label)
    if args.max_frames:
        tag += f'_first{args.max_frames}'
    X, t2l, l2t, n_frames = sw.load_embedding(emb, args.work, tag, args.max_frames)
    scorer = sw.Scorer(ekw, cfg, X.shape, t2l, l2t, n_frames, args.work, tag)
    lin_t = np.array([l2t[i][0] for i in range(X.shape[0])])
    device = None if args.device == 'cpu' else torch.device(args.device)
    emb_memo = sw.LRU(2)
    print(f"[worker {label} {args.device}] ready: {X.shape}", flush=True)
    idle_since = time.time()
    while True:
        if os.path.exists(os.path.join(args.exp_dir, 'STOP')):
            print(f"[worker {label}] STOP file, exiting", flush=True)
            return
        mine = sorted(glob.glob(os.path.join(d['jobs'], f'*__{label}__s*.json')))
        claimed = None
        for j in mine:
            dst = os.path.join(d['claimed'], os.path.basename(j))
            try:
                os.rename(j, dst)  # atomic claim
                claimed = dst
                break
            except FileNotFoundError:
                continue
        if claimed is None:
            if args.exit_idle and time.time() - idle_since > args.exit_idle:
                print(f"[worker {label}] idle {args.exit_idle}s, exiting", flush=True)
                return
            time.sleep(10)
            continue
        idle_since = time.time()
        job = json.load(open(claimed))
        name = os.path.basename(claimed)
        seed, params = int(job['seed']), job['params']
        try:
            v = lp_core.full_variant(params)
            times = {}
            tt = time.time()
            Xs, _ = emb_memo.get_or(lp_core.stage_key(v, 'embed', seed), lambda: lp_core.embed(X, v, seed))
            times['embed_s'] = round(time.time() - tt, 1)
            tt = time.time()
            graph = lp_core.build_graph(Xs, t2l, v, seed, device=device, gt_obj=scorer.gt_obj)
            times['graph_s'] = round(time.time() - tt, 1)
            tt = time.time()
            tracker = WormClusterTracker(Xs, t2l, linear_ind_to_t_and_seg_id=l2t, seed=seed, verbose=0)
            seed_times = lp_core.select_seeds(t2l, v, seed, tracker.num_frames)
            with torch.no_grad():
                labs, probs = lp_core.propagate(graph, t2l, Xs.shape[0], seed_times, v, device=device)
            if device is not None and device.type == 'cuda':
                torch.cuda.synchronize()
            times['prop_s'] = round(time.time() - tt, 1)
            tt = time.time()
            syn = lp_core.sync(labs, probs, t2l, v)
            times['sync_s'] = round(time.time() - tt, 1)
            tt = time.time()
            labels, conf = lp_core.finalize(syn, t2l, v, Xs=Xs, graph=graph)
            times['post_s'] = round(time.time() - tt, 1)
            tt = time.time()
            rec = scorer.score(tracker, labels, conf)
            times['score_s'] = round(time.time() - tt, 1)
            rec.update(scorer.fast(labels))
            cons = labelprop_consistency(labs, syn['perms'], labels, conf, lin_t, n_frames)
            out = dict(case=tag, label=label, variant=f"ax{job['trial']}", seed=seed, params=params, **rec,
                       consistency={k: cons[k] for k in ('agree_mean', 'agree_p10', 'conf_mean', 'assigned_frac',
                                                         'dup_frac')},
                       graph_info=graph['info'], n_objects=int(X.shape[0]), n_frames=n_frames,
                       device=str(args.device), **times)
            with open(os.path.join(args.exp_dir, 'records.jsonl'), 'a') as f:
                f.write(json.dumps(out) + '\n')
            _write_json(os.path.join(d['results'], name), dict(ok=True, accuracy=rec['accuracy'],
                                                               acc_fast=rec['acc_fast'], **times))
            print(f"[worker {label}] {name}: acc {rec['accuracy']:.4f} | {times}", flush=True)
            del graph, labs, probs, syn
        except Exception as e:
            traceback.print_exc()
            _write_json(os.path.join(d['results'], name), dict(ok=False, error=repr(e)[:500]))
        finally:
            if os.path.exists(claimed):
                os.remove(claimed)
            if device is not None and device.type == 'cuda':
                torch.cuda.empty_cache()


# ---------------------------------------------------------------- driver

def _import_records(path, labels, seeds):
    """Completed trials of an earlier round from its records.jsonl: (name, params, objective, per_case).

    A trial counts only if it has a record for every (label, seed); parameters missing from the earlier
    space take their BASE value (round 1 had no temporal_w, i.e. 0)."""
    by = {}
    for line in open(path):
        r = json.loads(line)
        if r['label'] in labels and r['seed'] in seeds:
            by.setdefault(r['variant'], {})[(r['label'], r['seed'])] = r
    out = []
    names = [p['name'] for p in PARAMETERS]
    for v, recs in sorted(by.items(), key=lambda kv: int(kv[0][2:]) if kv[0][2:].isdigit() else 0):
        if len(recs) < len(labels) * len(seeds):
            continue
        p = dict(BASE, **{k: x for k, x in next(iter(recs.values()))['params'].items() if k in names})
        per_case = {lab: float(np.mean([recs[(lab, s)]['accuracy'] for s in seeds])) for lab in labels}
        out.append((v, p, float(np.mean(list(per_case.values()))), per_case))
    return out


def driver(args):
    from ax.service.ax_client import AxClient, ObjectiveProperties
    from ax.core.base_trial import TrialStatus
    d = dirs(args.exp_dir)
    state = os.path.join(args.exp_dir, 'ax_state.json')
    labels = [c.split(':')[2] for c in args.cases]
    seeds = args.seeds
    log = open(os.path.join(args.exp_dir, 'driver.log'), 'a')

    def say(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()

    if os.path.exists(state):
        ax = AxClient.load_from_json_file(state)
        say(f"resumed: {len(ax.experiment.trials)} trials")
        # trials from before per-trial seeds (round 2 up to trial 17) ran on the base seeds
        for t in ax.experiment.trials.values():
            if 'seeds' not in t.run_metadata:
                t.update_run_metadata({'seeds': list(seeds)})
        # claimed jobs without results belonged to workers that died: requeue them
        for c in ([] if args.no_requeue else glob.glob(os.path.join(d['claimed'], '*.json'))):
            if not os.path.exists(os.path.join(d['results'], os.path.basename(c))):
                os.replace(c, os.path.join(d['jobs'], os.path.basename(c)))
                say(f"requeued {os.path.basename(c)}")
    else:
        ax = AxClient(random_seed=args.ax_seed, verbose_logging=False)
        ax.create_experiment(name='labelprop', parameters=[dict(p) for p in PARAMETERS],
                             objectives={'accuracy': ObjectiveProperties(minimize=False)},
                             choose_generation_strategy_kwargs=dict(num_initialization_trials=args.n_sobol,
                                                                    max_parallelism_override=args.parallel))
        seen = []
        for path in args.import_records or []:
            for nm, p, y, per_case in _import_records(path, labels, seeds):
                _, idx = ax.attach_trial(parameters=p)
                ax.experiment.trials[idx].update_run_metadata({'start_name': nm, 'imported_from': path})
                ax.complete_trial(trial_index=idx, raw_data={'accuracy': (y, None)})
                seen.append(p)
                say(f"imported trial {idx} ({nm}) accuracy {y:.4f} | "
                    + ' '.join(f"{k} {v:.4f}" for k, v in per_case.items()) + f" | {p}")
        for nm, ov in START:
            p = dict(BASE, **ov)
            if p in seen:
                say(f"skip start {nm}: already imported")
                continue
            _, idx = ax.attach_trial(parameters=p)
            ax.experiment.trials[idx].update_run_metadata({'start_name': nm})
            say(f"attached trial {idx} ({nm}): {p}")
        ax.save_to_json_file(state)

    def trial_seeds(i):
        """Seeds for trial i: the k-th repeat of an arm (same parameters as k earlier trials; GPEI re-suggests
        its incumbent) runs on the next block of seeds, so a repeat is a new noisy sample of the same objective
        (mean over RNG seeds) instead of a deterministic re-run. Stored in run metadata (resume)."""
        t = ax.experiment.trials[i]
        if 'seeds' in t.run_metadata:
            return list(t.run_metadata['seeds'])
        p = t.arm.parameters
        k = sum(1 for j, u in ax.experiment.trials.items() if j < i and u.arm.parameters == p)
        out = [s + k * (max(seeds) + 1) for s in seeds]
        t.update_run_metadata({'seeds': out})
        return out

    def write_jobs(idx, params):
        for lab in labels:
            for s in trial_seeds(idx):
                nm = job_name(idx, lab, s)
                if not any(os.path.exists(os.path.join(d[k], nm)) for k in ('jobs', 'claimed', 'results')):
                    _write_json(os.path.join(d['jobs'], nm), dict(trial=idx, seed=s, params=params))

    while True:
        trials = ax.experiment.trials
        running = [i for i, t in trials.items() if t.status == TrialStatus.RUNNING]
        done = [i for i, t in trials.items() if t.status in (TrialStatus.COMPLETED, TrialStatus.FAILED,
                                                              TrialStatus.ABANDONED)]
        for i in running:
            write_jobs(i, trials[i].arm.parameters)  # idempotent (resume)
            res = {}
            tseeds = trial_seeds(i)
            for lab in labels:
                for s in tseeds:
                    p = os.path.join(d['results'], job_name(i, lab, s))
                    if os.path.exists(p):
                        res[(lab, s)] = json.load(open(p))
            if len(res) < len(labels) * len(tseeds):
                continue
            bad = [k for k, r in res.items() if not r.get('ok')]
            if bad:
                ax.log_trial_failure(trial_index=i)
                say(f"trial {i} FAILED on {bad}: {[res[k].get('error') for k in bad][:2]}")
            else:
                per_case = {lab: float(np.mean([res[(lab, s)]['accuracy'] for s in tseeds])) for lab in labels}
                y = float(np.mean(list(per_case.values())))
                ax.complete_trial(trial_index=i, raw_data={'accuracy': (y, None)})
                say(f"trial {i} accuracy {y:.4f} (seeds {tseeds}) | " + ' '.join(f"{k} {v:.4f}" for k, v in per_case.items())
                    + f" | {trials[i].arm.parameters}")
            ax.save_to_json_file(state)
        trials = ax.experiment.trials
        running = [i for i, t in trials.items() if t.status == TrialStatus.RUNNING]
        n_total = len(trials)
        while len(running) < args.parallel and n_total < args.total_trials:
            try:
                params, idx = ax.get_next_trial()
            except Exception as e:  # e.g. not enough data for the model yet
                say(f"get_next_trial deferred: {repr(e)[:200]}")
                break
            write_jobs(idx, params)
            running.append(idx)
            n_total += 1
            say(f"trial {idx} started: {params}")
            ax.save_to_json_file(state)
        if not running and n_total >= args.total_trials:
            break
        time.sleep(args.poll)
    best = ax.get_best_parameters()
    say(f"done. best (model): {best}")
    rows = []
    for i, t in ax.experiment.trials.items():
        if t.status == TrialStatus.COMPLETED:
            m = ax.experiment.fetch_trials_data([i]).df
            rows.append((float(m['mean'].iloc[0]), i, t.arm.parameters))
    rows.sort(reverse=True)
    for y, i, p in rows[:10]:
        say(f"  top: trial {i} {y:.4f} {p}")
    _write_json(os.path.join(args.exp_dir, 'summary.json'),
                dict(best_model=best[0] if best else None, top=[dict(trial=i, accuracy=y, params=p) for y, i, p in rows]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    dr = sub.add_parser('driver')
    dr.add_argument('--exp_dir', required=True)
    dr.add_argument('--cases', nargs='+', default=DEV_CASES)
    dr.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    dr.add_argument('--total_trials', type=int, default=40)
    dr.add_argument('--parallel', type=int, default=3)
    dr.add_argument('--n_sobol', type=int, default=6)
    dr.add_argument('--ax_seed', type=int, default=0)
    dr.add_argument('--no_requeue', action='store_true', help='restart the driver alone: workers still hold their claims')
    dr.add_argument('--poll', type=int, default=30)
    dr.add_argument('--import_records', nargs='*', default=None,
                    help="earlier rounds' records.jsonl: their complete trials are attached and completed first")
    wk = sub.add_parser('worker')
    wk.add_argument('--exp_dir', required=True)
    wk.add_argument('--work', required=True)
    wk.add_argument('--case', required=True)
    wk.add_argument('--device', default='cpu')
    wk.add_argument('--max_frames', type=int, default=None)
    wk.add_argument('--exit_idle', type=int, default=0, help='exit after this many idle seconds (0: never)')
    args = ap.parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)
    driver(args) if args.mode == 'driver' else worker(args)


if __name__ == '__main__':
    main()
