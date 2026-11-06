"""Shared-config sweep: one training recipe, trained and scored on each of several recordings.

The paper's recipe is trained anew on each recording (self-supervised), so the config to freeze is
the one that does well on every recording, not on one. Each Ax trial here is one config, trained
and scored separately on each TUNING recording (one GPU job per recording, as a per-recording
round's trial), and Ax maximizes a low quantile of the per-recording regret:

    objective = quantile_q( accuracy[label] - reference_accuracy[label] ),  over tuning_labels

reference_accuracy is each recording's best score from its own GT-tuned per-recording round
(wk2/wk3), so a saturated recording counts as much as a hard one; q = 0.25 by default (the user,
2026-11-05: the mean is dominated by saturated datasets, the minimum is too noisy). A trial in which
any recording fails (no checkpoint, NaN) is a failed trial: Ax leaves it out of the fit.

objective_metric picks what is regretted. 'accuracy' (default) needs GT on every tuning recording.
A GT-free label-propagation consistency metric (utils_consistency.METRICS, e.g. agree_mean: agreement
of the 25 per-seed runs with their consensus; the user, 2026-11-05) uses reference_<metric> instead,
e.g. reference_agree_mean: the best value of that metric on each recording, GT-free if taken as the
maximum over a round's trials (scratch/gt_free_proxy/proxy_references.py). Every cell still records
its GT accuracy, so the GT-free selection can be checked afterwards; it never steers the search.

HELD-OUT recordings (heldout_labels) are never trained or scored by the sweep; they are listed so
the template documents the split, and the dispatcher refuses an overlap. Once the sweep is done,
--score_heldout trains the chosen trial(s)' config on each held-out recording and scores it (GT
accuracy plus the consistency metrics), one GPU job per (trial, replicate, recording), into
trial_N/<heldout label>/ (replicate r > 0: trial_N/<label>_rep<r>/, tag trial<N>r<r>) under the usual
record tags, appending one line per cell to heldout_results.jsonl. heldout_reference_accuracy
(optional; e.g. each held-out recording's best accuracy in its own GT-tuned per-recording round)
adds the gap to it. Cells already recorded are skipped; cells with a checkpoint are only re-scored.

Per-recording settings come from make_per_recording_rounds.round_files, as in the per-recording
rounds: project_path, crop depth, visnet_z_scale (z_scale_from_voxels), and the objective's
source / GT / every-detection settings. Every (trial, recording) cell is recorded in
<round>/exp_results.jsonl under the benchmark tag <round>_trial<N>_<label>, with its embedding in
<round>/emb_cache, so the standard benchmark recognizes the cells.

Layout of a round directory (template: shared_template.yaml):
    trial_N/params.yaml            Ax parameters and seed of trial N
    trial_N/<label>/               one training directory per recording (train_config.yaml, ...)
    exp_results.jsonl, emb_cache/  per-cell records and embeddings
    shared_results.jsonl           one line per finished trial (per-label accuracy, regret, objective)

Resumable: a restart attaches every trial whose cells are all recorded, re-scores cells whose
checkpoint exists, and retrains the rest.

    python barlow_track/scripts/optimize_shared_config.py -p <round>/shared_template.yaml
    python barlow_track/scripts/optimize_shared_config.py -p <round> --run_locally --smoke   # tiny local test
    python barlow_track/scripts/optimize_shared_config.py -p <round> --score_heldout          # best trial
    python barlow_track/scripts/optimize_shared_config.py -p <round> --score_heldout --top 3 --replicates 2
"""
import argparse
import copy
import json
import logging
import os
import re
import time
from types import SimpleNamespace

import numpy as np
import yaml

# Importing the per-recording driver first applies its dispatcher thread caps
from barlow_track.scripts import optimize_hyperparameters as oh
from barlow_track.scripts.scratch.make_per_recording_rounds import round_files

DEFAULT_QUANTILE = 0.25


def metric_value(record, metric):
    """The objective metric of one cell's eval record (or train_and_score result); None if missing."""
    if not record:
        return None
    if metric == 'accuracy':
        return oh.objective_value(record.get('accuracy'))
    return oh.objective_value((record.get('consistency') or {}).get(metric))


def regret_objective(accuracy, reference, labels, q=DEFAULT_QUANTILE):
    """q-quantile (linear interpolation) of value - reference over labels; None if any is missing."""
    vals = []
    for label in labels:
        a = oh.objective_value(accuracy.get(label))
        if a is None:
            return None
        vals.append(a - float(reference[label]))
    return float(np.quantile(vals, q))


def cell_tag(round_dir, trial_num, label):
    return oh.record_tag(round_dir, trial_num, label)


def latest_records(results_jsonl):
    """tag -> latest record (the benchmark convention: the last line of a tag wins)."""
    out = {}
    if not os.path.exists(results_jsonl):
        return out
    with open(results_jsonl) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get('tag'):
                out[rec['tag']] = rec
    return out


def train_and_score(cfg, eval_kwargs, tag, results_jsonl, emb_dir, train=True, objective_metric=None):
    """One (trial, recording) cell, inside its own Slurm job: train, then score on that recording.

    Returns dict(accuracy, consistency) of the eval record, or None on any failure (logged), so one
    recording's crash fails the cell rather than the dispatcher.
    """
    args = SimpleNamespace(**cfg)
    try:
        test_losses = oh.train_barlow_network(args) if train else None
        ckpt = oh.checkpoint_path(args.project_dir)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint in {args.project_dir}")
        extra = dict(objective='shared_quantile_regret', trial_dir=os.path.abspath(args.project_dir))
        if objective_metric:
            extra['objective_metric'] = objective_metric
        if isinstance(test_losses, dict):
            for k in ('test_loss', 'test_loss_original', 'test_loss_transpose'):
                extra[k] = oh._finite_or_none(test_losses.get(k))
        extra['val_loss'] = oh._finite_or_none(oh.extract_val_from_json(args.project_dir, key='val_loss'))
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        kwargs = {k: v for k, v in eval_kwargs.items() if k not in ('gt', 'snapshots')}
        record = oh.evaluate_trained_checkpoint(
            weights=ckpt, project=args.project_path, gt=eval_kwargs.get('gt') or args.project_path,
            tag=tag, results_jsonl=results_jsonl, emb_dir=emb_dir, extra_record=extra, **kwargs)
        return dict(accuracy=record.get('accuracy'), consistency=record.get('consistency'))
    except oh.PretrainedArchitectureMismatchError:
        raise
    except Exception as e:
        logging.exception(f"[{tag}] failed: {e}")
        return None


class SharedSweep:
    def __init__(self, template_path, run_locally=False, smoke=False):
        if os.path.isdir(template_path):
            template_path = os.path.join(template_path, 'shared_template.yaml')
        self.round_dir = os.path.dirname(os.path.abspath(template_path))
        with open(template_path) as f:
            self.tpl = yaml.safe_load(f)
        t = self.tpl
        self.labels = list(t['tuning_labels'])
        self.heldout = list(t.get('heldout_labels') or [])
        overlap = set(self.labels) & set(self.heldout)
        if overlap:
            raise ValueError(f"tuning_labels and heldout_labels overlap: {sorted(overlap)}")
        from barlow_track.utils.utils_consistency import METRICS
        self.metric = str(t.get('objective_metric', 'accuracy'))
        # Ax maximizes; dup_frac is lower-is-better (and 0 after temporal uniqueness), so not an objective
        allowed = [m for m in METRICS if m != 'dup_frac']
        if self.metric != 'accuracy' and self.metric not in allowed:
            raise ValueError(f"objective_metric {self.metric!r}: use 'accuracy' or one of {allowed}")
        ref_key = f'reference_{self.metric}'
        self.reference = {k: float(v) for k, v in (t.get(ref_key) or {}).items()}
        missing = [l for l in self.labels if l not in self.reference]
        if missing:
            raise ValueError(f"{ref_key} missing for tuning labels {missing}")
        self.q = float(t.get('objective_quantile', DEFAULT_QUANTILE))
        base_path = t['base_train_config']
        if not os.path.isabs(base_path):
            base_path = os.path.join(self.round_dir, base_path)
        with open(base_path) as f:
            self.base = yaml.safe_load(f)
        self.smoke = smoke
        if smoke:
            self.base.update(num_frames=12, epochs=1, print_freq=5, val_match_frames=0)
        self.round_name = os.path.basename(self.round_dir)
        # Per-recording train config and objective settings, exactly as the per-recording rounds
        grid = {k: v for k, v in t.items() if k.startswith('objective_') or k.startswith('train_')}
        if smoke:
            grid['objective_max_frames'] = 12
        self.grid = grid
        self.cell_cfg, self.eval_kwargs = {}, {}
        for label in self.labels:
            self.add_label(label)
        # A record counts for this run only if it was scored at this run's length: a --smoke run in the
        # same directory scores 12 frames, and must never be resumed as (or by) a full run
        self.max_frames = grid.get('objective_max_frames')
        self.results_jsonl = os.path.join(self.round_dir, 'exp_results.jsonl')
        self.emb_dir = os.path.join(self.round_dir, 'emb_cache')
        self.shared_jsonl = os.path.join(self.round_dir, 'shared_results.jsonl')
        self.base_seed = int(self.base.get('seed', 43))
        self.run_locally = run_locally

    def add_label(self, label):
        """Per-recording train config and eval settings of one recording, as in its own round."""
        cfg, cell_tpl = round_files(self.round_name, label, self.base, self.grid,
                                    z_scale_from_voxels=bool(self.tpl.get('z_scale_from_voxels', True)))
        self.cell_cfg[label] = cfg
        self.eval_kwargs[label] = oh.resolve_objective(cell_tpl)[2]

    def usable(self, rec):
        """rec if it was scored at this run's length (max_frames), else None."""
        if rec is None:
            return None
        want = self.max_frames
        got = rec.get('max_frames')
        return rec if (got is None and want is None) or (got is not None and want is not None
                                                          and int(got) == int(want)) else None

    def executor(self):
        from submitit import AutoExecutor
        if self.run_locally:
            return AutoExecutor(folder='/tmp/submitit_runs_shared', cluster='debug')
        executor = AutoExecutor(folder=os.path.join(self.round_dir, 'slurm'), cluster='slurm')
        minutes = oh.job_budget_minutes(self.base['epochs'], int(self.tpl.get('objective_minutes') or 120),
                                        train_hours=self.tpl.get('train_hours'))
        executor.update_parameters(timeout_min=minutes)
        executor.update_parameters(**oh.train_job_params(self.tpl, f'shared_{self.round_name}'))
        print(f"Each cell's walltime: {oh.slurm_duration(minutes)}", flush=True)
        return executor

    # ---- trial folders ---------------------------------------------------------------------
    def trial_dir(self, n):
        return os.path.join(self.round_dir, f'trial_{n}')

    def existing_trials(self):
        out = []
        for d in os.listdir(self.round_dir):
            m = re.fullmatch(r'trial_(\d+)', d)
            if m and os.path.exists(os.path.join(self.round_dir, d, 'params.yaml')):
                out.append(int(m.group(1)))
        return sorted(out)

    def cell_config(self, n, label, params, seed, subdir=None):
        cfg = oh.expand_dotted_params({**copy.deepcopy(self.cell_cfg[label]), **params})
        cfg['seed'] = seed
        if self.smoke:  # after the sampled parameters, which may include these
            cfg.update(num_frames=12, epochs=1)
        d = os.path.join(self.trial_dir(n), subdir or label)
        cfg['project_dir'] = d
        return cfg

    def write_cell(self, cfg):
        d = cfg['project_dir']
        for sub in ('', 'log', 'checkpoints'):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        from ruamel.yaml import YAML
        with open(os.path.join(d, 'train_config.yaml'), 'w') as f:
            YAML().dump(cfg, f)

    # ---- main loop -------------------------------------------------------------------------
    def run(self, total_budget, num_parallel_trials):
        from ax.service.ax_client import AxClient
        from submitit import LocalJob, DebugJob
        ax = AxClient(enforce_sequential_optimization=False)
        params_spec = list(self.tpl['hyperparameters'])
        for p in params_spec:
            if p.get('type') == 'choice':
                p.setdefault('is_ordered', p.get('value_type') == 'int')
        oh.create_sweep_experiment(ax, params_spec, accuracy_objective=True)
        records = latest_records(self.results_jsonl)

        # state: ax index -> dict(n, params, seed, acc{label}, jobs{label: job})
        state = {}
        resubmit = []  # (ax index, label, train?)
        for n in self.existing_trials():
            with open(os.path.join(self.trial_dir(n), 'params.yaml')) as f:
                saved = yaml.safe_load(f)
            params, seed = saved['params'], int(saved['seed'])
            if bool(saved.get('smoke', False)) != bool(self.smoke):
                raise ValueError(f"trial_{n} was a {'smoke' if saved.get('smoke') else 'full'} trial and this is a "
                                 f"{'smoke' if self.smoke else 'full'} run: use a separate round directory "
                                 f"(or delete {self.trial_dir(n)} and its records)")
            acc, val = {}, {}
            for label in self.labels:
                rec = self.usable(records.get(cell_tag(self.round_dir, n, label)))
                if metric_value(rec, self.metric) is not None:
                    val[label] = metric_value(rec, self.metric)
                    if rec.get('accuracy') is not None:
                        acc[label] = float(rec['accuracy'])
            idx = oh._attach_trial(ax, oh.expand_dotted_params(dict(params)))
            state[idx] = dict(n=n, params=params, seed=seed, acc=acc, val=val, jobs={}, failed=set())
            obj = regret_objective(val, self.reference, self.labels, self.q)
            if obj is not None:
                ax.complete_trial(idx, raw_data={'result': obj})
                state[idx]['done'] = True
                print(f"Resumed trial_{n}: complete, objective {obj:+.4f}", flush=True)
                continue
            for label in self.labels:
                if label in val:
                    continue
                cdir = os.path.join(self.trial_dir(n), label)
                has_ckpt = oh.checkpoint_path(cdir) is not None
                resubmit.append((idx, label, not has_ckpt))
            print(f"Resumed trial_{n}: {len(val)}/{len(self.labels)} cells recorded; "
                  f"re-running the rest", flush=True)

        executor = self.executor()

        def submit(idx, label, train):
            st = state[idx]
            cfg = self.cell_config(st['n'], label, st['params'], st['seed'])
            if train:
                self.write_cell(cfg)
            tag = cell_tag(self.round_dir, st['n'], label)
            for v in oh._DISPATCH_THREAD_VARS:
                os.environ.pop(v, None)
            job = executor.submit(train_and_score, cfg, self.eval_kwargs[label], tag, self.results_jsonl,
                                  self.emb_dir, train, self.metric)
            for v in oh._DISPATCH_THREAD_VARS:
                os.environ[v] = '1'
            st['jobs'][label] = job
            print(f"Submitted trial_{st['n']} / {label} ({'train + score' if train else 'score only'})",
                  flush=True)

        for idx, label, train in resubmit:
            submit(idx, label, train)

        guard = oh.ConsecutiveFailureGuard(self.tpl.get('max_consecutive_failures',
                                                        oh.DEFAULT_MAX_CONSECUTIVE_FAILURES))
        submitted = len(state)
        next_n = max([s['n'] for s in state.values()], default=-1) + 1
        t0 = time.time()
        while True:
            # Poll
            for idx, st in state.items():
                for label, job in list(st['jobs'].items()):
                    if job.done() or type(job) in (LocalJob, DebugJob):
                        try:
                            res = job.result()
                        except RuntimeError as e:
                            print(f"trial_{st['n']} / {label}: job error {e}", flush=True)
                            res = None
                        del st['jobs'][label]
                        value = metric_value(res, self.metric)
                        if value is None:
                            st['failed'].add(label)
                            print(f"trial_{st['n']} / {label}: FAILED", flush=True)
                        else:
                            st['val'][label] = value
                            a = oh.objective_value(res.get('accuracy'))
                            if a is not None:
                                st['acc'][label] = a
                            print(f"trial_{st['n']} / {label}: {self.metric} {value:.4f} "
                                  f"(reference {self.reference[label]:.4f}); accuracy {a}", flush=True)
                if st.get('done') or st['jobs']:
                    continue
                if not st['failed'] and len(st['val']) < len(self.labels):
                    continue
                obj = None if st['failed'] else regret_objective(st['val'], self.reference, self.labels, self.q)
                if obj is None:
                    oh.log_trial_failure(ax, idx)
                else:
                    ax.complete_trial(idx, raw_data={'result': obj})
                st['done'] = True
                line = dict(trial=st['n'], params=st['params'], seed=st['seed'], objective=obj,
                            metric=self.metric, quantile=self.q, value=st['val'], accuracy=st['acc'],
                            regret={l: st['val'][l] - self.reference[l] for l in st['val']},
                            failed=sorted(st['failed']))
                with open(self.shared_jsonl, 'a') as f:
                    f.write(json.dumps(line) + '\n')
                print(f"trial_{st['n']} done: objective {obj}", flush=True)
                if guard.record(obj is None):
                    raise RuntimeError(f"{guard.streak} trials failed in a row; stopping")
            running = sum(1 for s in state.values() if not s.get('done'))
            if submitted >= total_budget and running == 0:
                break
            # Submit new trials
            free = max(0, num_parallel_trials - running)
            n_new = min(free, total_budget - submitted)
            if n_new > 0:
                proposals, _ = ax.get_next_trials(max_trials=n_new)
                for idx, params in proposals.items():
                    n = next_n
                    next_n += 1
                    seed = oh.search_trial_seed(self.base_seed, n)
                    # a trial folder without params.yaml is a dispatcher that died right after mkdir
                    if os.path.exists(os.path.join(self.trial_dir(n), 'params.yaml')):
                        raise FileExistsError(f"{self.trial_dir(n)} already holds a trial")
                    os.makedirs(self.trial_dir(n), exist_ok=True)
                    with open(os.path.join(self.trial_dir(n), 'params.yaml'), 'w') as f:
                        yaml.safe_dump(dict(params=dict(params), seed=seed, ax_index=idx, smoke=bool(self.smoke)), f)
                    state[idx] = dict(n=n, params=dict(params), seed=seed, acc={}, val={}, jobs={}, failed=set())
                    submitted += 1
                    for label in self.labels:
                        submit(idx, label, True)
            print(f"t={time.time() - t0:.0f}s: {running} trials running, {submitted}/{total_budget} submitted",
                  flush=True)
            time.sleep(5 if self.run_locally else 600)
        self.report()

    # ---- held-out scoring ------------------------------------------------------------------
    def chosen_trials(self, top=1):
        """The `top` finished trials by objective (shared_results.jsonl), best first."""
        if not os.path.exists(self.shared_jsonl):
            raise FileNotFoundError(f"{self.shared_jsonl}: no finished trials to score on held-out data")
        with open(self.shared_jsonl) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        rows = [r for r in rows if r.get('objective') is not None]
        best = {}
        for r in rows:  # a resumed trial may appear twice; keep its latest line
            best[r['trial']] = r
        ranked = sorted(best.values(), key=lambda r: -r['objective'])
        return [r['trial'] for r in ranked[:top]]

    def score_heldout(self, trials, replicates=1, labels=None):
        """Train each trial's config on each held-out recording and score it (see module docstring)."""
        from submitit import LocalJob, DebugJob
        from barlow_track.utils.utils_seeding import replicate_seed
        labels = list(labels or self.heldout)
        if not labels:
            raise ValueError("no heldout_labels in the template")
        clash = set(labels) & set(self.labels)
        if clash:
            raise ValueError(f"{sorted(clash)} are tuning recordings, not held out")
        for label in labels:
            self.add_label(label)
        ref = {k: float(v) for k, v in (self.tpl.get('heldout_reference_accuracy') or {}).items()}
        out_jsonl = os.path.join(self.round_dir, 'heldout_results.jsonl')
        records = latest_records(self.results_jsonl)
        executor = self.executor()
        jobs, rows = {}, []
        for n in trials:
            with open(os.path.join(self.trial_dir(n), 'params.yaml')) as f:
                saved = yaml.safe_load(f)
            params, seed0 = saved['params'], int(saved['seed'])
            for r in range(replicates):
                seed = replicate_seed(seed0, r)
                tnum = str(n) if r == 0 else f'{n}r{r}'
                for label in labels:
                    tag = cell_tag(self.round_dir, tnum, label)
                    key = (n, r, label)
                    meta = dict(trial=n, replicate=r, seed=seed, label=label, tag=tag)
                    rec = self.usable(records.get(tag))
                    if rec is not None and oh.objective_value(rec.get('accuracy')) is not None:
                        rows.append(dict(meta, accuracy=float(rec['accuracy']),
                                         consistency=rec.get('consistency'), recorded=True))
                        print(f"{tag}: already recorded, accuracy {rec['accuracy']:.4f}", flush=True)
                        continue
                    cfg = self.cell_config(n, label, params, seed, subdir=label if r == 0 else f'{label}_rep{r}')
                    train = oh.checkpoint_path(cfg['project_dir']) is None
                    if train:
                        self.write_cell(cfg)
                    for v in oh._DISPATCH_THREAD_VARS:
                        os.environ.pop(v, None)
                    jobs[key] = (meta, executor.submit(train_and_score, cfg, self.eval_kwargs[label], tag,
                                                        self.results_jsonl, self.emb_dir, train, None))
                    for v in oh._DISPATCH_THREAD_VARS:
                        os.environ[v] = '1'
                    print(f"Submitted {tag} ({'train + score' if train else 'score only'})", flush=True)
        t0 = time.time()
        while jobs:
            for key, (meta, job) in list(jobs.items()):
                if not (job.done() or type(job) in (LocalJob, DebugJob)):
                    continue
                try:
                    res = job.result()
                except RuntimeError as e:
                    print(f"{meta['tag']}: job error {e}", flush=True)
                    res = None
                del jobs[key]
                acc = oh.objective_value((res or {}).get('accuracy'))
                row = dict(meta, accuracy=acc, consistency=(res or {}).get('consistency'), recorded=False)
                if acc is not None and meta['label'] in ref:
                    row['gap'] = acc - ref[meta['label']]
                with open(out_jsonl, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                rows.append(row)
                print(f"{meta['tag']}: accuracy {acc}" + (f" (reference {ref[meta['label']]:.4f})"
                                                          if meta['label'] in ref else ''), flush=True)
            if jobs:
                print(f"t={time.time() - t0:.0f}s: {len(jobs)} held-out cells running", flush=True)
                time.sleep(5 if self.run_locally else 600)
        self.heldout_summary(rows, labels, ref)
        return rows

    @staticmethod
    def heldout_summary(rows, labels, ref):
        print("\nHeld-out accuracy (mean over replicates; failed cells as None)", flush=True)
        for n in sorted({r['trial'] for r in rows}):
            parts = []
            for label in labels:
                accs = [r['accuracy'] for r in rows if r['trial'] == n and r['label'] == label
                        and r['accuracy'] is not None]
                if not accs:
                    parts.append(f"{label} None")
                    continue
                m = float(np.mean(accs))
                gap = f" ({m - ref[label]:+.4f})" if label in ref else ''
                parts.append(f"{label} {m:.4f}{gap}" + (f" n={len(accs)}" if len(accs) > 1 else ''))
            print(f"  trial_{n}: " + ', '.join(parts), flush=True)

    def report(self):
        if not os.path.exists(self.shared_jsonl):
            return
        with open(self.shared_jsonl) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        ok = [r for r in rows if r['objective'] is not None]
        if not ok:
            print("No completed trials", flush=True)
            return
        best = max(ok, key=lambda r: r['objective'])
        print(f"Best observed: trial_{best['trial']} objective {best['objective']:+.4f}; "
              f"params {best['params']}", flush=True)
        with open(os.path.join(self.round_dir, 'best_parameters.yaml'), 'w') as f:
            yaml.safe_dump(dict(trial=best['trial'], objective=best['objective'], metric=best.get('metric', 'accuracy'),
                                params=best['params'], value=best.get('value'), accuracy=best['accuracy']), f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-p', '--template', required=True, help='shared_template.yaml or its round directory')
    ap.add_argument('--total_budget', type=int, default=None, help='trials (default: template total_budget)')
    ap.add_argument('--num_parallel_trials', type=int, default=None,
                    help='trials in flight; each runs one GPU job per tuning recording')
    ap.add_argument('--run_locally', action='store_true')
    ap.add_argument('--smoke', action='store_true',
                    help='tiny trainings (12 frames, 1 epoch) scored on 12 frames: tests the plumbing only')
    ap.add_argument('--score_heldout', action='store_true',
                    help='train and score finished trial(s) on the held-out recordings instead of sweeping')
    ap.add_argument('--trials', type=int, nargs='+', default=None,
                    help='with --score_heldout: these trials (default: the --top best by objective)')
    ap.add_argument('--top', type=int, default=1, help='with --score_heldout: how many best trials')
    ap.add_argument('--replicates', type=int, default=1,
                    help='with --score_heldout: training seeds per (trial, recording); replicate 0 = the trial seed')
    ap.add_argument('--heldout_labels', nargs='+', default=None,
                    help="with --score_heldout: a subset of the template's heldout_labels")
    a = ap.parse_args()
    sweep = SharedSweep(a.template, run_locally=a.run_locally, smoke=a.smoke)
    if a.score_heldout:
        trials = a.trials or sweep.chosen_trials(a.top)
        print(f"Held-out scoring of {sweep.round_name} trials {trials} on {a.heldout_labels or sweep.heldout}, "
              f"{a.replicates} replicate(s)", flush=True)
        sweep.score_heldout(trials, replicates=a.replicates, labels=a.heldout_labels)
        return
    budget = a.total_budget or int(sweep.tpl.get('total_budget', 40))
    par = a.num_parallel_trials or int(sweep.tpl.get('num_parallel_trials', 2))
    print(f"Shared sweep {sweep.round_name}: tuning {sweep.labels}, held out {sweep.heldout}; objective "
          f"= {sweep.q:.2f}-quantile of ({sweep.metric} - reference); {budget} trials, {par} in flight "
          f"({par * len(sweep.labels)} GPU jobs)", flush=True)
    sweep.run(budget, par)


if __name__ == '__main__':
    main()
