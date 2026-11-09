"""Sensitivity tests around a frozen training recipe: one config, many one-factor variants, every recording.

The shared-config sweep (optimize_shared_config.py, round_shared_v1) picked ONE recipe for all
recordings. The paper's ablations (training volumes, each augmentation, the two Barlow loss terms,
hyperparameters, an untrained network) are then run around that recipe: each VARIANT is the frozen
trial's parameters plus a few overrides, trained anew on each recording and scored with the same
full-video tracking accuracy as the sweep. All variants share the training seeds of a replicate
(replicate r uses replicate_seed(trial seed, r)), so a variant is compared with the control in pairs.

    # 1. stage a round directory next to the shared round (copies base_train_config.yaml, writes
    #    sensitivity_template.yaml with the default variants, trial_<i>/params.yaml, variants.json, README.md)
    python barlow_track/scripts/sensitivity_sweep.py --init round_sensitivity_v1 --source_round round_shared_v1 --base_trial 59
    # 2. look at what would run (no jobs)
    python barlow_track/scripts/sensitivity_sweep.py -p round_sensitivity_v1 --dry_run
    # 3. launch (on the cluster through sbatch_dispatcher.sbatch with BARLOW_DISPATCHER=barlow_track/scripts/sensitivity_sweep.py)
    python barlow_track/scripts/sensitivity_sweep.py -p round_sensitivity_v1 --tier 1
    python barlow_track/scripts/sensitivity_sweep.py -p round_sensitivity_v1 --tier 2
    # 4. table of accuracy per variant and recording, paired differences against the control
    python barlow_track/scripts/sensitivity_sweep.py -p round_sensitivity_v1 --report

Cell = (variant i, replicate r, recording): trained in trial_<i>/<label>[_rep<r>]/, recorded in
exp_results.jsonl under the benchmark tag <round>_trial<i>[r<r>]_<label> (the tag convention of the
shared round's held-out scoring), embedding in emb_cache/. Resumable: recorded cells are skipped, cells
with a checkpoint are only re-scored, the rest retrain. At most --max_cells run at once.

Variant overrides are dotted config keys (global_augment.p_flip) with a value, or {mul: f} to scale the
recording's own value (target_sz_z 16 on Flavell stays proportional; lr; epochs).
"""
import argparse
import copy
import json
import logging
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import yaml

from barlow_track.scripts import optimize_hyperparameters as oh
from barlow_track.scripts.optimize_shared_config import (
    SharedSweep, cell_tag, latest_records, train_and_score)

CONTROL = 'control'

# Switches that turn each augmentation of the position pipeline off (key -> value)
AUG_OFF = {
    'affine': {'global_augment.p_global_affine': 0.0},      # rotation, scale, translation (and the flip)
    'elastic': {'global_augment.elastic_scale': 0.0},
    'noise': {'crop_photometric.p_noise': 0.0},
    'croprot': {'crop_photometric.crop_degrees_z': 0.0},
    'posaug': {'position_augment.dropout_p': 0.0, 'position_augment.spurious_p': 0.0},
}
# "only X": X stays at the recipe's value, the other image augmentations are off
_IMAGE_AUGS = ('affine', 'elastic', 'noise', 'croprot')


def _off(*names):
    out = {}
    for n in names:
        out.update(AUG_OFF[n])
    return out


def default_variants():
    """The paper's sensitivity tests as (name, group, tier, replicates, overrides); see the README written by --init.

    Tier 1: what the paper defends (loss terms x training volumes, fused and image-only encoder; training
    volumes), 3 replicates, 5 for the control. Tier 2: augmentation and hyperparameter ablations, an
    untrained network, a matched-step control for few training volumes: 1 replicate.
    """
    v = []

    def add(name, group, tier, overrides, replicates=None):
        v.append(dict(name=name, group=group, tier=tier, overrides=overrides,
                      replicates=replicates or (3 if tier == 1 else 1)))

    add(CONTROL, 'control', 1, {}, replicates=5)
    # loss terms x training volumes. lambd_obj 0 = feature-space Barlow only, 1 = object-space only;
    # 'rec' = the recipe's own value, which is the control at the recipe's num_frames. Image-only =
    # fusion visual_only (the position pipeline with the backbone output as descriptor; use_position
    # false would switch to the legacy per-crop pipeline and ignore global_augment)
    for enc, group, extra in (('', 'loss_x_volumes_fused', {}),
                              ('img_', 'loss_x_volumes_image_only', {'fusion': 'visual_only'})):
        for n in (16, 64, 'rec'):
            for lam in (0.0, 'rec', 1.0):
                if enc == '' and n == 'rec' and lam == 'rec':
                    continue  # the control
                name = f"{enc}lobj{'rec' if lam == 'rec' else int(lam)}_n{n}"
                o = dict(extra)
                if lam != 'rec':
                    o['lambd_obj'] = lam
                if n != 'rec':
                    o['num_frames'] = n
                add(name, group, 1, o)
    for n in (128, 256, 1000):
        add(f'n{n}', 'training_volumes', 1, {'num_frames': n})
    # tier 2
    add('aug_none', 'augmentation', 2, _off(*_IMAGE_AUGS, 'posaug'))
    for a in ('affine', 'elastic', 'noise', 'croprot', 'posaug'):
        add(f'aug_no_{a}', 'augmentation', 2, _off(a))
    for a in ('affine', 'elastic', 'noise'):
        add(f'aug_only_{a}', 'augmentation', 2, _off(*[b for b in _IMAGE_AUGS if b != a]))
    add('lr_x0.3', 'hyperparameter', 2, {'lr': {'mul': 0.3}})
    add('lr_x3', 'hyperparameter', 2, {'lr': {'mul': 3.0}})
    for d in (64, 128, 1024):
        add(f'emb{d}', 'hyperparameter', 2, {'embedding_dim': d})
    for d in (128, 2048):
        add(f'projf{d}', 'hyperparameter', 2, {'projector_final': d})
    add('crop_xy32', 'crop_size', 2, {'target_sz_xy': {'mul': 0.5}})
    add('crop_xy48', 'crop_size', 2, {'target_sz_xy': {'mul': 0.75}})
    add('crop_z_half', 'crop_size', 2, {'target_sz_z': {'mul': 0.5}})
    add('crop_z_x1.5', 'crop_size', 2, {'target_sz_z': {'mul': 1.5}})
    add('untrained', 'untrained', 2, {'epochs': 0})
    # 64 volumes for as many optimizer steps as the control's (371 frames x 30 epochs)
    add('n64_matched_steps', 'training_volumes', 2, {'num_frames': 64, 'epochs': {'mul': 371 / 64}})
    return v


def apply_overrides(cfg, overrides):
    """A copy of cfg with overrides applied: plain value = set; {mul: f} = f x the current value (int stays int)."""
    cfg = copy.deepcopy(cfg)
    plain = {}
    for key, val in overrides.items():
        if isinstance(val, dict) and set(val) == {'mul'}:
            cur = _get_dotted(cfg, key)
            if cur is None:
                raise KeyError(f"override {key}: {{mul}} needs a current value, the config has none")
            new = float(cur) * float(val['mul'])
            plain[key] = max(1, int(round(new))) if isinstance(cur, int) and not isinstance(cur, bool) else new
        else:
            plain[key] = val
    # a dotted key must reach the nested value, so expand after reading the multipliers
    return oh.expand_dotted_params({**cfg, **plain})


def _get_dotted(cfg, key):
    node = cfg
    for part in key.split('.'):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


class SensitivitySweep(SharedSweep):
    """SharedSweep's per-recording configs and eval settings, driven by variants instead of Ax trials."""

    def __init__(self, template_path, run_locally=False, smoke=False):
        if os.path.isdir(template_path):
            template_path = os.path.join(template_path, 'sensitivity_template.yaml')
        super().__init__(template_path, run_locally=run_locally, smoke=smoke)
        self.variants = list(self.tpl['variants'])
        names = [v['name'] for v in self.variants]
        if len(set(names)) != len(names) or CONTROL not in names:
            raise ValueError(f"variant names must be unique and include {CONTROL!r}")
        self.default_replicates = int(self.tpl.get('replicates', 3))
        self.base_params = dict(self.tpl['base_params'])
        self.base_seed_trial = int(self.tpl['base_seed'])

    def cell_config_variant(self, i, label, variant, seed, rep):
        cfg = oh.expand_dotted_params({**copy.deepcopy(self.cell_cfg[label]), **self.base_params})
        cfg = apply_overrides(cfg, variant['overrides'])
        cfg['seed'] = seed
        if self.smoke:  # caps only, so the untrained variant (epochs 0) stays untrained
            cfg.update(num_frames=min(12, int(cfg['num_frames'])), epochs=min(1, int(cfg['epochs'])))
        cfg['project_dir'] = os.path.join(self.trial_dir(i), label if rep == 0 else f'{label}_rep{rep}')
        return cfg

    def cells(self, tiers=None, names=None, labels=None, replicates=None):
        """Cells in launch order: replicate 0 of every variant first, so partial results already compare."""
        from barlow_track.utils.utils_seeding import replicate_seed
        labels = list(labels or self.labels)
        out = []
        for i, v in enumerate(self.variants):
            if tiers and v['tier'] not in tiers:
                continue
            if names and v['name'] not in names:
                continue
            n_rep = replicates or v.get('replicates') or self.default_replicates
            for r in range(int(n_rep)):
                for label in labels:
                    out.append(dict(i=i, variant=v, rep=r, label=label,
                                    seed=replicate_seed(self.base_seed_trial, r),
                                    trial=str(i) if r == 0 else f'{i}r{r}'))
        out.sort(key=lambda c: (c['rep'], c['i'], c['label']))
        return out

    def run_cells(self, cells, max_cells=24):
        from submitit import LocalJob, DebugJob
        for c in cells:
            if c['label'] not in self.cell_cfg:
                self.add_label(c['label'])
        records = latest_records(self.results_jsonl)
        executor = self.executor()
        pending, running, done, failed = [], {}, 0, []
        for c in cells:
            tag = cell_tag(self.round_dir, c['trial'], c['label'])
            rec = self.usable(records.get(tag))
            if rec is not None and oh.objective_value(rec.get('accuracy')) is not None:
                done += 1
                continue
            pending.append(dict(c, tag=tag))
        print(f"{len(cells)} cells: {done} already recorded, {len(pending)} to run, <= {max_cells} at once",
              flush=True)
        t0 = time.time()
        while pending or running:
            while pending and len(running) < max_cells:
                c = pending.pop(0)
                cfg = self.cell_config_variant(c['i'], c['label'], c['variant'], c['seed'], c['rep'])
                train = oh.checkpoint_path(cfg['project_dir']) is None
                if train:
                    self.write_cell(cfg)
                for var in oh._DISPATCH_THREAD_VARS:
                    os.environ.pop(var, None)
                job = executor.submit(train_and_score, cfg, self.eval_kwargs[c['label']], c['tag'],
                                      self.results_jsonl, self.emb_dir, train, None)
                for var in oh._DISPATCH_THREAD_VARS:
                    os.environ[var] = '1'
                running[c['tag']] = (c, job)
                print(f"Submitted {c['tag']} [{c['variant']['name']} rep {c['rep']}] "
                      f"({'train + score' if train else 'score only'})", flush=True)
            for tag, (c, job) in list(running.items()):
                if not (job.done() or type(job) in (LocalJob, DebugJob)):
                    continue
                try:
                    res = job.result()
                except RuntimeError as e:
                    print(f"{tag}: job error {e}", flush=True)
                    res = None
                del running[tag]
                acc = oh.objective_value((res or {}).get('accuracy'))
                if acc is None:
                    failed.append(tag)
                    print(f"{tag}: FAILED (no accuracy)", flush=True)
                else:
                    print(f"{tag}: accuracy {acc:.4f}", flush=True)
            if pending or running:
                print(f"t={time.time() - t0:.0f}s: {len(running)} running, {len(pending)} waiting, "
                      f"{len(failed)} failed", flush=True)
                time.sleep(5 if self.run_locally else 600)
        print(f"Done. {len(failed)} failed cells (rerun the same command to retry them): {failed}", flush=True)
        return failed

    # ---- report ----------------------------------------------------------------------------
    def table(self):
        """{variant name: {label: [accuracy per replicate (None for a missing one)]}}, from the records."""
        recs = latest_records(self.results_jsonl)
        out = {}
        for c in self.cells(labels=self.labels):
            rec = self.usable(recs.get(cell_tag(self.round_dir, c['trial'], c['label'])))
            acc = oh.objective_value((rec or {}).get('accuracy'))
            out.setdefault(c['variant']['name'], {}).setdefault(c['label'], {})[c['rep']] = acc
        return out

    def report(self):
        tab = self.table()
        ctrl = tab[CONTROL]
        lines, rows = [], []
        header = ['variant', 'group', 'tier'] + self.labels + ['mean', 'd_vs_control(paired)', 'n_pairs']
        lines.append('\t'.join(header))
        groups = {v['name']: (v['group'], v['tier']) for v in self.variants}
        for v in self.variants:
            name = v['name']
            per_label, diffs = {}, []
            for label in self.labels:
                accs = tab[name].get(label, {})
                vals = [a for a in accs.values() if a is not None]
                per_label[label] = float(np.mean(vals)) if vals else None
                for r, a in accs.items():
                    c = ctrl.get(label, {}).get(r)
                    if a is not None and c is not None and name != CONTROL:
                        diffs.append(a - c)
            have = [x for x in per_label.values() if x is not None]
            mean = float(np.mean(have)) if len(have) == len(self.labels) else None
            fmt = lambda x: 'NA' if x is None else f'{x:.4f}'
            d = f'{np.mean(diffs):+.4f}' if diffs else ('-' if name == CONTROL else 'NA')
            lines.append('\t'.join([name, groups[name][0], str(groups[name][1])] +
                                   [fmt(per_label[l]) for l in self.labels] + [fmt(mean), d, str(len(diffs))]))
            rows.append(dict(variant=name, group=groups[name][0], tier=groups[name][1], accuracy=per_label,
                             mean=mean, paired_diff=float(np.mean(diffs)) if diffs else None,
                             paired_sd=float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
                             n_pairs=len(diffs)))
        text = '\n'.join(lines)
        print(text)
        with open(os.path.join(self.round_dir, 'sensitivity_summary.tsv'), 'w') as f:
            f.write(text + '\n')
        with open(os.path.join(self.round_dir, 'sensitivity_summary.json'), 'w') as f:
            json.dump(rows, f, indent=1)
        # replicate noise of the control: what a variant has to beat
        sds = [float(np.std([a for a in r.values() if a is not None], ddof=1)) for r in ctrl.values()
               if len([a for a in r.values() if a is not None]) > 1]
        if sds:
            print(f"\nControl seed sd per recording: mean {np.mean(sds):.4f}, max {np.max(sds):.4f} "
                  f"(n={len(sds)} recordings)", flush=True)
        return rows


# ---- staging --------------------------------------------------------------------------------
README = """# {name} (sensitivity tests around round_shared_v1 trial {trial}, staged {date}, NOT launched)

One frozen recipe (the parameters of {source}/trial_{trial}/params.yaml, base config copied from
{source}/base_train_config.yaml) with one-factor variants, trained anew on each recording and scored
by full-video tracking accuracy exactly as in {source}. Variants share training seeds per replicate
(replicate r: replicate_seed({seed}, r)), so each is compared with the control in pairs.

Groups (tier 1 = 3 replicates, control 5; tier 2 = 1 replicate): {groups}.
{n_cells} cells for the default labels {labels}.

Launch (cluster login node, after git pull):

    B=/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files
    R={name}
    BARLOW_DISPATCHER=barlow_track/scripts/sensitivity_sweep.py sbatch --job-name=disp_$R \\
        --output=$B/$R/dispatcher_%j.log \\
        /lisc/data/scratch/neurobiology/zimmer/wbfm/code/barlow_track/barlow_track/scripts/sbatch_dispatcher.sbatch \\
        -p $B/$R/sensitivity_template.yaml --tier 1

then the same with --tier 2. Add --max_cells N to change the number of simultaneous GPU jobs. Read with
--report. Notes: keypoint jitter (position_augment.jitter_std) stays on in every variant; crop sizes and
epochs are scaled from each recording's own value; Flavell's voxel size (z_scale 1.0) is unresolved
(issue i261105-3), so read the Flavell crop-size cells with care.
"""


def init_round(name, source_round, base_trial, parent=None, labels=None, replicates=3):
    """Stage round NAME (next to the source round unless parent is given) from the source's frozen trial."""
    src = os.path.abspath(source_round)
    parent = parent or os.path.dirname(src)
    round_dir = os.path.join(parent, name)
    os.makedirs(round_dir, exist_ok=True)
    with open(os.path.join(src, 'shared_template.yaml')) as f:
        shared = yaml.safe_load(f)
    with open(os.path.join(src, f'trial_{base_trial}', 'params.yaml')) as f:
        saved = yaml.safe_load(f)
    shutil.copy(os.path.join(src, shared['base_train_config']), os.path.join(round_dir, 'base_train_config.yaml'))
    keep = ('z_scale_from_voxels', 'objective_minutes', 'train_hours', 'train_gpu_job_cpus',
            'max_consecutive_failures')
    tpl = {k: shared[k] for k in shared if k.startswith('objective_') or k.startswith('train_') or k in keep}
    tpl.update(
        base_train_config='base_train_config.yaml', source_round=os.path.basename(src), base_trial=int(base_trial),
        base_params=saved['params'], base_seed=int(saved['seed']), replicates=int(replicates),
        tuning_labels=list(labels or shared['tuning_labels']), heldout_labels=[],
        objective_metric='accuracy', reference_accuracy=shared['reference_accuracy'],
        variants=default_variants())
    path = os.path.join(round_dir, 'sensitivity_template.yaml')
    with open(path, 'w') as f:
        f.write(f"# {name}: sensitivity tests around {os.path.basename(src)} trial {base_trial}; edit variants freely\n")
        yaml.safe_dump(tpl, f, sort_keys=False)
    sweep = SensitivitySweep(path)
    for i, v in enumerate(sweep.variants):  # trial_<i>/params.yaml: what a variant changes (for the record)
        d = sweep.trial_dir(i)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'params.yaml'), 'w') as f:
            yaml.safe_dump(dict(params=sweep.base_params, seed=sweep.base_seed_trial, variant=v['name'],
                                group=v['group'], overrides=v['overrides']), f, sort_keys=False)
    with open(os.path.join(round_dir, 'variants.json'), 'w') as f:
        json.dump({i: v['name'] for i, v in enumerate(sweep.variants)}, f, indent=1)
    groups = ', '.join(sorted({v['group'] for v in sweep.variants}))
    with open(os.path.join(round_dir, 'README.md'), 'w') as f:
        f.write(README.format(name=name, trial=base_trial, source=os.path.basename(src), date=time.strftime('%Y-%m-%d'),
                              seed=saved['seed'], groups=groups, n_cells=len(sweep.cells()), labels=sweep.labels))
    print(f"Staged {round_dir}: {len(sweep.variants)} variants, {len(sweep.cells())} cells")
    return round_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-p', '--template', help='sensitivity_template.yaml or its round directory')
    ap.add_argument('--init', metavar='NAME', help='stage a new round directory NAME next to --source_round')
    ap.add_argument('--source_round', help='with --init: the shared-config round (directory) holding the frozen trial')
    ap.add_argument('--base_trial', type=int, help='with --init: the frozen trial of the source round')
    ap.add_argument('--out_dir', help='with --init: parent directory of the new round (default: the source round\'s)')
    ap.add_argument('--tier', type=int, nargs='+', default=None, help='only these tiers (1 = core, 2 = ablations)')
    ap.add_argument('--variants', nargs='+', default=None, help='only these variant names')
    ap.add_argument('--labels', nargs='+', default=None, help='recordings (default: the template tuning_labels)')
    ap.add_argument('--replicates', type=int, default=None, help='override the replicates of every variant')
    ap.add_argument('--max_cells', type=int, default=24, help='GPU jobs at once')
    ap.add_argument('--dry_run', action='store_true', help='list the cells and their status, submit nothing')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--run_locally', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='tiny trainings and scoring (plumbing only)')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if a.init:
        if not (a.source_round and a.base_trial is not None):
            ap.error('--init needs --source_round and --base_trial')
        init_round(a.init, a.source_round, a.base_trial, parent=a.out_dir, labels=a.labels)
        return
    if not a.template:
        ap.error('-p is required')
    sweep = SensitivitySweep(a.template, run_locally=a.run_locally, smoke=a.smoke)
    if a.report:
        sweep.report()
        return
    cells = sweep.cells(tiers=a.tier, names=a.variants, labels=a.labels, replicates=a.replicates)
    if a.dry_run:
        recs = latest_records(sweep.results_jsonl)
        n_done = 0
        by_var = defaultdict(lambda: [0, 0])
        for c in cells:
            rec = sweep.usable(recs.get(cell_tag(sweep.round_dir, c['trial'], c['label'])))
            ok = oh.objective_value((rec or {}).get('accuracy')) is not None
            n_done += ok
            by_var[c['variant']['name']][0] += 1
            by_var[c['variant']['name']][1] += ok
        for v in sweep.variants:
            if v['name'] in by_var:
                n, d = by_var[v['name']]
                print(f"  {v['name']:<22} tier {v['tier']}  {v['group']:<28} {d}/{n} recorded  {v['overrides']}")
        print(f"{len(cells)} cells, {n_done} recorded, {len(cells) - n_done} to run")
        return
    print(f"Sensitivity round {sweep.round_name}: {len(cells)} cells on {a.labels or sweep.labels}", flush=True)
    sweep.run_cells(cells, max_cells=a.max_cells)


if __name__ == '__main__':
    main()
