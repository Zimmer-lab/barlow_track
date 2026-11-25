"""Score every network of a pooled-training round on all benchmark recordings.

A pooled round trains ONE network on several recordings (train_config.yaml `project_paths`; see
train_barlow_clusterer.resolve_recordings), typically a direct grid over num_frames (per recording)
x seed under objective: loss, so the round itself scores nothing. This script scores each finished
trial_N/resnet50.pth on every recording of the reference round (default the sibling round_shared_v1,
trial 59: the 12 benchmark recordings), exactly as transfer_matrix.py scores a cross-recording cell:

  * the evaluation settings of target T are those of T's own trial-59 cell (its record in
    <ref>/exp_results.jsonl: project or NWB, GT, objects, crop window and centres; cell_kwargs),
  * the ViSNet z_scale is T's (the visnet_z_scale its trial-59 network trained with); a pooled
    checkpoint saves none, and eval_accuracy refuses one without an explicit value,
  * the crop size is the checkpoint's own (one crop shape per network),
  * the label propagation recipe is --labelprop (default v2).

Records go to <round>/exp_results_pooled_<labelprop>.jsonl, embeddings to <round>/emb_cache_pooled/.
Tags: <round>_trial<N>_on_<T>. Finished cells are skipped on a rerun; unfinished trials (no
resnet50.pth, or stats.json short of the configured epochs) are left out until they finish.

    BARLOW_DISPATCHER=barlow_track/scripts/scratch/transfer/score_pooled_round.py sbatch --output=<log> \
        barlow_track/scripts/sbatch_dispatcher.sbatch --round_dir <pooled round> --slurm 8
    python .../score_pooled_round.py --round_dir <round> --list      # cells, crop, z_scale, done or not
    python .../score_pooled_round.py --round_dir <round> --report    # per trial, and per data amount over seeds
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
import zlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *['..'] * 4)))  # repo root

from barlow_track.scripts.scratch.gt_free_proxy.rescore_round import cell_kwargs, latest_records  # noqa: E402
from barlow_track.scripts.scratch.transfer.transfer_matrix import _z_scale, own_cell, source_networks  # noqa: E402

# The pooled networks' training recordings (round_shared_v1 README: the tuning split); every other
# reference recording is held out
TUNING = ('flavell_2022032402', 'flavell_2022033002', 'samuel_137', 'samuel_184', 'zimmer_1123', 'zimmer_1210')

# train_config keys that differ between trials by construction, not as a grid setting
_PER_RUN_KEYS = {'seed', 'project_dir', 'wandb_name', 'config_fname', 'model_fname'}


def ref_dir(a):
    return os.path.abspath(a.ref_round_dir or os.path.join(os.path.dirname(os.path.abspath(a.round_dir)),
                                                            'round_shared_v1'))


def ref_targets(a):
    """{label: T's own reference record} for every recording with a reference network and record."""
    rd = ref_dir(a)
    records = latest_records(os.path.join(rd, 'exp_results.jsonl'))
    out = {}
    for label in source_networks(rd, a.ref_trial):
        rec = own_cell(records, os.path.basename(rd), a.ref_trial, label)
        if rec is None:
            print(f"WARNING: {label}: no own record in {rd}; not a target", flush=True)
        elif not a.targets or label in a.targets:
            out[label] = rec
    return out


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def pooled_trials(a, verbose=True):
    """[(trial number, checkpoint, train_config)] of the round's finished trials, by number."""
    from barlow_track.utils.utils_ground_truth import check_training_finished
    rd = os.path.abspath(a.round_dir)
    out = []
    for d in os.listdir(rd):
        m = re.fullmatch(r'trial_(\d+)', d)
        if m is None or (a.trials and int(m.group(1)) not in a.trials):
            continue
        td = os.path.join(rd, d)
        ckpt, cfg_path = os.path.join(td, 'resnet50.pth'), os.path.join(td, 'train_config.yaml')
        if not (os.path.isfile(ckpt) and os.path.isfile(cfg_path)):
            if verbose:
                print(f"{d}: no checkpoint or train_config.yaml yet; left out", flush=True)
            continue
        cfg = _load_yaml(cfg_path)
        if not check_training_finished(td, int(cfg['epochs']) - 1):
            if verbose:
                print(f"{d}: training not finished; left out", flush=True)
            continue
        out.append((int(m.group(1)), ckpt, cfg))
    return sorted(out, key=lambda x: x[0])


def cell_tag(round_name, trial, target):
    return f'{round_name}_trial{trial}_on_{target}'


def cells(a, verbose=True):
    """[(tag, trial, target, weights, target record, z_scale, crop, train_config)], trial-major."""
    name = os.path.basename(os.path.abspath(a.round_dir))
    targets = ref_targets(a)
    out = []
    for n, ckpt, cfg in pooled_trials(a, verbose=verbose):
        crop = [cfg.get('target_sz_z'), cfg.get('target_sz_xy'), cfg.get('target_sz_xy')]
        for t in sorted(targets):
            out.append((cell_tag(name, n, t), n, t, ckpt, targets[t], _z_scale(targets[t]), crop, cfg))
    return out


def paths(a):
    rd = os.path.abspath(a.round_dir)
    return (rd, a.out_jsonl or os.path.join(rd, f'exp_results_pooled_{a.labelprop}.jsonl'),
            a.emb_dir or os.path.join(rd, 'emb_cache_pooled'))


def _pooled_labels(cfg):
    return [e.get('label') for e in (cfg.get('project_paths') or [])]


def run_shard(a, shard):
    rd, out_jsonl, emb_dir = paths(a)
    todo = cells(a)
    mine = [c for c in todo if zlib.crc32(c[0].encode()) % a.num_shards == shard]
    print(f"{len(todo)} pooled cells; shard {shard}/{a.num_shards}: {len(mine)}; -> {out_jsonl}", flush=True)
    os.makedirs(emb_dir, exist_ok=True)
    from barlow_track.scripts.eval_accuracy import evaluate_trained_checkpoint
    n_fail = 0
    for tag, n, t, weights, tgt, z, _, cfg in mine:
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
                extra_record=dict(pooled_round=os.path.basename(rd), pooled_trial=n, pooled_target=t,
                                  pooled_labels=_pooled_labels(cfg), pooled_num_frames=cfg.get('num_frames'),
                                  pooled_seed=cfg.get('seed'), target_own_tag=tgt['tag']),
                **cell_kwargs(tgt))
            print(f"{tag}: accuracy {rec.get('accuracy'):.4f} in {time.time() - t0:.0f}s", flush=True)
        except Exception:
            n_fail += 1
            print(f"{tag}: FAILED\n{traceback.format_exc()}", flush=True)
    print(f"shard {shard} finished, {n_fail} failed of {len(mine)}", flush=True)
    return n_fail


def dispatch(a):
    from barlow_track.scripts.scratch.transfer import transfer_matrix
    return transfer_matrix.dispatch(a, shard_fn=run_shard, report_fn=report, paths_fn=paths, job_prefix='pooled')


def _grid_keys(configs):
    """train_config keys whose values differ between the trials (the grid), the per-run ones aside."""
    keys = set().union(*configs) - _PER_RUN_KEYS if configs else set()
    canon = lambda v: json.dumps(v, sort_keys=True, default=str)
    return sorted(k for k in keys if len({canon(c.get(k)) for c in configs}) > 1)


def _mean(v):
    return sum(v) / len(v) if v else None


def _fmt(v, width=9):
    return f"{v:{width}.4f}" if v is not None else f"{'-':>{width}}"


def summarize(acc, trial_cfgs, labels, tuning):
    """Rows of the report as plain data (tested): per trial, and per grid setting over seeds.

    acc: {(trial, target): accuracy}; trial_cfgs: {trial: train_config}. A trial's tuning / held-out
    mean is over the targets it has; n_tuning / n_held_out say how many.
    """
    held = [t for t in labels if t not in tuning]
    keys = _grid_keys(list(trial_cfgs.values()))

    def split_means(get):
        tu = [get(t) for t in labels if t in tuning and get(t) is not None]
        ho = [get(t) for t in held if get(t) is not None]
        return _mean(tu), len(tu), _mean(ho), len(ho)

    trials = []
    for n in sorted(trial_cfgs):
        per = {t: acc.get((n, t)) for t in labels}
        tu, ntu, ho, nho = split_means(per.get)
        trials.append(dict(trial=n, setting=tuple((k, trial_cfgs[n].get(k)) for k in keys),
                           seed=trial_cfgs[n].get('seed'), acc=per, tuning=tu, n_tuning=ntu,
                           held_out=ho, n_held_out=nho))
    groups = {}
    for r in trials:
        groups.setdefault(r['setting'], []).append(r)
    settings = []
    for setting, rows in groups.items():
        per = {t: _mean([r['acc'][t] for r in rows if r['acc'][t] is not None]) for t in labels}
        tu, ntu, ho, nho = split_means(per.get)
        sd = lambda v: (sum((x - _mean(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else None
        settings.append(dict(setting=setting, seeds=[r['seed'] for r in rows], acc=per, tuning=tu, n_tuning=ntu,
                             held_out=ho, n_held_out=nho,
                             tuning_sd=sd([r['tuning'] for r in rows if r['tuning'] is not None]),
                             held_out_sd=sd([r['held_out'] for r in rows if r['held_out'] is not None])))

    def order(s):
        return tuple((str(type(v)), v if isinstance(v, (int, float)) else str(v)) for _, v in s['setting'])
    return keys, trials, sorted(settings, key=order)


def report(a):
    rd, out_jsonl, _ = paths(a)
    name = os.path.basename(rd)
    records = latest_records(out_jsonl)
    targets = ref_targets(a)
    labels = sorted(targets)
    trial_cfgs = {n: cfg for n, _, cfg in pooled_trials(a, verbose=False)}
    acc = {}
    for n in trial_cfgs:
        for t in labels:
            r = records.get(cell_tag(name, n, t))
            if r is not None and r.get('accuracy') is not None:
                acc[n, t] = float(r['accuracy'])
    tuning = set(a.tuning)
    for n, cfg in trial_cfgs.items():
        pl = set(_pooled_labels(cfg))
        if pl and pl != tuning:
            print(f"WARNING: trial_{n} trained on {sorted(pl)}, not the --tuning set {sorted(tuning)}")
    keys, trials, settings = summarize(acc, trial_cfgs, labels, tuning)
    short = {l: l.replace('flavell_2022', 'fl_').replace('flavell_', 'fl_').replace('samuel_', 'sa_')
             .replace('zimmer_', 'zi_').replace('leifer_stardist_strict', 'leifer') for l in labels}
    cols = ''.join(f"{short[t] + ('*' if t in tuning else ''):>11}" for t in labels)
    print(f"{name}: {len(acc)} cells, labelprop {a.labelprop}; * = a training (tuning) recording; "
          f"grid keys {keys}")

    # Reference: each recording's own per-recording network (trial 59), same recipe
    rref = ref_dir(a)
    diag = latest_records(os.path.join(rref, 'exp_results.jsonl' if a.labelprop == 'legacy'
                                       else f'exp_results_{a.labelprop}.jsonl'))
    ref_acc = {t: (own_cell(diag, os.path.basename(rref), a.ref_trial, t) or {}).get('accuracy') for t in labels}
    ref_tu = _mean([v for t, v in ref_acc.items() if t in tuning and v is not None])
    ref_ho = _mean([v for t, v in ref_acc.items() if t not in tuning and v is not None])

    head = f"{'':>28}{cols}{'tuning':>10}{'held-out':>10}"
    print("\nper trial")
    print(head)
    print(f"{'own network (t' + str(a.ref_trial) + ')':>28}" + ''.join(_fmt(ref_acc[t], 11) for t in labels)
          + _fmt(ref_tu, 10) + _fmt(ref_ho, 10))
    for r in trials:
        lab = f"trial_{r['trial']} " + ' '.join(f"{k}={v}" for k, v in r['setting']) + f" s{r['seed']}"
        print(f"{lab[:28]:>28}" + ''.join(_fmt(r['acc'][t], 11) for t in labels)
              + _fmt(r['tuning'], 10) + _fmt(r['held_out'], 10)
              + f"  ({r['n_tuning']}+{r['n_held_out']} cells)")
    print("\nper grid setting: mean over seeds per recording; tuning / held-out mean, and their SD over seeds")
    print(head + f"{'sd_tu':>8}{'sd_ho':>8}  seeds")
    for s in settings:
        lab = ' '.join(f"{k}={v}" for k, v in s['setting']) or 'all trials'
        print(f"{lab[:28]:>28}" + ''.join(_fmt(s['acc'][t], 11) for t in labels)
              + _fmt(s['tuning'], 10) + _fmt(s['held_out'], 10)
              + _fmt(s['tuning_sd'], 8) + _fmt(s['held_out_sd'], 8) + f"  {len(s['seeds'])}")
    print(f"\nreference (own per-recording networks): tuning {_fmt(ref_tu, 0)}, held-out {_fmt(ref_ho, 0)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--round_dir', required=True, help='the pooled round (its trial_N/ hold the networks)')
    ap.add_argument('--ref_round_dir', default=None,
                    help="round whose trial-<ref_trial> records define the targets and their eval settings "
                         "(default: round_shared_v1 next to --round_dir)")
    ap.add_argument('--ref_trial', type=int, default=59)
    ap.add_argument('--trials', type=int, nargs='+', default=None, help='only these trials of the pooled round')
    ap.add_argument('--targets', nargs='+', default=None, help='only these recordings scored')
    ap.add_argument('--tuning', nargs='+', default=list(TUNING),
                    help='the recordings the networks trained on (report: tuning vs held-out means)')
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
        _, out_jsonl, _ = paths(a)
        done = latest_records(out_jsonl)
        for tag, n, t, w, tgt, z, crop, _ in cells(a):
            acc = (done.get(tag) or {}).get('accuracy')
            print(f"{tag}  crop {crop}  z_scale {z}  {'done %.4f' % acc if acc is not None else 'todo'}  "
                  f"({tgt.get('source')}: {tgt.get('project') or tgt.get('gt')})")
        return
    if a.report:
        report(a)
        return
    sys.exit(1 if (dispatch(a) if a.slurm else run_shard(a, a.shard)) else 0)


if __name__ == '__main__':
    main()
