"""Tracking accuracy of spectral-sync variants on cached sweep embeddings, scored as the sweep scores.

For each (round, trial, label): copy the trial's cached embedding into a scratch emb_dir (so the eval
reuses it and never writes into the round), then run eval_accuracy's evaluate_trained_checkpoint with
the label's own objective settings (make_per_recording_rounds.round_files + resolve_objective) once
per spectral_sync_from_topk variant. Label propagation runs once per case (memoised); only the
synchronisation differs between variants. Writes one JSON line per (case, variant) to --out.

    python barlow_track/scripts/scratch/spectral_sync/compare_sync.py --out sync.jsonl \
        --case round_bayes_wk2_samuel_137:63:samuel_137 ...
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

C = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files'
VARIANTS = {
    'independent': dict(rounding='independent', refine_iters=0),
    'refine': dict(rounding='independent', refine_iters=10),
    'cpqr_refine': dict(rounding='cpqr', refine_iters=10),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', nargs='+', required=True,
                    help='round:trial:label, or round:trial:label:weights:emb_npz for an embedding elsewhere')
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True, help='scratch directory (embedding copies, eval records)')
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS))
    args = ap.parse_args()

    import yaml
    import barlow_track.utils.utils_tracking as ut
    import barlow_track.utils.utils_spectral_relabeling as ss
    from barlow_track.scripts import optimize_hyperparameters as oh
    from barlow_track.scripts.scratch.make_per_recording_rounds import round_files

    memo = {}
    orig_msp = ut.multi_seed_propagation

    def memo_msp(X, *a, **k):
        key = (X.shape, float(np.asarray(X, dtype=np.float64).sum()))
        if key not in memo:
            memo.clear()
            memo[key] = orig_msp(X, *a, **k)
        return memo[key]
    ut.multi_seed_propagation = memo_msp
    orig_sync = ss.spectral_sync_from_topk

    for case in args.case:
        parts = case.split(':')
        rnd, trial, label = parts[0], int(parts[1]), parts[2]
        tag = f'{rnd}_trial{trial}_{label}'
        if len(parts) > 3:
            weights, emb = parts[3], parts[4]
        else:
            weights = os.path.join(C, rnd, f'trial_{trial}', 'resnet50.pth')
            emb = glob.glob(os.path.join(C, rnd, 'emb_cache', f'emb_*_trained_*{tag}.npz'))[0]
        emb_dir = os.path.join(args.work, 'emb', tag)
        os.makedirs(emb_dir, exist_ok=True)
        if not os.path.exists(os.path.join(emb_dir, os.path.basename(emb))):
            shutil.copy(emb, emb_dir)
        with open(os.path.join(os.path.dirname(weights), 'train_config.yaml')) as f:
            cfg = yaml.safe_load(f)
        tpl = yaml.safe_load(open(os.path.join(C, rnd, 'hyperparameter_search_template.yaml'))) \
            if os.path.exists(os.path.join(C, rnd, 'hyperparameter_search_template.yaml')) else {}
        _, cell_tpl = round_files(rnd, label, cfg, {k: v for k, v in tpl.items()
                                                    if k.startswith('objective_') and k not in
                                                    ('objective_dataset', 'objective_source', 'objective_gt')})
        ekw = oh.resolve_objective(cell_tpl)[2]
        ekw.update(track_device='cuda')
        gt = ekw.pop('gt', None) or cfg['project_path']
        ekw.pop('snapshots', None)
        for name in args.variants:
            kw = VARIANTS[name]
            ss.spectral_sync_from_topk = lambda *a, _kw=kw, **k: orig_sync(*a, **{**k, **_kw})
            ut.spectral_sync_from_topk = ss.spectral_sync_from_topk
            t0 = time.time()
            rec = oh.evaluate_trained_checkpoint(
                weights=weights, project=cfg['project_path'], gt=gt, tag=tag,
                results_jsonl=os.path.join(args.work, 'eval_records.jsonl'), emb_dir=emb_dir,
                extra_record=dict(sync_variant=name), **ekw)
            out = dict(case=tag, label=label, variant=name, accuracy=rec['accuracy'],
                       misses=rec['misses'], mismatches=rec['mismatches'], total=rec['total'],
                       consistency=rec.get('consistency'), seconds=round(time.time() - t0, 1))
            with open(args.out, 'a') as f:
                f.write(json.dumps(out) + '\n')
            print(f"RESULT {tag} {name}: accuracy {rec['accuracy']:.4f} agree "
                  f"{(rec.get('consistency') or {}).get('agree_mean')}", flush=True)
        ss.spectral_sync_from_topk = orig_sync
        ut.spectral_sync_from_topk = orig_sync


if __name__ == '__main__':
    main()
