"""Tracking accuracy of label-propagation variants on cached sweep embeddings, scored as the sweep scores.

Generalises scratch/spectral_sync/compare_sync.py to every stage of the tracker (graph, seeds,
propagation, sync, post-processing; see lp_core.DEFAULTS) and to several RNG seeds, and loads each
case's embedding, project and ground truth ONCE (compare_sync calls evaluate_trained_checkpoint per
variant). Stages are memoised, so variants that differ only in a late stage reuse the propagation.

Scoring replicates eval_accuracy.run_eval after tracking (cluster_obj2dataframe, metadata join for
project sources, NWB GT arrays / every-detection mapping, align, pad, match_columns_to_gt,
calculate_accuracy) with the label's own objective settings (round_files + resolve_objective, as
compare_sync). The RNG seed plays the role of eval_accuracy --seed (TruncatedSVD, seed volumes, kNN
graph). With variant 'base' and seed 0 this is the sweep's own number under the current default sync.

Writes one JSON line per (case, variant, seed) to --out (resumable: done triples are skipped).
Besides accuracy/misses/mismatches it records the GT-free consistency metrics (utils_consistency),
a fast GT-optimal-matching accuracy and an error decomposition (unlabelled / swapped to another GT
neuron / on an unmatched 'fragment' label / GT cell without a tracked detection), stage timings and
graph statistics (same-volume edge fraction, hubness, edge purity).

    python sweep_labelprop.py --work /path/scratch --out batch1.jsonl --batch batch1 --seeds 0 1 \
        --case round_bayes_wk2_samuel_137:50:samuel_137 round_bayes_wk3_zimmer_1210:36:zimmer_1210
    # smoke test: first 120 volumes, CPU, two variants, baseline check against the tracker
    python sweep_labelprop.py --work /tmp/x --out smoke.jsonl --variants base sym --max_frames 120 \
        --device cpu --check_baseline --case round_bayes_wk2_samuel_137:63:samuel_137
"""
import argparse
import collections
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_core  # noqa: E402  (also puts the repo on sys.path)
from variants import BATCHES  # noqa: E402

C = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files'
os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')


class LRU(collections.OrderedDict):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def get_or(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return self[key], True
        val = fn()
        self[key] = val
        while len(self) > self.size:
            self.popitem(last=False)
        return val, False


def case_settings(rnd, trial, label):
    """(weights, emb path, train cfg, eval kwargs) exactly as compare_sync derives them."""
    import yaml
    from barlow_track.scripts import optimize_hyperparameters as oh
    from barlow_track.scripts.scratch.make_per_recording_rounds import round_files
    tag = f'{rnd}_trial{trial}_{label}'
    weights = os.path.join(C, rnd, f'trial_{trial}', 'resnet50.pth')
    emb = glob.glob(os.path.join(C, rnd, 'emb_cache', f'emb_*_trained_*{tag}.npz'))[0]
    with open(os.path.join(os.path.dirname(weights), 'train_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    tp = os.path.join(C, rnd, 'hyperparameter_search_template.yaml')
    tpl = yaml.safe_load(open(tp)) if os.path.exists(tp) else {}
    _, cell_tpl = round_files(rnd, label, cfg, {k: v for k, v in tpl.items()
                                                if k.startswith('objective_') and k not in
                                                ('objective_dataset', 'objective_source', 'objective_gt')})
    ekw = oh.resolve_objective(cell_tpl)[2]
    return tag, weights, emb, cfg, ekw


def load_embedding(emb, work, tag, max_frames=None):
    """Local copy of the cached embedding (the CIFS mount is flaky), optionally cut to the first volumes."""
    local = os.path.join(work, 'emb', os.path.basename(emb))
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if not os.path.exists(local):
        shutil.copy(emb, local + '.part')
        os.replace(local + '.part', local)
    with np.load(local, allow_pickle=True) as d:
        X = d['X']
        t2l = d['time_to_lin'].item()
        l2t = d['lin_to_t_seg'].item()
        n_frames = int(d['n_frames']) if 'n_frames' in d else len(t2l)
    if max_frames:
        keep_t = {t: v for t, v in t2l.items() if t < max_frames}
        idx = np.sort(np.concatenate([np.asarray(list(v), dtype=np.int64) for v in keep_t.values()]))
        if not np.array_equal(idx, np.arange(len(idx))):
            raise ValueError("linear indices are not time-ordered; cannot truncate by slicing")
        X = X[:len(idx)]
        t2l = keep_t
        l2t = {i: l2t[i] for i in range(len(idx))}
        n_frames = min(n_frames, max_frames)
    return X, {t: list(v) for t, v in t2l.items()}, l2t, n_frames


class Scorer:
    """run_eval's post-tracking scoring, with the GT loaded once."""

    def __init__(self, ekw, cfg, X_shape, t2l, l2t, n_frames, work, tag):
        from wbfm.utils.projects.finished_project_data import ProjectData
        from barlow_track.scripts import eval_accuracy as ea
        from barlow_track.utils.utils_ground_truth import align_gt_pred_time_index
        import pandas as pd
        self.source = ekw.get('source', 'project')
        self.gt_matching = ekw.get('gt_matching', 'nandist')
        gt_path = ekw.get('gt') or cfg['project_path']
        self.n_frames = n_frames
        t0 = time.time()
        if self.source == 'nwb':
            src = ProjectData.load_final_project_data(gt_path, allow_hybrid_loading=True, verbose=0)
            arrays = ea._load_nwb_arrays(src)
            nwb_all = ekw.get('nwb_objects', 'gt') == 'all'
            df_gt = arrays['df_gt']
            try:
                _df_fin, _ = src.get_final_tracks_only_finished_neurons()
            except Exception:
                _df_fin = None
            if _df_fin is not None and not _df_fin.empty:
                df_gt = ea._ensure_ind_level(_df_fin)
            if nwb_all:
                cache = os.path.join(work, f'gtlabel_{tag}_{ekw.get("nwb_gt_mapping", "centroid")}_{n_frames}.npy')
                if os.path.exists(cache):
                    gt_label = np.load(cache)
                else:
                    gt_label = ea._nwb_all_objects(src, arrays, list(range(n_frames)),
                                                   gt_mapping=ekw.get('nwb_gt_mapping', 'centroid'))[1]
                    np.save(cache, gt_label)
                df_gt = df_gt.copy()
                if len(df_gt) != len(gt_label):
                    # truncated smoke runs: map the frames we have, NaN the rest
                    full = np.full((len(df_gt), gt_label.shape[1]), np.nan)
                    full[:len(gt_label)] = gt_label[:len(df_gt)]
                    gt_label = full
                for n in df_gt.columns.get_level_values(0).unique():
                    df_gt[(n, 'raw_neuron_ind_in_list')] = gt_label[:, arrays['neurons'].index(n)]
            self.match_col = 'raw_neuron_ind_in_list'
            self.seg_meta = None
        else:
            src = ProjectData.load_final_project_data(cfg['project_path'], allow_hybrid_loading=True, verbose=0)
            self.seg_meta = src.segmentation_metadata
            df_gt = ea._load_project_gt(gt_path)
            self.match_col = 'raw_segmentation_id'
        self.df_gt = df_gt
        self.load_s = round(time.time() - t0, 1)
        # object-level GT identity for the fast diagnostics (GT-optimal matching, error decomposition)
        dummy = pd.DataFrame(index=pd.RangeIndex(n_frames))
        g, _ = align_gt_pred_time_index(df_gt, dummy)
        col_gt = g.loc[:, (slice(None), self.match_col)].droplevel(1, axis=1)
        self.total_gt_fast = int(np.isfinite(col_gt.to_numpy(dtype=float)).sum())
        lookup = {}
        arr = col_gt.to_numpy(dtype=float)
        for ti, t in enumerate(col_gt.index):
            for j in np.flatnonzero(np.isfinite(arr[ti])):
                lookup[(int(t), int(arr[ti, j]))] = j
        n = X_shape[0]
        which = 2 if self.source != 'nwb' else 1
        self.gt_obj = np.array([lookup.get((int(l2t[i][0]), int(l2t[i][which])), -1) for i in range(n)])
        self.n_gt_neurons = arr.shape[1]
        print(f"[{tag}] GT loaded in {self.load_s}s: {self.n_gt_neurons} neurons, {self.total_gt_fast} cells, "
              f"{int((self.gt_obj >= 0).sum())} of {n} objects are GT", flush=True)

    def score(self, tracker, labels, conf):
        """The official accuracy (as run_eval)."""
        from barlow_track.utils.utils_ground_truth import (pad_with_nan_rows, align_gt_pred_time_index,
                                                           match_columns_to_gt, calculate_accuracy)
        df_pred = tracker.cluster_obj2dataframe({'labels': labels, 'probabilities': conf}, start_volume=0,
                                                n_vols=tracker.num_frames, labels_are_in_feature_order=True)
        if self.seg_meta is not None:
            from wbfm.utils.projects.utils_redo_steps import add_metadata_to_df_raw_ind
            df_pred = add_metadata_to_df_raw_ind(df_pred, self.seg_meta)
        df_gt, df_pred = align_gt_pred_time_index(self.df_gt, df_pred)
        L = max(len(df_gt), len(df_pred))
        df_pred = pad_with_nan_rows(df_pred, L)
        df_gt = pad_with_nan_rows(df_gt, L)
        r = match_columns_to_gt(df_gt, df_pred, self.match_col, gt_matching=self.gt_matching)
        if 'unmatched_neuron' in r.columns.get_level_values(0):
            r = r.drop(columns='unmatched_neuron')
        cg = df_gt.loc[:, (slice(None), self.match_col)].droplevel(1, axis=1)
        cp = r.loc[:, (slice(None), self.match_col)].droplevel(1, axis=1)
        s = calculate_accuracy(cg, cp)
        return dict(accuracy=float(s['accuracy']), misses=int(s['total_misses']),
                    mismatches=int(s['total_mismatches']), total=int(s['total_ground_truth']))

    def fast(self, labels):
        """GT-optimal label->neuron matching on object identities, and where the errors are."""
        from scipy.optimize import linear_sum_assignment
        g = self.gt_obj
        lab = np.asarray(labels)
        on = (g >= 0) & (lab >= 1)
        ul, li = np.unique(lab[on], return_inverse=True)
        Cm = np.zeros((len(ul), self.n_gt_neurons))
        np.add.at(Cm, (li, g[on]), 1)
        r, c = linear_sum_assignment(-Cm)
        correct = int(Cm[r, c].sum())
        lab2gt = dict(zip(ul[r], c))
        n_gt_obj = int((g >= 0).sum())
        unl = int(((g >= 0) & (lab < 1)).sum())
        mapped = np.array([lab2gt.get(x, -2) for x in lab[on]])
        swapped = int(((mapped >= 0) & (mapped != g[on])).sum())
        frag = int((mapped == -2).sum())
        T = self.total_gt_fast
        # perfect track merging: every label goes to its majority GT neuron (several labels may share one);
        # the gain over acc_fast bounds what fragment merging could recover
        merge = float(Cm.max(1).sum()) if len(ul) else 0.0
        n_frag_labels = int(len(ul) - len(r))
        return dict(acc_fast=correct / T, acc_merge_bound=merge / T, n_labels=int(len(ul)),
                    n_unmatched_labels=n_frag_labels, err_unlabelled=unl / T, err_swapped=swapped / T, err_fragment=frag / T,
                    err_no_detection=(T - n_gt_obj) / T)


def run_case(case, args, variants):
    import torch
    from barlow_track.utils.utils_tracking import WormClusterTracker
    from barlow_track.utils.utils_consistency import labelprop_consistency
    rnd, trial, label = case.split(':')[:3]
    tag, weights, emb, cfg, ekw = case_settings(rnd, int(trial), label)
    if args.max_frames:
        tag += f'_first{args.max_frames}'
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    x = json.loads(line)
                    done.add((x['case'], x['variant'], x['seed']))
                except ValueError:
                    pass
    todo = [(s, n) for s in args.seeds for n in variants if (tag, n, s) not in done]
    if not todo:
        print(f"[{tag}] all done", flush=True)
        return
    t0 = time.time()
    X, t2l, l2t, n_frames = load_embedding(emb, args.work, tag, args.max_frames)
    print(f"[{tag}] embedding {X.shape}, {len(t2l)} volumes ({time.time() - t0:.0f}s)", flush=True)
    if ekw.get('svd', 'scaled') != 'scaled':
        print(f"[{tag}] NOTE objective svd={ekw['svd']}; variants' svd_mode decides here", flush=True)
    scorer = Scorer(ekw, cfg, X.shape, t2l, l2t, n_frames, args.work, tag)
    lin_t = np.array([l2t[i][0] for i in range(X.shape[0])])
    device = None if args.device == 'cpu' else torch.device(args.device)
    emb_memo, graph_memo, prop_memo, sync_memo = LRU(2), LRU(2), LRU(args.prop_cache), LRU(2)

    # order: variants sharing early stages run consecutively (memo hits)
    def order(item):
        s, n = item
        v = lp_core.full_variant(variants[n])
        return (s, repr(lp_core.stage_key(v, 'graph', s)), repr(lp_core.stage_key(v, 'prop', s)), n)
    for seed, name in sorted(todo, key=order):
        v = lp_core.full_variant(variants[name])
        times = {}
        tt = time.time()
        Xs, _ = emb_memo.get_or(lp_core.stage_key(v, 'embed', seed), lambda: lp_core.embed(X, v, seed))
        times['embed_s'] = round(time.time() - tt, 1)
        tt = time.time()
        graph, _ = graph_memo.get_or(lp_core.stage_key(v, 'graph', seed),
                                     lambda: lp_core.build_graph(Xs, t2l, v, seed, device=device,
                                                                          gt_obj=scorer.gt_obj))
        times['graph_s'] = round(time.time() - tt, 1)
        tt = time.time()
        tracker = WormClusterTracker(Xs, t2l, linear_ind_to_t_and_seg_id=l2t, seed=seed, verbose=0)
        seed_times = lp_core.select_seeds(t2l, v, seed, tracker.num_frames)

        def do_prop():
            with torch.no_grad():
                out = lp_core.propagate(graph, t2l, Xs.shape[0], seed_times, v, device=device)
            if device is not None and device.type == 'cuda':
                torch.cuda.synchronize()
            return out
        (labs, probs), hit = prop_memo.get_or(lp_core.stage_key(v, 'prop', seed), do_prop)
        times['prop_s'] = round(time.time() - tt, 1)
        tt = time.time()
        syn, _ = sync_memo.get_or(lp_core.stage_key(v, 'sync', seed),
                                  lambda: lp_core.sync(labs, probs, t2l, v))
        times['sync_s'] = round(time.time() - tt, 1)
        tt = time.time()
        labels, conf = lp_core.finalize(syn, t2l, v, Xs=Xs, graph=graph)
        times['post_s'] = round(time.time() - tt, 1)
        tt = time.time()
        rec = scorer.score(tracker, labels, conf)
        times['score_s'] = round(time.time() - tt, 1)
        rec.update(scorer.fast(labels))
        cons = labelprop_consistency(labs, syn['perms'], labels, conf, lin_t, n_frames)
        if args.check_baseline and name == 'base':
            tr = WormClusterTracker(Xs, t2l, linear_ind_to_t_and_seg_id=l2t, seed=seed, verbose=0)
            tr.track_using_label_propagation_clusterer(num_seeds=v['num_seeds'], device=device)
            ref = tr.labelprop_runs['final_labels']
            rec['baseline_label_agreement'] = float(np.mean(ref == labels))
            ref_rec = scorer.score(tr, ref, tr.labelprop_runs['confidence'])
            rec['baseline_tracker_accuracy'] = ref_rec['accuracy']
            print(f"  baseline check: labels equal {rec['baseline_label_agreement']:.6f}, tracker accuracy "
                  f"{ref_rec['accuracy']:.4f} vs harness {rec['accuracy']:.4f}", flush=True)
        out = dict(case=tag, label=label, variant=name, seed=seed, params={k: v[k] for k in variants[name]},
                   **rec, consistency={k: cons[k] for k in ('agree_mean', 'agree_p10', 'conf_mean',
                                                            'assigned_frac', 'dup_frac')},
                   graph_info=graph['info'], seed_times=[int(t) for t in seed_times][:5],
                   n_objects=int(X.shape[0]), n_frames=n_frames, prop_cached=bool(hit), device=str(args.device),
                   **times)
        with open(args.out, 'a') as f:
            f.write(json.dumps(out) + '\n')
        print(f"RESULT {tag} {name} seed{seed}: acc {rec['accuracy']:.4f} fast {rec['acc_fast']:.4f} "
              f"agree {cons['agree_mean']:.3f} | {times}", flush=True)
        if device is not None and device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', nargs='+', required=True, help='round:trial:label (as compare_sync)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True, help='scratch dir: local embedding copies, GT-label caches')
    ap.add_argument('--batch', default=None, help=f'variant batch from variants.py: {sorted(BATCHES)}')
    ap.add_argument('--variants', nargs='*', default=None, help='variant names (default: the whole batch)')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max_frames', type=int, default=None, help='first volumes only (smoke tests)')
    ap.add_argument('--prop_cache', type=int, default=3, help='propagation results kept in memory')
    ap.add_argument('--check_baseline', action='store_true',
                    help="for 'base': also run the library tracker and record label/accuracy agreement")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    pool = {}
    for b in BATCHES.values():
        pool.update(b)
    if args.batch:
        variants = dict(BATCHES[args.batch])
        if args.variants:
            variants = {n: variants[n] for n in args.variants}
    else:
        variants = {n: pool[n] for n in (args.variants or ['base'])}
    for n, v in variants.items():
        lp_core.full_variant(v)  # fail early on a typo
    for case in args.case:
        run_case(case, args, variants)


if __name__ == '__main__':
    main()
