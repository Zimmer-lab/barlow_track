"""Factorial check: which paper-vs-benchmark difference moves tracking accuracy.

Re-tracks a SAVED embedding with label propagation, varying one factor at a time,
and scores every run with the current (corrected) paper recipe, same as
eval_accuracy.run_eval: align GT/pred time, pad, rename_columns_using_matching on
raw_segmentation_id, calculate_accuracy.

What the paper actually ran (flavell untrained, verified against its saved files):
  - features: worm_tracker_barlow.pickle X_svd = left singular vectors u[:, :50] of the
    raw 64-d embedding (unit-norm columns, NO S scaling; matches to ~1e-7). The first
    singular value is ~2140 vs ~11 for the second, so U*S is dominated by one direction.
  - num_seeds: default min(100, int(0.5*num_frames)) = 100; tracker seed None (unseeded).
  - code: v0.1.0 label propagation + spectral relabeling (relabeling changed since).

Factors:
  --code      head: current repo code; paper: v0.1.0 tracking code, run in a subprocess
              (extracted with `git archive v0.1.0` to --paper_code_root, outside the
              repo); scoring always uses the current code.
  --svd       saved: the paper's own X_svd (--source paper only);
              paper: u[:, :50] recomputed (same branch as the paper above 10000 rows);
              truncated: sklearn TruncatedSVD fit_transform (U*S), as eval_accuracy does.
  --num_seeds integer, or 'paper' = min(100, int(0.5 * num_frames)).
  --source    paper: the paper's saved embedding.zarr + pickles (segmentation
              detections, untrained paper weights);
              npz: an eval_accuracy cache (X, time_to_lin, lin_to_t_seg), e.g. a
              GT-centered NWB cache. Weights are whatever produced the cache.

A 'paper_stored' control scores the paper's saved df_barlow_tracks.h5 directly
(flavell: 0.6961, matching docs/accuracy_evaluation.md).

Usage (wbfm env, CPU; run from the repo root):
  python barlow_track/scripts/scratch/paper_factorial_check.py --max_frames 40 --num_seeds 5 --svd saved   # smoke test
  python barlow_track/scripts/scratch/paper_factorial_check.py --code paper --svd saved --num_seeds paper

Each labelprop run on the full flavell video (~286k rows) is roughly 30-120 min on CPU.
Records are appended as JSON lines to --out (default: ~/.cache/barlow_scratch/).
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

# Numba/matplotlib caches go outside the repo (conftest.py redirects them the same way for tests).
CACHE_ROOT = os.path.join(os.path.expanduser('~'), '.cache', 'barlow_scratch')
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(CACHE_ROOT, 'numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(CACHE_ROOT, 'mpl'))

import numpy as np
import pandas as pd

PAPER_DIR = ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects/'
             'flavell/untrained/2025_07_01trial_0/3-tracking/barlow_tracker')
FLAVELL_NWB = ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/'
               'flavell_data/images_for_charlie/flavell_data.nwb')
DEFAULT_OUT = os.path.join(CACHE_ROOT, 'paper_factorial_results.jsonl')
PAPER_TAG = 'v0.1.0'
DEFAULT_PAPER_CODE_ROOT = os.path.join(CACHE_ROOT, f'code_{PAPER_TAG}')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_paper_source(paper_dir):
    """The paper's saved embedding plus its row -> (t, raw, seg) and time -> rows maps."""
    import zarr
    X = np.asarray(zarr.open(os.path.join(paper_dir, 'embedding.zarr'), 'r'))
    with open(os.path.join(paper_dir, 'time_index_to_linear_feature_indices.pickle'), 'rb') as f:
        time_to_lin = pickle.load(f)
    with open(os.path.join(paper_dir, 'linear_ind_to_t_and_seg_id.pickle'), 'rb') as f:
        lin_to_t_seg = pickle.load(f)
    return X, time_to_lin, lin_to_t_seg


def load_paper_saved_svd(paper_dir):
    """X_svd of the tracker the paper's tracking step loaded (worm_tracker_barlow.pickle)."""
    with open(os.path.join(paper_dir, 'worm_tracker_barlow.pickle'), 'rb') as f:
        return np.asarray(pickle.load(f).X_svd)


def load_npz_source(path):
    """An eval_accuracy embedding cache (same layout as the paper's pickles)."""
    d = np.load(path, allow_pickle=True)
    return d['X'], d['time_to_lin'].item(), d['lin_to_t_seg'].item()


def restrict_to_frames(time_to_lin, lin_to_t_seg, max_frames):
    """Rows of frames < max_frames. Rows are time-ordered in both caches, so they form a prefix; returns its length."""
    if max_frames is None:
        return len(lin_to_t_seg), time_to_lin, lin_to_t_seg
    n_keep = sum(len(v) for t, v in time_to_lin.items() if t < max_frames)
    assert all(lin_to_t_seg[i][0] < max_frames for i in range(n_keep)), 'rows are not time-ordered'
    time_to_lin = {t: v for t, v in time_to_lin.items() if t < max_frames}
    lin_to_t_seg = {i: lin_to_t_seg[i] for i in range(n_keep)}
    return n_keep, time_to_lin, lin_to_t_seg


def svd_features(X, svd, seed, X_saved=None, k=50):
    """Tracking features for one SVD factor level."""
    if svd == 'saved':
        if X_saved is None:
            raise SystemExit("--svd saved needs --source paper (the paper's saved tracker)")
        return X_saved
    if svd == 'paper':
        if X.shape[0] > 10000:
            # Same as da.linalg.svd(...)[0][:, :k] on 10000-row chunks, up to sign per column (distances are sign-invariant).
            u, _, _ = np.linalg.svd(X, full_matrices=False)
            return u[:, :k]
        from sklearn.decomposition import TruncatedSVD
        return TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)
    if svd == 'truncated':
        from sklearn.decomposition import TruncatedSVD
        return TruncatedSVD(n_components=min(k, X.shape[1] - 1), random_state=seed).fit_transform(X)
    raise ValueError(svd)


def paper_default_num_seeds(n_frames):
    return min(100, int(0.5 * n_frames))


def pred_to_segmentation_ids(df_pred, lin_to_t_seg):
    """Map the tracker's raw_neuron_ind_in_list cells to raw_segmentation_id via the saved (t, raw, seg) rows."""
    lookup = {(t, raw): seg for (t, raw, seg) in lin_to_t_seg.values()}
    names = list(df_pred.columns.get_level_values(0).unique())
    out = {}
    times = list(df_pred.index)
    for name in names:
        raw_col = df_pred[(name, 'raw_neuron_ind_in_list')].values
        segs = np.full(len(times), np.nan)
        for i, (t, raw) in enumerate(zip(times, raw_col)):
            if pd.notna(raw):
                segs[i] = lookup.get((int(t), int(raw)), np.nan)
        out[(name, 'raw_segmentation_id')] = segs
    return pd.DataFrame(out, index=df_pred.index)


def score(df_gt, df_pred_seg):
    """Paper recipe (see eval_accuracy.run_eval, the non-NWB-ind branch)."""
    from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
    from barlow_track.utils.utils_ground_truth import (
        calculate_accuracy, pad_with_nan_rows, align_gt_pred_time_index)
    match_col = 'raw_segmentation_id'
    df_gt, df_pred = align_gt_pred_time_index(df_gt, df_pred_seg)
    max_len = max(len(df_gt), len(df_pred))
    df_pred = pad_with_nan_rows(df_pred, max_len)
    df_gt = pad_with_nan_rows(df_gt, max_len)
    df_pred_r, _, _, _ = rename_columns_using_matching(df_gt, df_pred, column=match_col, try_to_fix_inf=True)
    if 'unmatched_neuron' in df_pred_r.columns.get_level_values(0):
        df_pred_r = df_pred_r.drop(columns='unmatched_neuron')
    col_gt = df_gt.loc[:, (slice(None), match_col)].droplevel(1, axis=1)
    col_pr = df_pred_r.loc[:, (slice(None), match_col)].droplevel(1, axis=1)
    stats = calculate_accuracy(col_gt, col_pr)
    return dict(accuracy=float(stats['accuracy']), misses=int(stats['total_misses']),
                mismatches=int(stats['total_mismatches']), total=int(stats['total_ground_truth']))


def load_gt(nwb_path):
    """GT as eval_accuracy scores it: finished neurons if available, else all final_tracks.

    Not truncated here: score() aligns GT and prediction on the time index.
    """
    from barlow_track.scripts.eval_accuracy import _load_project_gt
    return _load_project_gt(nwb_path)


def track_head(X_svd, time_to_lin, lin_to_t_seg, num_seeds, seed):
    from barlow_track.utils.utils_seeding import seed_all
    from barlow_track.utils.utils_tracking import WormClusterTracker
    seed_all(seed)
    tracker = WormClusterTracker(X_svd, dict(time_to_lin), linear_ind_to_t_and_seg_id=lin_to_t_seg, seed=seed)
    return tracker.track_using_label_propagation_clusterer(num_seeds=num_seeds, device=None, seed=seed)


def ensure_paper_code(root):
    """Extract the paper tag's barlow_track package outside the repo (read-only use of git)."""
    if os.path.isfile(os.path.join(root, 'barlow_track', 'utils', 'utils_tracking.py')):
        return root
    os.makedirs(root, exist_ok=True)
    archive = subprocess.Popen(['git', '-C', REPO_ROOT, 'archive', PAPER_TAG, 'barlow_track'], stdout=subprocess.PIPE)
    subprocess.check_call(['tar', '-x', '-C', root], stdin=archive.stdout)
    if archive.wait() != 0:
        raise RuntimeError(f'git archive {PAPER_TAG} failed')
    return root


def track_paper_code(X_svd, time_to_lin, lin_to_t_seg, num_seeds, seed, paper_code_root):
    """Track with the paper tag's code in a subprocess (it cannot share a process with the current package)."""
    root = ensure_paper_code(paper_code_root)
    with tempfile.TemporaryDirectory(dir=CACHE_ROOT) as tmp:
        f_in, f_out = os.path.join(tmp, 'in.pickle'), os.path.join(tmp, 'out.pickle')
        with open(f_in, 'wb') as f:
            pickle.dump(dict(X_svd=X_svd, time_to_lin=time_to_lin, lin_to_t_seg=lin_to_t_seg,
                             num_seeds=num_seeds, seed=seed), f, protocol=4)
        env = dict(os.environ, PYTHONPATH=root)
        # Run outside the repo so the current package is not found via the working directory.
        subprocess.check_call([sys.executable, '-u', os.path.abspath(__file__), '--_worker', f_in, f_out],
                              env=env, cwd=tmp)
        with open(f_out, 'rb') as f:
            return pickle.load(f)


def paper_code_worker(f_in, f_out):
    """Subprocess body: v0.1.0 has no seed arguments, so seed the global streams it reads
    (random.shuffle of seed times, NNDescent's default numpy RNG, torch)."""
    import random
    import torch
    import barlow_track
    from barlow_track.utils.utils_tracking import WormClusterTracker
    print(f'[paper-code worker] barlow_track from {barlow_track.__file__}', flush=True)
    with open(f_in, 'rb') as f:
        d = pickle.load(f)
    random.seed(d['seed'])
    np.random.seed(d['seed'])
    torch.manual_seed(d['seed'])
    # Same options as v0.1.0 embed_using_barlow_from_config passed (opt_db/opt_umap came from the
    # tracking config, default {}; v0.1.0 __post_init__ indexes opt_db, so None crashes).
    tracker = WormClusterTracker(d['X_svd'], dict(d['time_to_lin']), linear_ind_to_t_and_seg_id=d['lin_to_t_seg'],
                                 svd_components=50, cluster_directly_on_svd_space=True,
                                 n_clusters_per_window=3, n_volumes_per_window=120, opt_db={}, opt_umap={})
    df_pred = tracker.track_using_label_propagation_clusterer(num_seeds=d['num_seeds'])
    with open(f_out, 'wb') as f:
        pickle.dump(df_pred, f, protocol=4)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--_worker':
        import warnings
        warnings.filterwarnings('ignore')
        paper_code_worker(sys.argv[2], sys.argv[3])
        return None

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', choices=['paper', 'npz'], default='paper')
    ap.add_argument('--npz', default=None, help='embedding cache for --source npz')
    ap.add_argument('--paper_dir', default=PAPER_DIR)
    ap.add_argument('--nwb', default=FLAVELL_NWB, help='GT NWB (flavell)')
    ap.add_argument('--code', nargs='+', choices=['head', 'paper'], default=['head'])
    ap.add_argument('--paper_code_root', default=DEFAULT_PAPER_CODE_ROOT)
    ap.add_argument('--svd', nargs='+', choices=['saved', 'paper', 'truncated'], default=['saved', 'truncated'])
    ap.add_argument('--num_seeds', nargs='+', default=['25', 'paper'],
                    help="integers and/or 'paper' (= min(100, int(0.5*num_frames)))")
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--no_stored_control', action='store_true')
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(CACHE_ROOT, exist_ok=True)

    import warnings
    warnings.filterwarnings('ignore')

    X_saved = None
    if args.source == 'paper':
        X, time_to_lin, lin_to_t_seg = load_paper_source(args.paper_dir)
        if 'saved' in args.svd:
            X_saved = load_paper_saved_svd(args.paper_dir)
            assert X_saved.shape[0] == X.shape[0], 'saved X_svd and embedding.zarr disagree on rows'
        source_label = f'paper:{args.paper_dir}'
    else:
        if not args.npz:
            raise SystemExit('--source npz needs --npz')
        X, time_to_lin, lin_to_t_seg = load_npz_source(args.npz)
        source_label = f'npz:{args.npz}'
    n_keep, time_to_lin, lin_to_t_seg = restrict_to_frames(time_to_lin, lin_to_t_seg, args.max_frames)
    X = X[:n_keep]
    if X_saved is not None:
        X_saved = X_saved[:n_keep]
    n_frames = len(time_to_lin)
    print(f'[{source_label}] X {X.shape}, frames {n_frames}', flush=True)

    df_gt = load_gt(args.nwb)

    records = []
    if not args.no_stored_control and args.source == 'paper':
        stored = pd.read_hdf(os.path.join(args.paper_dir, 'df_barlow_tracks.h5'))
        if args.max_frames is not None:
            stored = stored.iloc[:args.max_frames]
        rec = dict(control='paper_stored', source=source_label, max_frames=args.max_frames, **score(df_gt, stored))
        records.append(rec)
        print(json.dumps(rec), flush=True)
        with open(args.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    for code in args.code:
        for svd in args.svd:
            for ns_arg in args.num_seeds:
                num_seeds = paper_default_num_seeds(n_frames) if ns_arg == 'paper' else int(ns_arg)
                for seed in args.seeds:
                    t0 = time.time()
                    X_svd = svd_features(X, svd, seed, X_saved=X_saved)
                    if code == 'head':
                        df_pred = track_head(X_svd, time_to_lin, lin_to_t_seg, num_seeds, seed)
                    else:
                        df_pred = track_paper_code(X_svd, time_to_lin, lin_to_t_seg, num_seeds, seed,
                                                   args.paper_code_root)
                    df_pred_seg = pred_to_segmentation_ids(df_pred, lin_to_t_seg)
                    rec = dict(control=None, source=source_label, code=code, svd=svd, num_seeds=num_seeds,
                               num_seeds_arg=ns_arg, seed=seed, n_frames=n_frames, n_rows=int(X.shape[0]),
                               max_frames=args.max_frames, minutes=(time.time() - t0) / 60,
                               **score(df_gt, df_pred_seg))
                    records.append(rec)
                    print(json.dumps(rec), flush=True)
                    with open(args.out, 'a') as f:
                        f.write(json.dumps(rec) + '\n')
    return records


if __name__ == '__main__':
    main()
