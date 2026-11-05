"""Time each phase of the label-propagation eval on a cached embedding, on CPU (several thread counts) and GPU.

Phases, as eval_accuracy --cluster labelprop runs them (utils_tracking.track_using_label_propagation_clusterer):
  svd         TruncatedSVD to 50 dims (eval_accuracy)
  nndescent   pynndescent kNN graph (k = 20)
  edge_list   the Python double loop turning the neighbour array into edge_index
  propagate   per seed: 100 layers of sparse propagation + top-k; reported per seed
  spectral    spectral_sync_from_topk over all seeds' top-k labelings
For each phase: wall time and process CPU time, so cpu / wall ~ the cores it actually kept busy.

    python .../labelprop_phases.py --emb .../emb_...npz --threads 1 16 --seeds 4 [--cuda]
"""
import argparse
import json
import os
import time

import numpy as np


def timed(fn):
    w0, c0 = time.perf_counter(), time.process_time()
    out = fn()
    w, c = time.perf_counter() - w0, time.process_time() - c0
    return out, dict(wall_s=round(w, 2), cpu_s=round(c, 2), cores=round(c / max(w, 1e-9), 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--emb', required=True)
    ap.add_argument('--threads', type=int, nargs='+', default=[1, 16])
    ap.add_argument('--seeds', type=int, default=4, help='seed times to propagate per setting (eval uses 25)')
    ap.add_argument('--cuda', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    import torch
    import numba
    from sklearn.decomposition import TruncatedSVD
    from pynndescent import NNDescent
    from barlow_track.utils.utils_label_propagation import (normalized_adjacency, make_seed_labels_no_dict,
                                                            run_label_propagation)
    from barlow_track.utils.utils_spectral_relabeling import spectral_sync_from_topk
    from barlow_track.utils.utils_tracking import select_seed_times

    with np.load(args.emb, allow_pickle=True) as d:
        X = d['X']
        t2l = d['time_to_lin'].item()
    res = dict(emb=os.path.abspath(args.emb), n_objects=int(X.shape[0]), n_frames=len(t2l),
               cpu_count=os.cpu_count(), settings={})
    seeds = select_seed_times(t2l, num_seeds=args.seeds, seed=0)
    settings = [('cpu', n) for n in args.threads] + ([('cuda', max(args.threads))] if args.cuda else [])
    for dev, n in settings:
        torch.set_num_threads(n)
        numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))
        os.environ['OMP_NUM_THREADS'] = str(n)
        r = {}
        Xs, r['svd'] = timed(lambda: TruncatedSVD(n_components=50, random_state=0).fit_transform(X))
        idx, r['nndescent'] = timed(lambda: NNDescent(Xs, n_neighbors=20, metric='euclidean',
                                                      random_state=0).neighbor_graph[0])

        def edges():
            rows, cols = [], []
            for i in range(len(idx)):
                for j in idx[i]:
                    rows.append(i)
                    cols.append(j)
            return torch.tensor([rows, cols], dtype=torch.long)
        ei, r['edge_list'] = timed(edges)
        device = torch.device(dev)

        def prep():
            e = ei.to(device)
            return normalized_adjacency(e)
        (e, norm), r['normalize'] = timed(prep)
        labs, probs, per = [], [], []
        for t in seeds:
            y = make_seed_labels_no_dict(t2l[t], num_timepoints=X.shape[0])

            def one():
                p, q = run_label_propagation(e, y, edge_norm=norm, device=device, return_top_k=2, num_layers=100,
                                             softmax=False, tau=0.02)
                if dev == 'cuda':
                    torch.cuda.synchronize()
                return p.cpu().numpy(), q.cpu().numpy()
            (p, q), tt = timed(one)
            labs.append(p)
            probs.append(q)
            per.append(tt)
        r['propagate_per_seed'] = dict(wall_s=round(float(np.mean([x['wall_s'] for x in per])), 2),
                                       cores=round(float(np.mean([x['cores'] for x in per])), 2))
        _, r['spectral'] = timed(lambda: spectral_sync_from_topk(labs, probs, time_index_to_linear_feature_indices=t2l,
                                                                 input_probability_threshold=0.1, verbose=False))
        r['projected_25_seed_wall_s'] = round(sum(r[k]['wall_s'] for k in ('svd', 'nndescent', 'edge_list',
                                                                          'normalize', 'spectral'))
                                              + 25 * r['propagate_per_seed']['wall_s'], 1)
        res['settings'][f'{dev}_{n}threads'] = r
        print(f'{dev} {n} threads', json.dumps(r), flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
