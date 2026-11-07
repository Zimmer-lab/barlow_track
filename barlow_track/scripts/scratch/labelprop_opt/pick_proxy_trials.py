"""Pick sweep trials at evenly spaced accuracy ranks per round, for re-checking the GT-free consistency proxy
(agree_mean) under a new label propagation recipe.

For every round, the trials with a cached embedding and a recorded accuracy (exp_results.jsonl, latest record
per tag wins) are sorted by accuracy and N are taken at evenly spaced ranks (always the best and the worst).
Leifer StarDist trials (EXTRA_LABELS in sweep_labelprop.py) have no recorded accuracy; all of them with an
embedding are listed.

Prints one harness case per line, round:trial:label, for run_local.sh ONLY_CASE.

    python pick_proxy_trials.py --n 16 > $WORK/proxy_cases.txt
"""
import argparse
import glob
import json
import math
import os
import re

C = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files'
ROUNDS = [('round_bayes_wk2_samuel_137', 'samuel_137'), ('round_bayes_wk3_samuel_184', 'samuel_184'),
          ('round_bayes_wk3_zimmer_1210', 'zimmer_1210'), ('round_bayes_wk3_zimmer_1123', 'zimmer_1123'),
          ('round_bayes_wk3_flavell_2022032402', 'flavell_2022032402'),
          ('round_bayes_wk3_flavell_2022033002', 'flavell_2022033002'),
          ('round_bayes_wk2_leifer_reexport', 'leifer_reexport')]
STARDIST_EMB = os.path.expanduser('~/.cache/barlow_scratch/train/leifer_reseg/benchmark_emb')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--no_stardist', action='store_true')
    args = ap.parse_args()
    for rnd, label in ROUNDS:
        acc = {}
        for line in open(os.path.join(C, rnd, 'exp_results.jsonl')):
            r = json.loads(line)
            if r.get('tag', '').endswith('_' + label) and r.get('accuracy') is not None:
                a = float(r['accuracy'])
                if math.isfinite(a):
                    acc[r['tag']] = a
        trials = []
        for tag, a in acc.items():
            m = re.match(rf'{rnd}_trial(\d+)_{label}$', tag)
            if m and glob.glob(os.path.join(C, rnd, 'emb_cache', f'emb_*_trained_*{tag}.npz')):
                trials.append((a, int(m.group(1))))
        trials.sort()
        k = min(args.n, len(trials))
        idx = sorted({round(i * (len(trials) - 1) / max(k - 1, 1)) for i in range(k)})
        for i in idx:
            print(f'{rnd}:{trials[i][1]}:{label}')
    if not args.no_stardist:
        for f in sorted(glob.glob(os.path.join(STARDIST_EMB, 'emb_*_trained_round_bayes_wk2_leifer_reexport_trial*_leifer_stardist.npz'))):
            t = int(re.search(r'_trial(\d+)_', f).group(1))
            print(f'round_bayes_wk2_leifer_reexport:{t}:leifer_stardist_strict')


if __name__ == '__main__':
    main()
