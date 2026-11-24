"""TEST_PLAN T2: where do the image crops matter? Fused vs position-only vs image-only on the six tuning recordings.

Reads the round_sens_* rounds only (nothing is trained or embedded). A trial's variant is how its train_config.yaml
differs from the frozen recipe (sensitivity_report.variant_name); compared here: control (fused), fusion=position_only,
fusion=visual_only (image-only) and epochs=0 (untrained). Replicates are networks with different training seeds;
trials sharing a seed (the control in loss_volume and again in crop_depth) are averaged first.

1. t2_modality_table.tsv: per recording x variant x tracker (v2 = exp_results_v2.jsonl from sens_rescore, legacy =
   exp_results.jsonl): number of seeds, mean, min, max accuracy. v2 and legacy are never mixed in one row.
2. t2_neurons_<lab>.tsv and t2_neuron_summary.tsv: per GT neuron, accuracy of each variant from the v2 tracks (mean
   over seeds), delta_pos = fused - position-only, delta_img = fused - image-only, and neuron features: median NN
   distance to other GT neurons and local GT density (GT neighbours within 2 NN) in units of the detection NN, the fraction of frames without GT (gaps), and the
   non-rigid residual at lag 10 (t1_deformation_table.py). Summary: the aggregate gap (frame-weighted, equals the
   accuracy difference), the share of it carried by the top 10% of neurons, and Spearman of delta with each feature.
3. t2_frame_deciles.tsv: per-frame delta_pos by decile of T1's d_novel (needs t1_frames_<lab>.tsv), with each
   decile's error of both variants and their ratio (position-only / fused). A growing point gap alone can be a
   ceiling effect; only a rising ratio says the images matter specifically on deformed frames. The 'spearman' row
   holds rank correlations with deformation (delta_pos over frames; err_ratio over the deciles).

Brightness is not among the features yet (needs the red channel).

Usage (wbfm env, from the repo root; run t1_deformation_table.py first for the deformation features):
  python barlow_track/scripts/paper_scripts/revision/t2_modality_contribution.py \
      --out /lisc/.../barlow_track_revisions/revision_analysis
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from barlow_track.scripts.paper_scripts.revision.gt_positions import ROUNDS, TUNING, aligned, detection_nn, load_positions, track_accuracy
from barlow_track.scripts.sensitivity_report import BASE, variant_name

VARIANTS = {'': 'fused', 'fusion=position_only': 'position_only', 'fusion=visual_only': 'image_only',
            'epochs=0': 'untrained'}


def latest(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('tag'):
                out[r['tag']] = r
    return out


def collect_cells(base):
    """[dict(lab, variant, seed, tracker, accuracy, tracks)] for the compared variants of every round_sens_* round."""
    cells = []
    for d in sorted(glob.glob(os.path.join(ROUNDS, 'round_sens_*'))):
        for tracker, fn in (('v2', 'exp_results_v2.jsonl'), ('legacy', 'exp_results.jsonl')):
            for tag, r in latest(os.path.join(d, fn)).items():
                m = re.search(r'_trial(\d+)_(.+)$', tag)
                cfg_path = os.path.join(d, f'trial_{m.group(1)}', 'train_config.yaml') if m else None
                if not m or r.get('accuracy') is None or not os.path.exists(cfg_path):
                    continue
                lab = m.group(2)
                var = variant_name(yaml.safe_load(open(cfg_path)), base, lab)
                if var not in VARIANTS:
                    continue
                cfg = yaml.safe_load(open(cfg_path))
                tracks = os.path.join(d, 'tracks_v2', f'{tag}.h5')
                cells.append(dict(lab=lab, variant=VARIANTS[var], seed=int(cfg['seed']), tracker=tracker,
                                  accuracy=float(r['accuracy']),
                                  tracks=tracks if tracker == 'v2' and os.path.exists(tracks) else None))
    return pd.DataFrame(cells)


def modality_table(cells):
    per_seed = cells.groupby(['lab', 'variant', 'tracker', 'seed'], as_index=False)['accuracy'].mean()
    t = per_seed.groupby(['lab', 'variant', 'tracker'])['accuracy'].agg(['count', 'mean', 'min', 'max']).reset_index()
    t = t.rename(columns={'count': 'n_seeds'})
    fused = t[t.variant == 'fused'].set_index(['lab', 'tracker'])['mean']
    t['delta_vs_fused_pt'] = [100 * (r['mean'] - fused.get((r['lab'], r['tracker']), np.nan)) for _, r in t.iterrows()]
    return t.round(4)


def neuron_features(lab, valid, out_dir):
    """Per GT neuron (valid's columns): median NN distance and density in NN units, gap fraction, lag-10 deformation."""
    pos, names, index = load_positions(lab, os.path.join(out_dir, 'cache'))
    nn = detection_nn(lab, os.path.join(out_dir, 'cache'))['nn_det']  # same unit as t1_deformation_table.py
    p, coverage = aligned(pos, names, index, valid)
    nn_i, dens = [[] for _ in range(p.shape[1])], [[] for _ in range(p.shape[1])]
    for t in range(0, len(p), 5):
        ok = np.flatnonzero(np.isfinite(p[t]).all(1))
        if len(ok) < 3:
            continue
        d = np.linalg.norm(p[t, ok][:, None] - p[t, ok][None], axis=-1) / nn
        np.fill_diagonal(d, np.inf)
        for j, i in enumerate(ok):
            nn_i[i].append(d[j].min())
            dens[i].append((d[j] < 2).sum())
    f = pd.DataFrame(dict(neuron=[str(c) for c in valid.columns],
                          nn_dist=[np.median(v) if v else np.nan for v in nn_i],
                          density_2nn=[np.median(v) if v else np.nan for v in dens],
                          gap_frac=1 - valid.mean(0).to_numpy()))
    dfile = os.path.join(out_dir, f'neuron_deformation_{lab}.npz')
    if os.path.exists(dfile):
        z = np.load(dfile)
        f['nonrigid_lag10'] = f.neuron.map(dict(zip([str(n) for n in z['names']], z['nonrigid_lag10'])))
    return f, coverage


def variant_accuracy(files):
    """Per-neuron and per-frame accuracy, mean over seeds (files), and the GT validity mask."""
    per_n, per_t, valid = [], [], None
    for f in files:
        c, v = track_accuracy(f)
        per_n.append(c.sum(0) / v.sum(0).replace(0, np.nan))
        per_t.append(c.sum(1) / v.sum(1).replace(0, np.nan))
        valid = v
    return pd.concat(per_n, axis=1).mean(1), pd.concat(per_t, axis=1).mean(1), valid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--labs', nargs='+', default=TUNING)
    args = ap.parse_args()
    base = yaml.safe_load(open(BASE))

    cells = collect_cells(base)
    cells = cells[cells.lab.isin(args.labs)]
    table = modality_table(cells)
    table.to_csv(os.path.join(args.out, 't2_modality_table.tsv'), sep='\t', index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(table, flush=True)

    summary, frame_rows = [], []
    for lab in args.labs:
        v2 = cells[(cells.lab == lab) & (cells.tracker == 'v2') & cells.tracks.notna()]
        # one track file per seed (the first trial of a repeated seed), so every seed weighs the same
        files = {var: g.drop_duplicates('seed').tracks.tolist() for var, g in v2.groupby('variant')}
        if 'fused' not in files or 'position_only' not in files:
            print(f'{lab}: no v2 tracks for fused and position-only, skipped', flush=True)
            continue
        acc_n, acc_t, n_seeds = {}, {}, {}
        valid = None
        for var, fl in files.items():
            acc_n[var], acc_t[var], valid = variant_accuracy(fl)
            n_seeds[var] = len(fl)
        feats, coverage = neuron_features(lab, valid, args.out)
        for var in acc_n:
            feats[f'acc_{var}'] = acc_n[var].reindex(valid.columns).to_numpy()
        feats['n_frames_gt'] = valid.sum(0).to_numpy()
        feats['delta_pos'] = feats.acc_fused - feats.acc_position_only
        if 'image_only' in acc_n:
            feats['delta_img'] = feats.acc_fused - feats.acc_image_only
        feats.to_csv(os.path.join(args.out, f't2_neurons_{lab}.tsv'), sep='\t', index=False)

        for delta in [c for c in ('delta_pos', 'delta_img') if c in feats]:
            f = feats.dropna(subset=[delta])
            w = f.n_frames_gt / f.n_frames_gt.sum()
            gap = float((w * f[delta]).sum())
            contrib = (w * f[delta]).sort_values(ascending=False)
            top10 = contrib.iloc[:max(1, int(round(0.1 * len(contrib))))].sum()
            row = dict(lab=lab, delta=delta, n_neurons=len(f), seeds=str(n_seeds), position_coverage=round(coverage, 4),
                       gap_pt=round(100 * gap, 2), top10pct_neurons_share=round(float(top10 / gap), 2) if gap > 0 else np.nan,
                       delta_median_pt=round(100 * float(f[delta].median()), 2),
                       delta_p90_pt=round(100 * float(f[delta].quantile(0.9)), 2))
            for feat in ('nn_dist', 'density_2nn', 'gap_frac', 'nonrigid_lag10'):
                if feat in f and f[feat].notna().sum() > 5:
                    row[f'rho_{feat}'] = round(float(spearmanr(f[feat], f[delta], nan_policy='omit').correlation), 3)
            summary.append(row)
            print(row, flush=True)

        t1 = os.path.join(args.out, f't1_frames_{lab}.tsv')
        if os.path.exists(t1):
            fr = pd.read_csv(t1, sep='\t').set_index('frame')
            af = acc_t['fused'].reindex(fr.index)
            apo = acc_t['position_only'].reindex(fr.index)
            ok = fr.d_novel.notna() & af.notna() & apo.notna()
            s, f_, p_ = fr.d_novel[ok].to_numpy(), af[ok].to_numpy(), apo[ok].to_numpy()
            d = f_ - p_
            q = np.clip(np.searchsorted(np.percentile(s, np.linspace(0, 100, 11)), s, side='right') - 1, 0, 9)
            # Ceiling check: a constant relative weakness of position-only also grows in points where both fail
            # more, so report the error ratio (position-only errors / fused errors) per decile next to the gap.
            # Only a ratio that rises with deformation means the images matter specifically under deformation.
            ratios = []
            for k in range(10):
                ef, ep = 1 - f_[q == k].mean(), 1 - p_[q == k].mean()
                ratios.append(ep / ef if ef > 0 else np.nan)
                frame_rows.append(dict(lab=lab, decile=k + 1, n_frames=int((q == k).sum()),
                                       delta_pos_pt=round(100 * float(d[q == k].mean()), 2),
                                       err_fused_pt=round(100 * float(ef), 2), err_pos_pt=round(100 * float(ep), 2),
                                       err_ratio=round(float(ratios[-1]), 2)))
            frame_rows.append(dict(lab=lab, decile='spearman', n_frames=int(ok.sum()),
                                   delta_pos_pt=round(float(spearmanr(s, d).correlation), 3),
                                   err_ratio=round(float(spearmanr(np.arange(10), ratios, nan_policy='omit').correlation), 3)))

    pd.DataFrame(summary).to_csv(os.path.join(args.out, 't2_neuron_summary.tsv'), sep='\t', index=False)
    if frame_rows:
        pd.DataFrame(frame_rows).to_csv(os.path.join(args.out, 't2_frame_deciles.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    main()
