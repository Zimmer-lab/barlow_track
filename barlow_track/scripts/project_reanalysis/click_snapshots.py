"""Image crops around GT clicks for a manual check: is the click on the right object?

Reads a click list (neuron_removal_scores.py consensus_error_clicks.csv: neuron, t, click_z/x/y in project voxels,
gt_label, consensus_label) and, per neuron, draws up to --per_neuron clicks spread over the recording. Each panel
is the red channel around the click (max projection over z +- --dz planes, xy +- --half px), with the outline of
the GT's object (green), the object every tracker chose (magenta), other masks (thin grey) and the click (cyan +).
A second row per click shows the single z plane of the click. One PNG per neuron, <out>/<neuron>.png, plus
index.csv listing what each panel shows.

    python .../click_snapshots.py --project .../stardist/project_config.yaml --clicks consensus_error_clicks.csv \
        --neurons neuron_023 neuron_072 --out DIR
"""
import argparse
import os

import numpy as np
import pandas as pd


def _outline(mask):
    from scipy import ndimage as ndi
    return mask & ~ndi.binary_erosion(mask)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', required=True)
    ap.add_argument('--clicks', required=True)
    ap.add_argument('--neurons', nargs='*', default=None, help='default: the 10 with most clicks in the list')
    ap.add_argument('--per_neuron', type=int, default=16)
    ap.add_argument('--half', type=int, default=24, help='half width of the crop in xy pixels')
    ap.add_argument('--dz', type=int, default=2)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from wbfm.utils.projects.finished_project_data import ProjectData
    prj = ProjectData.load_final_project_data(args.project, allow_hybrid_loading=True, verbose=0)
    df = pd.read_csv(args.clicks)
    neurons = args.neurons or list(df.neuron.value_counts().index[:10])
    rows = []
    for n in neurons:
        d = df[df.neuron == n].sort_values('t')
        if not len(d):
            continue
        pick = d.iloc[np.unique(np.linspace(0, len(d) - 1, min(args.per_neuron, len(d))).round().astype(int))]
        k = len(pick)
        fig, axes = plt.subplots(2, k, figsize=(1.9 * k, 4.0), squeeze=False)
        for c, r in enumerate(pick.itertuples()):
            red = np.asarray(prj.red_data[r.t], dtype=float)
            seg = np.asarray(prj.raw_segmentation[r.t])
            Z, X, Y = red.shape
            z, x, y = int(round(r.click_z)), int(round(r.click_x)), int(round(r.click_y))
            x0, x1 = max(0, x - args.half), min(X, x + args.half + 1)
            y0, y1 = max(0, y - args.half), min(Y, y + args.half + 1)
            z0, z1 = max(0, z - args.dz), min(Z, z + args.dz + 1)
            for row, (zz0, zz1, title) in enumerate(((z0, z1, f't {r.t}'), (z, z + 1, f'z {z}'))):
                img = red[zz0:zz1, x0:x1, y0:y1].max(0)
                sg = seg[zz0:zz1, x0:x1, y0:y1]
                lo, hi = np.percentile(img, [1, 99.5])
                rgb = np.repeat(((img - lo) / max(hi - lo, 1e-9)).clip(0, 1)[..., None], 3, axis=2)
                for lab in np.unique(sg):
                    if lab == 0:
                        continue
                    o = _outline((sg == lab).any(0))
                    col = (0.1, 1.0, 0.1) if lab == r.gt_label else (1.0, 0.2, 1.0) if lab == r.consensus_label \
                        else (0.55, 0.55, 0.55)
                    if lab in (r.gt_label, r.consensus_label) or row == 1:
                        rgb[o] = col
                ax = axes[row, c]
                ax.imshow(rgb, interpolation='nearest')
                ax.plot([y - y0], [x - x0], '+', color='cyan', ms=9, mew=1.5)
                ax.set_title(title, fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            rows.append(dict(neuron=n, column=c, t=int(r.t), click_zxy=(z, x, y), gt_label=int(r.gt_label),
                             consensus_label=int(r.consensus_label)))
        fig.suptitle(f'{n}: click (cyan +), GT object (green), object every tracker chose (magenta); '
                     f'top: max over z +-{args.dz}, bottom: click plane', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f'{n}.png'), dpi=120)
        plt.close(fig)
        print(n, k, 'panels', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'index.csv'), index=False)


if __name__ == '__main__':
    main()
