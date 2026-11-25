"""Trace fidelity of trial 59 BarlowTrack and the paired five-template fDNC baseline. Run as a module from the repository root."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
import yaml
import zarr

from barlow_track.scripts.paper_scripts.revision.gt_positions import ROUNDS, VPS

RECORDINGS = {
    'zimmer_1128': 'Zimmer · 1128, worm 1',
    'leifer_stardist_strict': 'Leifer · NeRVE worm 1',
    'samuel': 'Samuel · 153',
    'flavell': 'Flavell · 2022-03-31-01 (GFP)',
}
SOURCE_TABLES = Path(__file__).with_name('trace_sources')
DEFAULT_ROUND = Path(ROUNDS) / 'round_shared_v1'
OVERVIEW_SECONDS = 120
DETAIL_SECONDS = 8


def track_files(round_dir, recording):
    pattern = re.compile(rf'round_shared_v1_trial59(?:r(\d+))?_{recording}')
    files = {}
    for path in (round_dir / 'tracks_v2').glob('*.h5'):
        match = pattern.fullmatch(path.stem)
        if match:
            files[int(match.group(1) or 0)] = path
    if 0 not in files:
        raise FileNotFoundError(f'Missing original trial 59 for {recording}')
    return dict(sorted(files.items()))


def sources(record):
    cfg_path = Path(record['train_config']['project_path'])
    cfg = yaml.safe_load(cfg_path.read_text())
    root = cfg_path.parent
    sub = cfg['subfolder_configs']
    pre = yaml.safe_load((root / sub['preprocessing']).read_text())
    info = dict(background=float(pre['reset_background_per_pixel']), configuration=str(cfg_path))
    if record['source'] == 'nwb':
        suffix = 'Untracked' if record['nwb_objects'] == 'all' else ''
        info.update(nwb=record['gt'], masks=f'processing/CalciumActivity/CalciumSeriesSegmentation{suffix}/data',
                    images='acquisition/CalciumImageSeries/data')
    else:
        seg = yaml.safe_load((root / sub['segmentation']).read_text())
        info['masks'] = str(root / seg['output_masks'])
        for channel in ('green', 'red'):
            key = 'preprocessed_' + channel
            info[channel] = str(root / (cfg.get(key) or pre[key + '_fname']))
    return info


def integrate_frames(span, source):
    """Return area and both integrated channels indexed by (frame, positive mask label)."""
    rows = []
    with ExitStack() as stack:
        if 'nwb' in source:
            f = stack.enter_context(h5py.File(source['nwb'], 'r'))
            masks, images = f[source['masks']], f[source['images']]
        else:
            masks = zarr.open(source['masks'], mode='r')
            channels = [zarr.open(stack.enter_context(zarr.ZipStore(source[c], mode='r')), mode='r')
                        for c in ('red', 'green')]
        for t in span:
            mask = masks[t]
            if 'nwb' in source:
                red, green = (images[t, ..., i] for i in (0, 1))
            else:
                red, green = (z[t] for z in channels)
            if red.shape != mask.shape or green.shape != mask.shape:
                raise ValueError(f'Image/mask shape mismatch at {t}')
            pixels = np.flatnonzero(mask)
            labels = mask.ravel()[pixels].astype(np.intp)
            area = np.bincount(labels)
            ids = np.flatnonzero(area)
            green_sum = np.bincount(labels, green.ravel()[pixels])
            red_sum = np.bincount(labels, red.ravel()[pixels])
            rows.append(np.column_stack((np.full(len(ids), t), ids, area[ids], green_sum[ids], red_sum[ids])))
    return np.concatenate(rows)


def extract(source, frames, workers):
    spans = [frames[i:i+40] for i in range(0, len(frames), 40)]
    with ProcessPoolExecutor(workers) as pool:
        values = np.concatenate(list(pool.map(partial(integrate_frames, source=source), spans)))
    return pd.DataFrame(values.astype('int64'), columns=['frame', 'mask', 'area', 'green', 'red'])



def channel_correct(values):
    """wbfm's per-channel exponential division, including its failed-fit fallback."""
    from wbfm.utils.external.bleach_correction import detrend_exponential_lmfit
    finite = values[np.isfinite(values)]
    if len(finite) < 2 or np.ptp(finite) == 0:
        return values.copy(), 'insufficient' if len(finite) < 2 else 'constant'
    corrected, (_, _, fitted) = detrend_exponential_lmfit(pd.Series(values.copy()), restore_mean_value=True)
    return np.asarray(corrected), 'fitted' if fitted else 'fallback'


def make_trace(area, green, red, background):
    from wbfm.utils.visualization.filtering_traces import remove_outliers_using_std, filter_rolling_mean
    observed = np.isfinite(area) & (area > 0) & np.isfinite(green) & np.isfinite(red)
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = np.divide(green, red, out=np.full_like(green, np.nan), where=observed & (red != 0))
    g, gfit = channel_correct(green - area * background)
    r, rfit = channel_correct(red - area * background)
    ratio = np.divide(g, r, out=np.full_like(g, np.nan), where=observed & np.isfinite(r) & (r != 0))
    valid = observed & np.isfinite(ratio)
    baseline = np.median(ratio[valid]) if valid.any() else np.nan
    processed = np.full_like(g, np.nan)
    flat = valid.any() and np.ptp(ratio[valid]) == 0
    if baseline != 0 and np.isfinite(baseline):
        scaled = pd.Series((ratio - baseline) / baseline)
        clean = remove_outliers_using_std(scaled, std_factor=5)
        smoothed = filter_rolling_mean(clean, window=3).to_numpy()
        # A rolling mean can create values at missing masks and rejected samples.
        keep = valid & np.isfinite(clean)
        processed[keep] = smoothed[keep]
    return dict(observed=observed, processed=processed, raw=raw, constant_ratio=flat, zero_baseline=baseline == 0,
                green_fit=gfit, red_fit=rfit)


def traces_for_track(objects, labels, background):
    ids = labels.to_numpy(dtype=float)
    if (np.isfinite(ids) & ((ids <= 0) | (ids != np.floor(ids)))).any():
        raise ValueError('Expected positive frame-local integer mask IDs')
    keys = pd.MultiIndex.from_arrays([np.repeat(labels.index, labels.shape[1]), ids.ravel()],
                                     names=['frame', 'mask'])
    values = objects.reindex(keys).to_numpy().reshape(*ids.shape, 3)
    if not np.array_equal(np.isfinite(ids), np.isfinite(values[:, :, 0])):
        raise ValueError('An assigned frame-local ID is absent from the segmentation')
    return {name: make_trace(*values[:, j].T, background) for j, name in enumerate(labels)}


def correlation(a, b, different, undefined_reason=None):
    from barlow_track.scripts.project_reanalysis.compare_traces import _r
    shared = np.isfinite(a) & np.isfinite(b)
    n = int(shared.sum())
    if n < 10:
        status = undefined_reason or 'insufficient_overlap'
        r = np.nan
    elif np.ptp(a[shared]) == 0 or np.ptp(b[shared]) == 0:
        r, status = np.nan, 'constant'
    else:
        r, count = _r(a, b)
        if not np.isfinite(r) or count != n:
            raise ValueError('Nonfinite correlation for a varying trace')
        status = 'evaluable'
    return dict(r=r, n=n, n_different=int((different & shared).sum()), status=status)


def reference_snr(raw):
    """Single-volume noise estimate from successive observed samples of the raw ratio."""
    finite = np.isfinite(raw)
    adjacent = finite[1:] & finite[:-1]
    if adjacent.sum() < 10:
        return np.nan
    noise = 1.4826 * np.median(np.abs(np.diff(raw)[adjacent])) / np.sqrt(2)
    # Winsorize the amplitude estimate so one intensity spike cannot select a neuron.
    low, high = np.percentile(raw[finite], [5, 95])
    return np.std(np.clip(raw[finite], low, high)) / noise if noise > 0 else np.nan


def zimmer_identities(config):
    path = Path(config).parent / '3-tracking/manual_annotation/manual_annotation.xlsx'
    ids = pd.read_excel(path).set_index('Neuron ID')
    primary = ids.loc[(ids.Certainty >= 2) & ids.ID1.notna() & ids.ID2.isna(), 'ID1']
    return primary[~primary.duplicated(keep=False)].to_dict()


def samuel_identities(nwb, names):
    from scipy.spatial.distance import cdist
    published = pd.read_csv(SOURCE_TABLES / 'samuel_153_viewer_centroids.csv')
    with h5py.File(nwb, 'r') as f:
        group = f['processing/CalciumActivity/NeuronCentroids']
        xyz = np.array([group[name]['data'][4] for name in names])
    # The viewer's static annotation image corresponds to zero-based NWB frame 4.
    # Its coordinates are one-based XYZ voxel coordinates scaled by (3, 3, 8).
    distance = cdist((xyz + 1) * [3, 3, 8], published[['x','y','z']])
    i, j = np.where(distance < .001)
    if len(i) != 95 or len(np.unique(i)) != len(i) or len(np.unique(j)) != len(j):
        raise ValueError('Samuel viewer-to-NWB centroid match failed')
    return {names[a]: published.neuron_id[b] for a, b in zip(i,j)
            if not re.fullmatch(r'Q\d+', published.neuron_id[b])}


def behavior_intervals(recording, config, nframes):
    """Half-open spans on the volume time axis; never extend over a missing annotation."""
    if recording == 'zimmer_1128':
        cfg = yaml.safe_load(Path(config).read_text())
        folder = Path(cfg['red_bigtiff_fname']).parent.parent
        path, = folder.glob('*Ch0-BH/beh_annotation_manual_corrected_timeseries.csv')
        labels = pd.read_csv(path).set_index('Volume', verify_integrity=True)
        acquisition = yaml.safe_load((folder / 'config.yaml').read_text())
        stride = acquisition['num_z_planes'] + acquisition['num_flyback_planes_discarded']
        start = cfg['dataset_params']['bigtiff_start_volume']
        # wbfm WormFullVideoPosture.subsample_indices takes the middle exposure.
        indices = (start + np.arange(nframes)) * stride + stride // 2
        states = labels.loc[indices, 'Description']
        backward = states.eq('Rev').to_numpy()
    elif recording == 'leifer_stardist_strict':
        from scipy.io import loadmat
        root = Path(config).parents[4] / 'barlow_track_paper/leifer_data/Nguyen2017NeuronTracking/worm1'
        data = loadmat(root / 'hiResData.mat', squeeze_me=True, struct_as_record=False)['dataAll']
        ends = pd.Series(data.frameTime).groupby(data.stackIdx).last().loc[1:nframes]
        np.testing.assert_array_equal(ends.index, np.arange(1, nframes+1))
        end_ms = np.rint(1000*ends.to_numpy()).astype(int)
        heat = loadmat(root / 'outputFiles/heatData.mat', squeeze_me=True,
                       variable_names=['hasPointsTime', 'ethoTrack'])
        times = np.rint(1000*heat['hasPointsTime']).astype(int)
        # The first published time is the end of stack 3. Check EVERY timestamp
        # against the acquisition clock, including gaps, before applying any label.
        absolute = times + end_ms[2]
        indices = np.searchsorted(end_ms, absolute)
        np.testing.assert_array_equal(end_ms[indices], absolute)
        if not np.all(np.diff(indices) > 0):
            raise ValueError('Leifer behavior timestamps must identify distinct, ordered volumes')
        backward = np.zeros(nframes, dtype=bool)
        backward[indices] = heat['ethoTrack'] == -1
    elif recording == 'samuel':
        bouts = pd.read_csv(SOURCE_TABLES / 'samuel_153_ventral_contact.csv')
        # Susoy et al. report one-based inclusive volumes. No label for the final GT frame.
        return list(zip(bouts.first_volume - 1, bouts.last_volume))
    else:
        return []
    changes = np.diff(np.r_[0, backward, 0].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def barlow_tracks(files, records, gt, source):
    for rep, path in files.items():
        rec = records[path.stem]
        if rec['labelprop'] != 'v2' or rec['cluster'] != 'labelprop' or sources(rec) != source:
            raise ValueError(f'Inconsistent evaluation inputs for {path}')
        pd.testing.assert_frame_equal(pd.read_hdf(path, 'gt'), gt)
        yield rep+1, path.stem, rec['train_config']['seed'], rec['accuracy'], pd.read_hdf(path, 'pred_matched')


def fdnc_track(directory, recording, baseline, gt, embedding):
    """Recover the benchmark's whole-recording identity assignment in frame-local mask IDs."""
    from barlow_track.utils.utils_ground_truth import match_columns_to_gt
    rec = baseline['fdnc_record']
    if not baseline['checks']['all_identical'] or rec['variant'] != 'multi' or rec['gt_matching'] != 'nandist':
        raise ValueError(f'Unexpected fDNC benchmark record for {recording}')
    scored = pd.read_hdf(directory / (recording + '_scored_gt.h5'), 'gt')
    pd.testing.assert_frame_equal(scored.loc[gt.index], gt, check_index_type=False)
    if scored.loc[scored.index.difference(gt.index)].notna().any().any():
        raise ValueError('Extra fDNC reference observations outside the compared recording')
    path = rec['reused_from'] or directory / (rec['tag'] + '_df.pickle')
    pred = pd.read_pickle(path).xs('raw_neuron_ind_in_list', axis=1, level=1)
    if not pred.index.equals(scored.index):
        raise ValueError(f'fDNC track and reference frames differ for {recording}')
    col = rec['match_col']
    if col == 'raw_segmentation_id':
        with np.load(embedding, allow_pickle=True) as archive:
            objects = pd.DataFrame(archive['lin_to_t_seg'].item().values(), columns=['frame','raw_ind','mask'])
        mapping = objects.set_index(['frame','raw_ind'], verify_integrity=True)['mask']
        keys = pd.MultiIndex.from_arrays([np.repeat(pred.index, pred.shape[1]), pred.to_numpy().ravel()])
        labels = mapping.reindex(keys).to_numpy().reshape(pred.shape)
        np.testing.assert_array_equal(np.isfinite(labels), pred.notna().to_numpy())
        pred = pd.DataFrame(labels, index=pred.index, columns=pred.columns)
    # Flavell and Samuel benchmark predictions already carry the NWB mask labels.
    ref = scored.copy()
    for frame in (ref, pred):
        frame.columns = pd.MultiIndex.from_product([frame.columns, [col]])
    matched = match_columns_to_gt(ref, pred, col, gt_matching=rec['gt_matching']).xs(col, axis=1, level=1)
    matched = matched.drop(columns='unmatched_neuron', errors='ignore').reindex(index=gt.index, columns=gt.columns)
    missing = (gt.notna() & matched.isna()).sum().sum()
    wrong = (gt.notna() & matched.notna() & gt.ne(matched)).sum().sum()
    np.testing.assert_array_equal([missing, wrong, gt.count().sum()], [rec['misses'], rec['mismatches'], rec['total']])
    return 1, rec['tag'], None, rec['accuracy'], matched


def analyze(recording, runs, gt, objects, source):
    lookup = objects.set_index(['frame', 'mask'], verify_integrity=True)[['area', 'green', 'red']]
    reference = traces_for_track(lookup, gt, source['background'])
    identities = {}
    if recording == 'zimmer_1128':
        identities = zimmer_identities(source['configuration'])
    elif recording == 'samuel':
        identities = samuel_identities(source['nwb'], gt.columns)
    snr = {name: reference_snr(trace['raw']) for name, trace in reference.items()}
    rows, examples = [], {}
    for rep, tag, seed, expected_accuracy, matched in runs:
        if not matched.index.equals(gt.index) or not matched.columns.is_unique:
            raise ValueError(f'Invalid matched identity table in {tag}')
        matched = matched.reindex(columns=gt.columns)
        accuracy = (gt == matched).sum().sum() / gt.count().sum()
        np.testing.assert_allclose(accuracy, expected_accuracy, atol=1e-8)
        prediction = traces_for_track(lookup, matched, source['background'])
        for name, ref in reference.items():
            pred = prediction[name]
            shared = ref['observed'] & pred['observed']
            different = shared & gt[name].ne(matched[name]).to_numpy()
            same = shared & ~different
            np.testing.assert_array_equal(ref['raw'][same], pred['raw'][same],
                                          err_msg=f'{tag}, {name}: same-mask ratio differs')
            nref = int(ref['observed'].sum())
            n_correct = int(same.sum())
            row = dict(recording=recording, replicate=rep, track_tag=tag, training_seed=seed, neuron=name,
                       anatomical_id=identities.get(name), reference_snr=snr[name],
                       n_ref_assigned=gt[name].count(), n_pred_assigned=matched[name].count(),
                       n_ref=nref, n_pred=int(pred['observed'].sum()), n_shared=int(shared.sum()),
                       n_different_masks=int(different.sum()), n_correct_masks=n_correct,
                       mask_accuracy=n_correct/nref if nref else np.nan,
                       coverage=shared.sum()/nref if nref else np.nan)
            for signal in ('processed', 'raw'):
                undefined = None
                if signal == 'processed' and shared.sum() >= 10:
                    if ref['zero_baseline'] or pred['zero_baseline']:
                        undefined = 'zero_baseline'
                    if ref['constant_ratio'] or pred['constant_ratio']:
                        undefined = 'constant'
                result = correlation(ref[signal], pred[signal], different, undefined)
                row.update({signal+'_'+key: val for key, val in result.items()})
            for side, trace in [('ref', ref), ('pred', pred)]:
                for channel in ['green_fit', 'red_fit']:
                    row[side+'_'+channel] = trace[channel]
            rows.append(row)
        if rep == 1:
            first = pd.DataFrame(rows).dropna(subset=['raw_r']).sort_values(
                ['raw_r', 'neuron']).reset_index(drop=True)
            selected = {}
            for key, q in [('Median', .5), ('10th percentile', .1)]:
                dist = (first.raw_r - first.raw_r.quantile(q)).abs()
                ties = first.index[dist == dist.min()].to_numpy()
                nearest = ties[np.argmin(np.abs(ties - q*(len(first)-1)))]
                selected[key] = first.loc[nearest]
            ranked = first.sort_values(['reference_snr','neuron'], ascending=[False,True]).dropna(subset=['reference_snr'])
            ranked = ranked[~ranked.neuron.isin([r.neuron for r in selected.values()])]
            if recording == 'zimmer_1128':
                # One member per anatomical class shown in the manuscript's Fig. 4b/e/h.
                classes = ranked.anatomical_id.str.extract(r'^(AVA|BAG|RME|RID)[LRVD]?$', expand=False)
                ranked = ranked.assign(cell_class=classes).dropna(subset=['cell_class']).drop_duplicates('cell_class')
                ranked = ranked.head(3)
            elif recording == 'samuel':
                ranked = ranked.dropna(subset=['anatomical_id']).head(2)
            elif recording == 'flavell':
                ranked = ranked.iloc[:0]
            else:
                ranked = ranked.head(2)
            for i, (_, row) in enumerate(ranked.iterrows(), 1):
                selected[f'High SNR {i}'] = row
            for key, row in selected.items():
                name = row.neuron
                examples[key] = (row, reference[name]['raw'], prediction[name]['raw'],
                                 gt[name].ne(matched[name]).to_numpy())
        print(f'Analyzed {tag}', flush=True)
    result = pd.DataFrame(rows)
    result['example'] = ''
    for key, (row, *_) in examples.items():
        result.loc[(result.replicate == 1) & (result.neuron == row.neuron), 'example'] = key
    return result, examples


def summarize(df):
    rows = []
    for (recording, rep, tag), group in df.groupby(['recording', 'replicate', 'track_tag'], sort=False):
        row = dict(recording=recording, replicate=rep, track_tag=tag, training_seed=group.training_seed.iloc[0],
                   n_reference_ids=len(group),
                   mask_accuracy=group.n_correct_masks.sum() / group.n_ref.sum(),
                   coverage_median=group.coverage.median(), coverage_q25=group.coverage.quantile(.25),
                   coverage_q75=group.coverage.quantile(.75), coverage_pooled=group.n_shared.sum()/group.n_ref.sum())
        for signal in ('processed', 'raw'):
            r = group[signal+'_r'].dropna()
            status = group[signal+'_status']
            metrics = dict(median=r.median(), q25=r.quantile(.25), q75=r.quantile(.75),
                           n_below_0_9=(r < .9).sum(), n_evaluable=len(r),
                           n_constant=(status == 'constant').sum(), n_zero_baseline=(status == 'zero_baseline').sum(),
                           n_insufficient_overlap=(status == 'insufficient_overlap').sum())
            if len(r) + sum(metrics['n_'+k] for k in ['constant','zero_baseline','insufficient_overlap']) != len(group):
                raise ValueError('Incomplete exclusion accounting')
            row.update({signal+'_'+k: v for k, v in metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)



def disagreement_window(ref, pred, different, time):
    shared = (time <= OVERVIEW_SECONDS) & np.isfinite(ref) & np.isfinite(pred)
    if not shared.any():
        raise ValueError('No observed pair in the displayed interval for the detail example')
    candidates = shared & different
    if not candidates.any():
        candidates = shared
    peak = time[np.argmax(np.where(candidates, np.abs(ref - pred), -np.inf))]
    start = int(np.clip(round(peak - DETAIL_SECONDS / 2), 0, OVERVIEW_SECONDS - DETAIL_SECONDS))
    return start, start + DETAIL_SECONDS


def plot_figures(method, df, summary, examples, behavior, out, windows, rank_floor, bt_first):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'barlowtrack-trace-fidelity', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .65, 'xtick.major.size': 2.5,
                         'ytick.major.size': 2.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'axes.edgecolor': '#555e64', 'text.color': '#17252c', 'axes.labelcolor': '#17252c'})
    ref_color, bt_color = '#285a78', '#b74b22'
    styles = [dict(color=ref_color, lw=1.35), dict(color=bt_color, lw=.85, ls=(0, (3.0, 2.1)))]
    ink = '#244451'
    names = list(RECORDINGS)
    short = [RECORDINGS[n].split(' ·')[0] for n in names]

    def save(fig, key):
        fig.savefig(out / (key + '.svg'), metadata={'Date': None})
        fig.savefig(out / (key + '.png'), dpi=240)
        plt.close(fig)

    def panel_heading(fig, letter, title, x, y, gap=.03):
        fig.text(x, y, letter, fontsize=14, weight='bold', va='baseline')
        fig.text(x+gap, y, title, fontsize=10, weight='bold', va='baseline')

    is_barlow = method == 'BarlowTrack'
    basename = 'figure_' + method.lower()
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator
    fig = plt.figure(figsize=(9.6, 10.6))
    fig.text(.025, .973, f'{method} fluorescence traces against reference tracks', fontsize=13, weight='bold', va='top')
    shade = '#dedede'
    contact = '#cce6db'
    handles = [Line2D([], [], **style) for style in styles] + [Patch(fc=c, ec='none') for c in (shade,contact)]
    fig.legend(handles, ['Reference', method, 'Backward crawling (Zimmer, Leifer)', 'Ventral contact (Samuel)'],
               loc='upper left', bbox_to_anchor=(.018,.954), ncol=4, frameon=False,
               handlelength=2.6, columnspacing=2)
    blocks = fig.add_gridspec(len(names), 1, left=.248, right=.982, top=.888, bottom=.051,
                               hspace=.46, height_ratios=[len(examples[name]) for name in names])
    for i, name in enumerate(names):
        rate = VPS[name]
        rows = examples[name]
        _, ref10, _, _ = rows['10th percentile']
        t = np.arange(len(ref10)) / rate
        if t[-1] < OVERVIEW_SECONDS:
            raise ValueError(f'{name} ends before the common display interval')
        window = windows[name]
        zoom_color = '#7853a2'
        grid = blocks[i].subgridspec(len(rows), 2, width_ratios=[3.6,1], hspace=.05, wspace=.16)
        # Put the biological examples together above the two distribution landmarks.
        order = [key for key in rows if key.startswith('High SNR')] + ['Median', '10th percentile']
        for k, key in enumerate(order):
            row, ref, pred, _ = rows[key]
            for j, (xmin, xmax) in enumerate([(0, OVERVIEW_SECONDS), window]):
                # One adjacent sample at each boundary preserves segments crossing the edge.
                view = slice(max(0, np.searchsorted(t, xmin)-1), np.searchsorted(t, xmax, side='right')+1)
                ax = fig.add_subplot(grid[k,j])
                xv = t[view]
                for start, end in behavior[name]:
                    left, right = (start-.5)/rate, (end-.5)/rate
                    if right >= xv[0] and left <= xv[-1]:
                        ax.axvspan(left, right, color=contact if name == 'samuel' else shade, lw=0, zorder=-2)
                for y, style in zip([ref, pred], styles):
                    v = y[view]
                    ax.plot(xv, v, **style)
                    finite = np.isfinite(v)
                    isolated = finite & ~np.r_[False, finite[:-1]] & ~np.r_[finite[1:], False]
                    ax.plot(xv[isolated], v[isolated], '.', ms=2, color=style['color'])
                ax.set_xlim(xmin, xmax)
                ax.margins(y=.12)
                if j == 0:
                    ax.axvspan(*window, facecolor=zoom_color, alpha=.14, lw=0, zorder=-1)
                    for bound in window:
                        ax.axvline(bound, color=zoom_color, lw=.95)
                    if k in (0, len(rows)-1):
                        edge = 1 if k == 0 else 0
                        ax.plot(window, [edge,edge], transform=ax.get_xaxis_transform(),
                                color=zoom_color, lw=1.4, clip_on=False)
                lo, hi = ax.get_ylim()
                ticks = MaxNLocator(nbins=3, steps=[1,2,5,10]).tick_values(lo,hi)
                inside = ticks[(ticks > lo+.04*(hi-lo)) & (ticks < hi-.04*(hi-lo))]
                if len(inside) >= 2:
                    inside = inside[[0,-1]]
                ax.yaxis.set_major_locator(FixedLocator(inside))
                ax.tick_params(axis='y', labelsize=6.5, pad=2, length=2)
                last = k == len(rows)-1
                ax.tick_params(axis='x', bottom=last, labelbottom=last, pad=2)
                ax.spines['bottom'].set_visible(last)
                ax.spines['left'].set_color('#bac2c6')
                ax.set_xticks([0, 30, 60, 90, 120] if j == 0 else [xmin, (xmin+xmax)//2, xmax])
                box = ax.get_position()
                if k == 0:
                    if j == 0:
                        panel_heading(fig, 'abcd'[i], RECORDINGS[name], .025, box.y1+.009)
                    else:
                        fig.text(box.x0, box.y1+.009, f'Detail   {xmin}–{xmax} s', fontsize=8.5,
                                 color=zoom_color, weight='bold')
                if j == 0:
                    yc = (box.y1 + box.y0)/2
                    identity = (f'{row.anatomical_id}  ·  {row.neuron}' if pd.notna(row.anatomical_id)
                                else str(row.neuron))
                    selection = {'Median': 'median', '10th percentile': '10th pct.'}.get(key)
                    if selection:
                        identity += f'  ({selection})'
                    fig.text(.025, yc+.009, identity, fontsize=8.2, weight='bold', va='center')
                    values = (f'{row.raw_r:.3f}', f'{100*row.mask_accuracy:.1f}%', f'{100*row.coverage:.1f}%')
                    for x, value in zip((.025, .099, .169), values):
                        fig.text(x, yc-.007, value, fontsize=7.6, va='center')
    fig.text(.612,.012,'Elapsed time (s)', ha='center', fontsize=9)
    for x, label in zip((.025, .099, .169), ('Pearson r', 'Accuracy', 'Coverage')):
        fig.text(x, .916, label, weight='bold', fontsize=8, color=ink)
    fig.text(.248, .916, 'Raw green/red ratio   ·   first 120 s', weight='bold', fontsize=8.5, color=ink)
    save(fig, basename + '_overlays')

    fig = plt.figure(figsize=(9.4, 5.3))
    fig.text(.025, .973, f'{method} raw-ratio agreement across neurons', fontsize=13, weight='bold', va='top')
    first = df[df.replicate == 1]
    first_summary = summary[summary.replicate == 1].set_index('recording')
    grid = fig.add_gridspec(1, 4, left=.075, right=.984, bottom=.565, top=.822, wspace=.18)
    marks = {'Median': 'o', '10th percentile': '^'}
    for j, name in enumerate(names):
        ax = fig.add_subplot(grid[0,j])
        neurons = first[first.recording == name].dropna(subset=['raw_r']).sort_values(
            ['raw_r','neuron']).reset_index(drop=True)
        ranks = np.linspace(0, 100, len(neurons))
        ax.axhline(.9, color='#777', lw=.7, ls=(0,(2,3)), zorder=0)
        ax.plot(ranks, neurons.raw_r, lw=.8, color=ink, marker='.', markersize=2)
        for key, marker in marks.items():
            row = examples[name][key][0]
            k = neurons.index[neurons.neuron == row.neuron].item()
            ax.plot(ranks[k], neurons.raw_r[k], marker=marker, markersize=5.8, color=ink,
                    markerfacecolor=ink if marker == 'o' else 'white', clip_on=False)
        box = ax.get_position()
        panel_heading(fig, 'abcd'[j], short[j]+(' (GFP)' if name=='flavell' else ''), box.x0, box.y1+.02)
        ax.set(xlim=(-2,102), ylim=(rank_floor-.025,1.035), xticks=[0,50,100])
        if j == 0:
            ax.set_ylabel('Pearson r')
        else:
            ax.tick_params(labelleft=False)
        ax.spines['left'].set_visible(j == 0)
    fig.text(.526,.476,'Neuron percentile within full recording' + (' · first replicate' if is_barlow else ''), ha='center', fontsize=9)
    legend = [Line2D([],[],marker=m,ls='',color=ink,markerfacecolor=ink if m=='o' else 'white') for m in marks.values()]
    fig.legend(legend, list(marks), loc='upper right',
               bbox_to_anchor=(.994,.93), ncol=2, frameon=False, handlelength=1)
    ax = fig.add_axes([.108, .102 if is_barlow else .145, .273, .263 if is_barlow else .22])
    panel_heading(fig, 'e', 'Median of each replicate' if is_barlow else 'Correct mask assignments', .027, .391)
    for i,name in enumerate(names):
        group = summary[summary.recording == name].sort_values('replicate')
        if is_barlow:
            offsets = np.linspace(-.29,.29,len(group))
            for k,(med,offset) in enumerate(zip(group.raw_median,offsets)):
                ax.plot(med, i+offset, marker='o', ms=4.1, color=ink, mew=.85,
                        markerfacecolor=ink if k==0 else 'white', zorder=3)
        else:
            values = [100*group.mask_accuracy.item(), 100*bt_first.loc[name, 'mask_accuracy']]
            ax.plot(values,[i,i], color='#aab6bc', lw=1)
            ax.plot(values[0], i, 'o', color=bt_color, ms=4, label='fDNC' if i==0 else None)
            ax.plot(values[1], i, 'o', color=ink, mfc='white', ms=4, label='BarlowTrack' if i==0 else None)
    if not is_barlow:
        fig.legend(*ax.get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.108, .004),
                   ncol=2, frameon=False, fontsize=7.5, handlelength=1, borderaxespad=0)
    ax.set(yticks=range(4), yticklabels=short, ylim=(3.6,-.6),
           xlabel='Median Pearson r' if is_barlow else 'Observed reference frames correct (%)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.grid(axis='x', color='#dce2e5', lw=.6)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    table = fig.add_axes([.437,.087,.548,.277])
    table.axis('off')
    panel_heading(fig, 'f', 'First replicate' if is_barlow else 'Five-template vote', .437, .391)
    xs = [0,.23,.41,.63,.81,.99]
    headers = ['Lab','Median r','Q1–Q3','r < 0.9 / N','Coverage','Excl.']
    for k, (x, label) in enumerate(zip(xs, headers)):
        table.text(x,.89,label,fontsize=8,ha='left' if k==0 else 'right',weight='bold')
    table.axhline(.825,color=ink,lw=.7)
    for i,name in enumerate(names):
        s = first_summary.loc[name]
        cells = [short[i],f'{s.raw_median:.3f}',f'{s.raw_q25:.3f}–{s.raw_q75:.3f}',
                 f'{s.raw_n_below_0_9} / {s.raw_n_evaluable}',f'{100*s.coverage_median:.1f}%',
                 f'{s.raw_n_constant}/{s.raw_n_insufficient_overlap}']
        y = .68 - i*.19
        for k,(x,value) in enumerate(zip(xs,cells)):
            table.text(x,y,value,fontsize=8,ha='left' if k==0 else 'right')
    table.text(0,-.11,'Coverage = median across neurons.  Excl. = constant / overlap < 10.',
               fontsize=7.4)
    save(fig, basename + '_agreement')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round-dir', type=Path, default=DEFAULT_ROUND)
    parser.add_argument('--fdnc-dir', type=Path, default=Path.home() / '.cache/barlow_scratch/fdnc/r1m2_trial59')
    parser.add_argument('--out', type=Path, default=Path('docs/paper/trace_fidelity'))
    parser.add_argument('--cache', type=Path, help='Optional reusable directory for per-mask intensities')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--extract-only', action='store_true')
    return parser.parse_args()


def main():
    args = arguments()
    with (args.round_dir / 'exp_results_v2.jsonl').open() as fh:
        records = {rec['tag']: rec for rec in map(json.loads, fh)}
    if args.extract_only and not args.cache:
        raise ValueError('--extract-only requires --cache')
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
    methods = ('BarlowTrack', 'fDNC')
    results = {m: [] for m in methods}
    all_examples = {m: {} for m in methods}
    behavior, windows = {}, {}
    if not args.extract_only:
        paired = json.loads((args.fdnc_dir / 'paired_table_details.json').read_text())['recordings']
    for recording in RECORDINGS:
        files = track_files(args.round_dir, recording)
        rec = records[files[0].stem]
        source = sources(rec)
        gt = pd.read_hdf(files[0], 'gt')
        frames = gt.index.to_numpy(dtype='int64')
        np.testing.assert_array_equal(gt.index, np.arange(len(gt)))
        cache = args.cache / (recording + '.h5') if args.cache else None
        if cache and cache.exists():
            with pd.HDFStore(cache, 'r') as store:
                if json.loads(store.get_storer('objects').attrs.source) != source:
                    raise ValueError(f'Stale cache {cache}')
                objects = store['objects']
        else:
            print('Extracting', recording, len(frames), 'volumes', flush=True)
            objects = extract(source, frames, args.workers)
            if cache:
                tmp = cache.with_suffix('.tmp')
                with pd.HDFStore(tmp, 'w', complevel=1) as store:
                    store.put('objects', objects)
                    store.get_storer('objects').attrs.source = json.dumps(source)
                tmp.replace(cache)
        print(recording, len(objects), 'frame-local masks', flush=True)
        if not args.extract_only:
            runs = {'BarlowTrack': barlow_tracks(files, records, gt, source),
                    'fDNC': [fdnc_track(args.fdnc_dir, recording, paired[recording], gt, rec['emb_path'])]}
            behavior[recording] = behavior_intervals(recording, source['configuration'], len(gt))
            for method in methods:
                result, examples = analyze(recording, runs[method], gt, objects, source)
                results[method].append(result)
                all_examples[method][recording] = examples
            _, ref, pred, different = all_examples['BarlowTrack'][recording]['10th percentile']
            windows[recording] = disagreement_window(ref, pred, different, frames / VPS[recording])
    if args.extract_only:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    data = {m: pd.concat(results[m], ignore_index=True) for m in methods}
    summaries = {m: summarize(data[m]) for m in methods}
    rank_floor = np.floor(min(d.loc[d.replicate == 1, 'raw_r'].min() for d in data.values()) * 5) / 5
    bt_first = summaries['BarlowTrack'].query('replicate == 1').set_index('recording')
    for method in methods:
        prefix = '' if method == 'BarlowTrack' else 'fdnc_'
        data[method].to_csv(args.out / (prefix + 'per_neuron.csv'), index=False)
        summaries[method].to_csv(args.out / (prefix + 'per_replicate.csv'), index=False)
        plot_figures(method, data[method], summaries[method], all_examples[method],
                     behavior, args.out, windows, rank_floor, bt_first)


if __name__ == '__main__':
    main()
