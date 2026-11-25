"""Export a reviewer movie and a displacement-vector debug copy for each benchmark recording.

Run from the repository root using the wbfm Python. FFmpeg with libx264 is required.
Each nonempty pred_all cell indexes a detection within ONE volume. Only the column
name supplies the identity across volumes. See manifest.json for the coordinate joins.
"""
import argparse
import colorsys
import hashlib
import json
import math
import shlex
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import nrrd
import numpy as np
import pandas as pd
import yaml
import zarr
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import regionprops

from barlow_track.scripts.paper_scripts.revision.gt_positions import ROUNDS, VPS

SCRIPT = 'barlow_track/scripts/paper_scripts/revision/export_supplementary_videos.py'
ROOT = Path(ROUNDS).parent
# Window start in t1_segments.tsv, recording label and reason following image inspection.
CLIPS = {
    'zimmer_1123': (1277, 'Zimmer · 2022-11-23, worm 11',
                    'Highest mean novelty in Zimmer. The head turns within the field.'),
    'zimmer_1128': (485, 'Zimmer · 1128, worm 1',
                    'Second-highest mean novelty in this recording. Nuclei remain visible through the turn; '
                    'the head becomes indistinct during the highest-ranked window.'),
    'zimmer_1210': (1418, 'Zimmer · 2022-12-10, worm 1',
                    'Highest mean novelty in this recording. The head curls through the field before straightening.'),
    'flavell': (1494, 'Flavell · 2022-03-31-01',
                'Second-highest mean novelty. The head returns into view after nearing the crop edge; '
                'the follow-through of the highest-ranked window cuts off the head.'),
    'flavell_2022031603': (319, 'Flavell · 2022-03-16-03',
                          'Third-highest mean novelty in Flavell. The neuron group stays in view before and '
                          'after the bend. In the two higher windows on 2022-03-31-01, the head meets the image edge.'),
    'flavell_2022032402': (1331, 'Flavell · 2022-03-24-02',
                           'Highest mean novelty in this recording. The curved row of nuclei swings across the field.'),
    'flavell_2022033001': (112, 'Flavell · 2022-03-30-01',
                          'Highest mean novelty in this recording. The head arches and relaxes within the crop.'),
    'flavell_2022033002': (161, 'Flavell · 2022-03-30-02',
                          'Highest mean novelty in this recording. The direction of the bend changes in the XY view.'),
    'leifer_stardist_strict': (776, 'Leifer · NeRVE worm 1, StarDist',
                              'Highest mean novelty in Leifer. The head changes direction within the field.'),
    'samuel': (725, 'Samuel · recording 153',
               'Highest mean novelty in Samuel. The curved group of nuclei straightens and bends again.'),
    'samuel_137': (524, 'Samuel · recording 137',
                   'Second-highest mean novelty. The row of nuclei stays visible through the bend; '
                   'in the highest-ranked window the edge of the imaging field crosses the group.'),
    'samuel_184': (1208, 'Samuel · recording 184-004',
                   'Highest mean novelty in this recording. The bright group bends and recovers in the XY view.'),
}
PANEL = 900
MARGIN, GUTTER = 16, 14
HEADER = 106
BG, FG, MUTED = '#14191e', '#f4f6f6', '#bdc9d2'
PLAYBACK_FPS = 2  # one encoded frame per source volume
STATUS_KEY = ('correct', 'incorrect', 'no_ground_truth')


def region_centres(mask, intensity=None):
    props = regionprops(mask, intensity_image=intensity)
    coords = [p.centroid for p in props] if intensity is None else [p.weighted_centroid for p in props]
    return pd.DataFrame(coords, index=[p.label for p in props], columns=['z', 'x', 'y'])


def project_source(rec, info, stack):
    path = Path(rec['project'])
    cfg = yaml.safe_load(path.read_text())
    folder = path.parent
    pre = yaml.safe_load((folder / cfg['subfolder_configs']['preprocessing']).read_text())
    seg = yaml.safe_load((folder / cfg['subfolder_configs']['segmentation']).read_text())
    red = folder / (cfg.get('preprocessed_red') or pre['preprocessed_red_fname'])
    meta = folder / seg['output_metadata']
    masks = folder / seg['output_masks']
    store = stack.enter_context(zarr.ZipStore(str(red), mode='r'))
    images = zarr.open(store, mode='r')
    try:
        metadata = pd.read_pickle(meta)
    except EOFError:
        # Match wbfm SegmentationMetadata's fallback on Zimmer 1128.
        meta = meta.with_name('backup_metadata.pickle')
        metadata = pd.read_pickle(meta)
    mask = zarr.open(str(masks), mode='r')
    units = cfg['physical_units']
    info.update(image_source=str(red), detection_source=str(meta), mask_source=str(masks),
                id_join='pred_all value = metadata.label (raw_segmentation_id), metadata.centroids in ZXY',
                xy_um=units['zimmer_fluroscence_um_per_pixel_xy'], z_um=units['zimmer_um_per_pixel_z'],
                scale_source=str(path) + ' physical_units', image_axes='TZXY')

    def read(t, check):
        v = images[t]
        md = metadata[t].set_index('label')
        c = pd.DataFrame(md.centroids.tolist(), index=md.index, columns=['z', 'x', 'y'])
        if check is not None:
            seg_t = mask[t]
            independent = region_centres(seg_t, v).loc[c.index]
            distances = np.linalg.norm(c.values - independent.values, axis=1)
            check['metadata_vs_red_weighted_mask_max_centroid_distance_vox'] = float(max(distances))
            if max(distances) > 0.01:
                raise ValueError(f'Project mask/metadata frame or axes differ at {t}')
        return v, c
    return read


def nwb_source(rec, info, frames, stack):
    path = Path(rec['gt'])
    f = stack.enter_context(h5py.File(path, 'r'))
    activity = 'processing/CalciumActivity/'
    series = 'acquisition/CalciumImageSeries/'
    images = f[series + 'data']
    info.update(image_source=f'{path}::{series}data[..., 0]', image_axes='TXYZC, channel 0 (reference)',
                nwb_rate_hz=float(f[series + 'starting_time'].attrs['rate']))
    if rec['nwb_objects'] == 'all':
        seg_key = activity + 'CalciumSeriesSegmentationUntracked/data'
        masks = f[seg_key]
        original = path.parent / 'NRRD_cropped'
        info.update(detection_source=f'{path}::{seg_key}', id_join='pred_all value = per-frame watershed mask label',
                    scale_source=f'{original}/*_ch2.nrrd space directions (original source image)',
                    frame_numbering='Benchmark/NWB frame t corresponds to original NRRD t+1; both shown.',
                    rate_note='Nominal 1.7 volumes/s from benchmark T1 and Atanas et al. (2023). '
                              'The local NWB converter defaults to 1 Hz. No measured timestamps used.',
                    nwb_spacing_xyz=f['general/optophysiology/CalciumImVol/grid_spacing'][:].tolist(),
                    acquisition_reference='https://doi.org/10.1016/j.cell.2023.07.035')

        def read(t, check):
            xyz = images[t, ..., 0]
            mask = masks[t].transpose(2, 0, 1)
            if check is not None:
                raw, hdr = nrrd.read(str(next(original.glob(f'*_t{t+1:04d}_ch2.nrrd'))))
                np.testing.assert_array_equal(raw, xyz)
                original_mask, _ = nrrd.read(str(path.parent/'img_roi_watershed'/f'{t+1}.nrrd'))
                np.testing.assert_array_equal(original_mask.transpose(2, 0, 1), mask)
                xyz_scale = np.linalg.norm(hdr['space directions'], axis=1)
                np.testing.assert_allclose(xyz_scale, [0.54] * 3)
                info.update(xy_um=float(xyz_scale[0]), z_um=float(xyz_scale[2]))
                check['source_nrrd_image_and_mask_equal_voxelwise'] = t + 1
            return xyz.transpose(2, 0, 1), region_centres(mask)
        return read

    if not rec['lab'].startswith('samuel') or rec['nwb_objects'] != 'gt':
        raise ValueError('Unknown NWB object ID space')
    # The source stores each reference object's time series under a name. Recover
    # the frame-local detection lookup as in eval_accuracy._load_nwb_arrays.
    # Discard the source names before joining to the independent predicted columns.
    group = f[activity + 'NeuronCentroids']
    ids = f[activity + 'NeuronSegmentationID']
    sl = slice(frames.start, frames.stop)
    names = sorted(group.keys())
    for n in names:
        np.testing.assert_array_equal(group[n]['timestamps'][sl], frames)
    xyz = np.stack([group[n]['data'][sl] for n in names], axis=1)
    labels = np.stack([ids[n][sl] for n in names], axis=1)
    raw_path = path.with_suffix('.h5')
    if path.parent.name == 'thresholded_segmentation':
        raw_path = path.parent.parent/raw_path.name
    raw = stack.enter_context(h5py.File(raw_path, 'r'))
    spacing = f['general/optophysiology/CalciumImVol/grid_spacing'][:]
    info.update(detection_source=f'{path}::{activity}NeuronCentroids + NeuronSegmentationID',
                id_join='pred_all value = reference detection ID at t; NWB XYZ points are voxel indices '
                        'as used in evaluation, joined by the ID stored for that object at t',
                scale_source=f'{path}::general/optophysiology/CalciumImVol/grid_spacing',
                xy_um=float(spacing[0]), z_um=float(spacing[2]))
    np.testing.assert_allclose(info['nwb_rate_hz'], VPS[rec['lab']])
    if rec['lab'] == 'samuel_184':
        info.update(xy_um=None, scale_source='Native XY pixels in the reference-channel image',
                    scale_note='The NWB gives 0.30 µm/px in XY, whereas the raw H5 pixel_scale is 1:1:4 '
                               'and the reported z step is 1.75 µm. The displayed bar is 40 native XY pixels.',
                    nwb_spacing_xyz=spacing.tolist())

    def read(t, check):
        i = t - frames.start
        valid = np.isfinite(labels[i]) & np.isfinite(xyz[i]).all(1)
        c = pd.DataFrame(xyz[i, valid][:, [2, 0, 1]], columns=['z', 'x', 'y'], index=labels[i, valid])
        v = images[t, ..., 0]
        if check is not None:
            np.testing.assert_array_equal(raw[f'{t}/frame'][0], v)
            np.testing.assert_array_equal(raw['points'][t], xyz[i])
            pos = np.rint(c[['x', 'y', 'z']].values).astype(int)
            mask = f[activity + 'CalciumSeriesSegmentation/data'][t]
            check['fraction_reference_points_in_own_watershed_mask'] = (
                float(np.mean(mask[tuple(pos.T)] == c.index)) if len(pos) else None)
            check['source_h5_volume_and_points_equal'] = True
        return v.transpose(2, 0, 1), c
    return read


def scored_identities(pred, matched):
    """Recover the benchmark's fixed column renaming from full-length saved predictions."""
    if not pred.index.equals(matched.index) or not matched.columns.is_unique:
        raise ValueError('Ambiguous score table frame or identity labels')
    identities = {}
    for identity, series in matched.items():
        t = series.first_valid_index()
        if t is None:
            continue
        # Per-frame exclusivity makes one nonmissing detection sufficient to
        # locate a candidate column. Verify the entire series including gaps.
        candidates = pred.columns[pred.loc[t].eq(series.at[t])]
        if len(candidates) != 1:
            raise ValueError(f'Matched identity {identity} does not identify one predicted track')
        track = candidates[0]
        if track in identities or not pred[track].equals(series):
            raise ValueError(f'Inconsistent full-recording score mapping for {track}')
        identities[track] = identity
    return identities


def score_marks(points, identities, gt):
    """Annotate predicted points without changing their IDs, ordering or coordinates."""
    if not gt.is_unique:
        raise ValueError('Ground truth assigns two identities to one detection in a frame')
    at_detection = pd.Series(gt.index, index=gt.values, dtype=object)
    claimed = points.predicted_track_id.map(identities)
    actual = points.detection_id.map(at_detection)
    expected = claimed.map(gt)
    correct = points.detection_id.eq(expected)
    # Either endpoint can establish a contradiction: the claimed neuron is
    # annotated elsewhere, or this detection belongs to another reference ID.
    known = expected.notna() | actual.notna()
    points['matched_gt_identity'] = claimed
    points['gt_identity_at_detection'] = actual
    points['expected_detection_id'] = expected
    points['status'] = np.select([correct, known], STATUS_KEY[:2], default='no_ground_truth')


def volume_scores(reference, matched):
    """Benchmark accuracy uses every nonmissing reference entry, including misses."""
    correct = (reference.notna() & reference.eq(matched.reindex_like(reference))).sum(axis=1)
    annotated = reference.notna().sum(axis=1)
    return pd.DataFrame(dict(correct=correct, annotated=annotated,
                             accuracy=correct.div(annotated.where(annotated > 0))))


def add_vector_targets(points, detections):
    """Look up the current reference detection for the identity claimed at each cross."""
    ids = points.expected_detection_id.where(points.status == 'incorrect')
    target = detections.loc[:, ['z', 'x', 'y']].reindex(ids).to_numpy(dtype=float)
    if not np.isfinite(target[ids.notna()]).all():
        raise ValueError('A scored target detection has no coordinate in its volume')
    points[['target_z', 'target_x', 'target_y']] = target


def collect(rec, frames, checks, info):
    with np.load(rec['emb_path'], allow_pickle=True) as cache:
        embedded = defaultdict(set)
        for t, raw_ind, seg_id in cache['lin_to_t_seg'].item().values():
            if t in frames:
                embedded[t].add(seg_id if rec['source'] == 'project' else raw_ind)
    pred = pd.read_hdf(info['track_source'], 'pred_all')
    if not pred.columns.is_unique or not pred.index.is_unique:
        raise ValueError('Ambiguous track IDs or frame labels')
    matched = pd.read_hdf(info['track_source'], 'pred_matched')
    ground_truth = pd.read_hdf(info['track_source'], 'gt')
    if (not ground_truth.index.equals(pred.index) or not ground_truth.columns.is_unique or
            not matched.columns.isin(ground_truth.columns).all()):
        raise ValueError('The score tables lack a common frame or identity index')
    identities = scored_identities(pred, matched)
    reference = ground_truth.reindex(frames)
    scores = volume_scores(reference, matched)
    projections, rows = [], []
    with ExitStack() as stack:
        project = rec['source'] == 'project'
        read = project_source(rec, info, stack) if project else nwb_source(rec, info, frames, stack)
        for t in frames:
            check = checks.get(t)
            v, det = read(t, check)
            if not det.index.is_unique or v.ndim != 3:
                raise ValueError(f'Ambiguous detections or axes at frame {t}')
            assigned = pred.loc[t].dropna() if t in pred.index else pd.Series(dtype=float)
            if (not assigned.is_unique or not assigned.isin(embedded[t]).all()
                    or not np.array_equal(assigned, assigned.astype(int))):
                raise ValueError(f'Inconsistent pred_all/embedding IDs at frame {t}')
            pts = det.loc[assigned.values].copy()
            pts.insert(0, 'detection_id', assigned.astype(int).values)
            pts.insert(0, 'predicted_track_id', assigned.index)
            pts.insert(0, 'frame', t)
            score_marks(pts, identities, reference.loc[t].dropna())
            if (pts.status == 'correct').sum() != scores.at[t, 'correct']:
                raise ValueError(f'Correct markers and benchmark numerator disagree at frame {t}')
            add_vector_targets(pts, det)
            coords = pts[['z', 'x', 'y']].to_numpy(dtype=float)
            # Continuous centroids may fall between the last pixel centre and
            # its outer face. Validate voxel support without clipping the point.
            if not (np.isfinite(coords).all() and (coords >= -0.5).all() and (coords < np.array(v.shape)-0.5).all()):
                raise ValueError(f'Out-of-volume detection at frame {t}')
            rows.append(pts.reset_index(drop=True))
            top = v.max(axis=0)
            projections.append(top if project else top.T)
            if check is not None:
                pix = v[tuple(np.rint(coords).astype(int).T)]
                check.update(assigned=len(pts), detections=len(det), volume_zxy=list(v.shape),
                             median_intensity_at_mark=float(np.median(pix)) if len(pix) else None,
                             volume_background_median=float(np.median(v)))
    return projections, rows, scores


@lru_cache(None)
def font(size, bold=False):
    return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', size)


@lru_cache(None)
def colour(track):
    i = int(track.split('_')[-1])
    rgb = colorsys.hsv_to_rgb((i * 0.61803398875) % 1, 0.77, 1.0)
    return tuple(round(x * 255) for x in rgb)


def label_tracks(points):
    # Fixed, spatially separated examples from the first populated volume. No later
    # occupancy, correctness or displacement enters this choice.
    p = points[['x', 'y']].to_numpy()
    median = np.median(p, axis=0)
    dist = np.linalg.norm(p - median, axis=1)
    central = np.flatnonzero(dist <= np.percentile(dist, 72))
    chosen = [dist.argmin()]
    while len(chosen) < min(6, len(central)):
        d = np.linalg.norm(p[central, None] - p[chosen], axis=2).min(axis=1)
        if d.max() == 0:
            break
        chosen.append(central[d.argmax()])
    return points.iloc[chosen].predicted_track_id.tolist()


def draw_marker(draw, x, y, status, color, radius):
    if status == 'correct':
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=2)
    elif status == 'incorrect':
        r = radius + 2
        for width, ink in ((5, '#151820'), (3, color)):
            draw.line((x-r, y-r, x+r, y+r), fill=ink, width=width)
            draw.line((x-r, y+r, x+r, y-r), fill=ink, width=width)
    elif status == 'no_ground_truth':
        draw.ellipse((x-3, y-3, x+3, y+3), fill=color, outline='#151820')
    else:
        raise ValueError(f'Unknown marker status {status}')


def panel_xy(x, y, project, width, height, arr_shape):
    a, b = (y, x) if project else (x, y)
    return (round((a + .5) * width / arr_shape[1] - .5),
            round((b + .5) * height / arr_shape[0] - .5))


def draw_arrow(draw, start, tip, color):
    """Keep the tip at the actual projected target, including very short displacements."""
    x, y = tip
    dx, dy = x - start[0], y - start[1]
    length = math.hypot(dx, dy)
    if not length:
        return  # Pure depth displacement or subpixel XY separation.
    ux, uy = dx/length, dy/length
    head = min(10, length * .55)
    left = (x - head*ux - .48*head*uy, y - head*uy + .48*head*ux)
    right = (x - head*ux + .48*head*uy, y - head*uy - .48*head*ux)
    draw.line([start, tip], fill='#151820', width=4)
    draw.line([start, tip], fill=color, width=2)
    draw.polygon([tip, left, right], fill=color)


def draw_view(arr, points, labels, limits, project, debug):
    lo, hi = limits
    gray = (255 * np.clip((arr.astype('f4') - lo) / (hi - lo), 0, 1)**0.75).astype('u1')
    height = max(1, round(arr.shape[0] * PANEL / arr.shape[1]))
    base = Image.fromarray(gray).resize((PANEL, height), Image.Resampling.BILINEAR).convert('RGB')
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)
    marked = []
    # Paint crosses last so failures remain visible in crowded projections.
    for status in ('no_ground_truth', 'correct', 'incorrect'):
        for row in points[points.status == status].itertuples(index=False):
            x, y = panel_xy(row.x, row.y, project, PANEL, height, arr.shape)
            c = colour(row.predicted_track_id)
            draw_marker(draw, x, y, status, c, 4 if project else 5)
            if row.predicted_track_id in labels:
                marked.append((x, y, row.predicted_track_id, c))
    if debug:
        # Keep short vectors visible even when a target and a cross overlap.
        for row in points.loc[points.target_x.notna()].itertuples(index=False):
            origin = panel_xy(row.x, row.y, project, PANEL, height, arr.shape)
            target = panel_xy(row.target_x, row.target_y, project, PANEL, height, arr.shape)
            draw_arrow(draw, origin, target, colour(row.predicted_track_id))
    for x, y, track, c in marked:
        draw.text((min(PANEL-40, x+6), max(1, y-20)), track.split('_')[-1], font=font(15, True),
                  fill=c, stroke_width=2, stroke_fill='black')
    return base, overlay


def compose(lab, title, t, projection, points, labels, limits, info, score, debug):
    left, right = draw_view(projection, points, labels, limits, info['source'] == 'project', debug)
    bottom = HEADER + left.height
    h = bottom + 52 + (34 if debug else 0)
    canvas = Image.new('RGB', (PANEL*2+MARGIN*2+GUTTER, h+h%2), BG)
    d = ImageDraw.Draw(canvas)
    d.text((MARGIN, 12), title, font=font(30, True), fill=FG)
    rate = VPS[lab]
    frame = f'Frame {t}' + (f'  (NRRD {t+1})' if lab.startswith('flavell') else '')
    clock = f'{frame}     {t/rate:.1f} s from start     {info["playback_speed"]:.2g}× playback'
    d.text((canvas.width-MARGIN, 23), clock, font=font(23), anchor='ra', fill=FG)
    right_x = MARGIN + PANEL + GUTTER
    prediction_heading = 'BarlowTrack predictions' + (' · debug' if debug else '')
    if score.annotated:
        accuracy_text = f'Volume accuracy  {score.accuracy:.1%}  ({int(score.correct)}/{int(score.annotated)})'
    else:
        accuracy_text = 'Volume accuracy  N/A  (0 scored)'
    for x, heading, im in ((MARGIN, 'Reference channel', left), (right_x, prediction_heading, right)):
        d.text((x, 68), heading, font=font(22, True), fill=FG)
        canvas.paste(im, (x, HEADER))
    d.text((MARGIN+PANEL, 72), 'XY projection', font=font(17), anchor='ra', fill=MUTED)
    d.text((right_x+PANEL, 68), accuracy_text, font=font(21, True), anchor='ra', fill=FG)
    bar_pixels, bar_title = (20/info['xy_um'], '20 µm') if info['xy_um'] else (40, '40 px')
    bar = round(bar_pixels * PANEL/projection.shape[1])
    d.line((MARGIN, bottom+26, MARGIN+bar, bottom+26), fill=FG, width=4)
    d.text((MARGIN+bar+10, bottom+13), bar_title, font=font(20), fill=FG)
    d.text((MARGIN+PANEL, bottom+14), f'Acquired at {rate:g} volumes/s   ·   Playing at {PLAYBACK_FPS} volumes/s',
           font=font(18), fill=MUTED, anchor='ra')
    for offset, status, label in zip((9, 155, 318), STATUS_KEY, ('Correct', 'Incorrect', 'No ground truth')):
        x = right_x + offset
        draw_marker(d, x, bottom+26, status, FG, 6)
        d.text((x+15, bottom+13), label, font=font(19), fill=FG)
    d.text((canvas.width-MARGIN, bottom+14), 'Color = predicted track ID', anchor='ra', font=font(18), fill=MUTED)
    if debug:
        count = points.target_x.notna().sum()
        crosses = points.status.eq('incorrect').sum()
        d.text((right_x, bottom+49), 'X → reference position in this volume', font=font(18), fill=FG)
        d.text((right_x+PANEL, bottom+49), f'Known targets: {count}/{crosses} crosses',
               font=font(18), anchor='ra', fill=MUTED)
    return canvas


def encode(path, lab, title, frames, projections, rows, scores, info, debug=False):
    # A fixed display window for the whole clip, shared by both panels.
    samples_xy = np.stack([p[::3, ::3] for p in projections])
    limits = np.percentile(samples_xy, [18, 99.85]).tolist()
    anchor = next((pts for pts in rows if not pts.empty), None)
    info.update(intensity_limits=limits, display_gamma=0.75,
                numbered_tracks=label_tracks(anchor) if anchor is not None else [])
    fps = info['encoded_fps']
    proc = None
    samples = {}
    try:
        for t, projection, pts in zip(frames, projections, rows):
            img = compose(lab, title, t, projection, pts, info['numbered_tracks'], limits, info,
                          scores.loc[t], debug)
            if t in info['checks']:
                samples[t] = np.array(img)
            if proc is None:
                proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                         '-s', f'{img.width}x{img.height}', '-r', str(fps), '-i', '-',
                                         '-c:v', 'libx264', '-threads', '4', '-preset', 'fast',
                                         '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
                                        stdin=subprocess.PIPE)
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError(f'FFmpeg failed for {path}')
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
    probe = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                     'stream=codec_name,pix_fmt,width,height,avg_frame_rate,duration,nb_frames',
                                     '-of', 'json', str(path)])
    stream = json.loads(probe)['streams'][0]
    if int(stream['nb_frames']) != len(frames) or stream['avg_frame_rate'] != f'{fps}/1' or stream['codec_name'] != 'h264':
        raise ValueError(f'Encoded stream differs from scheduled volumes: {stream}')
    info['debug_encoded_stream' if debug else 'encoded_stream'] = stream
    for t, reference in samples.items():
        index = t-frames.start
        decoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '1', '-ss', f'{index/fps:.8f}',
                                           '-i', str(path), '-frames:v', '1', '-threads', '1', '-f', 'rawvideo',
                                           '-pix_fmt', 'rgb24', '-'])
        pixels = np.frombuffer(decoded, np.uint8).reshape(reference.shape)
        diff = np.abs(pixels.astype('i2') - reference)
        info['checks'][t]['encoded_frame'] = index
        key = 'debug_mean_absolute_rgb_error' if debug else 'mean_absolute_rgb_error'
        info['checks'][t][key] = float(diff.mean())
        if diff.mean() > 5:
            raise ValueError(f'Encoded frame mismatch for {lab}, frame {t}')


def export(lab, rec, cand, anal, output):
    start, title, reason = CLIPS[lab]
    win = cand[(cand.lab == lab) & (cand.frame_start == start)].iloc[0]
    end = int(win.frame_end)
    count = round(20*VPS[lab])
    first = start - (count-(end-start+1))//2
    frames = range(first, first+count)
    scores = pd.read_csv(anal/f't1_frames_{lab}.tsv', sep='\t', usecols=['frame', 'd_lag1']).set_index('frame')
    peak = int(scores.loc[start:end].d_lag1.idxmax())
    checks = {t: {} for t in sorted({first, peak, peak+1, frames[-1]})}
    round_dir = anal.parent/'config_files/round_shared_v1'
    tracks = round_dir/'tracks_v2'/(rec['tag']+'.h5')
    assert rec['labelprop'] == 'v2' and rec['cluster'] == 'labelprop' and Path(rec['weights']).parent.parent.name == 'trial_59'
    family = lab.split('_')[0]
    project = rec['source'] == 'project'
    axes = 'XY (horizontal Y, vertical X)' if project else 'XY (horizontal X, vertical Y)'
    caption = (f'{title}. Reference-channel fluorescence is shown at left and trial-59 predictions at right '
               f'for volumes {first}–{frames[-1]}, including the high-deformation window {start}–{end}. ')
    extra = {'zimmer': 'The XY view shows the head turning.',
             'flavell': 'The head bends within the original cropped field. Both NWB and NRRD frame numbers are shown.',
             'leifer': 'Markers follow the associations assigned to StarDist detections during the turn.',
             'samuel': 'Training, evaluation and visualization all use reference detections. The displayed associations are BarlowTrack predictions.'}
    info = dict(recording=title, benchmark_id=lab, frame_range_inclusive=[first, frames[-1]],
                frame_numbering='Zero-based benchmark volume index, identical to image and prediction frame.',
                window_inclusive=[start,end], d_novel_mean=float(win.d_novel_mean), selection_reason=reason,
                clip_acquisition_seconds=count/VPS[lab], acquisition_volumes_per_second=VPS[lab],
                time_source='T1 benchmark nominal volume rate; elapsed time from recording frame zero',
                playback_speed=PLAYBACK_FPS/VPS[lab], encoded_fps=PLAYBACK_FPS,
                playback_volumes_per_second=PLAYBACK_FPS, playback_seconds=count/PLAYBACK_FPS,
                projection='Fixed maximum-intensity projection, full image field. '+axes,
                source=rec['source'], score_record=rec['tag'], checkpoint=rec['weights'],
                training_project=rec['train_config']['project_path'], embedding_map=rec['emb_path']+'::lin_to_t_seg',
                track_source=str(tracks), track_key='pred_all', status_keys=['pred_matched', 'gt'], ground_truth_source=rec['gt'], checks=checks,
                peak_movement_frame=peak, peak_definition='Largest d_lag1(t) within the T1 window, transition t to t+1.',
                file=f'{lab}.mp4', debug_file=f'debug/{lab}_vectors.mp4', caption=caption+extra[family])
    if family == 'leifer':
        info['caption'] += ' The accuracy denominator is the number of reference clicks assigned to StarDist detections in that volume.'
    if lab == 'samuel_184':
        info['caption'] += ' The scale bar is in native image pixels because the lateral calibration is uncertain.'
    projections, rows, scores = collect(rec, frames, checks, info)
    empty = [t for t, pts in zip(frames, rows) if pts.empty]
    if empty:
        info['frames_without_assignments'] = empty
        info['caption'] += f' {len(empty)} volumes in this clip have no predicted assignments and show no markers.'
    encode(output/info['file'], lab, title, frames, projections, rows, scores, info)
    encode(output/info['debug_file'], lab, title, frames, projections, rows, scores, info, debug=True)
    trace = pd.concat(rows, ignore_index=True)
    trace.insert(0, 'recording', lab)
    info['n_drawn_assignments'] = len(trace)
    info['marker_counts'] = {status: int((trace.status == status).sum()) for status in STATUS_KEY}
    info['crosses_with_known_target'] = int(trace.target_x.notna().sum())
    scores.index.name = 'frame'
    scores.insert(0, 'recording', lab)
    print(f"{lab}: {count} volumes, {len(trace)} assignments, {info['encoded_stream']['duration']} s", flush=True)
    return info, trace, scores.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--revision-dir', type=Path, default=ROOT)
    ap.add_argument('--out', type=Path, help='Default: REVISION_DIR/revision_analysis/supplementary_videos')
    ap.add_argument('--jobs', type=int, choices=range(1, 5), default=1, help='Recordings to render concurrently')
    args = ap.parse_args()
    analysis = args.revision_dir.resolve()/'revision_analysis'
    out = args.out or analysis/'supplementary_videos'
    out.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, SCRIPT, '--revision-dir', str(analysis.parent)])
    # Select by deformation and image visibility alone. Never load accuracy columns here.
    cand = pd.read_csv(analysis/'t1_segments.tsv', sep='\t', usecols=['lab','frame_start','frame_end','d_novel_mean'])
    if set(CLIPS) != set(cand.lab):
        raise ValueError('Clip selection must cover every recording in t1_segments.tsv')
    index = analysis.parent/'config_files/round_shared_v1/exp_results_v2.jsonl'
    with index.open() as f:
        records = {r['tag']: r for r in map(json.loads, f)}
    manifest = dict(export_command=command, working_directory='barlow_track repository root',
                    export_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    selection_source=str(analysis/'t1_segments.tsv'), score_records=str(index),
                    selection_metric='20-volume mean of d_novel, the smallest median non-rigid residual to a '
                                     'sampled reference posture after a similarity fit, in median detection '
                                     'nearest-neighbour distances. Larger values indicate more novel configurations.',
                    overlay='All nonmissing pred_all cells, no filtering, interpolation or gap filling. '
                            'Column name determines marker color and numeric label. Each frame is joined independently.',
                    marker_status='Circle: correct, the assigned detection equals the reference for the scored '
                                  'identity of this track. Cross: incorrect, the matched identity is annotated '
                                  'elsewhere or the marked detection belongs to a different annotated identity. '
                                  'Dot: no ground truth at either endpoint in this volume. Only predictions are drawn. '
                                  'Counts describe plotted assignments; benchmark accuracy also counts misses.',
                    traces='assignments.tsv.gz gives the frame, predicted track, detection ID, ZXY voxel coordinates, '
                           'matched reference identity, annotated identity at the detection, expected ID, status and '
                           'same-frame target ZXY coordinates for incorrect assignments with known destinations.',
                    volume_accuracy='volume_accuracy.tsv contains correct / annotated using pred_matched and gt from '
                                    'the scored trial-59 recording. Missing predictions count as misses. Volumes with '
                                    'no scored annotations display N/A.',
                    debug_vectors='Arrows connect an incorrect predicted detection to the detection assigned to its '
                                  'matched reference identity in the same volume. Unresolved crosses have no arrow. '
                                  'Vectors are projected into XY at image scale; depth-only errors can have zero length.',
                    recordings=[])
    traces = []
    score_tables = []
    # Publish only complete, probed movies and the matching trace/manifest set.
    with TemporaryDirectory(dir=out) as tmp:
        stage = Path(tmp)
        (stage/'debug').mkdir()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            jobs = [pool.submit(export, lab, records['round_shared_v1_trial59_'+lab], cand, analysis, stage)
                    for lab in CLIPS]
            for job in jobs:
                info, trace, scores = job.result()
                manifest['recordings'].append(info)
                traces.append(trace)
                score_tables.append(scores)
        pd.concat(traces, ignore_index=True).to_csv(stage/'assignments.tsv.gz', sep='\t', index=False)
        pd.concat(score_tables, ignore_index=True).to_csv(stage/'volume_accuracy.tsv', sep='\t', index=False,
                                                         na_rep='NA')
        (stage/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        (stage/Path(SCRIPT).name).write_bytes(Path(__file__).read_bytes())
        (stage/'captions.txt').write_text('\n\n'.join(i['caption'] for i in manifest['recordings'])+
            f'\n\nAll movies show XY maximum-intensity projections at {PLAYBACK_FPS} source volumes per playback second. '
            'Circles mark correct assignments, crosses mark incorrect assignments, and dots mark predictions '
            'with no ground truth. '
            'Colors and numbers follow predicted track IDs within each movie. Missing assignments leave gaps. '
            'Volume accuracy is the fraction of scored reference identities assigned to the correct detection; '
            'it includes missed predictions in the denominator. N/A means there is no scored ground truth. '
            'Debug copies add same-volume XY arrows from crosses to their known reference detections. '
            'Maximum-intensity projections can superimpose neurons at different depths.\n')
        (out/'debug').mkdir(exist_ok=True)
        for path in stage.rglob('*'):
            if path.is_file():
                path.replace(out/path.relative_to(stage))


if __name__ == '__main__':
    main()
