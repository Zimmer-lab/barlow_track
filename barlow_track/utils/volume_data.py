"""Step 1 data pipeline: augment the FULL volume first, then extract crops.

Legacy path (`barlow_lightning.get_crops_from_project`) applies affine/elastic
transforms to already-tight (e.g. 8x64x64) crops, pulling zero-padding into the
borders (Step 0 baseline: ~7% exact-zero voxels). Here geometric augmentation
(affine about the volume center) is applied to the whole volume AND to the
centroid coordinates with the same matrix, so subsequently extracted crops see
real neighborhood instead of padding.

Only photometric transforms (blur/noise/intensity) are applied per-crop after
extraction; those are position-agnostic.

Legacy classes are untouched; use `VolumeCoordsDataModule` to opt in.
"""
import contextlib
import logging
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchio as tio
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from tqdm.auto import tqdm

from barlow_track.utils.data_loading import get_3d_crop_using_bbox_or_centroid
from barlow_track.utils.superglue import normalize_keypoints
from barlow_track.utils.utils_seeding import TORCH_SEED_BITS, derive_seed


DEFAULT_GLOBAL_ARGS = dict(
    p_global_affine=1.0,
    max_degrees_z=180.0,
    scale_jitter=0.1,
    max_translation=(2, 8, 8),  # (z, x, y) voxels
    p_flip=0.0,  # extra 180 deg in-plane flip; 180 deg range already covers it
    # Smooth non-rigid deformation of volume AND points (one field per view).
    # ViSNet position features ignore rotation/translation, so without this the
    # two views' layouts differ only by scale, jitter and dropout. On flavell,
    # elastic_std (1, 4, 4), elastic_length 40 with position jitter_std 0.3
    # matches how much neighbor distances change between real frames, from
    # 8 to 140 voxels (scratch/local_geometry_change.py); jitter_std 1.0 alone
    # over-perturbs close pairs and under-perturbs distant ones. Strong short
    # fields (e.g. std 3 at length 15) fold.
    elastic_std=0.0,  # voxels, per-axis std of the field; float or (z, x, y). 0 = off
    elastic_length=40.0,  # voxels, distance at which the field's correlation is 1/e
    # Multiplies elastic_std: a scalar knob for sweeps (Ax choices must be
    # scalars, elastic_std is per axis). 0 = off.
    elastic_scale=1.0,
    # False moves only the keypoints (affine, elastic, jitter); crops come from
    # the untransformed volume at the original centroids, so image views differ
    # only by the per-crop transform (crop_augment). Decouples position
    # augmentation from what the image branch sees.
    apply_to_volume=True,
)
ELASTIC_KEYS = ('elastic_std', 'elastic_length', 'elastic_scale')
GLOBAL_FLAG_KEYS = ('apply_to_volume',)

DEFAULT_CROP_PHOTOMETRIC_ARGS = dict(
    p_blur=0.0,
    p_noise=0.1,
    std_noise=0.25,
    # Per-crop rotation about each crop's own centre and z axis, as the legacy
    # crop pipeline does (one angle per view). With the global rotation reduced
    # (max_degrees_z), image features still learn rotation invariance without
    # rotating a long, narrow volume (flavell 284 x 120) and losing or
    # off-centring the objects near its edges. 0 = off.
    crop_degrees_z=0.0,
)

DEFAULT_POSITION_ARGS = dict(
    jitter_std=0.0,  # voxels; float (isotropic), (z, x, y), or 'auto' (from GT-free dataset stats)
    dropout_p=0.0,  # per-object independent drop prob, per view
    min_keep=2,  # always keep at least this many objects per view (if available)
    # Synthetic segmentation errors, per view, before the global transform (all off
    # by default). Synthetic detections carry id -1, so the loss never pairs them;
    # they are context for the real ones. Real Zimmer frames hold ~20% detections
    # without a GT name and miss ~14% of the GT neurons.
    spurious_p=0.0,  # expected extra detections per object, each near a random object
    spurious_std=(1.0, 5.0, 5.0),  # voxels (z, x, y): offset of a spurious point from its object
    merge_p=0.0,  # per-object prob of merging with its nearest neighbour into the midpoint
    split_p=0.0,  # per-object prob of splitting into two fragments
    split_dist=(1.0, 4.0, 4.0),  # voxels (z, x, y): distance between the two fragments
)


def build_photometric_transform(photometric_args=None):
    """Shared per-crop photometric transform (blur/noise/intensity).

    Used by `VolumeCoordsDataset` during training and by the napari
    augmentation viewer, so the two cannot drift apart.
    """
    photo = {**DEFAULT_CROP_PHOTOMETRIC_ARGS, **(photometric_args or {})}
    rotation = []
    if photo['crop_degrees_z']:
        # (N, Z, X, Y) tensor: torchio's first spatial axis is z, as in barlow.Transform
        rotation = [tio.RandomAffine(degrees=(float(photo['crop_degrees_z']), 0, 0), scales=0)]
    return tio.Compose(rotation + [
        tio.RandomBlur(p=photo['p_blur']),
        tio.RandomNoise(std=photo['std_noise'], p=photo['p_noise']),
        # (5, 99.5): the max (100) lets one hot voxel compress contrast for
        # the whole frame; must stay in sync with all inference normalizers.
        tio.RescaleIntensity(percentiles=(5, 99.5)),
    ])


def sample_global_affine(rng, p_global_affine=1.0, max_degrees_z=180.0, scale_jitter=0.1,
                         max_translation=(2, 8, 8), p_flip=0.0):
    """Sample a forward (z, x, y) transform: x' = R(x - C) + C + t.

    Returns (R (3,3), t (3,)). Identity when the Bernoulli(p_global_affine) draw fails.
    """
    if rng.random() > p_global_affine:
        return np.eye(3), np.zeros(3)
    angle = rng.uniform(-max_degrees_z, max_degrees_z)
    if rng.random() < p_flip:
        angle += 180.0
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    scale = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    R = scale * np.array([[1, 0, 0],
                          [0, c, -s],
                          [0, s, c]])
    t = np.array([rng.uniform(-m, m) for m in max_translation])
    return R, t


def apply_stack_affine(crops, R, t, order=1):
    """Apply one forward map x' = R(x - C) + C + t to a (N, Z, X, Y) crop stack.

    Same convention as apply_global_affine (rotation about the crop center,
    border-extended like ndimage mode='nearest'), but batched over the stack
    with grid_sample so the legacy image-only path gets view-consistent
    ("global") geometry without needing the full volume. Returns same
    shape/dtype/device as the input.
    """
    import torch.nn.functional as F
    N, Z, X, Y = crops.shape
    dev = crops.device
    f64 = torch.float64
    Rm = torch.as_tensor(np.asarray(R, dtype=np.float64), device=dev, dtype=f64)
    tm = torch.as_tensor(np.asarray(t, dtype=np.float64), device=dev, dtype=f64)
    # grid_sample orders the trailing grid dim as (x, y, z) = (W, H, D), but
    # R/t are in (z, x, y); permute everything to (y, x, z) to match.
    P = torch.tensor([2, 1, 0], device=dev)
    Rinv = torch.linalg.inv(Rm)[P][:, P]
    tm = tm[P]
    C = torch.tensor([(Z - 1) / 2.0, (X - 1) / 2.0, (Y - 1) / 2.0], device=dev, dtype=f64)[P]
    s = torch.tensor([Z, X, Y], device=dev, dtype=f64)[P]
    one = torch.ones(3, device=dev, dtype=f64)
    off = C - Rinv @ (C + tm)
    # Normalized grid: un = (2x + 1) / s - 1  <=>  x = ((un + 1) s - 1) / 2,
    # with x_in = Rinv x_out + off.
    A = Rinv * s[None, :] / s[:, None]
    b = (Rinv @ (s - one) + 2.0 * off + one) / s - one
    theta = torch.cat([A, b[:, None]], dim=1).to(crops.dtype).expand(N, 3, 4)
    grid = F.affine_grid(theta, (N, 1, Z, X, Y), align_corners=False)
    out = F.grid_sample(crops.unsqueeze(1).to(crops.dtype), grid,
                        mode='bilinear' if order == 1 else 'nearest',
                        padding_mode='border', align_corners=False)
    return out.squeeze(1)


def split_global_args(global_args):
    """global_augment dict -> (affine kwargs for sample_global_affine, elastic kwargs).

    Flags (GLOBAL_FLAG_KEYS) are dropped; callers read them from the dict.
    """
    args = dict(global_args or {})
    elastic = {k: args.pop(k) for k in ELASTIC_KEYS if k in args}
    for k in GLOBAL_FLAG_KEYS:
        args.pop(k, None)
    return args, elastic


class ElasticField:
    """Backward displacement field v (voxels, (z, x, y)) on a coarse grid of spacing `step`.

    Output voxel y of the warp shows the input at y + v(y), so the volume warp
    is exact by construction; a point p moves to the y solving y + v(y) = p,
    found per point by fixed-point iteration (forward_points). The field is
    smooth on the scale of `step`; linear interpolation gives it anywhere.
    """

    def __init__(self, coarse, step):
        self.coarse = np.asarray(coarse, dtype=np.float32)  # (3, cz, cx, cy)
        self.step = np.asarray(step, dtype=float)  # (3,)

    def at(self, coords):
        """v at (3, ...) voxel coordinates -> (3, ...)."""
        coords = np.asarray(coords, dtype=np.float32)
        c = coords / self.step.reshape((3,) + (1,) * (coords.ndim - 1))
        return np.stack([ndimage.map_coordinates(f, c, order=1, mode='nearest') for f in self.coarse])

    def forward_points(self, points_zxy, iterations=50):
        """(N, 3) input points -> (N, 3) output positions y with y + v(y) = p.

        y <- p - v(y) contracts where |grad v| < 1, i.e. where the warp does not
        fold; cheap because it only runs on the N points.
        """
        p = np.asarray(points_zxy, dtype=float).T
        y = p - self.at(p)
        for _ in range(iterations):
            y = p - self.at(y)
        return y.T


def sample_elastic_field(rng, shape, elastic_std=0.0, elastic_length=15.0, elastic_scale=1.0):
    """Smooth random ElasticField over a (D, H, W) volume, or None when off.

    Gaussian-filtered white noise: with filter sigma = elastic_length / 2 the
    field's spatial correlation is exp(-d^2 / elastic_length^2), i.e. 1/e at
    elastic_length. Each component is rescaled to std elastic_std (per axis).
    Draws nothing from rng when off, so disabled configs keep their streams.
    The grid spacing is ~sigma/2 voxels, fine enough for a field this smooth.
    """
    std = _parse_jitter_std(elastic_std) * float(elastic_scale)
    if bool((std == 0).all()):
        return None
    length = _parse_jitter_std(elastic_length)
    shape = np.asarray(shape)
    sigma = length / 2.0
    step = np.maximum(1, np.floor(sigma / 2.0)).astype(int)
    coarse_shape = tuple(np.ceil((shape - 1) / step).astype(int) + 1)
    coarse = np.empty((3,) + coarse_shape, dtype=np.float32)
    for a in range(3):
        smooth = ndimage.gaussian_filter(rng.normal(size=coarse_shape), sigma=sigma / step, mode='reflect')
        coarse[a] = smooth - smooth.mean()
    field = ElasticField(coarse, step)
    # Normalize the std of the interpolated field (linear interpolation between
    # nodes lowers it a few percent), measured at fixed off-grid probe points
    # so the caller's rng stream is not touched.
    probes = np.random.RandomState(0).uniform(0, 1, size=(3, 4096)) * (shape - 1)[:, None]
    field.coarse *= (std / (field.at(probes).std(1) + 1e-12)).astype(np.float32)[:, None, None, None]
    return field


def apply_global_affine(volume, points_zxy, R, t, order=1, field=None):
    """Apply forward transform x' = R(x - C) + C + t to volume and points.

    volume: (D, H, W); points_zxy: (N, 3). Returns (augmented volume, moved points).
    With an ElasticField v (see sample_elastic_field), output voxel y shows the
    affine image at y + v(y), and points move on accordingly.
    """
    center = (np.array(volume.shape) - 1) / 2.0
    R_inv = np.linalg.inv(R)
    pts_aug = transform_points(points_zxy, volume.shape, R, t)
    if field is None:
        offset = center - R_inv @ (center + t)
        vol_aug = ndimage.affine_transform(np.asarray(volume, dtype=np.float32),
                                           R_inv, offset=offset, order=order, mode='nearest')
        return vol_aug, pts_aug
    # Output voxel y shows the affine image at w = y + v(y), i.e. the input
    # voxel R^-1 (w - C - t) + C: one resampling for both maps.
    y = np.indices(volume.shape, dtype=np.float32)
    w = y + field.at(y)
    src = np.tensordot(R_inv.astype(np.float32), w - (center + t).astype(np.float32)[:, None, None, None],
                       axes=1) + center.astype(np.float32)[:, None, None, None]
    vol_aug = ndimage.map_coordinates(np.asarray(volume, dtype=np.float32), src, order=order, mode='nearest')
    if len(pts_aug):
        pts_aug = field.forward_points(pts_aug)
    return vol_aug, pts_aug


def transform_points(points_zxy, vol_shape, R, t, field=None):
    """The point half of apply_global_affine, without resampling a volume."""
    center = (np.array(vol_shape) - 1) / 2.0
    pts_aug = (np.asarray(points_zxy, dtype=float) - center) @ R.T + center + t
    if field is not None and len(pts_aug):
        pts_aug = field.forward_points(pts_aug)
    return pts_aug


def _parse_jitter_std(jitter_std):
    """Float or (z, x, y) -> (3,) float array."""
    arr = np.asarray(jitter_std, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(3, float(arr[0]))
    if arr.size == 3:
        return arr
    raise ValueError(f"jitter_std must be a float or (z, x, y), got {jitter_std!r}")


def apply_position_jitter(points_zxy, rng, jitter_std=0.0):
    """Independent Gaussian jitter per point (voxel units).

    Applied AFTER the global affine so each view gets its own centroid noise.
    Callers use the jittered points for BOTH crop extraction and keypoints,
    keeping crops and position channel consistent within a view.
    """
    std = _parse_jitter_std(jitter_std)
    if len(points_zxy) == 0 or bool((std == 0).all()):
        return np.asarray(points_zxy, dtype=float).copy()
    noise = rng.normal(0.0, 1.0, size=np.shape(points_zxy)) * std
    return np.asarray(points_zxy, dtype=float) + noise


def sample_dropout_keep(n, rng, dropout_p=0.0, min_keep=2):
    """Independent per-object keep mask; returns (bool mask (N,), keep_idx (K,)).

    Guarantees at least min(n, min_keep) survivors so tiny volumes stay usable.
    """
    n = int(n)
    if n == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=int)
    if not dropout_p or dropout_p <= 0.0:
        return np.ones((n,), dtype=bool), np.arange(n, dtype=int)
    if dropout_p >= 1.0:
        keep = np.zeros((n,), dtype=bool)
    else:
        keep = rng.random(n) >= dropout_p
    need = min(n, int(min_keep))
    if keep.sum() < need:
        # Top-up with random non-kept indices (rng-driven, reproducible)
        candidates = np.where(~keep)[0]
        rng.shuffle(candidates)
        keep[candidates[:need - int(keep.sum())]] = True
    return keep, np.where(keep)[0].astype(int)


def apply_detection_errors(points_zxy, rng, spurious_p=0.0, spurious_std=(1.0, 5.0, 5.0),
                           merge_p=0.0, split_p=0.0, split_dist=(1.0, 4.0, 4.0)):
    """Synthetic segmentation errors on one view's detections; returns (points (M, 3), ids (M,)).

    ids are the original row of each surviving real point, or -1 for synthetic ones:
    the midpoint of a merged pair, the two fragments of a split, and spurious points
    (each a Gaussian offset from a random object). Merges pair an object with its
    nearest neighbour, skipping objects already merged. Draws no random numbers when
    every probability is 0, so existing runs reproduce.
    """
    pts = np.asarray(points_zxy, dtype=float)
    n = len(pts)
    ids = np.arange(n, dtype=int)
    if n < 2 or not (merge_p > 0 or split_p > 0 or spurious_p > 0):
        return pts.copy(), ids
    used = np.zeros(n, dtype=bool)
    extra = []
    if merge_p > 0:
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        nearest = d.argmin(1)
        for i in rng.permutation(np.where(rng.random(n) < merge_p)[0]):
            j = nearest[i]
            if used[i] or used[j]:
                continue
            used[i] = used[j] = True
            extra.append((pts[i] + pts[j]) / 2)
    if split_p > 0:
        half = _parse_jitter_std(split_dist) / 2
        for i in np.where(rng.random(n) < split_p)[0]:
            if used[i]:
                continue
            used[i] = True
            u = rng.normal(size=3)
            u /= max(np.linalg.norm(u), 1e-12)
            extra += [pts[i] + u * half, pts[i] - u * half]
    if spurious_p > 0:
        k = rng.binomial(n, min(float(spurious_p), 1.0))
        if k:
            extra += list(pts[rng.randint(n, size=k)] + rng.normal(size=(k, 3)) * _parse_jitter_std(spurious_std))
    out = np.concatenate([pts[~used], np.reshape(extra, (-1, 3))], 0)
    return out, np.concatenate([ids[~used], np.full(len(extra), -1, dtype=int)])


def _centroids_from_metadata_table(project_data, t):
    """get_centroids_for_volume read straight from the per-frame metadata table, or None.

    wbfm's get_all_neuron_metadata_for_single_time evaluates volumes_with_no_neurons, a
    property that scans every frame, on each call (2.3 s a frame on Zimmer, so setup's
    scan of all frames was quadratic). The centroids and labels are columns of
    segmentation_metadata[t]; this returns the same arrays. None (use the wbfm call)
    when the table is not in that form or has no frame t.
    """
    table = getattr(project_data.segmentation_metadata, 'segmentation_metadata', None)
    if not isinstance(table, dict) or t not in table:
        return None
    df = table[t]
    if not isinstance(df, pd.DataFrame) or not {'centroids', 'label'} <= set(df.columns):
        return None
    if len(df) == 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)
    zxy = np.array(df['centroids'].values.tolist(), dtype=float)
    if zxy.ndim > 2:  # multiple channels: wbfm averages them
        zxy = zxy.mean(axis=-1)
    seg_ids = df['label'].to_numpy()
    ok = ~np.isnan(zxy).any(axis=1)
    return zxy[ok], seg_ids[ok].astype(int)


def get_centroids_for_volume(project_data, t):
    """Centroids (z, x, y) + raw segmentation ids for one timepoint.

    Same metadata source as `get_bbox_data_for_volume_with_label`, but without
    track filtering: the new pipeline needs ALL detections plus positions.

    Volumes with no detections yield empty arrays (wbfm returns ([], []));
    callers treat these as skippable, mirroring the legacy empty-volume path.
    """
    fast = _centroids_from_metadata_table(project_data, t)
    if fast is not None:
        return fast
    row_data, column_names = project_data.segmentation_metadata.get_all_neuron_metadata_for_single_time(
        t, as_dataframe=False)
    if len(column_names) == 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)
    mdata = pd.DataFrame(dict(zip(column_names, row_data)))
    mdata = mdata.dropna(subset=['z', 'x', 'y'])
    zxy = mdata[['z', 'x', 'y']].to_numpy(dtype=float)
    seg_ids = mdata['raw_segmentation_id'].to_numpy(dtype=int)
    return zxy, seg_ids


def load_volume(project_data, t):
    vol = project_data.red_data[t, ...]
    if hasattr(vol, 'compute'):
        vol = vol.compute()
    return np.asarray(vol, dtype=np.float32)


def in_bounds_mask(points_zxy, vol_shape, rounded=False):
    """Bool mask of points whose centroid lies inside the volume.

    Non-square (H, W) volumes rotate points completely outside on large
    z-rotations; their clipped crops would be nonsensical edge duplicates
    with out-of-range keypoints, so callers drop them (keeping original
    indices for intersection alignment).

    rounded tests the rounded centroid, which is what extract_crops crops
    around: without it a point within half a voxel of the far edge (y 119.6
    in a 120-wide volume) is kept and then gets a zero placeholder crop.
    """
    pts = np.asarray(points_zxy, dtype=float)
    if pts.size == 0:
        return np.zeros((0,), dtype=bool)
    if rounded:
        pts = np.round(pts)
    shape = np.asarray(vol_shape, dtype=float)
    return ((pts[:, 0] >= 0) & (pts[:, 0] < shape[0]) &
            (pts[:, 1] >= 0) & (pts[:, 1] < shape[1]) &
            (pts[:, 2] >= 0) & (pts[:, 2] < shape[2]))


def extract_crops(volume, points_zxy, target_sz):
    """Extract a target_sz crop centered on each (z, x, y) point.

    Preserves input order/N so callers can index by original position
    (viewer/inference). Points that fail (out-of-range) yield a zero crop
    with a warning instead of a silent edge-duplicate; training callers
    already drop out-of-bounds points via `in_bounds_mask` before calling.
    """
    target_sz = np.array(target_sz)
    sz = np.array([1, *volume.shape])  # mimic full-video 4d shape for clipping
    crops = []
    for p in points_zxy:
        dat, _ = get_3d_crop_using_bbox_or_centroid(p, sz, target_sz, volume)
        if dat is None:
            logging.warning(f"extract_crops: skipping out-of-range point {np.asarray(p)}; using zero placeholder")
            dat = np.zeros(tuple(int(v) for v in target_sz), dtype=np.float32)
        crops.append(dat)
    return np.stack(crops, 0) if crops else np.zeros((0, *tuple(int(v) for v in target_sz)), dtype=np.float32)


@contextlib.contextmanager
def _seeded_torch_rng(seed):
    """Deterministic GLOBAL torch RNG for the wrapped block.

    Needed because torchio's random transforms (blur/noise) draw from torch's
    global stream, not from a RandomState we own. `devices=[]` keeps the fork
    CPU-only; the crop tensors are CPU here.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


class VolumeCoordsDataset(Dataset):
    """Lazy dataset of full volumes + coordinates; crops extracted AFTER augmentation.

    __getitem__ returns (y1, y2, kpts1, kpts2, idx1, idx2):
        y1/y2: (N1/N2, 1, Z, X, Y) float32 torch tensors (two global-aug views)
        kpts1/kpts2: (N1/N2, 3) normalized (z, x, y) torch tensors, augmentation-consistent
        idx1/idx2: (N1/N2,) long tensors of original object indices (-1: a synthetic
            detection, position_args spurious_p/merge_p/split_p), so the
            training loss can align the intersection when per-view object
            dropout keeps different subsets (N1 != N2). Objects whose
            transformed centroid falls outside the volume are also dropped
            here (same idx mechanism), so N1/N2 vary with the augmentation.
    Volumes are loaded on demand (NOT pre-stacked) so RAM stays ~1 volume.

    Reproducibility: every augmentation draw for item `i` comes from a private
    ``np.random.RandomState`` derived from (seed, epoch, i), and the torchio
    photometrics from a torch seed derived the same way. So one (seed, epoch,
    index) always produces the same two views, whatever the DataLoader worker
    count or sampling order. A single long-lived RandomState would not: each
    DataLoader worker would fork an identical copy (workers duplicating each
    other's augmentation) and the stream would drift with the number of
    workers. ``set_epoch`` is what rotates augmentation between epochs.
    """

    def __init__(self, project_data, frame_indices, target_sz,
                 global_args=None, photometric_args=None, position_args=None, seed=0, epoch=0,
                 round_bounds=False, crop_augment='photometric', legacy_transform_args=None,
                 keypoints_only=False):
        self.project_data = project_data
        # in_bounds_mask(rounded=...): off by default so earlier runs reproduce
        self.round_bounds = bool(round_bounds)
        # Position-only networks never read the crops: skip loading the volume and
        # return (N, 1, 1, 1, 1) zero placeholders, so only N (the object count) is
        # real. Keypoint augmentation draws exactly as with crops, so the views are
        # unchanged. Needs apply_to_volume false (crops would follow the volume).
        self.keypoints_only = bool(keypoints_only)
        if self.keypoints_only and bool({**DEFAULT_GLOBAL_ARGS, **(global_args or {})}.get(
                'apply_to_volume', True)):
            raise ValueError("keypoints_only needs global_augment.apply_to_volume: false")
        self._volume_shape = tuple(int(s) for s in project_data.red_data.shape[1:]) \
            if self.keypoints_only else None
        self.frame_indices = list(frame_indices)
        self.target_sz = np.array(target_sz)
        self.global_args = {**DEFAULT_GLOBAL_ARGS, **(global_args or {})}
        self.position_args = {**DEFAULT_POSITION_ARGS, **(position_args or {})}
        self.seed = int(seed)
        self.epoch = int(epoch)
        # Kept only as a fallback for anything reading self.rng outside
        # __getitem__; the per-item stream is set there.
        self.rng = np.random.RandomState(derive_seed(self.seed, self.epoch))
        # Per-view crop transforms. 'photometric': the same blur/noise/normalize
        # for both views. 'legacy': the preprint crop pipeline's two view
        # transforms (barlow.Transform: view 1 rotation + blur, view 2 rotation,
        # flip, elastic, noise; probabilities from legacy_transform_args, i.e.
        # the training config's p_* keys), one torchio draw per crop stack.
        if crop_augment == 'photometric':
            self.crop_transform = build_photometric_transform(photometric_args)
            self.view_transforms = (self.crop_transform, self.crop_transform)
        elif crop_augment == 'legacy':
            from barlow_track.utils.barlow import Transform  # barlow imports this module
            legacy_args = {k: v for k, v in dict(legacy_transform_args or {}).items()
                           if k != 'global_augment'}
            legacy = Transform(legacy_args)
            self.crop_transform = legacy.transform
            self.view_transforms = (legacy.transform, legacy.transform_prime)
        else:
            raise ValueError(f"Unknown crop_augment '{crop_augment}'; use 'photometric' or 'legacy'")
        # Precompute (cheap) centroids; volumes stay on disk until __getitem__
        self._centroids, self._seg_ids = [], []
        for t in tqdm(self.frame_indices, desc="Loading centroids", leave=False):
            try:
                zxy, seg = get_centroids_for_volume(project_data, int(t))
            except (KeyError, IndexError, FileNotFoundError) as e:
                logging.warning(f"Skipping frame {t}: {e}")
                zxy, seg = np.zeros((0, 3)), np.zeros((0,), dtype=int)
            self._centroids.append(zxy)
            self._seg_ids.append(seg)
        if isinstance(self.position_args.get('jitter_std'), str):
            self.position_args['jitter_std'] = self._jitter_from_stats(self.position_args['jitter_std'])

    def _jitter_from_stats(self, mode, n_pairs=50):
        """jitter_std 'auto': per-axis centroid noise estimated from the segmentation alone
        (no GT; utils/dataset_stats.py), from up to n_pairs (t, t+1) pairs of this dataset's
        frames. Off by default: it reads n_pairs extra metadata frames at start."""
        if mode != 'auto':
            raise ValueError(f"jitter_std must be a float, (z, x, y) or 'auto', got {mode!r}")
        from barlow_track.utils.dataset_stats import centroid_noise_from_pairs
        n_frames = int(getattr(self.project_data, 'num_frames', 0) or 0)
        cand = [i for i, t in enumerate(self.frame_indices) if int(t) + 1 < n_frames]
        pick = [cand[j] for j in np.unique(np.linspace(0, len(cand) - 1, min(n_pairs, len(cand))).astype(int))] \
            if cand else []
        pairs = [(self._centroids[i], get_centroids_for_volume(self.project_data, int(self.frame_indices[i]) + 1)[0])
                 for i in pick]
        noise = centroid_noise_from_pairs(pairs)
        if noise is None:
            logging.warning("jitter_std 'auto': too few lag-1 matches; using 1.0")
            return 1.0
        print(f"jitter_std 'auto' from {len(pairs)} frame pairs: (z, x, y) {np.round(noise, 3).tolist()}", flush=True)
        return [float(v) for v in noise]

    def __len__(self):
        return len(self.frame_indices)

    def num_objects(self, idx):
        return len(self._centroids[idx])

    def set_epoch(self, epoch):
        """Rotate the augmentation stream for a new epoch (see class docstring)."""
        self.epoch = int(epoch)

    def _item_seeds(self, idx):
        """(numpy, torch) seeds for this item: (base seed, epoch, index)."""
        # Modulo folds negative indices (Dataset allows them) onto positions.
        pos = int(idx) % max(len(self.frame_indices), 1)
        return (derive_seed(self.seed, self.epoch, pos),
                derive_seed(self.seed, self.epoch, pos, 'torch', bits=TORCH_SEED_BITS))

    def __getitem__(self, idx):
        t = int(self.frame_indices[idx])
        # A shape-only stand-in when keypoints_only: _augmented_view reads only
        # volume.shape on that path
        volume = np.empty(self._volume_shape, dtype=np.float32) if self.keypoints_only \
            else load_volume(self.project_data, t)
        points = self._centroids[idx].astype(float)
        np_seed, torch_seed = self._item_seeds(idx)
        # Fresh, index-specific RNGs (see class docstring): the two views still
        # differ from each other because they draw from the same fresh stream
        # in sequence, but the pair is reproducible across workers and epochs.
        self.rng = np.random.RandomState(np_seed)
        with _seeded_torch_rng(torch_seed):
            y1, k1, i1 = self._augmented_view(volume, points, view=0)
            y2, k2, i2 = self._augmented_view(volume, points, view=1)
        return y1, y2, k1, k2, i1, i2

    def _augmented_view(self, volume, points, view=0):
        affine_args, elastic_args = split_global_args(self.global_args)
        to_volume = bool(self.global_args.get('apply_to_volume', True))
        # Segmentation errors happen in the frame, before any motion; ids map rows
        # back to original objects (-1: synthetic)
        pa = self.position_args
        points, ids = apply_detection_errors(
            points, self.rng, spurious_p=pa.get('spurious_p', 0.0),
            spurious_std=pa.get('spurious_std', DEFAULT_POSITION_ARGS['spurious_std']),
            merge_p=pa.get('merge_p', 0.0), split_p=pa.get('split_p', 0.0),
            split_dist=pa.get('split_dist', DEFAULT_POSITION_ARGS['split_dist']))
        R, t_vec = sample_global_affine(self.rng, **affine_args)
        field = sample_elastic_field(self.rng, volume.shape, **elastic_args)
        if to_volume:
            vol_aug, pts_aug = apply_global_affine(volume, points, R, t_vec, field=field)
        else:
            vol_aug, pts_aug = volume, transform_points(points, volume.shape, R, t_vec, field=field)
        # Position-only jitter: shift crop centers AND keypoints together
        # (keypoints only when the volume stays put)
        pts_aug = apply_position_jitter(pts_aug, self.rng,
                                        self.position_args.get('jitter_std', 0.0))
        # Drop objects rotated/translated completely out of frame: their
        # clipped crops would be edge duplicates with out-of-range keypoints.
        # Keep original indices so the loss can still align the intersection.
        # With the volume untransformed every crop is valid, so keep them all.
        n_raw = len(pts_aug)
        if to_volume:
            valid = in_bounds_mask(pts_aug, vol_aug.shape, rounded=self.round_bounds)
        else:
            valid = np.ones(n_raw, dtype=bool)
        n_inbounds = int(np.asarray(valid).sum()) if n_raw else 0
        valid_orig_idx = np.where(valid)[0]
        pts_valid = pts_aug[valid] if len(pts_aug) else pts_aug
        # Position-only dropout: independent subset per view (of survivors)
        _, keep_rel = sample_dropout_keep(
            len(pts_valid), self.rng,
            dropout_p=self.position_args.get('dropout_p', 0.0),
            min_keep=self.position_args.get('min_keep', 2))
        keep_idx = valid_orig_idx[keep_rel] if len(valid_orig_idx) else valid_orig_idx
        pts_kept = pts_valid[keep_rel] if len(pts_valid) else pts_valid
        logging.debug(
            f"_augmented_view: raw={n_raw} in_bounds={n_inbounds} "
            f"dropped_bounds={n_raw - n_inbounds} kept_after_dropout={len(pts_kept)}"
        )
        crop_pts = pts_kept if to_volume else (points[keep_idx] if len(keep_idx) else points[:0])
        keep_ids = ids[keep_idx] if len(keep_idx) else keep_idx
        if self.keypoints_only:
            kpts = torch.from_numpy(np.asarray(pts_kept, dtype=np.float32))
            return (torch.zeros((len(pts_kept), 1, 1, 1, 1)), self._normalize(kpts, vol_aug.shape),
                    torch.as_tensor(np.asarray(keep_ids, dtype=np.int64)))
        crops = extract_crops(vol_aug, crop_pts, self.target_sz)
        # 4D (N,Z,X,Y) with N as the channel dim, exactly like the legacy
        # NeuronAugmentedImagePairDataset path (torchio Image convention)
        if len(crops) == 0:
            x = torch.from_numpy(crops)
        else:
            x = self.view_transforms[view](torch.from_numpy(crops))
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x))
        x = x.float().unsqueeze(1)  # (N,1,Z,X,Y)
        kpts = torch.from_numpy(np.asarray(pts_kept, dtype=np.float32))
        kpts = self._normalize(kpts, vol_aug.shape)
        return x, kpts, torch.as_tensor(np.asarray(keep_ids, dtype=np.int64))

    @staticmethod
    def _normalize(kpts_vox, vol_shape):
        # (D,H,W) -> image_shape (1,1,D,H,W); kpts (N,3) -> (1,1,N,3)
        img_shape = (1, 1) + tuple(vol_shape)
        k = kpts_vox.reshape(1, 1, -1, 3)
        normed = normalize_keypoints(k, img_shape)
        return normed.reshape(-1, 3)


def _num_centroids_safe(project_data, t):
    """Number of usable centroids for frame t, or 0 if the frame is unusable."""
    try:
        return len(get_centroids_for_volume(project_data, t)[0])
    except (KeyError, IndexError, FileNotFoundError) as e:
        logging.warning(f"Skipping frame {t}: {e}")
        return 0


def make_worker_init_fn(base_seed, epoch):
    """`worker_init_fn` factory: reseed each worker's GLOBAL streams.

    The dataset's own geometry RNG is per-item and needs nothing here, but
    torchio transforms and the legacy paths read the global numpy / python /
    torch streams, which every worker would otherwise inherit identically from
    the forking parent. (torch seeds its workers too, but from the main
    process's torch stream, which is only reproducible if the entry point seeded
    it -- see utils_seeding.seed_all.)
    """
    def _init(worker_id):
        np.random.seed(derive_seed(base_seed, epoch, worker_id))
        random.seed(derive_seed(base_seed, epoch, worker_id, 'py'))
        torch.manual_seed(derive_seed(base_seed, epoch, worker_id, 'torch',
                                      bits=TORCH_SEED_BITS))
    return _init
  
  
def _resolve_split_count(frac_or_count, n, name="split"):
    """Fraction (<=1.0) -> int(n * f); absolute count (>1) -> int(f), clamped to n."""
    try:
        f = float(frac_or_count)
    except (TypeError, ValueError):
        logging.warning(f"Invalid {name}={frac_or_count!r}, using 0")
        return 0
    if f < 0:
        logging.warning(f"Negative {name}={f}, clamping to 0")
        return 0
    if f <= 1.0:
        return int(n * f)
    count = int(f)
    if count > n:
        logging.warning(f"{name}={count} exceeds dataset size {n}, clamping")
        return n
    return count


class VolumeCoordsDataModule(LightningDataModule):
    """Train/val/test splits over frames, mirroring NeuronCropImageDataModule.

    One `seed` fixes the frame selection, the train/val/test split, and the
    augmentation stream (`num_workers` included, since augmentation is keyed by
    (seed, epoch, index) rather than by a stream that workers fork).
    """

    def __init__(self, project_data=None, num_frames=100, batch_size=1,
                 train_fraction=0.8, val_fraction=0.1, target_sz=(8, 64, 64),
                 global_args=None, photometric_args=None, position_args=None, seed=0,
                 num_workers=0, round_bounds=False, crop_augment='photometric',
                 legacy_transform_args=None, keypoints_only=False):
        super().__init__()
        self.round_bounds = bool(round_bounds)
        self.keypoints_only = bool(keypoints_only)
        self.crop_augment = crop_augment
        self.legacy_transform_args = legacy_transform_args
        self.project_data = project_data
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.target_sz = target_sz
        self.global_args = global_args
        self.photometric_args = photometric_args
        self.position_args = position_args
        self.seed = int(seed)
        self.num_workers = int(num_workers)

    def _current_epoch(self):
        """Epoch Lightning is currently in; 0 before/without a trainer."""
        trainer = getattr(self, 'trainer', None)
        return int(getattr(trainer, 'current_epoch', 0) or 0)

    def _set_epoch(self, epoch):
        ds = getattr(self.train_dataset, 'dataset', self.train_dataset)  # unwrap Subset
        if hasattr(ds, 'set_epoch'):
            ds.set_epoch(epoch)

    def set_epoch(self, epoch):
        """Rotate augmentation for loops that drive epochs themselves (no Lightning
        trainer, so _current_epoch() is always 0). The split Subsets share one dataset,
        so validation views rotate too; loaders built earlier see the change."""
        self._set_epoch(epoch)

    def setup(self, stage: Optional[str] = None):
        max_frames = self.project_data.num_frames
        sampled = random.Random(self.seed).sample(range(max_frames), max_frames)
        # Same validity rule as legacy get_crops_from_project (1, 200].
        # Guarded per frame: a single unreadable volume must not kill the trial
        # (same convention as VolumeCoordsDataset, which skips such frames).
        valid = [t for t in sampled if 1 < _num_centroids_safe(self.project_data, t) <= 200]
        if len(valid) < self.num_frames:
            logging.warning(f"Requested {self.num_frames} volumes, found {len(valid)}; continuing")
        frames = valid[:self.num_frames]
        print(f"Number of frames selected: {len(frames)}")

        alldata = VolumeCoordsDataset(self.project_data, frames, self.target_sz,
                                      global_args=self.global_args,
                                      photometric_args=self.photometric_args,
                                      position_args=self.position_args, seed=self.seed,
                                      epoch=self._current_epoch(), round_bounds=self.round_bounds,
                                      crop_augment=self.crop_augment,
                                      legacy_transform_args=self.legacy_transform_args,
                                      keypoints_only=self.keypoints_only)
        n = len(alldata)
        if n == 0:
            raise ValueError("VolumeCoordsDataModule.setup: no valid frames found")
        
        n_train = _resolve_split_count(self.train_fraction, n, name="train_fraction")
        n_val = _resolve_split_count(self.val_fraction, n, name="val_fraction")
        if n_train >= n:
            if n_train > n:
                logging.warning(f"train split {n_train} > n={n}, clamping")
            n_train = n
            n_val = 0
            n_test = 0
            logging.warning("train_fraction takes the full dataset; val and test splits are empty")
        else:
            if n_train + n_val > n:
                logging.warning(
                    f"train ({n_train}) + val ({n_val}) > n ({n}), clamping val to {n - n_train}"
                )
                n_val = n - n_train
            n_test = n - n_train - n_val
        splits = [n_train, n_val, n_test]
        
        # random_split draws from a torch Generator; without one the split
        # changes every run even with a fixed seed.
        split_gen = torch.Generator()
        split_gen.manual_seed(derive_seed(self.seed, 'split', bits=TORCH_SEED_BITS))
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            alldata, splits, generator=split_gen)
        self.alldata = alldata

    def _dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=_collate_single,
                          worker_init_fn=make_worker_init_fn(self.seed, self._current_epoch()))

    def train_dataloader(self):
        # Augmentation rotates per epoch (Lightning rebuilds the loader each
        # epoch), but stays a deterministic function of (seed, epoch, index).
        self._set_epoch(self._current_epoch())
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)


def _collate_single(batch):
    # batch_size is 1 volume; drop the list wrapper (variable N per volume)
    assert len(batch) == 1
    return batch[0]
