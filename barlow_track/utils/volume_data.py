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


DEFAULT_GLOBAL_ARGS = dict(
    p_global_affine=1.0,
    max_degrees_z=180.0,
    scale_jitter=0.1,
    max_translation=(2, 8, 8),  # (z, x, y) voxels
    p_flip=0.0,  # extra 180 deg in-plane flip; 180 deg range already covers it
)

DEFAULT_CROP_PHOTOMETRIC_ARGS = dict(
    p_blur=0.0,
    p_noise=0.1,
    std_noise=0.25,
)


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


def apply_global_affine(volume, points_zxy, R, t, order=1):
    """Apply forward transform x' = R(x - C) + C + t to volume and points.

    volume: (D, H, W); points_zxy: (N, 3). Returns (augmented volume, moved points).
    """
    center = (np.array(volume.shape) - 1) / 2.0
    R_inv = np.linalg.inv(R)
    offset = center - R_inv @ (center + t)
    vol_aug = ndimage.affine_transform(np.asarray(volume, dtype=np.float32),
                                       R_inv, offset=offset, order=order, mode='nearest')
    pts_aug = (points_zxy - center) @ R.T + center + t
    return vol_aug, pts_aug


def get_centroids_for_volume(project_data, t):
    """Centroids (z, x, y) + raw segmentation ids for one timepoint.

    Same metadata source as `get_bbox_data_for_volume_with_label`, but without
    track filtering: the new pipeline needs ALL detections plus positions.
    """
    row_data, column_names = project_data.segmentation_metadata.get_all_neuron_metadata_for_single_time(
        t, as_dataframe=False)
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


def extract_crops(volume, points_zxy, target_sz):
    """Extract a target_sz crop centered on each (z, x, y) point."""
    sz = np.array([1, *volume.shape])  # mimic full-video 4d shape for clipping
    crops = [get_3d_crop_using_bbox_or_centroid(p, sz, np.array(target_sz), volume)[0]
             for p in points_zxy]
    return np.stack(crops, 0) if crops else np.zeros((0, *target_sz), dtype=np.float32)


class VolumeCoordsDataset(Dataset):
    """Lazy dataset of full volumes + coordinates; crops extracted AFTER augmentation.

    __getitem__ returns (y1, y2, kpts1, kpts2):
        y1/y2: (N, 1, Z, X, Y) float32 torch tensors (two global-aug views)
        kpts1/kpts2: (N, 3) normalized (z, x, y) torch tensors, augmentation-consistent
    Volumes are loaded on demand (NOT pre-stacked) so RAM stays ~1 volume.
    """

    def __init__(self, project_data, frame_indices, target_sz,
                 global_args=None, photometric_args=None, seed=0):
        self.project_data = project_data
        self.frame_indices = list(frame_indices)
        self.target_sz = np.array(target_sz)
        self.global_args = {**DEFAULT_GLOBAL_ARGS, **(global_args or {})}
        self.rng = np.random.RandomState(seed)
        photo = {**DEFAULT_CROP_PHOTOMETRIC_ARGS, **(photometric_args or {})}
        self.crop_transform = tio.Compose([
            tio.RandomBlur(p=photo['p_blur']),
            tio.RandomNoise(std=photo['std_noise'], p=photo['p_noise']),
            tio.RescaleIntensity(percentiles=(5, 100)),
        ])
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

    def __len__(self):
        return len(self.frame_indices)

    def num_objects(self, idx):
        return len(self._centroids[idx])

    def __getitem__(self, idx):
        t = int(self.frame_indices[idx])
        volume = load_volume(self.project_data, t)
        points = self._centroids[idx].astype(float)
        y1, k1 = self._augmented_view(volume, points)
        y2, k2 = self._augmented_view(volume, points)
        return y1, y2, k1, k2

    def _augmented_view(self, volume, points):
        R, t_vec = sample_global_affine(self.rng, **self.global_args)
        vol_aug, pts_aug = apply_global_affine(volume, points, R, t_vec)
        crops = extract_crops(vol_aug, pts_aug, self.target_sz)
        # 4D (N,Z,X,Y) with N as the channel dim, exactly like the legacy
        # NeuronAugmentedImagePairDataset path (torchio Image convention)
        x = self.crop_transform(torch.from_numpy(crops))
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x))
        x = x.float().unsqueeze(1)  # (N,1,Z,X,Y)
        kpts = torch.from_numpy(np.asarray(pts_aug, dtype=np.float32))
        kpts = self._normalize(kpts, vol_aug.shape)
        return x, kpts

    @staticmethod
    def _normalize(kpts_vox, vol_shape):
        # (D,H,W) -> image_shape (1,1,D,H,W); kpts (N,3) -> (1,1,N,3)
        img_shape = (1, 1) + tuple(vol_shape)
        k = kpts_vox.reshape(1, 1, -1, 3)
        normed = normalize_keypoints(k, img_shape)
        return normed.reshape(-1, 3)


class VolumeCoordsDataModule(LightningDataModule):
    """Train/val/test splits over frames, mirroring NeuronCropImageDataModule."""

    def __init__(self, project_data=None, num_frames=100, batch_size=1,
                 train_fraction=0.8, val_fraction=0.1, target_sz=(8, 64, 64),
                 global_args=None, photometric_args=None, seed=0):
        super().__init__()
        self.project_data = project_data
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.target_sz = target_sz
        self.global_args = global_args
        self.photometric_args = photometric_args
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        max_frames = self.project_data.num_frames
        sampled = random.Random(self.seed).sample(range(max_frames), max_frames)
        # Same validity rule as legacy get_crops_from_project (1, 200]
        valid = [t for t in sampled
                 if 1 < len(get_centroids_for_volume(self.project_data, t)[0]) <= 200]
        if len(valid) < self.num_frames:
            logging.warning(f"Requested {self.num_frames} volumes, found {len(valid)}; continuing")
        frames = valid[:self.num_frames]
        print(f"Number of frames selected: {len(frames)}")

        alldata = VolumeCoordsDataset(self.project_data, frames, self.target_sz,
                                      global_args=self.global_args,
                                      photometric_args=self.photometric_args, seed=self.seed)
        n = len(alldata)
        n_train = int(n * self.train_fraction) if self.train_fraction < 1.0 else int(self.train_fraction)
        n_val = int(n * self.val_fraction) if self.val_fraction < 1.0 else int(self.val_fraction)
        splits = [n_train, n_val, n - n_train - n_val]
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(alldata, splits)
        self.alldata = alldata

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=_collate_single)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=_collate_single)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=_collate_single)


def _collate_single(batch):
    # batch_size is 1 volume; drop the list wrapper (variable N per volume)
    assert len(batch) == 1
    return batch[0]
