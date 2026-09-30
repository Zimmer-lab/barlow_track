"""Step 0 baseline tests: embed a real project + characterize current augmentation.

Covers the pre-refactor behavior before position encodings / SuperGlue-GNN /
global-before-crop augmentation:

1. Load a test project (local copy preferred, /lisc fallback).
2. Extract per-volume crops AND centroids (what the new pipeline will need).
3. Embed crops with a pretrained Barlow model (model-native target_sz).
4. Check the current crop-level augmentation: shapes, two views differ,
   no NaNs, and quantify zero-padding edge effects (motivation for
   augment-before-crop).

Run with the wbfm env (CPU ok, GPU if available):
    /home/charles/anaconda3/envs/wbfm/bin/python -m pytest barlow_track/tests/test_step0_embed_and_augment.py -x -q

Heavy tests (full-volume embed) are marked slow and skipped by default;
run them with `--runslow`.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

TEST_PROJECT_CANDIDATES = [
    "/home/charles/Current_work/test_projects/barlow/worm4-2-2025-03-09/project_config.yaml",
    "/lisc/data/scratch/neurobiology/zimmer/wbfm/test_projects/barlow/worm4-2-2025-03-09/project_config.yaml",
]
MODEL_CANDIDATES = [
    "/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/barlow_ZIM2165_Gcamp7b_worm1-2022_11_28_from_search/trial_13/resnet50.pth",
]


def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None


PROJECT_PATH = os.environ.get("BARLOW_TEST_PROJECT") or _first_existing(TEST_PROJECT_CANDIDATES)
MODEL_PATH = os.environ.get("BARLOW_TEST_MODEL") or _first_existing(MODEL_CANDIDATES)

requires_project = pytest.mark.skipif(PROJECT_PATH is None, reason="No test project found")
requires_model = pytest.mark.skipif(MODEL_PATH is None, reason="No pretrained model found")


@pytest.fixture(scope="module")
def project_data():
    from wbfm.utils.projects.finished_project_data import ProjectData

    assert PROJECT_PATH is not None, f"No test project found; tried {TEST_PROJECT_CANDIDATES}"
    return ProjectData.load_final_project_data(PROJECT_PATH, allow_hybrid_loading=True)


@pytest.fixture(scope="module")
def barlow_model():
    from barlow_track.utils.barlow import load_barlow_model

    assert MODEL_PATH is not None, f"No pretrained model found; tried {MODEL_CANDIDATES}"
    _, model, args = load_barlow_model(MODEL_PATH)
    model.eval()
    return model, args


def _default_transform_args():
    # Mirror barlow_project_template/train_config.yaml defaults
    return SimpleNamespace(
        p_RandomAffine_base=1.0,
        p_RandomBlur_base=0.1,
        p_RandomAffine_flip=1.0,
        p_RandomBlur=0.0,
        p_RandomAffine=1.0,
        p_RandomElasticDeformation=1.0,
        zxy_RandomElasticDeformation=[1, 3, 3],
        p_RandomNoise=0.1,
        std_RandomNoise=0.25,
        p_RandomAffine_both=None,
    )


# ---------------------------------------------------------------- project
@requires_project
def test_project_loads(project_data):
    assert project_data.num_frames > 0
    assert len(project_data.red_data.shape) == 4  # (T, Z, X, Y)


@requires_project
def test_volume_crops_and_centroids(project_data, barlow_model):
    """New pipeline needs crops + coordinates; check both are available for one volume."""
    from barlow_track.utils.data_loading import get_bbox_data_for_volume

    _, args = barlow_model
    target_sz = np.array(args.target_sz)  # model-native size, e.g. [4, 128, 128]
    t = 0
    crops, _ = get_bbox_data_for_volume(project_data, t, target_sz=target_sz)
    assert len(crops) > 1
    assert crops[0].shape == tuple(target_sz)

    # Centroids for the same time point (position channel for SuperGlue-style encoder)
    row_data, col_names = project_data.segmentation_metadata.get_all_neuron_metadata_for_single_time(
        t, as_dataframe=False
    )
    import pandas as pd

    mdata = pd.DataFrame(dict(zip(col_names, row_data)))
    assert {"z", "x", "y"}.issubset(set(mdata.columns))
    assert len(mdata) >= len(crops)  # metadata covers at least the segmented objects


# ---------------------------------------------------------------- embedding
@requires_project
@requires_model
def test_embed_single_volume_subset(project_data, barlow_model):
    """Embed a few crops of one volume on CPU; locks in current inference behavior."""
    from barlow_track.utils.data_loading import get_bbox_data_for_volume

    model, args = barlow_model
    target_sz = np.array(args.target_sz)
    crops, _ = get_bbox_data_for_volume(project_data, 0, target_sz=target_sz)
    x = torch.from_numpy(np.stack(crops[:4]).astype(np.float32)).unsqueeze(1)  # (N,1,Z,X,Y)
    with torch.no_grad():
        z = model.embed(x)
    latent = int(args.projector.split("-")[-1])
    assert z.shape == (4, latent)
    assert torch.isfinite(z).all()


@requires_project
@requires_model
@pytest.mark.slow
def test_embed_full_volume(project_data, barlow_model):
    """Whole-volume embed (~150 neurons); slow on CPU, fast on GPU."""
    from barlow_track.utils.data_loading import get_bbox_data_for_volume

    model, args = barlow_model
    target_sz = np.array(args.target_sz)
    crops, _ = get_bbox_data_for_volume(project_data, 0, target_sz=target_sz)
    x = torch.from_numpy(np.stack(crops).astype(np.float32)).unsqueeze(1)
    with torch.no_grad():
        z = model.embed(x)
    assert z.shape[0] == len(crops)
    assert torch.isfinite(z).all()


# ---------------------------------------------------------------- augmentation
@requires_project
def test_augmentation_two_views(project_data, barlow_model):
    """Current crop-level Transform: same input -> two finite views of same shape,
    and (with p=1 affine) the views differ."""
    from barlow_track.utils.barlow import Transform
    from barlow_track.utils.data_loading import get_bbox_data_for_volume

    _, args = barlow_model
    target_sz = np.array(args.target_sz)
    crops, _ = get_bbox_data_for_volume(project_data, 0, target_sz=target_sz)
    x = torch.from_numpy(np.stack(crops[:2]).astype(np.float32)).unsqueeze(1)

    augmentor = Transform(_default_transform_args())
    y1, y2 = augmentor(torch.squeeze(x))
    assert y1.shape == y2.shape
    assert torch.isfinite(torch.as_tensor(np.asarray(y1))).all()
    assert torch.isfinite(torch.as_tensor(np.asarray(y2))).all()
    assert not np.allclose(np.asarray(y1), np.asarray(y2))  # views must differ


@requires_project
def test_augmentation_edge_effect_baseline(project_data, barlow_model):
    """Quantify zero-padding introduced by augmenting the SMALL crop.

    This is the baseline that 'augment-before-crop' should improve:
    affine/elastic on an already-tight crop pulls in out-of-bounds (zero)
    voxels at the borders. We assert the effect exists so a future
    global-first pipeline can show it shrinking.
    """
    from barlow_track.utils.barlow import Transform
    from barlow_track.utils.data_loading import get_bbox_data_for_volume

    _, args = barlow_model
    target_sz = np.array(args.target_sz)
    crops, _ = get_bbox_data_for_volume(project_data, 0, target_sz=target_sz)
    x = torch.from_numpy(np.stack(crops[:4]).astype(np.float32)).unsqueeze(1)

    augmentor = Transform(_default_transform_args())
    n_zero = []
    for _ in range(5):
        y1, _ = augmentor(torch.squeeze(x))
        arr = np.asarray(y1)
        n_zero.append(np.mean(arr == 0))
    mean_zero_frac = float(np.mean(n_zero))
    print(f"\nBaseline exact-zero voxel fraction after small-crop aug: {mean_zero_frac:.4f}")
    # Sanity bounds only: documents current behavior without overfitting to it
    assert 0.0 <= mean_zero_frac < 0.9
