"""Unit tests for BarlowWithPosition (Step 2) and BarlowSuperGlue (Step 3).

Synthetic data + tiny backbone: fast on CPU, no test project needed.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest barlow_track/tests/test_barlow_position.py -q --assert=plain
"""
from types import SimpleNamespace

import pytest
import torch

from barlow_track.utils.barlow_superglue import BarlowSuperGlue, BarlowWithPosition, both_correlation_matrices
from barlow_track.utils.siamese import ResidualEncoder3D


def _args(**overrides):
    base = dict(
        embedding_dim=16,
        projector='32-32',
        projector_final=8,
        lambd=0.0051,
        lambd_obj=0.5,
        fusion='add',
        keypoint_encoder_layers=[16, 32],
        gnn_layers=['self', 'cross'],
        match_loss_weight=1.0,
        sinkhorn_iterations=5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _backbone_kwargs():
    import numpy as np
    # Note: embedding_dim is injected by the model itself; do not pass it here
    return dict(in_channels=1, num_levels=2, f_maps=2, crop_sz=np.array([4, 16, 16]))


@pytest.fixture
def batch():
    torch.manual_seed(0)
    n = 6
    y1 = torch.randn(n, 1, 4, 16, 16)
    y2 = torch.randn(n, 1, 4, 16, 16)
    k1 = torch.randn(n, 3) * 0.5
    k2 = torch.randn(n, 3) * 0.5
    return y1, y2, k1, k2


def test_correlation_matrices_shapes():
    z1, z2 = torch.randn(6, 8), torch.randn(6, 8)
    cf, co = both_correlation_matrices(z1, z2)
    assert cf.shape == (8, 8) and co.shape == (6, 6)


def test_fusion_add_forward_backward(batch):
    y1, y2, k1, k2 = batch
    model = BarlowWithPosition(_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    loss, lo, lt = model(y1, y2, k1, k2)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.kenc.encoder[0].weight.grad is not None  # position path gets gradients


def test_fusion_concat_shapes(batch):
    y1, y2, k1, k2 = batch
    model = BarlowWithPosition(_args(fusion='concat'), backbone=ResidualEncoder3D, **_backbone_kwargs())
    z = model.embed_with_position(y1, k1)
    assert z.shape == (6, 8)
    loss, _, _ = model(y1, y2, k1, k2)
    assert torch.isfinite(loss)


def test_position_changes_embedding(batch):
    y1, _, k1, _ = batch
    model = BarlowWithPosition(_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    model.eval()
    with torch.no_grad():
        z_a = model.embed_with_position(y1, k1)
        k_perturbed = k1.clone()
        k_perturbed[0] += 0.5  # move a single neuron: relative geometry changes
        z_b = model.embed_with_position(y1, k_perturbed)
    assert not torch.allclose(z_a, z_b)  # keypoints actually matter


def test_explicit_scores(batch):
    y1, y2, k1, k2 = batch
    model = BarlowWithPosition(_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    s = torch.ones(6)
    loss, _, _ = model(y1, y2, k1, k2, s, s)
    assert torch.isfinite(loss)


def test_superglue_forward_backward(batch):
    y1, y2, k1, k2 = batch
    model = BarlowSuperGlue(_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    total, lo, lt, lm = model(y1, y2, k1, k2)
    for v in (total, lo, lt, lm):
        assert torch.isfinite(v)
    total.backward()
    assert model.gnn.layers[0].attn.merge.weight.grad is not None


def test_superglue_match_scores_shape(batch):
    y1, _, k1, _ = batch
    model = BarlowSuperGlue(_args(), backbone=ResidualEncoder3D, **_backbone_kwargs())
    model.eval()
    with torch.no_grad():
        d = model.fused_descriptors(y1, k1)
        scores = model.calculate_match_scores(d, d)
    assert scores.shape == (1, 7, 7)  # N=6 + dustbin
    idx0, _, ms0, _ = model.matches_from_scores(scores)
    assert idx0.shape == (6,) and ms0.shape == (6,)
