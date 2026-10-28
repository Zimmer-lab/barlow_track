"""ViSNetPositionEncoder: invariances and the optional cutoff in nearest-neighbour units."""
import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('torch_geometric')

from barlow_track.utils.barlow_superglue import BarlowWithPosition, ViSNetPositionEncoder


def _points(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, generator=g) * torch.tensor([0.2, 1.0, 0.4]) - 0.3


def _encoder(**kwargs):
    torch.manual_seed(0)
    return ViSNetPositionEncoder(16, hidden_channels=16, num_layers=2, **kwargs).eval()


def _rotate_z(x, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    R = torch.tensor([[1.0, 0, 0], [0, c, -s], [0, s, c]])
    return x @ R.T


def test_rotation_and_translation_invariant():
    enc, x = _encoder(cutoff=0.35), _points()
    with torch.no_grad():
        a = enc(x)
        b = enc(_rotate_z(x, 73.0) + torch.tensor([0.05, -0.1, 0.2]))
    assert torch.allclose(a, b, atol=1e-4)


def test_default_encoder_is_blind_to_distances_at_small_cutoff():
    # Documents the pyg RBF scale problem rbf_rescale fixes: at cutoff 0.35 the
    # untrained encoder gives every node the same output.
    enc, x = _encoder(cutoff=0.35), _points()
    with torch.no_grad():
        assert enc(x).std(0).max() < 1e-5


def test_rbf_rescale_sees_distances():
    enc, x = _encoder(cutoff=0.35, rbf_rescale=True), _points()
    with torch.no_grad():
        out = enc(x)
        assert out.std(0).mean() > 1e-2
        assert torch.allclose(out, enc(_rotate_z(x, 73.0) + 0.1), atol=1e-4)
        assert not torch.allclose(out, enc(1.1 * x), atol=1e-3)  # absolute units stay scale-sensitive


def test_rbf_rescale_matches_a_native_cutoff_five_encoder():
    x = _points()
    torch.manual_seed(0)
    native = ViSNetPositionEncoder(16, hidden_channels=16, num_layers=2, cutoff=5.0).eval()
    rescaled = _encoder(cutoff=0.35, rbf_rescale=True)
    rescaled.load_state_dict(native.state_dict())
    with torch.no_grad():
        assert torch.allclose(rescaled(x), native(x * (5.0 / 0.35)), atol=1e-5)


def test_cutoff_nn_makes_the_encoder_scale_invariant():
    enc, x = _encoder(cutoff=0.35, cutoff_nn=4.0, rbf_rescale=True), _points()
    assert enc(x).std(0).mean() > 1e-2
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(1.1 * x), atol=1e-4)
        assert torch.allclose(enc(x), enc(0.5 * _rotate_z(x, 30.0)), atol=1e-4)


def test_cutoff_nn_default_off_and_wired_from_args():
    base = dict(embedding_dim=16, projector='32', projector_final=16, lambd=0.0051,
                position_encoder='visnet', fusion='position_only', visnet_hidden=16, visnet_layers=2)
    from barlow_track.utils.siamese import ResidualEncoder3D
    kw = dict(backbone=ResidualEncoder3D, in_channels=1, num_levels=2, f_maps=4, crop_sz=(8, 16, 16))
    default = BarlowWithPosition(SimpleNamespace(**base), **kw).visnet_enc
    assert default.cutoff_nn is None and not default.rbf_rescale
    model = BarlowWithPosition(SimpleNamespace(**base, visnet_cutoff_nn=3, visnet_rbf_rescale=True), **kw)
    assert model.visnet_enc.cutoff_nn == 3.0 and model.visnet_enc.rbf_rescale


def test_radius_graph_keeps_the_nearest_neighbours_like_the_cluster():
    # The cluster has no torch-cluster and uses the pure-torch fallback (nearest
    # max_num_neighbors within r); where the compiled package exists it must be
    # replaced, or capped neighbourhoods differ between local and cluster runs.
    from torch_geometric.nn import radius_graph
    from barlow_track.utils.barlow_superglue import _pure_torch_radius_graph
    x = torch.rand(120, 3, generator=torch.Generator().manual_seed(0))  # every node capped
    got = radius_graph(x, r=0.35, max_num_neighbors=8)
    want = _pure_torch_radius_graph(x, 0.35, None, False, 8)
    as_set = lambda e: set(map(tuple, e.t().tolist()))
    assert as_set(got) == as_set(want)
    d = torch.cdist(x, x)
    for i in range(5):
        nbrs = got[0][got[1] == i] if got.shape[1] else got[0]
        others = d[i].clone()
        others[i] = float('inf')
        within = int((others <= 0.35).sum())
        assert set(nbrs.tolist()) == set(others.topk(min(8, within), largest=False).indices.tolist())
    assert (torch.bincount(got[1], minlength=len(x)) == 8).sum() > 60  # mostly capped
