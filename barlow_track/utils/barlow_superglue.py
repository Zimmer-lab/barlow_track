"""Position-aware Barlow models: visual + SuperGlue-style position fusion, optional GNN matching.

Step 2 - BarlowWithPosition:
    crop --backbone--> visual desc --norm--+
                                           +--> fuse (add | concat+MLP) --> projector --> Barlow loss
    kpts --KeypointEncoder--> pos desc --norm--+

    Per-branch normalization (fusion_norm: none | layernorm | l2, default none
    for checkpoint back-compat) lets training balance the two streams instead
    of freezing in a scale by hand. Trained with the SAME two-view Barlow loss
    as BarlowTwins3d, except the two views come from VolumeCoordsDataset
    (global-before-crop augmentation) with augmentation-consistent keypoints.

Attention variant - BarlowVolumeAttention (extends fusion):
    fused descs --intra-volume self-attention--> contextualized descs --> projector
    Self-attention runs WITHIN one volume, so contextualize() yields a genuine
    per-neuron embedding from a single frame (usable at inference), unlike the
    cross-volume matching below which is inherently pair-dependent.

Step 3 - BarlowSuperGlue: STUB, intentionally not used. Pair-only
    cross-volume matching has no inference path (tracking embeds one volume
    at a time), so only the intra-volume attention above is implemented for
    real. The stub class exists so old 'superglue' checkpoints still load;
    its forward() raises.

Legacy BarlowTwins3d is untouched; all new behavior is opt-in via
use_position / use_attention training flags.
"""
import torch
from torch import nn

from barlow_track.utils.barlow import BarlowTwins3d, off_diagonal
from barlow_track.utils.siamese import Siamese
from barlow_track.utils.superglue import (
    AttentionalGNN,
    KeypointEncoder,
    log_optimal_transport,
    process_scores_into_matches,
)


def both_correlation_matrices(z1, z2):
    """Feature-space (DxD) and object-space (NxN) cross-correlation matrices."""
    z1_norm = (z1 - z1.mean(0)) / z1.std(0)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)
    c_features = torch.matmul(z1_norm.T, z2_norm) / z1.shape[0]

    z1_t = ((z1.T - z1.mean(1)) / z1.std(1)).T
    z2_t = ((z2.T - z2.mean(1)) / z2.std(1)).T
    c_objects = torch.matmul(z1_t, z2_t.T) / z1.shape[1]
    return c_features, c_objects


class L2Normalize(nn.Module):
    """Unit-norm over the feature dim (per neuron)."""

    def forward(self, x):
        return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)


def _make_norm(kind, dim):
    if kind in (None, 'none'):
        return nn.Identity()
    if kind == 'layernorm':
        return nn.LayerNorm(dim)
    if kind == 'l2':
        return L2Normalize()
    raise ValueError(f"Unknown fusion_norm '{kind}'; use 'none', 'layernorm' or 'l2'")


class BarlowWithPosition(BarlowTwins3d):
    """BarlowTwins3d + KeypointEncoder position fusion (Step 2)."""

    def __init__(self, args, backbone=Siamese, fusion=None, keypoint_layers=None,
                 fusion_norm=None, **backbone_kwargs):
        super().__init__(args, backbone=backbone, **backbone_kwargs)
        embedding_dim = args.embedding_dim
        self.embedding_dim = embedding_dim
        self.fusion = fusion or getattr(args, 'fusion', 'concat')
        self.fusion_norm = fusion_norm or getattr(args, 'fusion_norm', 'none')
        layers = keypoint_layers or list(getattr(args, 'keypoint_encoder_layers', [32, 64]))
        self.kenc = KeypointEncoder(embedding_dim, layers)
        self.norm_visual = _make_norm(self.fusion_norm, embedding_dim)
        self.norm_pos = _make_norm(self.fusion_norm, embedding_dim)
        if self.fusion == 'concat':
            self.fuse_mlp = nn.Sequential(nn.Linear(2 * embedding_dim, embedding_dim), nn.ReLU())
        elif self.fusion == 'position_only':
            pass  # visual branch unused; see fused_descriptors
        elif self.fusion != 'add':
            raise ValueError(f"Unknown fusion '{self.fusion}'; use 'add', 'concat' or 'position_only'")

    def encode_position(self, kpts, scores=None):
        """(N,3) normalized keypoints -> (N,D) position descriptors.

        Frames with fewer than 2 detections carry no relative geometry
        (InstanceNorm needs N > 1), so they fall back to zeros, i.e. the
        fused embedding degrades gracefully to visual-only.
        """
        if kpts.shape[0] < 2:
            return kpts.new_zeros((kpts.shape[0], self.embedding_dim))
        if scores is None:
            scores = kpts.new_ones(kpts.shape[0])
        k = kpts.reshape(1, 1, -1, 3)
        s = scores.reshape(1, 1, -1)
        return self.kenc(k, s).squeeze(0).transpose(0, 1)

    def fused_descriptors(self, y, kpts, scores=None):
        pos = self.norm_pos(self.encode_position(kpts, scores))
        if self.fusion == 'position_only':
            return pos  # visual branch unused: pure geometry benchmark
        visual = self.norm_visual(self.backbone(y))
        if self.fusion == 'add':
            return visual + pos
        return self.fuse_mlp(torch.cat([visual, pos], dim=1))

    def embed_with_position(self, y, kpts, scores=None):
        return self.projector(self.fused_descriptors(y, kpts, scores))

    def forward(self, y1, y2, kpts1, kpts2, scores1=None, scores2=None):
        z1 = self.embed_with_position(y1, kpts1, scores1)
        z2 = self.embed_with_position(y2, kpts2, scores2)
        c_features, c_objects = both_correlation_matrices(z1, z2)

        loss_transpose = torch.tensor(0.0, device=y1.device)
        loss_original = torch.tensor(0.0, device=y1.device)
        if self.args.lambd_obj < 1:
            loss_original = self.loss_from_correlation_matrix(c_features)
        if self.args.lambd_obj > 0:
            loss_transpose = self.loss_from_correlation_matrix(c_objects)

        loss = (1.0 - self.args.lambd_obj) * loss_original + self.args.lambd_obj * loss_transpose
        return loss, loss_original, loss_transpose


class BarlowVolumeAttention(BarlowWithPosition):
    """Fusion + INTRA-volume self-attention (the actual per-neuron embedding).

    contextualize() runs self-attention over the neurons of ONE volume, so
    unlike cross-volume matching it yields a standalone embedding per neuron
    from a single frame: backbone -> fuse -> norm -> self-attend -> projector.
    This is what inference (embed_volumes_with_position) uses.
    """

    def __init__(self, args, backbone=Siamese, self_layers=None, **backbone_kwargs):
        # Pop our own kwargs before delegating (backbone would choke on them)
        fusion = backbone_kwargs.pop('fusion', None)
        keypoint_layers = backbone_kwargs.pop('keypoint_layers', None)
        fusion_norm = backbone_kwargs.pop('fusion_norm', None)
        super().__init__(args, backbone=backbone, fusion=fusion,
                         keypoint_layers=keypoint_layers, fusion_norm=fusion_norm,
                         **backbone_kwargs)
        n_self = (self_layers if self_layers is not None
                  else int(getattr(args, 'self_layers', 2)))
        self.self_layer_names = ['self'] * n_self
        self.self_gnn = (AttentionalGNN(args.embedding_dim, self.self_layer_names)
                         if n_self > 0 else nn.Identity())

    def contextualize(self, d):
        """(N,D) fused descriptors -> (N,D) volume-contextualized descriptors.

        Single detections bypass attention (nothing to attend to, and the
        propagation MLP's InstanceNorm needs N > 1).
        """
        if d.shape[0] < 2 or isinstance(self.self_gnn, nn.Identity):
            return d
        batch = d.transpose(0, 1).unsqueeze(0)
        out, _ = self.self_gnn(batch, batch)
        return out.squeeze(0).transpose(0, 1)

    def contextual_descriptors(self, y, kpts, scores=None):
        return self.contextualize(self.fused_descriptors(y, kpts, scores))

    def embed_with_position(self, y, kpts, scores=None):
        return self.projector(self.contextual_descriptors(y, kpts, scores))

    def forward(self, y1, y2, kpts1, kpts2, scores1=None, scores2=None):
        z1 = self.embed_with_position(y1, kpts1, scores1)
        z2 = self.embed_with_position(y2, kpts2, scores2)
        c_features, c_objects = both_correlation_matrices(z1, z2)

        loss_transpose = torch.tensor(0.0, device=y1.device)
        loss_original = torch.tensor(0.0, device=y1.device)
        if self.args.lambd_obj < 1:
            loss_original = self.loss_from_correlation_matrix(c_features)
        if self.args.lambd_obj > 0:
            loss_transpose = self.loss_from_correlation_matrix(c_objects)

        loss = (1.0 - self.args.lambd_obj) * loss_original + self.args.lambd_obj * loss_transpose
        return loss, loss_original, loss_transpose


class BarlowSuperGlue(BarlowVolumeAttention):
    """STUB - pair-only cross-volume matching is intentionally NOT used.

    The original SuperGlue matches two volumes with cross-attention +
    Sinkhorn, but inference (tracking) embeds ONE volume at a time, so a
    pair-dependent head can never feed back into an embedding. The useful
    part - intra-volume self-attention over fused descriptors - lives in
    BarlowVolumeAttention; use that (use_attention) instead.

    Kept only so old 'superglue' checkpoints still load. Instantiating and
    loading weights is fine, but forward() raises.
    """

    def __init__(self, args, backbone=Siamese, gnn_layers=None, **backbone_kwargs):
        super().__init__(args, backbone=backbone, **backbone_kwargs)
        embedding_dim = args.embedding_dim
        default_gnn = ['self', 'cross'] * 3
        self.gnn_layer_names = gnn_layers or list(getattr(args, 'gnn_layers', default_gnn))
        self.gnn = AttentionalGNN(embedding_dim, self.gnn_layer_names)
        self.final_proj = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1, bias=True)
        self.bin_score = nn.Parameter(torch.tensor(1.0))
        self.loss_epsilon = 1e-6
        self.match_loss_weight = float(getattr(args, 'match_loss_weight', 1.0))
        self.sinkhorn_iterations = int(getattr(args, 'sinkhorn_iterations', 50))

    def forward(self, y1, y2, kpts1=None, kpts2=None, scores1=None, scores2=None):
        raise NotImplementedError(
            "BarlowSuperGlue.forward is a stub: pair-only cross-volume matching "
            "has no inference path (tracking embeds one volume at a time). "
            "Use BarlowVolumeAttention (use_attention) instead.")
