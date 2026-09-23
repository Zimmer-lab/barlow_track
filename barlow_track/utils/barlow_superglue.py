"""Position-aware Barlow models: visual + SuperGlue-style position fusion, optional GNN matching.

Step 2 - BarlowWithPosition:
    crop --backbone--> visual desc --+
                                     +--> fuse (add | concat+MLP) --> projector --> Barlow loss
    kpts --KeypointEncoder--> pos desc -+

    Trained with the SAME two-view Barlow loss as BarlowTwins3d, except the two
    views come from VolumeCoordsDataset (global-before-crop augmentation) and
    carry augmentation-consistent normalized keypoints.

Step 3 - BarlowSuperGlue (extends fusion):
    fused descs --AttentionalGNN--> final_proj --> Sinkhorn scores --> NLL
    identity-match loss (two views show the same neurons in the same order).
    Total = barlow_loss + match_loss_weight * match_loss.

Legacy BarlowTwins3d is untouched; all new behavior is opt-in via
use_position / use_gnn training flags.
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


class BarlowWithPosition(BarlowTwins3d):
    """BarlowTwins3d + KeypointEncoder position fusion (Step 2)."""

    def __init__(self, args, backbone=Siamese, fusion=None, keypoint_layers=None, **backbone_kwargs):
        super().__init__(args, backbone=backbone, **backbone_kwargs)
        embedding_dim = args.embedding_dim
        self.fusion = fusion or getattr(args, 'fusion', 'add')
        layers = keypoint_layers or list(getattr(args, 'keypoint_encoder_layers', [32, 64]))
        self.kenc = KeypointEncoder(embedding_dim, layers)
        if self.fusion == 'concat':
            self.fuse_mlp = nn.Sequential(nn.Linear(2 * embedding_dim, embedding_dim), nn.ReLU())
        elif self.fusion != 'add':
            raise ValueError(f"Unknown fusion '{self.fusion}'; use 'add' or 'concat'")

    def encode_position(self, kpts, scores=None):
        """(N,3) normalized keypoints -> (N,D) position descriptors."""
        if scores is None:
            scores = kpts.new_ones(kpts.shape[0])
        k = kpts.reshape(1, 1, -1, 3)
        s = scores.reshape(1, 1, -1)
        return self.kenc(k, s).squeeze(0).transpose(0, 1)

    def fused_descriptors(self, y, kpts, scores=None):
        visual = self.backbone(y)
        pos = self.encode_position(kpts, scores)
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


class BarlowSuperGlue(BarlowWithPosition):
    """Fusion model + AttentionalGNN identity matching (Step 3)."""

    def __init__(self, args, backbone=Siamese, gnn_layers=None, **backbone_kwargs):
        # Pop our own kwargs before delegating (backbone would choke on them)
        fusion = backbone_kwargs.pop('fusion', None)
        keypoint_layers = backbone_kwargs.pop('keypoint_layers', None)
        super().__init__(args, backbone=backbone, fusion=fusion,
                         keypoint_layers=keypoint_layers, **backbone_kwargs)
        embedding_dim = args.embedding_dim
        default_gnn = ['self', 'cross'] * 3
        self.gnn_layer_names = gnn_layers or list(getattr(args, 'gnn_layers', default_gnn))
        self.gnn = AttentionalGNN(embedding_dim, self.gnn_layer_names)
        self.final_proj = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1, bias=True)
        self.bin_score = nn.Parameter(torch.tensor(1.0))
        self.loss_epsilon = 1e-6
        self.match_loss_weight = float(getattr(args, 'match_loss_weight', 1.0))
        self.sinkhorn_iterations = int(getattr(args, 'sinkhorn_iterations', 50))

    def calculate_match_scores(self, d0, d1):
        """(N,D) fused descriptors -> (1,N+1,N+1) log-space assignment scores."""
        g0, g1 = self.gnn(d0.transpose(0, 1).unsqueeze(0), d1.transpose(0, 1).unsqueeze(0))
        m0, m1 = self.final_proj(g0), self.final_proj(g1)
        scores = torch.einsum('bdn,bdm->bnm', m0, m1) / self.args.embedding_dim ** 0.5
        return log_optimal_transport(scores, self.bin_score, iters=self.sinkhorn_iterations)

    def identity_match_loss(self, scores):
        n = scores.shape[1] - 1
        idx = torch.arange(n, device=scores.device)
        return (-torch.log(scores[0, idx, idx].exp() + self.loss_epsilon)).mean()

    def matches_from_scores(self, scores, match_threshold=0.2):
        idx0, idx1, ms0, ms1 = process_scores_into_matches(scores, match_threshold)
        return idx0[0], idx1[0], ms0[0], ms1[0]

    def forward(self, y1, y2, kpts1, kpts2, scores1=None, scores2=None):
        d1 = self.fused_descriptors(y1, kpts1, scores1)
        d2 = self.fused_descriptors(y2, kpts2, scores2)

        z1, z2 = self.projector(d1), self.projector(d2)
        c_features, c_objects = both_correlation_matrices(z1, z2)
        loss_transpose = torch.tensor(0.0, device=y1.device)
        loss_original = torch.tensor(0.0, device=y1.device)
        if self.args.lambd_obj < 1:
            loss_original = self.loss_from_correlation_matrix(c_features)
        if self.args.lambd_obj > 0:
            loss_transpose = self.loss_from_correlation_matrix(c_objects)
        barlow_loss = ((1.0 - self.args.lambd_obj) * loss_original
                       + self.args.lambd_obj * loss_transpose)

        match_scores = self.calculate_match_scores(d1, d2)
        match_loss = self.identity_match_loss(match_scores)

        total = barlow_loss + self.match_loss_weight * match_loss
        return total, loss_original, loss_transpose, match_loss
