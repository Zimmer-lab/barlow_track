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
    z1_norm = (z1 - z1.mean(0)) / z1.std(0, unbiased=False).clamp_min(1e-6)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0, unbiased=False).clamp_min(1e-6)
    c_features = torch.matmul(z1_norm.T, z2_norm) / z1.shape[0]

    z1_t = ((z1.T - z1.mean(1)) / z1.std(1, unbiased=False).clamp_min(1e-6)).T
    z2_t = ((z2.T - z2.mean(1)) / z2.std(1, unbiased=False).clamp_min(1e-6)).T
    c_objects = torch.matmul(z1_t, z2_t.T) / z1.shape[1]
    return c_features, c_objects


def standardization_health(z1, z2, clamp_thresh=1e-6):
    """Pre-standardization batch statistics for the Barlow correlation.

    z1, z2: (N, D) embeddings about to be correlated (same pair the loss
    uses). Returns plain floats: frac of dims whose bottleneck std hits the
    loss clamp (std <= thresh in either view), plus min/median/mean/max of
    the per-dim std spectrum. A rising clamp rate means dims are dying
    (constant across the batch, so the loss standardizes noise); a
    collapsing spectrum with a flat loss means the loss has no signal left
    to descend on. No grad; safe to call on any device.
    """
    with torch.no_grad():
        s1 = z1.float().std(dim=0, unbiased=False)
        s2 = z2.float().std(dim=0, unbiased=False)
        s = torch.minimum(s1, s2)  # bottleneck view per dim
        finite = torch.isfinite(s)
        if int(finite.sum()) == 0:
            return dict(clamp_frac=float('nan'), std_min=float('nan'),
                        std_median=float('nan'), std_mean=float('nan'),
                        std_max=float('nan'))
        s = s[finite]
        return dict(clamp_frac=float((s <= clamp_thresh).float().mean().cpu()),
                    std_min=float(s.min().cpu()),
                    std_median=float(s.median().cpu()),
                    std_mean=float(s.mean().cpu()),
                    std_max=float(s.max().cpu()))


def intersection_gather(idx1, idx2):
    """Align two independent dropout views on their surviving intersection.

    idx1/idx2: (N1,)/(N2,) original object ids (sorted, unique). Returns
    (sel1, sel2): index tensors gathering the COMMON ids in the same order,
    so z1[sel1] <-> z2[sel2] are paired. Empty when there is no overlap.
    """
    if idx1 is None or idx2 is None:
        return None, None
    i1 = idx1.reshape(-1).tolist()
    i2 = idx2.reshape(-1).tolist()
    set2 = set(i2)
    common = [v for v in i1 if v in set2]
    if not common:
        dev = idx1.device
        return (torch.empty(0, dtype=torch.long, device=dev),
                torch.empty(0, dtype=torch.long, device=dev))
    pos2 = {v: j for j, v in enumerate(i2)}
    sel1 = torch.as_tensor([j for j, v in enumerate(i1) if v in set2],
                           dtype=torch.long, device=idx1.device)
    sel2 = torch.as_tensor([pos2[v] for v in common],
                           dtype=torch.long, device=idx2.device)
    return sel1, sel2


def _zero_losses(device):
    z = torch.tensor(0.0, device=device)
    return z, z.clone(), z.clone()


@torch.no_grad()
def descriptor_health(C, dead_rel_tol=0.01):
    """Anti-collapse health metrics for one (N, D) pre-projector descriptor set.

    Returns dict with:
      eff_rank: exp(entropy of normalized singular values), in [1, min(N,D)]
      offdiag_corr: mean |off-diagonal| of the DxD feature correlation matrix
      dead_frac: fraction of dims with std < dead_rel_tol * max dim std
      mean_frac: ||per-volume mean|| / mean per-row ||.|| in [0, 1]; high
        means descriptors are dominated by a shared volume-mean component
        (the diagnosed dense-volume collapse signature c_i ~= m + r_i).
    Returns NaNs when N < 2 or D < 2.
    """
    C = C.detach().float()
    n, d = C.shape
    nan = float('nan')
    if n < 2 or d < 2:
        return dict(eff_rank=nan, offdiag_corr=nan, dead_frac=nan, mean_frac=nan)
    centered = C - C.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(centered)
    except RuntimeError:
        return dict(eff_rank=nan, offdiag_corr=nan, dead_frac=nan, mean_frac=nan)
    s = s[s > 0]
    if s.numel() == 0:
        eff_rank = 1.0
    else:
        p = s / s.sum()
        eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()).cpu())
    std = centered.std(dim=0)
    denom = (centered * centered).mean().sqrt().clamp_min(1e-12)
    corr = (centered.T @ centered) / n / (denom * denom)
    corr = corr.clamp(-1.0, 1.0)
    mask = ~torch.eye(d, dtype=torch.bool, device=C.device)
    offdiag = float(corr[mask].abs().mean().cpu())
    max_std = float(std.max().cpu())
    if max_std <= 0:
        dead_frac = 1.0
    else:
        dead_frac = float((std < dead_rel_tol * max_std).float().mean().cpu())
    mean_vec = C.mean(dim=0)
    row_norms = C.norm(dim=1).mean().clamp_min(1e-12)
    mean_frac = float((mean_vec.norm() / row_norms).cpu())
    return dict(eff_rank=eff_rank, offdiag_corr=offdiag,
                dead_frac=dead_frac, mean_frac=mean_frac)


@torch.no_grad()
def attention_entropy_normed(model):
    """Mean attention entropy / log(N), averaged over stored probs.

    Reads the prob lists that AttentionalGNN stashed during the last
    contextualize() call. Returns NaN when no probs are stored (e.g.
    self_layers=0 or single-detection passthrough). Near 1.0 means
    diffuse/uniform attention (averaging risk); near 0.0 means peaked.
    """
    probs = []
    gnn = getattr(model, 'self_gnn', None)
    for layer in getattr(gnn, 'layers', []) or []:
        for p in getattr(getattr(layer, 'attn', None), 'prob', []) or []:
            probs.append(p.detach().float())
    if not probs:
        return float('nan')
    ents = []
    for p in probs:
        n = p.shape[-1]
        if n < 2:
            continue
        p = p.clamp_min(1e-12)
        ent = -(p * p.log()).sum(dim=-1).mean() / torch.tensor(n, dtype=torch.float32).log()
        ents.append(float(ent.cpu()))
    return float(sum(ents) / len(ents)) if ents else float('nan')


@torch.no_grad()
def position_jitter_sensitivity(model, kpts, eps=0.02):
    """Relative change in position descriptors under small keypoint noise.

    ||P(k+e) - P(k)||_F / ||P(k)||_F. NaN for N < 2 (fallback path) or
    zero-norm descriptors. Healthy models respond to geometry but do not
    explode; ~0 with large tracking error means the position branch is dead.
    """
    if kpts.shape[0] < 2 or not hasattr(model, 'encode_position'):
        return float('nan')
    try:
        base = model.encode_position(kpts).float()
        denom = base.norm().clamp_min(1e-12)
        if not torch.isfinite(denom) or float(denom) == 0:
            return float('nan')
        noisy = model.encode_position(
            kpts + torch.randn_like(kpts) * eps).float()
        return float(((noisy - base).norm() / denom).cpu())
    except (RuntimeError, ValueError):
        return float('nan')


@torch.no_grad()
def volume_descriptor_diagnostics(model, y, kpts, scores=None):
    """One-volume anti-collapse report on pre-projector descriptors.

    Runs fused (+ contextual when available) descriptors without gradients
    and returns a flat dict: fused_*/contextual_* health, attention_entropy,
    attn_gate (when gated), and position jitter sensitivity. Used by the
    training validation loop; N < 2 volumes yield NaNs rather than crashing.
    """
    out = {}
    try:
        fused = model.fused_descriptors(y, kpts, scores) if hasattr(
            model, 'fused_descriptors') else model.backbone(y)
    except (RuntimeError, ValueError):
        return out
    for k, v in descriptor_health(fused).items():
        out[f'fused_{k}'] = v
    if hasattr(model, 'contextual_descriptors'):
        try:
            ctx = model.contextual_descriptors(y, kpts, scores)
        except (RuntimeError, ValueError):
            ctx = None
        if ctx is not None:
            for k, v in descriptor_health(ctx).items():
                out[f'contextual_{k}'] = v
    out['attention_entropy'] = attention_entropy_normed(model)
    gate = getattr(model, 'attn_gate', None)
    if gate is not None:
        out['attn_gate'] = float(torch.sigmoid(gate.detach()).cpu())
    out['position_jitter_sensitivity'] = position_jitter_sensitivity(
        model, kpts)
    return out


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


def canonicalize_keypoints(kpts, eps=1e-6):
    """Per-volume data centering + scaling: (p - mu) / (s + eps).

    mu = volume centroid (removes translation), s = mean radius (removes
    overall scale). Stateless and identical in train and eval. This is NOT
    full affine invariance -- rotation and shear remain -- it is the stable
    baseline that stops asking a pointwise MLP to learn translation
    invariance from augmentation. Applied at the KENC input (see
    BarlowWithPosition.encode_position) so all callers share it.
    """
    mu = kpts.mean(dim=0, keepdim=True)
    s = (kpts - mu).norm(dim=1).mean().clamp_min(eps)
    return (kpts - mu) / s


class ViSNetPositionEncoder(nn.Module):
    """Rotation/translation-invariant position descriptors via ViSNet.

    Each neuron is an "atom" of a single pseudo-element; invariant scalar
    node features from ViSNetBlock (relative vectors + spherical harmonics)
    are projected to the descriptor dim. Invariant to translation and
    rotation by construction; NOT scale-invariant (inputs are the usual
    volume-normalized keypoints, so scale is fixed per dataset -- combine
    with canonicalize_keypoints for scale removal). Replaces KeypointEncoder
    when position_encoder='visnet'. cutoff covers the normalized volume
    diameter, so the graph is fully connected (no isolated nodes).
    """
    def __init__(self, output_dim, hidden_channels=64, num_layers=3, cutoff=4.0,
                 max_num_neighbors=64):
        super().__init__()
        from torch_geometric.nn.models import ViSNet
        self.repr = ViSNet(lmax=1, num_heads=4, num_layers=num_layers,
                           hidden_channels=hidden_channels, cutoff=cutoff,
                           max_num_neighbors=max_num_neighbors).representation_model
        self.head = nn.Linear(hidden_channels, output_dim)

    def forward(self, kpts):
        # kpts: (N, 3) float; returns (N, output_dim)
        dev = kpts.device
        z = torch.zeros(kpts.shape[0], dtype=torch.long, device=dev)
        batch = torch.zeros(kpts.shape[0], dtype=torch.long, device=dev)
        x, _ = self.repr(z, kpts.float(), batch)
        return self.head(x)


class BarlowWithPosition(BarlowTwins3d):
    """BarlowTwins3d + KeypointEncoder position fusion (Step 2)."""

    def __init__(self, args, backbone=Siamese, fusion=None, keypoint_layers=None,
                 fusion_norm=None, **backbone_kwargs):
        super().__init__(args, backbone=backbone, **backbone_kwargs)
        embedding_dim = args.embedding_dim
        self.embedding_dim = embedding_dim
        self.fusion = fusion or getattr(args, 'fusion', 'concat')
        # Default 'layernorm': per-branch LayerNorm stops the position branch
        # from dominating via scale (~25:1 init imbalance with 'none').
        # 'none' is kept for loading legacy checkpoints (see load_barlow_model
        # migration, which pins missing fields to legacy values).
        self.fusion_norm = fusion_norm or getattr(args, 'fusion_norm', 'layernorm')
        layers = keypoint_layers or list(getattr(args, 'keypoint_encoder_layers', [32, 64]))
        self.kenc = KeypointEncoder(embedding_dim, layers)
        # Alternate rotation-invariant position branch (default keypoint_mlp
        # keeps legacy behavior; see ViSNetPositionEncoder).
        self.pos_encoder = getattr(args, 'position_encoder', 'keypoint_mlp')
        if self.pos_encoder == 'visnet':
            self.visnet_enc = ViSNetPositionEncoder(
                embedding_dim,
                hidden_channels=int(getattr(args, 'visnet_hidden', 64)),
                num_layers=int(getattr(args, 'visnet_layers', 3)))
        elif self.pos_encoder != 'keypoint_mlp':
            raise ValueError(f"Unknown position_encoder '{self.pos_encoder}'; use 'keypoint_mlp' or 'visnet'")
        self.norm_visual = _make_norm(self.fusion_norm, embedding_dim)
        self.norm_pos = _make_norm(self.fusion_norm, embedding_dim)
        if self.fusion == 'concat':
            # LayerNorm-terminated MLP (no bare terminal ReLU): the final
            # LayerNorm IS the post-fusion normalization, so no separate
            # norm_fused layer is needed. This fixes the dead-dim failure
            # mode of the old Linear+ReLU head and stops the position
            # branch from dominating via scale.
            self.fuse_mlp = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
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
        if getattr(self.args, 'canonicalize_keypoints', False):
            kpts = canonicalize_keypoints(kpts)
        if getattr(self, 'pos_encoder', 'keypoint_mlp') == 'visnet':
            return self.visnet_enc(kpts)
        if scores is None:
            scores = kpts.new_ones(kpts.shape[0])
        k = kpts.reshape(1, 1, -1, 3)
        s = scores.reshape(1, 1, -1)
        return self.kenc(k, s).squeeze(0).transpose(0, 1)

    def fused_descriptors(self, y, kpts, scores=None):
        if kpts.shape[0] < 2:
            # Truly visual-only fallback: skip norm_pos entirely. LayerNorm
            # of a zero vector would otherwise inject the learned bias as a
            # constant offset after training.
            pos = self.encode_position(kpts, scores)
        else:
            pos = self.norm_pos(self.encode_position(kpts, scores))
        if self.fusion == 'position_only':
            return pos  # visual branch unused: pure geometry benchmark
        visual = self.norm_visual(self.backbone(y))
        if self.fusion == 'add':
            return visual + pos
        return self.fuse_mlp(torch.cat([visual, pos], dim=1))

    def embed_with_position(self, y, kpts, scores=None):
        return self.projector(self.fused_descriptors(y, kpts, scores))

    def _paired_for_loss(self, z1, z2, idx1=None, idx2=None):
        """Full-context embeddings -> intersection-paired rows for the loss.

        Attention/contextualization already saw each view's full neighbor set;
        the Barlow loss itself needs paired rows, so gather the common ids.
        Returns None when the intersection has < 2 objects (no usable pairs).
        """
        if idx1 is None or idx2 is None:
            return z1, z2
        sel1, sel2 = intersection_gather(idx1, idx2)
        if sel1 is None or len(sel1) < 2:
            return None
        return z1[sel1], z2[sel2]

    def forward(self, y1, y2, kpts1, kpts2, scores1=None, scores2=None,
                idx1=None, idx2=None):
        # Views that lost (almost) all objects to out-of-bounds filtering
        # carry no usable pairs; skip before embedding (avoids NaN std).
        if y1.shape[0] < 2 or y2.shape[0] < 2:
            dev = y1.device if isinstance(y1, torch.Tensor) else kpts1.device
            return _zero_losses(dev)
        z1 = self.embed_with_position(y1, kpts1, scores1)
        z2 = self.embed_with_position(y2, kpts2, scores2)
        paired = self._paired_for_loss(z1, z2, idx1, idx2)
        if paired is None:
            return _zero_losses(y1.device)
        z1, z2 = paired
        c_features, c_objects = both_correlation_matrices(z1, z2)

        feat_w, obj_w = self._offdiag_weights()
        loss_transpose = torch.tensor(0.0, device=y1.device)
        loss_original = torch.tensor(0.0, device=y1.device)
        if self.args.lambd_obj < 1:
            loss_original = self.loss_from_correlation_matrix(c_features, feat_w)
        if self.args.lambd_obj > 0:
            loss_transpose = self.loss_from_correlation_matrix(c_objects, obj_w)

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
        # C2: pre-attention volume-mean centering. A shared per-volume
        # component (c_i ~= m + r_i) is frame identity, not neuron identity:
        # it cancels within a frame but gates the cross-frame kNN budget and
        # gets metric-amplified by SVD50. Subtracting the per-volume mean
        # before self-attention kills the span(1)/DC attractor the softmax
        # otherwise diffuses toward. Per-row LayerNorm (norm_context) does NOT
        # do this (it removes per-row means, leaving centered-m shared).
        # Gated residual: attention starts "almost off" (sigmoid(-4) ~= 0.018)
        # and training must prove it useful.
        self.center_per_volume = bool(getattr(args, 'center_per_volume', True))
        gate_init = float(getattr(args, 'attn_gate_init', -4.0))
        self.attn_gate = nn.Parameter(torch.tensor(gate_init))
        self.norm_context = nn.LayerNorm(args.embedding_dim)

    def attention_gate_value(self):
        """Scalar gate in (0, 1); near 0 means attention is ~identity."""
        return float(torch.sigmoid(self.attn_gate).detach().cpu())

    def contextualize(self, d):
        """(N,D) fused descriptors -> (N,D) volume-contextualized descriptors.

        With center_per_volume (default), the per-volume mean is subtracted
        before self-attention so a shared frame component cannot become the
        attention attractor; the residual + LayerNorm then operate in centered
        space. Single detections bypass attention (nothing to attend to, and
        the propagation MLP's InstanceNorm needs N > 1).
        """
        if d.shape[0] < 2 or isinstance(self.self_gnn, nn.Identity):
            return d
        dc = d - d.mean(dim=0, keepdim=True) if self.center_per_volume else d
        batch = dc.transpose(0, 1).unsqueeze(0)
        out, _ = self.self_gnn(batch, batch)
        out = out.squeeze(0).transpose(0, 1)
        gate = torch.sigmoid(self.attn_gate)
        return self.norm_context(dc + gate * (out - dc))

    def contextual_descriptors(self, y, kpts, scores=None):
        return self.contextualize(self.fused_descriptors(y, kpts, scores))

    def embed_with_position(self, y, kpts, scores=None):
        return self.projector(self.contextual_descriptors(y, kpts, scores))

    def forward(self, y1, y2, kpts1, kpts2, scores1=None, scores2=None,
                idx1=None, idx2=None):
        if y1.shape[0] < 2 or y2.shape[0] < 2:
            dev = y1.device if isinstance(y1, torch.Tensor) else kpts1.device
            return _zero_losses(dev)
        z1 = self.embed_with_position(y1, kpts1, scores1)
        z2 = self.embed_with_position(y2, kpts2, scores2)
        paired = self._paired_for_loss(z1, z2, idx1, idx2)
        if paired is None:
            return _zero_losses(y1.device)
        z1, z2 = paired
        c_features, c_objects = both_correlation_matrices(z1, z2)

        feat_w, obj_w = self._offdiag_weights()
        loss_transpose = torch.tensor(0.0, device=y1.device)
        loss_original = torch.tensor(0.0, device=y1.device)
        if self.args.lambd_obj < 1:
            loss_original = self.loss_from_correlation_matrix(c_features, feat_w)
        if self.args.lambd_obj > 0:
            loss_transpose = self.loss_from_correlation_matrix(c_objects, obj_w)

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
