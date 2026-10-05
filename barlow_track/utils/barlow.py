# From: http://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf
import concurrent.futures
import gc
from operator import is_
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
import torchio as tio
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from wbfm.utils.general.utils_filenames import pickle_load_binary

from barlow_track.utils.siamese import Siamese
import math
import logging

from barlow_track.utils.data_loading import get_bbox_data_for_volume_with_label
from barlow_track.utils.volume_data import apply_stack_affine, sample_global_affine


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins3d(nn.Module):
    def __init__(self, args, backbone=Siamese, **backbone_kwargs):
        super().__init__()
        self.args = args

        embedding_dim = args.embedding_dim
        self.backbone = backbone(embedding_dim=embedding_dim, **backbone_kwargs)
        self.backbone.fc = nn.Identity()

        # projector
        # NOTE: LayerNorm (per-neuron), NOT BatchNorm. A training batch is a
        # single volume (rows = neurons of that volume), so BatchNorm would
        # normalize each feature across the volume — destroying volume-level
        # information before the loss ever sees it. The Barlow loss would then
        # be blind to shared volume-mean components and could neither train
        # on them nor train them away in the pre-projector (tracking) space.
        # LayerNorm keeps inter-neuron structure intact so the loss pressures
        # the descriptors we actually track with.
        sizes = [embedding_dim] + list(map(int, args.projector.split('-')))
        if 'projector_final' in vars(args):
            # Otherwise assume it's all in the original projector string
            sizes += [args.projector_final]

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

    def embed(self, _y):
        return self.projector(self.backbone(_y))

    def forward(self, y1, y2):
        # Degenerate volumes carry no pairs (std over N<2 is NaN); return
        # grad-free zeros so the training loop skips the optimizer step.
        if y1.shape[0] < 2 or y2.shape[0] < 2:
            z = torch.tensor(0.0, device=y1.device)
            return z, z.clone(), z.clone()
        # Shape of z: neurons x features
        c_features, c_objects = self.calculate_both_correlation_matrices(y1, y2)
        feat_w, obj_w = self._offdiag_weights()
        loss_transpose = torch.tensor(0.0, device=y1.device)
        loss_original = torch.tensor(0.0, device=y1.device)
        if self.args.lambd_obj < 1:
            # Original loss
            loss_original = self.loss_from_correlation_matrix(c_features, feat_w)
        if self.args.lambd_obj > 0:
            # New object loss; use same lambd to combine on and off diagonal
            loss_transpose = self.loss_from_correlation_matrix(c_objects, obj_w)
        
        loss = (1.0-self.args.lambd_obj) * loss_original + self.args.lambd_obj * loss_transpose

        return loss, loss_original, loss_transpose

    def loss_from_correlation_matrix(self, c_features, offdiag_weight=None):
        if offdiag_weight is None:
            offdiag_weight = float(getattr(self.args, 'lambd', 0.0051))
        on_diag = torch.diagonal(c_features).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c_features).pow_(2).sum()
        loss_features = (on_diag + offdiag_weight * off_diag) / c_features.numel()
        return loss_features

    def _offdiag_weights(self):
        """(feature_w, object_w) off-diagonal weights.

        object_w defaults to lambd (back-compat); set lambd_obj_offdiag
        (~0.05-0.3 suggested) to make object-space collapse actually costly
        once the projector is LayerNorm (post-LN the shared mean no longer
        leaks through BN, so the tiny 0.0051 default under-penalizes it).
        """
        feat_w = float(getattr(self.args, 'lambd', 0.0051))
        obj_w = getattr(self.args, 'lambd_obj_offdiag', None)
        obj_w = float(obj_w) if obj_w is not None else feat_w
        return feat_w, obj_w

    def calculate_correlation_matrix(self, y1, y2):
        z1 = self.embed(y1)
        z2 = self.embed(y2)
        # empirical cross-correlation matrix; unbiased=False so that
        # identical views give exactly 1 on the diagonal (std with N-1
        # would give (N-1)/N and a spurious on-diagonal loss).
        z1_norm = (z1 - z1.mean(0)) / z1.std(0, unbiased=False).clamp_min(1e-6)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0, unbiased=False).clamp_min(1e-6)
        this_batch_sz = z1.shape[0]

        c = torch.matmul(z1_norm.T, z2_norm) / this_batch_sz  # D x D (feature space)
        return c

    def calculate_both_correlation_matrices(self, y1, y2):
        """
        Given two images (original and augmented), calculate the correlation matrices for both the feature space and
        the object space.

        Parameters
        ----------
        y1 - torch.Tensor of shape (num_objects (pseudo-batch size), batch_size (1), z, x, y)
        y2 - same as y1

        Returns
        -------

        """
        z1 = self.embed(y1)
        z2 = self.embed(y2)
        # empirical cross-correlation matrix
        z1_norm = (z1 - z1.mean(0)) / z1.std(0, unbiased=False).clamp_min(1e-6)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0, unbiased=False).clamp_min(1e-6)
        this_batch_sz = z1.shape[0]
        c_features = torch.matmul(z1_norm.T, z2_norm) / this_batch_sz  # D x D (feature space)

        # empirical cross-correlation matrix
        z1_norm = ((z1.T - z1.mean(1)) / z1.std(1, unbiased=False).clamp_min(1e-6)).T
        z2_norm = ((z2.T - z2.mean(1)) / z2.std(1, unbiased=False).clamp_min(1e-6)).T
        this_num_features = z1.shape[1]
        c_objects = torch.matmul(z1_norm, z2_norm.T) / this_num_features  # N x N (object space)

        return c_features, c_objects


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class Transform:
    def __init__(self, args=None):
        if args is None:
            args = dict()
        elif not isinstance(args, dict):
            args = vars(args)  # Try to convert from simplenamespace
        if args.get('p_RandomAffine_both', None) is not None:
            args['p_RandomAffine_base'] = args['p_RandomAffine_both']
            args['p_RandomAffine_flip'] = args['p_RandomAffine_both']

        # This normalization should get rid of the noise floor (~100) and keep the actual peak values.
        # (5, 99.5): the 100th percentile (= max) lets one hot voxel compress
        # contrast for the whole frame; kept in sync with volume_data + inference.
        self.final_normalization = tio.RescaleIntensity(percentiles=(5, 99.5))
        self.final_normalization_no_copy = tio.RescaleIntensity(percentiles=(5, 99.5), copy=False)

        # View-consistent ("global") geometry, shared by the whole crop stack
        # per view (same R, t for all crops, preserving relative layout).
        # Active whenever the global_augment settings are present -- including
        # for the image-only path, where the legacy per-crop transforms below
        # still apply afterwards. Absent settings = legacy behavior unchanged.
        self.global_args = args.get('global_augment', None)

        self.transform = tio.transforms.Compose([
            tio.RandomAffine(degrees=(180, 0, 0), p=args.get('p_RandomAffine_base', 1.0)),
            tio.RandomBlur(p=args.get('p_RandomBlur_base', 0.1)),
            self.final_normalization
        ])
        self.transform_prime = tio.transforms.Compose([
            # tio.RandomFlip(axes=(1, 2), p=args.get('p_RandomFlip', 0.0)),  # Do not flip z
            tio.RandomBlur(p=args.get('p_RandomBlur', 0.0)),
            tio.RandomAffine(degrees=(180, 0, 0), p=args.get('p_RandomAffine', 0.1)), 
            tio.RandomAffine(degrees=(180, 180, 0, 0, 0, 0), p=args.get('p_RandomAffine_flip', 0.1)), # A 180 degree rotation, like a flip
            tio.RandomElasticDeformation(max_displacement=args.get('zxy_RandomElasticDeformation', (1, 3, 3)), p=args.get('p_RandomElasticDeformation', 0.0)),
            tio.RandomNoise(std=args.get('std_RandomNoise', 0.25), p=args.get('p_RandomNoise', 0.1)),
            # tio.ZNormalization()
            self.final_normalization
        ])

        print("Initialized Transformation class with augmentation and probabilities:")
        print([(_t.name, _t.probability) for _t in self.transform])
        print([(_t.name, _t.probability) for _t in self.transform_prime])

    def __call__(self, x):
        y1 = self.transform(self._global_view(x))
        y2 = self.transform_prime(self._global_view(x))
        return y1, y2

    def _global_view(self, x):
        """One view-consistent affine over the whole (N, Z, X, Y) crop stack.

        Returns x unchanged when the global_augment settings are absent (or
        its Bernoulli draw fails), so configs without the key behave exactly
        as before.
        """
        if not self.global_args:
            return x
        R, t = sample_global_affine(np.random, **self.global_args)
        return apply_stack_affine(x, R, t)

    def normalize(self, img):
        return self.final_normalization(img)


class NeuronImageWithGTDatasetDense(Dataset):
    """
    Preloads all image data into memory, then returns it as needed. Useful for small datasets.

    See also: NeuronImageWithGTDataset for lazy loading

    """
    def __init__(self, dict_of_neurons_of_volumes, dict_of_ids_of_volumes, which_neurons):
        # In order to synchronize the normalization used
        self._transform = Transform()
        def _normalize(x):
            # Note: applied to crops, not full volumes
            t = self._transform.final_normalization_no_copy
            return t(torch.as_tensor(x.astype(float), dtype=torch.float32))

        logging.info("Normalizing data, can take a while")
        self.dict_all_volume_crops = {i: _normalize(this_vol) for i, this_vol in
                                      tqdm(dict_of_neurons_of_volumes.items())}
        self.dict_of_ids_of_volumes = dict_of_ids_of_volumes
        self.which_neurons = which_neurons

    def __getitem__(self, idx):
        if idx not in self.dict_of_ids_of_volumes:
            raise IndexError   # Make basic looping work with pytorch
        x = torch.unsqueeze(self.dict_all_volume_crops[idx], 0)
        gt_id = self.dict_of_ids_of_volumes[idx]
        return x, gt_id

    def __len__(self):
        return len(self.dict_all_volume_crops)

    @staticmethod
    def load_from_project(project_data, num_frames, target_sz):
        # TODO: refactor to be a lazy loader of the cropped data
        project_data.project_config.logger.info("Loading image data from project")
        if num_frames is None:
            num_frames = project_data.num_frames

        dict_of_neurons_of_volumes, dict_of_ids_of_volumes = {}, {}

        def parallel_func(_t):
            all_dat_dict, all_seg_dict, which_neurons = get_bbox_data_for_volume_with_label(project_data, _t,
                                                                                            target_sz=target_sz)
            keys = list(all_dat_dict.keys())  # Need to enforce ordering?
            if len(keys) > 0:
                keys.sort()
                dict_of_ids_of_volumes[_t] = keys  # strings
                dict_of_neurons_of_volumes[_t] = np.stack([all_dat_dict[k] for k in keys], 0)
            else:
                dict_of_ids_of_volumes[_t] = []
                dict_of_neurons_of_volumes[_t] = np.zeros((0, *target_sz))

        which_neurons = project_data.get_list_of_finished_neurons()[1]

        with tqdm(total=num_frames) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(parallel_func, i): i for i in list(range(num_frames))}
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    pbar.update(1)

        return NeuronImageWithGTDatasetDense(dict_of_neurons_of_volumes, dict_of_ids_of_volumes, which_neurons)


class NeuronImageWithGTDataset(Dataset):
    """
    Lazy loaded version of NeuronImageWithGTDatasetDense

    """
    def __init__(self, project_data, num_frames, target_sz, include_untracked=False):
        # In order to synchronize the normalization used
        self._transform = Transform()
        self.num_frames = int(num_frames)
        self.project_data = project_data
        self.target_sz = target_sz
        self.which_neurons = project_data.get_list_of_finished_neurons()[1]
        self.include_untracked = include_untracked

        try:
            _ = self.project_data.segmentation_metadata
        except FileNotFoundError:
            assert project_data.intermediate_global_tracks is not None, "Need either raw segmentation metadata or intermediate_global_tracks"
            logging.warning("No raw segmentation metadata found, using intermediate_global_tracks instead")

    def _normalize(self, x):
        # Note: applied to crops, not full volumes
        t = self._transform.final_normalization_no_copy
        return t(torch.as_tensor(x.astype(float), dtype=torch.float32))

    def __getitem__(self, idx):
        if idx > self.num_frames - 1:
            raise IndexError   # Make basic looping work with pytorch
        # Get data from the lazy loader
        x, gt_id = self.get_neurons_single_volume(idx)

        # Normalize, unsqueeze, and return
        x = torch.unsqueeze(self._normalize(x), 0)

        return x, gt_id

    def __len__(self):
        return self.num_frames

    @staticmethod
    def _fix_empty_volume(neurons_in_single_volume, target_sz):
        keys = list(neurons_in_single_volume.keys())  # Need to enforce ordering?
        if len(keys) > 0:
            keys.sort()
            ids = keys  # strings
            cropped_neuron_data = np.stack([neurons_in_single_volume[k] for k in keys], 0)
        else:
            ids = []
            cropped_neuron_data = np.zeros((0, *target_sz))
        return cropped_neuron_data, ids

    def get_neurons_single_volume(self, _t):
        neurons_in_single_volume, _, _ = get_bbox_data_for_volume_with_label(self.project_data, _t,
                                                                             target_sz=self.target_sz,
                                                                             include_untracked=self.include_untracked)
        return self._fix_empty_volume(neurons_in_single_volume, self.target_sz)
    
    def __repr__(self):
        return f"NeuronImageWithGTDataset(num_frames={self.num_frames}, target_sz={self.target_sz}, " \
               f"include_untracked={self.include_untracked})"


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def print_all_on_gpu():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def _get_arg(args, name, default=None):
    """Read an arch field from a SimpleNamespace or dict (checkpoints predate some keys)."""
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _set_arg(args, name, value):
    """Set an arch field on a SimpleNamespace or dict (legacy migration)."""
    if isinstance(args, dict):
        args[name] = value
    else:
        setattr(args, name, value)


def _backbone_params(args):
    """(num_levels, f_maps) with the same defaults as fresh training."""
    kwargs = _get_arg(args, 'backbone_kwargs', None) or {}
    if not isinstance(kwargs, dict):
        try:
            kwargs = vars(kwargs)
        except TypeError:
            kwargs = {}
    return kwargs.get('num_levels', 2), kwargs.get('f_maps', 4)


def _target_sz_tuple(args):
    """(z, xy) handling the legacy target_sz array fallback."""
    z = _get_arg(args, 'target_sz_z', None)
    xy = _get_arg(args, 'target_sz_xy', None)
    if z is not None and xy is not None:
        return z, xy
    legacy = _get_arg(args, 'target_sz', None)
    if legacy is not None:
        try:
            return legacy[0], legacy[1]
        except (TypeError, IndexError, KeyError):
            pass
    return None, None


def _expected_model_type(args):
    if _get_arg(args, 'use_attention', False):
        return 'attention'
    if _get_arg(args, 'use_position', False):
        return 'position'
    return _get_arg(args, 'model_type', 'barlow') or 'barlow'


class PretrainedArchitectureMismatchError(ValueError):
    """Checkpoint architecture differs from the current config (fine-tuning setup error)."""


def compare_model_architecture(expected_args, pretrained_args):
    """List human-readable mismatches between config and checkpoint architecture.

    Returns [] when architectures match. Each entry is
    "field: config=<...> vs checkpoint=<...>".
    """
    mismatches = []
    pre_type = _get_arg(pretrained_args, 'model_type', 'barlow') or 'barlow'
    exp_type = _expected_model_type(expected_args)
    if exp_type != pre_type:
        mismatches.append(f"model_type: config={exp_type!r} vs checkpoint={pre_type!r} "
                          f"(config use_position={_get_arg(expected_args, 'use_position', False)!r}, "
                          f"use_attention={_get_arg(expected_args, 'use_attention', False)!r})")
    for field in ('embedding_dim', 'projector', 'projector_final'):
        curr, pre = _get_arg(expected_args, field), _get_arg(pretrained_args, field)
        if curr is not None and pre is not None and curr != pre:
            mismatches.append(f"{field}: config={curr!r} vs checkpoint={pre!r}")
    curr_lv, curr_fm = _backbone_params(expected_args)
    pre_lv, pre_fm = _backbone_params(pretrained_args)
    if curr_lv != pre_lv:
        mismatches.append(f"backbone num_levels: config={curr_lv!r} vs checkpoint={pre_lv!r}")
    if curr_fm != pre_fm:
        mismatches.append(f"backbone f_maps: config={curr_fm!r} vs checkpoint={pre_fm!r}")
    curr_z, curr_xy = _target_sz_tuple(expected_args)
    pre_z, pre_xy = _target_sz_tuple(pretrained_args)
    if curr_z is not None and pre_z is not None and curr_z != pre_z:
        mismatches.append(f"target_sz_z: config={curr_z!r} vs checkpoint={pre_z!r}")
    if curr_xy is not None and pre_xy is not None and curr_xy != pre_xy:
        mismatches.append(f"target_sz_xy: config={curr_xy!r} vs checkpoint={pre_xy!r}")
    position_family = {'position', 'attention', 'superglue'}
    if exp_type in position_family or pre_type in position_family:
        # Fresh-training default is layernorm; legacy checkpoints without the
        # field keep 'none' (pinned in load_barlow_model).
        curr_fn = _get_arg(expected_args, 'fusion_norm', 'layernorm')
        pre_fn = _get_arg(pretrained_args, 'fusion_norm', 'none')
        if curr_fn != pre_fn:
            mismatches.append(f"fusion_norm: config={curr_fn!r} vs checkpoint={pre_fn!r}")
        curr_f, pre_f = _get_arg(expected_args, 'fusion', 'concat'), _get_arg(pretrained_args, 'fusion', 'concat')
        if curr_f != pre_f:
            mismatches.append(f"fusion: config={curr_f!r} vs checkpoint={pre_f!r}")
        curr_k, pre_k = list(_get_arg(expected_args, 'keypoint_encoder_layers', [32, 64])), \
            list(_get_arg(pretrained_args, 'keypoint_encoder_layers', [32, 64]))
        if curr_k != pre_k:
            mismatches.append(f"keypoint_encoder_layers: config={curr_k!r} vs checkpoint={pre_k!r}")
    if exp_type == 'attention' or pre_type == 'attention':
        curr_s, pre_s = int(_get_arg(expected_args, 'self_layers', 2)), \
            int(_get_arg(pretrained_args, 'self_layers', 2))
        if curr_s != pre_s:
            mismatches.append(f"self_layers: config={curr_s!r} vs checkpoint={pre_s!r}")
        curr_c, pre_c = bool(_get_arg(expected_args, 'center_per_volume', True)), \
            bool(_get_arg(pretrained_args, 'center_per_volume', False))
        if curr_c != pre_c:
            mismatches.append(f"center_per_volume: config={curr_c!r} vs checkpoint={pre_c!r}")
    return mismatches


def load_barlow_model(model_fname, expected_args=None):
    """
    Loads a model directly from the weights file, and assumes the args are saved in the same folder as args.pickle

    If expected_args (current config) is given, the checkpoint architecture is
    compared against it first and a ValueError listing every mismatch is raised.
    """
    if model_fname is None or not Path(model_fname).exists():
        raise FileNotFoundError(f"Model file not found: {model_fname}")
    from barlow_track.utils.siamese import ResidualEncoder3D
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_fname, map_location=gpu)
    if state_dict.get('model', None) is not None:
        # Then this was a saved checkpoint, and we should load just the model
        state_dict = state_dict['model']
    logging.info(f"Using device: {gpu}")
    # Check if there are multiple .pickle files in the same folder, and give a warning if so
    for fname in Path(model_fname).parent.glob('*.pickle'):
        if fname.name != 'args.pickle':
            logging.warning(f"Found extra pickle file in model folder: {fname}... "
                            f"Make sure args.pickle is the correct file!")
    # Possible problem: multiple models saved in the same folder with different settings
    args_fname = Path(model_fname).with_name('args.pickle')
    args = pickle_load_binary(args_fname)
    logging.info(f"Loaded args from {args_fname}: {args}")
    # Legacy migration: checkpoints predating a field must keep legacy
    # behavior, not the current fresh-training default. For fusion_norm,
    # infer from the weights when possible: new checkpoints carry
    # norm_visual/norm_pos params, legacy ones do not.
    if _get_arg(args, 'fusion_norm', None) is None:
        _has_norm = any(
            k in ('norm_visual.weight', 'norm_pos.weight')
            or k.endswith('.norm_visual.weight')
            or k.endswith('.norm_pos.weight')
            for k in state_dict.keys()
        )
        _set_arg(args, 'fusion_norm', 'layernorm' if _has_norm else 'none')
    if _get_arg(args, 'center_per_volume', None) is None:
        _set_arg(args, 'center_per_volume', False)
    if expected_args is not None:
        mismatches = compare_model_architecture(expected_args, args)
        if mismatches:
            details = "\n".join(f"  - {m}" for m in mismatches)
            raise PretrainedArchitectureMismatchError(
                f"Pretrained model architecture does not match current config ({model_fname}).\n"
                f"{details}\n"
                f"Checkpoint args were loaded from {args_fname}.\n"
                f"Align the config with the checkpoint values above, or train from scratch "
                f"with pretrained_model_path set to null."
            )
    try:
        target_sz = np.array([args.target_sz_z, args.target_sz_xy, args.target_sz_xy])
    except AttributeError:
        target_sz = np.array(args.target_sz)
    num_levels, f_maps = _backbone_params(args)
    backbone_kwargs = dict(in_channels=1, num_levels=num_levels, f_maps=f_maps, crop_sz=target_sz)
    model_type = getattr(args, 'model_type', 'barlow')
    if model_type == 'superglue':
        from barlow_track.utils.barlow_superglue import BarlowSuperGlue
        model = BarlowSuperGlue(args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)
    elif model_type == 'attention':
        from barlow_track.utils.barlow_superglue import BarlowVolumeAttention
        model = BarlowVolumeAttention(args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)
    elif model_type == 'position':
        from barlow_track.utils.barlow_superglue import BarlowWithPosition
        model = BarlowWithPosition(args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)
    else:
        model = BarlowTwins3d(args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Backward compat: checkpoints predating gated attention (attn_gate,
        # norm_context) and the LayerNorm-terminated fusion MLP have missing
        # keys only. Load what matches and keep fresh inits for the rest
        # (gate starts near-identity by design, so old weights behave like
        # the old model on load). Checkpoints predating the projector
        # BatchNorm->LayerNorm switch additionally carry stale BN buffers
        # (running_mean/var); those are ignored with a warning. BN affine
        # weight/bias share key names and shapes with the new LayerNorm and
        # load directly — fine for backbone/fused/contextual eval (the
        # projector is discarded for tracking); retrain if you care about
        # the 'projected' stage.
        allowed_prefixes = ('attn_gate', 'norm_context.', 'fuse_mlp.2.',
                            'fuse_mlp.3.', 'fuse_mlp.4.')
        allowed_stale_suffixes = ('.running_mean', '.running_var',
                                  '.num_batches_tracked')
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            missing = list(missing)
            stale_bn = [u for u in unexpected
                        if u.startswith('projector.') and u.endswith(allowed_stale_suffixes)]
            unexpected = [u for u in unexpected if u not in stale_bn]
            if unexpected or any(not m.startswith(allowed_prefixes) for m in missing):
                raise RuntimeError(
                    f"Could not load weights from {model_fname} into a '{model_type}' model "
                    f"(embedding_dim={getattr(args, 'embedding_dim', '?')}). "
                    f"If you changed the architecture, align the config with the checkpoint "
                    f"or train from scratch with pretrained_model_path set to null. "
                    f"Original error: {e}"
                ) from e
            logging.warning(f"Checkpoint {model_fname} predates gated attention / "
                            f"LayerNorm fusion MLP; {len(missing)} new params "
                            f"kept at init: {sorted(missing)}")
            if stale_bn:
                logging.warning(f"Checkpoint {model_fname} predates the projector "
                                f"BatchNorm->LayerNorm switch; {len(stale_bn)} stale "
                                f"BN buffers ignored: {sorted(stale_bn)}")
        except RuntimeError as e2:
            # If the strict=False path itself raised a different error
            # (e.g. shape mismatch), report the original failure.
            if 'Could not load weights' in str(e2):
                raise
            raise RuntimeError(
                f"Could not load weights from {model_fname} into a '{model_type}' model "
                f"(embedding_dim={getattr(args, 'embedding_dim', '?')}). "
                f"If you changed the architecture, align the config with the checkpoint "
                f"or train from scratch with pretrained_model_path set to null. "
                f"Original error: {e}"
            ) from e
    return gpu, model, args
