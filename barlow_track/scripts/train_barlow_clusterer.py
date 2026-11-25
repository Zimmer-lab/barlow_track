# Load a project and data, then train a Siamese network
import argparse
import json
import logging
import math
import os
import pickle
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

# Headless NWB loads keep the segmentation lazy (wbfm reads one volume at a time instead of the
# whole array: 35 GB for the Leifer re-export, 7 GB for Flavell, on every ProjectData load).
# Ignored by wbfm versions without the flag; set WBFM_NWB_LAZY_SEGMENTATION=0 to turn it off.
os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
import numpy as np
import torch
import wandb
from ruamel.yaml import YAML

from barlow_track.utils.barlow import BarlowTwins3d, load_barlow_model
from barlow_track.utils.barlow_lightning import NeuronCropImageDataModule
from barlow_track.utils.barlow_superglue import BarlowVolumeAttention, BarlowWithPosition
from barlow_track.utils.barlow_visualize import visualize_model_performance
from barlow_track.utils.siamese import ResidualEncoder3D

from barlow_track.utils.utils_seeding import derive_seed

from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.general.utils_filenames import get_sequential_filename


def _keypoints_only(args):
    """keypoints_only (default off): skip volumes and crops in the position data path.

    Only position-only networks may set it; any other model would train on zero crops.
    """
    if not getattr(args, 'keypoints_only', False):
        return False
    if getattr(args, 'fusion', None) != 'position_only' or getattr(args, 'use_attention', False):
        raise ValueError("keypoints_only needs fusion: position_only (and no attention): "
                         "every other model reads the crops")
    return True


# Keys that configs carry but training never reads, with what decides instead
UNREAD_CONFIG_KEYS = {
    'training_fraction': "the data modules' train_fraction default (0.8)",
    'p_RandomNoise_base': 'p_RandomNoise (legacy crops) or photometric p_noise (position pipeline)',
}


def _warn_unread_config_keys(args):
    for key, instead in UNREAD_CONFIG_KEYS.items():
        if hasattr(args, key):
            logging.warning(f"Config key {key}={getattr(args, key)!r} is never read; {instead} decides")
    if getattr(args, 'batch_size', 1) != 1:
        logging.warning(f"Config key batch_size={args.batch_size!r} is ignored: a batch is one volume")


def _auto_num_workers(args, data_module):
    """DataLoader workers for the position pipeline (on unless num_workers_auto: false).

    A config's num_workers > 0 is used as is. With 0 (the default), a Slurm GPU job that
    crops volumes from a zarr store gets SLURM_CPUS_PER_TASK - 2 workers: the
    full-volume augmentation, not the GPU, bounds those epochs (Zimmer ~3.5 s per item
    without workers), and the views are identical with any worker count (per-item seeds;
    check_workers_real_data.py on five recordings). Anything else keeps 0: CPU jobs
    need their cores for torch, keypoints_only reads no volume, and other stores (h5 /
    NWB) are not safe to read from forked workers. One batch is drawn with workers first;
    if that fails (e.g. a small /dev/shm) training falls back to 0. The global torch RNG
    is restored after that draw, so the run matches num_workers 0.
    """
    n = int(getattr(args, 'num_workers', 0) or 0)
    cpus = os.getenv('SLURM_CPUS_PER_TASK')
    if n > 0 or not bool(getattr(args, 'num_workers_auto', True)) or not cpus \
            or not torch.cuda.is_available() or _keypoints_only(args):
        return n
    import zarr
    red = data_module.alldata.project_data.red_data
    if not isinstance(red, zarr.Array):
        print(f"num_workers 0: red_data is {type(red).__name__}, not a zarr array", flush=True)
        return 0
    n = max(int(cpus) - 2, 0)
    if n == 0:
        return 0
    rng_state = torch.get_rng_state()
    data_module.num_workers = n
    try:
        next(iter(data_module.train_dataloader()))
    except Exception as e:
        logging.warning(f"DataLoader with {n} workers failed ({e!r}); training with num_workers 0")
        n = 0
    finally:
        torch.set_rng_state(rng_state)
    print(f"num_workers {n} (SLURM_CPUS_PER_TASK={cpus})", flush=True)
    return n


class _PhaseTimer:
    """timing_log (default off): wall seconds per training phase, printed once per epoch.

    Synchronizes CUDA at each phase boundary so GPU time lands in the phase that queued it
    (this slows a GPU run slightly, hence off by default).
    """

    def __init__(self, enabled, device):
        self.enabled = enabled
        self.sync = enabled and device.type == 'cuda'
        self.totals, self.counts = {}, {}
        self._t = None

    def start(self):
        if self.enabled:
            if self.sync:
                torch.cuda.synchronize()
            self._t = time.perf_counter()

    def lap(self, phase):
        """Charge the time since the last start/lap to phase."""
        if not self.enabled:
            return
        if self.sync:
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.totals[phase] = self.totals.get(phase, 0.0) + now - self._t
        self.counts[phase] = self.counts.get(phase, 0) + 1
        self._t = now

    def add_items(self, prefix, data_module):
        """Move the position dataset's per-item phase times (load, augmentation, crops) in
        under prefix; they are a breakdown of 'data' (train) or the val phases, not extra."""
        item_times = getattr(getattr(data_module, 'alldata', None), 'timings', None)
        if not self.enabled or item_times is None:
            return
        for k, v in item_times.items():
            self.totals[f"{prefix}_{k}"] = self.totals.get(f"{prefix}_{k}", 0.0) + v
        item_times.clear()

    def report(self, epoch):
        if not self.enabled:
            return
        stats = dict(timing_epoch=epoch, threads=torch.get_num_threads(),
                     **{f"{k}_s": round(v, 2) for k, v in self.totals.items()},
                     **{f"{k}_n": n for k, n in self.counts.items()})
        print(json.dumps(stats), flush=True)
        self.totals, self.counts = {}, {}


def reaugment_enabled(args, use_position):
    """Whether the position pipeline draws new augmented views every epoch.

    VolumeCoordsDataset draws each item's augmentation from (seed, epoch, index), and
    without a Lightning trainer its epoch stays 0 unless set_epoch advances it. Default
    on since 2026-11-05 (a261027-1). Position runs from 80a2bbd (2026-10-05) until then
    repeated one fixed view pair per frame every epoch (i261027-1); their saved configs
    lack the key, so reproducing one needs reaugment_each_epoch: false. The legacy crop
    pipeline draws from the global RNG and is unaffected.
    """
    return bool(getattr(args, 'reaugment_each_epoch', True)) and bool(use_position)


POOLED_ENTRY_KEYS = ('label', 'project_path', 'visnet_z_scale', 'num_frames')


def is_pooled(args):
    """Whether the config trains one network on several recordings (`project_paths`)."""
    return getattr(args, 'project_paths', None) is not None


def resolve_recordings(args, seed):
    """The recordings this run trains on, as plain dicts (label, project_path, visnet_z_scale,
    num_frames, seed).

    Without `project_paths` (the default) it is the single `project_path` with the run's own
    seed and num_frames, label None, and visnet_z_scale None, which here means "leave the
    model's z_scale as args built it" (the historical behaviour, unchanged).

    With `project_paths` (pooled training) every entry is a mapping with project_path, an
    optional label (default: the project folder's name), visnet_z_scale (required for a ViSNet
    position encoder: the z / xy voxel size is a property of the recording, so the training
    loop sets it on the encoder before every forward pass of that recording's batches) and an
    optional num_frames (default: the top-level num_frames, so a sweep over num_frames sweeps
    the amount of data PER RECORDING). The top-level project_path and visnet_z_scale must
    then be unset: either would be ambiguous, and the saved args keep visnet_z_scale None so
    that no evaluation can silently embed every recording with one recording's value.

    Per-recording data seeds: a pool of one entry keeps the run seed, so it is exactly the
    single-project run (tested). Larger pools use derive_seed(seed, 'recording', label): keyed by
    the label, not the position, so reordering the list does not change what a recording
    trains on; and distinct per recording, so recordings of equal length do not draw the same
    frames and the same augmentation for the same item index. The frame selection is a prefix
    of one seeded permutation (VolumeCoordsDataModule.setup), so for one seed a smaller
    num_frames selects a subset of a larger one's frames (train, val and test together; the
    train/val/test split of them is drawn afresh per size).
    """
    if not is_pooled(args):
        return [dict(label=None, project_path=args.project_path, visnet_z_scale=None,
                     num_frames=args.num_frames, seed=int(seed))]
    entries = list(args.project_paths)
    if not entries:
        raise ValueError("project_paths is empty: list the recordings to pool, or use project_path")
    if getattr(args, 'project_path', None):
        raise ValueError("Set either project_path (one recording) or project_paths (pooled training), "
                         f"not both (project_path={args.project_path!r})")
    if getattr(args, 'visnet_z_scale', None) is not None:
        raise ValueError("Pooled training (project_paths) takes visnet_z_scale per entry; set the top-level "
                         f"visnet_z_scale to null (got {args.visnet_z_scale!r})")
    if not (getattr(args, 'use_position', False) or getattr(args, 'use_attention', False)):
        raise ValueError("project_paths (pooled training) needs the position pipeline (use_position: true)")
    needs_z = getattr(args, 'position_encoder', 'keypoint_mlp') == 'visnet'
    out = []
    for i, e in enumerate(entries):
        if not isinstance(e, Mapping) or not e.get('project_path'):
            raise ValueError(f"project_paths[{i}] must be a mapping with a project_path, got {e!r}")
        unknown = sorted(set(e) - set(POOLED_ENTRY_KEYS))
        if unknown:
            raise ValueError(f"project_paths[{i}]: unknown keys {unknown}; allowed {list(POOLED_ENTRY_KEYS)}")
        if needs_z and 'visnet_z_scale' not in e:
            raise ValueError(f"project_paths[{i}] ({e['project_path']}) has no visnet_z_scale; a ViSNet "
                             f"position encoder needs each recording's z / xy voxel size (null = voxel units)")
        z = e.get('visnet_z_scale')
        out.append(dict(label=str(e.get('label') or Path(str(e['project_path'])).parent.name),
                        project_path=str(e['project_path']),
                        visnet_z_scale=None if z is None else float(z),
                        num_frames=int(e.get('num_frames', args.num_frames))))
    labels = [r['label'] for r in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"project_paths labels must be unique, got {labels}")
    for r in out:
        r['seed'] = int(seed) if len(out) == 1 else derive_seed(int(seed), 'recording', r['label'])
    return out


def interleaved_order(lengths, seed, epoch):
    """Recording index of every training batch of one epoch, in the order they are trained.

    Recording r appears lengths[r] times (each of its train batches once, in its own loader's
    order). The order of the recordings is a permutation seeded by (seed, 'interleave', epoch),
    so it is reproducible and changes every epoch; with one recording nothing is drawn.
    """
    order = np.repeat(np.arange(len(lengths)), [int(n) for n in lengths])
    if len(lengths) > 1:
        order = np.random.RandomState(derive_seed(int(seed), 'interleave', int(epoch))).permutation(order)
    return [int(r) for r in order]


def _train_batches(recordings, seed, epoch):
    """(recording index, batch) over every recording's train loader once, in interleaved_order.

    One DataLoader per recording, not a ConcatDataset, so each keeps its own worker count (zarr
    stores can use workers, NWB stores must not). With one recording this is exactly
    `for batch in loader`. Loaders start when first needed and are drained at the end, so their
    workers shut down as after a plain loop.
    """
    iters = [None] * len(recordings)
    for r in interleaved_order([len(rec.loader) for rec in recordings], seed, epoch):
        if iters[r] is None:
            iters[r] = iter(recordings[r].loader)
        yield r, next(iters[r])
    for it in iters:
        if it is not None:
            next(it, None)


def _use_recording(model, rec, pooled):
    """Before a forward pass on rec's data: set the ViSNet z_scale to rec's (pooled runs only).

    The z stretch is a per-recording constant held as one attribute of the encoder, not a
    weight, so it is switched per batch rather than baked into the keypoints: a checkpoint then
    means the same thing at evaluation, where eval_accuracy sets the TARGET recording's value.
    A single-project run never touches it (the model keeps args.visnet_z_scale).
    """
    if pooled:
        from barlow_track.utils.barlow_superglue import set_visnet_z_scale
        set_visnet_z_scale(model, rec.visnet_z_scale)


def _make_data_module(args, project_data, num_frames, seed, target_sz, use_position):
    """The data module of one recording (position pipeline, or the legacy crop pipeline)."""
    if use_position:
        from barlow_track.utils.volume_data import VolumeCoordsDataModule
        return VolumeCoordsDataModule(
            project_data=project_data, num_frames=num_frames, batch_size=1,
            train_fraction=args.train_fraction, val_fraction=args.val_fraction,
            target_sz=target_sz,
            global_args=getattr(args, 'global_augment', None),
            photometric_args=getattr(args, 'crop_photometric', None),
            position_args=getattr(args, 'position_augment', None),
            seed=seed, num_workers=int(getattr(args, 'num_workers', 0) or 0),
            round_bounds=bool(getattr(args, 'round_crop_bounds', False)),
            # 'legacy': the preprint per-crop view transforms, from this config's p_* keys
            crop_augment=getattr(args, 'crop_augment', 'photometric') or 'photometric',
            legacy_transform_args=vars(args),
            keypoints_only=_keypoints_only(args))
    return NeuronCropImageDataModule(project_data=project_data, num_frames=num_frames, batch_size=1,
                                     train_fraction=args.train_fraction,
                                     val_fraction=args.val_fraction,
                                     crop_kwargs=dict(target_sz=target_sz), transform_args=args)


def _mean_over_recordings(values, weights=None):
    """Mean of the finite values (optionally weighted), NaN if none: the unsuffixed summary of
    a per-recording diagnostic in a pooled run."""
    weights = [1.0] * len(values) if weights is None else weights
    pairs = [(float(v), float(w)) for v, w in zip(values, weights)
             if v is not None and np.isfinite(float(v)) and float(w) > 0]
    if not pairs:
        return float('nan')
    return float(sum(v * w for v, w in pairs) / sum(w for _, w in pairs))


def _pooled_descriptor_diagnostics(model, recordings, gpu, use_position):
    """_validation_descriptor_diagnostics on each recording (its own z_scale, its first usable
    validation volume): keys suffixed _<label>, plus the unsuffixed mean over recordings, so the
    stats keep the single-project key names. One recording failing does not drop the others."""
    out, per_key = {}, {}
    for rec in recordings:
        _use_recording(model, rec, True)
        try:
            diag = _validation_descriptor_diagnostics(model, rec.data_module, gpu, use_position)
        except (RuntimeError, ValueError, AttributeError, StopIteration) as e:
            logging.warning(f"Descriptor diagnostics failed on {rec.label}: {e}")
            continue
        for k, v in diag.items():
            out[f"{k}_{rec.label}"] = v
            per_key.setdefault(k, []).append(v)
    out.update({k: _mean_over_recordings(vs) for k, vs in per_key.items()})
    return out


def train_barlow_network(args):
    _warn_unread_config_keys(args)
    # Taken before training writes into args (model_type, pooled_recordings, ...): what a
    # checkpoint of this run is resumable by (see _resume_state)
    resume_config = training_config(args)
    fingerprint = config_fingerprint(resume_config)

    # One seed for the whole run. seed_all is the entry-point fallback for code
    # that still reads global streams (the legacy crop pipeline uses random /
    # torchio); the data module and the model get the seed explicitly.
    # See barlow_track/utils/utils_seeding.py.
    from barlow_track.utils.utils_seeding import seed_all
    seed = getattr(args, 'seed', None)
    seed = 43 if seed is None else int(seed)
    seed_all(seed)

    # One recording (project_path), or several pooled into one network (project_paths); see
    # resolve_recordings. Each gets its own project, data module and train loader.
    pooled = is_pooled(args)
    recordings = [SimpleNamespace(**r) for r in resolve_recordings(args, seed)]

    target_sz = np.array([args.target_sz_z, args.target_sz_xy, args.target_sz_xy])
    use_position = getattr(args, 'use_position', False)
    use_attention = getattr(args, 'use_attention', False)
    if getattr(args, 'use_gnn', False):
        raise ValueError("use_gnn was removed: pair-only matching has no inference path. "
                         "Use use_attention (BarlowVolumeAttention) instead.")
    # Attention models need coordinates even if use_position was left false
    use_position = use_position or use_attention
    for rec in recordings:
        # Load ground truth
        rec.project_data = ProjectData.load_final_project_data(rec.project_path, allow_hybrid_loading=True)
        print("Preparing cropped volumes..." if not pooled else
              f"Preparing cropped volumes of {rec.label} (seed {rec.seed}, z_scale {rec.visnet_z_scale})...")
        print(f"Setting up data module (position={use_position}, "
              f"frames={rec.num_frames})...", flush=True)
        rec.data_module = _make_data_module(args, rec.project_data, rec.num_frames, rec.seed, target_sz,
                                            use_position)
        rec.data_module.setup()
        if use_position:
            rec.data_module.num_workers = _auto_num_workers(args, rec.data_module)
        rec.loader = rec.data_module.train_dataloader()
    if pooled:
        # Provenance in the saved args (plain types): what each recording contributed
        for rec in recordings:
            dm = rec.data_module
            rec.n_frames = len(dm.alldata)
            rec.n_train, rec.n_val, rec.n_test = len(dm.train_dataset), len(dm.val_dataset), len(dm.test_dataset)
        args.pooled_recordings = [
            {k: getattr(rec, k) for k in ('label', 'project_path', 'visnet_z_scale', 'num_frames', 'seed',
                                          'n_frames', 'n_train', 'n_val', 'n_test')} for rec in recordings]
        print(f"Pooled training on {len(recordings)} recordings: "
              + ", ".join(f"{r['label']} ({r['n_train']} train / {r['n_val']} val / {r['n_test']} test, "
                          f"z_scale {r['visnet_z_scale']})" for r in args.pooled_recordings), flush=True)
    cuda_index = os.getenv("CUDA_VISIBLE_DEVICES", 0)
    gpu = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {gpu}")
    print(f"Using device: {gpu} (cuda available: {torch.cuda.is_available()})", flush=True)
    # Initialize model, loading from checkpoint if passed
    try:
        pretrained_model_path = args.pretrained_model_path
    except AttributeError:
        pretrained_model_path = None
    if pretrained_model_path is not None:
        logging.info(f"Loading model from {pretrained_model_path}")
        gpu, model, pretrained_args = load_barlow_model(pretrained_model_path, expected_args=args)
        logging.info(f"Loaded pretrained args: {pretrained_args}")
        # Architecture already validated inside load_barlow_model; keep training
        # hyperparameters from the current config on the loaded model.
        for k, v in vars(args).items():
            setattr(model.args, k, v)
        # Keep the class tag in sync so the re-saved args.pickle reloads correctly
        args.model_type = getattr(pretrained_args, 'model_type', 'barlow')
    else:
        # backbone_kwargs is usually a plain dict from yaml (vars() only works
        # on namespaces); fall back to defaults field by field.
        _bb = vars(args).get('backbone_kwargs', dict()) or dict()
        user_args = dict(_bb) if isinstance(_bb, dict) else vars(_bb)
        backbone_kwargs = dict(in_channels=1, num_levels=user_args.get('num_levels', 2), f_maps=user_args.get('f_maps', 4), crop_sz=target_sz)
        if use_attention:
            args.model_type = 'attention'
            model = BarlowVolumeAttention(args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)
            # Record the effective value: the class defaults to True, but
            # load_barlow_model pins a missing key to False (pre-centering checkpoints)
            args.center_per_volume = model.center_per_volume
        elif use_position:
            args.model_type = 'position'
            model = BarlowWithPosition(args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)
        else:
            args.model_type = 'barlow'
            model = BarlowTwins3d(args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)

    # NOTE: yaml.safe_load parses '1e-6' (no decimal point) as str, not float,
    # which used to crash Adam with "'<=' not supported between 'float' and 'str'".
    # Coerce here so old configs keep working; the template now uses 1.0e-6.
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr),
                                 weight_decay=float(getattr(args, 'weight_decay', 0.0)))

    # Actually train
    start_time = time.time()
    log_dir = os.path.join(args.project_dir, 'log')
    print(f"Starting training with args: {args}. Stats in folder: {args.project_dir}")
    if args.dryrun:
        print("Dryrun, therefore stopping before actual training")
        return
    # Readers (load_barlow_model, check_training_finished, top_k_trials) use the
    # fixed names, so an earlier run's outputs move aside rather than this run
    # writing resnet50-1.pth / args-1.pickle / stats-1.json next to them. An unfinished
    # run of this same config is continued instead (see _resume_state); then only what
    # its interrupted attempt wrote on the way out moves aside.
    resume = _resume_state(args, fingerprint, resume_config, model, keep=[pretrained_model_path])
    if resume is None:
        _archive_previous_run(args.project_dir, keep=[pretrained_model_path])
    else:
        _archive_previous_run(args.project_dir, keep=[pretrained_model_path], globs=STALE_ATTEMPT_GLOBS)
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
    stats_file = get_sequential_filename(os.path.join(log_dir, 'stats.json'))
    checkpoint_file = os.path.join(args.project_dir, CHECKPOINT_NAME)
    heartbeat = _Heartbeat(args.project_dir)
    heartbeat.beat()

    wandb_opt = dict(mode="disabled") if args.DEBUG else {}
    json_stats = []
    test_losses = None
    val_losses = None
    train_losses = None
    oom_error = None

    # Initialize wandb run, if the user enables it
    if args.wandb_name and args.wandb_username:
        wandb.login()
        run = wandb.init(project=args.wandb_name, entity=args.wandb_username, config=args, **wandb_opt)
        wandb.config.update(args)  # TODO
    else:
        run = None

    # Initial json entry: the wandb run name and id
    if run is not None:
        json_stats.append(dict(run_name=run.name, run_id=run.id))
    else:
        json_stats.append(dict(run_name="Non-wandb-run", run_id=None))

    # Held-out frames for the per-epoch matching proxy: evenly spaced over the
    # frames the data module did not sample (train, val and test alike)
    # (per recording in a pooled run: val_match_frames frames of each)
    match_stages = ('auto',)
    n_match = int(getattr(args, 'val_match_frames', 0) or 0)
    for rec in recordings:
        rec.match_frames = []
        if n_match > 0:
            from barlow_track.utils.val_matching import held_out_frames
            if use_position:
                used = rec.data_module.alldata.frame_indices
                # keypoints_only has placeholder crops, so there is no image stage to score
                match_stages = ('auto',) if _keypoints_only(args) else ('auto', 'backbone')
            else:
                used = rec.data_module.selected_frames
                match_stages = ('backbone',)
            rec.match_frames = held_out_frames(rec.project_data.num_frames, used, n_match)
            print(f"Frame-pair matching proxy on {len(rec.match_frames)} held-out frames"
                  f"{f' of {rec.label}' if pooled else ''}: {rec.match_frames}", flush=True)

    num_skipped_batches = 0
    prev_gate = None  # previous attention-gate value, for update magnitudes
    gate_grad_norm = None  # captured after backward, before grads are cleared
    reaugment = reaugment_enabled(args, use_position)
    timer = _PhaseTimer(bool(getattr(args, 'timing_log', False)), gpu)
    # epoch_snapshots (default none): epoch counts after which to keep the weights, e.g.
    # [1, 2, 5, 10, 20], each a loadable model folder (see _save_epoch_snapshot)
    snapshot_epochs = {int(e) for e in (getattr(args, 'epoch_snapshots', None) or [])}
    if timer.enabled and use_position:
        # Splits 'data' into volume reading, augmentation and crop phases (num_workers 0 only)
        for rec in recordings:
            rec.data_module.alldata.timings = {}
    n_train_batches = sum(len(rec.loader) for rec in recordings)
    start_epoch = 0
    if resume is not None:
        # Continue where the checkpoint left off: its stats become this run's (one stats.json
        # for the whole run, plus a marker entry), and the global RNG streams are restored last,
        # after everything above that may draw from them (model init, wandb, loader probes)
        start_epoch = int(resume['epoch'])
        json_stats = list(resume['json_stats']) + [dict(resumed_from_epoch=start_epoch, **json_stats[0])]
        num_skipped_batches = int(resume['num_skipped_batches'])
        prev_gate, gate_grad_norm = resume['prev_gate'], resume['gate_grad_norm']
        start_time = time.time() - float(resume['elapsed'])  # 'time' in the stats counts training time
        if reaugment:
            # The views of the last trained epoch, as the loop leaves them: what the test loss
            # sees when no epoch is left to train (the loop sets every epoch it runs itself)
            for rec in recordings:
                rec.data_module.set_epoch(max(start_epoch - 1, 0))
        _set_rng_state(resume['rng'])
        print(f"Resuming training from {checkpoint_file} at epoch {start_epoch} of {args.epochs}", flush=True)
    epoch = args.epochs - 1  # what the loop leaves behind; epochs=0 (an untrained network) skips the loop
    try:
        for epoch in range(start_epoch, args.epochs):
            if reaugment:
                for rec in recordings:
                    rec.data_module.set_epoch(epoch)
            timer.start()
            # Pooled runs interleave the recordings' batches in a seeded order per epoch
            for step, (rec_i, batch) in enumerate(_train_batches(recordings, seed, epoch),
                                                  start=epoch * n_train_batches):
                heartbeat.beat()
                _use_recording(model, recordings[rec_i], pooled)
                timer.lap('data')
                loss, loss_original, loss_transpose, loss_match = _run_forward(model, batch, gpu, use_position)
                timer.lap('forward')

                # Crash on non-finite loss: stepping on NaN poisons the whole
                # model (gate and all) and wastes the trial producing a
                # garbage checkpoint. Degenerate grad-free batches are still
                # skipped below; this only fires when a real step would run.
                if loss.requires_grad and not bool(torch.isfinite(loss).detach().cpu()):
                    raise RuntimeError(
                        f"Non-finite training loss at epoch {epoch} step {step}: "
                        f"loss={loss.item()}, original={loss_original.item()}, "
                        f"transpose={loss_transpose.item()}. Aborting the trial; "
                        f"no checkpoint is saved.")

                # Degenerate batches (a view with <2 objects, or an empty
                # dropout intersection) yield a grad-free zero loss; skip the
                # optimizer step but keep logging so one bad volume cannot
                # kill the whole trial.
                if loss.requires_grad:
                    # adjust_learning_rate(args, optimizer, loader, step)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Attention-gate diagnostic: capture before grads are cleared.
                    # (With self_layers=0 the gate is structurally unused, so a
                    # frozen gate there is expected, not saturation.)
                    gate_param = getattr(model, 'attn_gate', None)
                    if gate_param is not None and gate_param.grad is not None:
                        gate_grad_norm = float(gate_param.grad.norm().detach().cpu())
                    else:
                        gate_grad_norm = None
                    optimizer.zero_grad(set_to_none=True)
                else:
                    num_skipped_batches += 1
                timer.lap('backward')

                if step % args.print_freq == 0 or step == 0:
                    if args.rank == 0:
                        # Just print (train_* keys match the val_*/test_* convention)
                        stats = dict(epoch=epoch, step=step,
                                     train_loss=loss.item(), train_loss_original=loss_original.item(), train_loss_transpose=loss_transpose.item(),
                                     time=int(time.time() - start_time))
                        if loss_match is not None:
                            stats['train_loss_match'] = loss_match.item()
                        # Attention-gate diagnostics: raw value, sigmoid, grad
                        # norm, and update since last print. A gate frozen with
                        # ~zero grad norm is saturated/starved; frozen WITHOUT
                        # a gate in the graph (self_layers=0) is structural.
                        gate_param = getattr(model, 'attn_gate', None)
                        if gate_param is not None:
                            gate_val = float(gate_param.detach().cpu())
                            gate_sig = 1.0 / (1.0 + math.exp(-gate_val))
                            stats['attn_gate'] = gate_val
                            stats['attn_gate_sigmoid'] = gate_sig
                            stats['attn_gate_grad_norm'] = gate_grad_norm
                            stats['attn_gate_update'] = (abs(gate_val - prev_gate)
                                                         if prev_gate is not None else None)
                            prev_gate = gate_val
                        # Standardization health probe: one extra no-grad embedding
                        # pair per print; measures clamp-hit rate and per-dim
                        # variance spectrum of exactly what the loss correlates
                        # (see standardization_health). Cheap: runs print_freq.
                        try:
                            from barlow_track.utils.barlow_superglue import standardization_health
                            with torch.no_grad():
                                if use_position:
                                    pz1 = model.embed_with_position(batch[0].to(gpu), batch[2].to(gpu))
                                    pz2 = model.embed_with_position(batch[1].to(gpu), batch[3].to(gpu))
                                else:
                                    # Image-only model: embed() takes a single
                                    # view (cf. _correlation_for_plot below).
                                    vy1, vy2 = _format_vectors_on_gpu(batch[0], batch[1], gpu)
                                    pz1, pz2 = model.embed(vy1), model.embed(vy2)
                                for _k, _v in standardization_health(pz1, pz2).items():
                                    stats[f"train_{_k}"] = _v
                        except (RuntimeError, ValueError, IndexError, AttributeError) as e:
                            logging.warning(f"Standardization probe failed: {e}")
                        print(json.dumps(stats))
                        json_stats.append(stats)

                        # wandb logging
                        train_losses = {"train_loss": loss.item(), "train_loss_original": loss_original.item(), "train_loss_transpose": loss_transpose.item()}
                        if run is not None:
                            run.log(train_losses)

                        # More infrequently, plot embedding
                        if step % (100*args.print_freq) == 0:
                            with torch.no_grad():
                                c = _correlation_for_plot(model, batch, gpu, use_position)
                                if c is not None:
                                    save_fname = os.path.join(args.project_dir, 'log', f'correlation_matrix_{step}.png')
                                    fig = visualize_model_performance(c, save_fname=save_fname, vmin=-0.5, vmax=1)
                                    if run is not None:
                                        run.log({"chart": fig})
                    timer.lap('print_probe')
            for rec in recordings:
                timer.add_items('train_item', rec.data_module)

            if args.rank == 0:
                # Calculate validation loss. By default in train mode (batch-norm
                # projectors use and update batch stats, as before); val_eval_mode
                # scores it like the test loss. Tracking descriptors are taken
                # before the projector, so neither changes tracking accuracy.
                val_eval_mode = bool(getattr(args, 'val_eval_mode', False))
                if val_eval_mode:
                    model.eval()
                with torch.no_grad():
                    val_loss, val_loss_original, val_loss_transpose = 0, 0, 0
                    c = None
                    num_val_batches = 0
                    rec_val = {}  # pooled runs: label -> (summed loss, number of batches)
                    for rec in recordings:
                        _use_recording(model, rec, pooled)
                        rec_loss, rec_batches = 0.0, 0
                        for val_step, batch in enumerate(rec.data_module.val_dataloader()):
                            heartbeat.beat()
                            loss, loss_original, loss_transpose, _ = _run_forward(model, batch, gpu, use_position)
                            val_loss += loss.item()
                            val_loss_original += loss_original.item()
                            val_loss_transpose += loss_transpose.item()
                            num_val_batches += 1
                            rec_loss, rec_batches = rec_loss + loss.item(), rec_batches + 1
                            # Plot validation embedding
                            if run is not None:
                                c_batch = _correlation_for_plot(model, batch, gpu, use_position)
                                if c_batch is not None:
                                    c = c_batch if c is None else c + c_batch
                        rec_val[rec.label] = (rec_loss, rec_batches)
                    if run is not None and c is not None and num_val_batches > 0:
                        c /= num_val_batches  # Plot the average
                        fig = visualize_model_performance(c, save_fname=None, vmin=-0.5, vmax=1)
                        run.log({"validation_chart": fig})
                if val_eval_mode:
                    model.train()
                timer.lap('val_loss')  # the checkpoint is now written at the end of the epoch

                val_losses ={"val_loss": val_loss, "val_loss_original": val_loss_original, "val_loss_transpose": val_loss_transpose}
                # val_loss is a sum over val batches; the mean compares across
                # runs with different numbers of val frames.
                val_losses["mean_batch_loss"] = val_loss / num_val_batches if num_val_batches else float('nan')
                if pooled:
                    for label, (rec_loss, rec_batches) in rec_val.items():
                        val_losses[f"mean_batch_loss_{label}"] = rec_loss / rec_batches if rec_batches else float('nan')
                # Anti-collapse descriptor health (pre-projector): effective rank,
                # off-diag correlation, dead dims, per-volume mean fraction,
                # attention entropy/gate, position jitter sensitivity. SSL loss
                # alone cannot reject collapsed tracking descriptors, so select
                # configs by these + tracking accuracy instead.
                try:
                    if pooled:
                        _diag = _pooled_descriptor_diagnostics(model, recordings, gpu, use_position)
                    else:
                        _diag = _validation_descriptor_diagnostics(model, recordings[0].data_module, gpu, use_position)
                    val_losses.update({f"val_{k}": v for k, v in _diag.items()})
                except (RuntimeError, ValueError, AttributeError, StopIteration) as e:
                    logging.warning(f"Descriptor diagnostics failed: {e}")
                timer.lap('val_diagnostics')
                # Tracking proxy on frames training never sampled (val_matching.py);
                # off unless val_match_frames > 0. Draws no random numbers. Pooled runs score
                # each recording (keys suffixed _<label>) and report the pair-weighted mean
                # under the single-project key names.
                match_by_rec = {}  # label -> ({stage: accuracy}, number of pairs)
                for rec in recordings:
                    if not rec.match_frames:
                        continue
                    heartbeat.beat()
                    _use_recording(model, rec, pooled)
                    try:
                        from barlow_track.utils.val_matching import frame_pair_matching_accuracy
                        acc, n_pairs = frame_pair_matching_accuracy(
                            model, rec.project_data, rec.match_frames, target_sz, gpu, stages=match_stages,
                            crop_window=getattr(args, 'val_match_crop_window', 'pad'),
                            round_centers=bool(getattr(args, 'val_match_round_crop_centers', False)),
                            keypoints_only=_keypoints_only(args))
                        match_by_rec[rec.label] = (acc, n_pairs)
                    except (RuntimeError, ValueError, KeyError, IndexError, AttributeError) as e:
                        logging.warning(f"Frame-pair matching accuracy failed"
                                        f"{f' on {rec.label}' if pooled else ''}: {e}")
                if not pooled:
                    for acc, n_pairs in match_by_rec.values():
                        for stage, a in acc.items():
                            val_losses[f"val_match_accuracy_{stage}"] = a
                        val_losses["val_match_pairs"] = n_pairs
                elif match_by_rec:
                    for label, (acc, n_pairs) in match_by_rec.items():
                        for stage, a in acc.items():
                            val_losses[f"val_match_accuracy_{stage}_{label}"] = a
                        val_losses[f"val_match_pairs_{label}"] = n_pairs
                    for stage in match_stages:
                        val_losses[f"val_match_accuracy_{stage}"] = _mean_over_recordings(
                            [acc.get(stage) for acc, _ in match_by_rec.values()],
                            [n for _, n in match_by_rec.values()])
                    val_losses["val_match_pairs"] = sum(n for _, n in match_by_rec.values())
                timer.lap('val_match_proxy')
                for rec in recordings:
                    timer.add_items('val_item', rec.data_module)
                timer.report(epoch)
                if epoch + 1 in snapshot_epochs:
                    _save_epoch_snapshot(model, args, epoch + 1)
                if run is not None:
                    run.log(val_losses)
                # Printing
                stats = dict(epoch=epoch, val_loss=val_loss, time=int(time.time() - start_time),
                             skipped_batches=num_skipped_batches)
                stats.update({f"val_{k}": v for k, v in val_losses.items() if k != "val_loss"
                              and "val_loss_" not in k})
                print(json.dumps(stats))
                json_stats.append(stats)
                # Checkpoint at the very end of the epoch, so a run resumed from it continues
                # with exactly this state (validation in train mode updates batch-norm
                # statistics, and every DataLoader iterator draws from the global torch RNG)
                save_checkpoint(dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                     fingerprint=fingerprint, config=resume_config, rng=_rng_state(),
                                     json_stats=json_stats, num_skipped_batches=num_skipped_batches,
                                     prev_gate=prev_gate, gate_grad_norm=gate_grad_norm,
                                     elapsed=time.time() - start_time), checkpoint_file)
                heartbeat.beat()

        # Calculate the final test loss
        test_loss, test_loss_original, test_loss_transpose = 0, 0, 0
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for rec in recordings:
                _use_recording(model, rec, pooled)
                for _, batch in enumerate(rec.data_module.test_dataloader()):
                    heartbeat.beat()
                    loss, loss_original, loss_transpose, _ = _run_forward(model, batch, gpu, use_position)
                    test_loss += loss.item()
                    test_loss_original += loss_original.item()
                    test_loss_transpose += loss_transpose.item()
            # Package losses into a dictionary for return
            test_losses = dict(test_loss=test_loss, test_loss_original=test_loss_original, test_loss_transpose=test_loss_transpose)

        if run is not None:
            run.log(test_losses)
        # Printing
        stats = dict(epoch=epoch, test_loss=test_loss, time=int(time.time() - start_time))
        print(json.dumps(stats))
        json_stats.append(stats)

    except KeyboardInterrupt:
        print("Interrupted training, saving model")
    except torch.cuda.OutOfMemoryError as e:
        print("Out of memory error, saving the partial model as resnet50_oom.pth")
        print(e)
        oom_error = e
        # Last stats entry without 'epoch': check_training_finished reads it as unfinished
        json_stats.append(dict(out_of_memory=str(e)))
    finally:
        # Never save a model from a crashed trial (e.g. non-finite loss, or submitit's
        # walltime signal, which raises inside training): see the final saving below
        _exc = sys.exc_info()[1]
        _crashed = (_exc is not None
                    and not isinstance(_exc, (KeyboardInterrupt, torch.cuda.OutOfMemoryError)))
        if _crashed:
            # Last stats entry without 'epoch' (as for an OOM): check_training_finished reads it
            # as unfinished, also when the crash came in the last epoch
            json_stats.append(dict(crashed=repr(_exc)))
        if test_losses is None:
            # Then the run failed in some way, and an alternate value should be returned
            if val_losses is not None:
                test_losses = {k.replace('val', 'test'): v for k, v in val_losses.items()}
                logging.warning("Could not calculate test losses, using last validation loss instead")
            elif train_losses is not None:
                test_losses = {k.replace('train', 'test'): v for k, v in train_losses.items()}
                logging.warning("Could not calculate test losses, using last training loss instead")
            else:
                test_losses = dict(test_loss=np.inf, test_loss_original=np.inf, test_loss_transpose=np.inf)
                logging.warning("Could not calculate any loss, returning np.inf for test losses")
                
        # Clean up the wandb run
        if run is not None:
            run.finish()

        # Final saving
        with open(stats_file, 'w') as f:
            print(json.dumps(json_stats), file=f)

        if args.rank == 0:
            # Never save a model from a crashed trial (e.g. non-finite loss):
            # a poisoned checkpoint would silently pass the "has resnet50.pth"
            # check and get benchmarked as if trained. Interrupt keeps the
            # legacy save-what-we-have behavior; an OOM saves under a name no
            # reader takes for a finished trial, and is re-raised below.
            if _crashed:
                logging.warning(f"Not saving model: trial crashed with {_exc!r}")
            else:
                # save final model (not in checkpoint dir)
                name = 'resnet50_oom.pth' if oom_error is not None else 'resnet50.pth'
                fname = get_sequential_filename(os.path.join(args.project_dir, name))
                torch.save(model.state_dict(), fname)
                args.model_fname = fname

        # Also save the args namespace
        fname = get_sequential_filename(args.project_dir + '/args.pickle')
        with open(fname, 'wb') as f:
            pickle.dump(args, f)

        print("Training complete")

    if oom_error is not None:
        # A partial model must not count as a finished trial (the sweep scores
        # an exception as a failure, never as its last validation loss)
        raise oom_error
    return test_losses


RUN_OUTPUT_GLOBS = ('resnet50*.pth', 'args*.pickle', 'log/stats*.json', 'checkpoints/checkpoint*.pth',
                    'epoch_snapshots/*/*')


def epoch_snapshot_dir(project_dir, epoch):
    return os.path.join(project_dir, 'epoch_snapshots', f'epoch_{int(epoch):03d}')


def _save_epoch_snapshot(model, args, epoch):
    """Weights after `epoch` epochs as <project_dir>/epoch_snapshots/epoch_NNN/{resnet50.pth,
    args.pickle}: load_barlow_model and the trained-mode eval read it like a finished run."""
    d = epoch_snapshot_dir(args.project_dir, epoch)
    os.makedirs(d, exist_ok=True)
    fname = os.path.join(d, 'resnet50.pth')
    torch.save(model.state_dict(), fname)
    snap_args = SimpleNamespace(**{**vars(args), 'model_fname': fname, 'snapshot_epoch': int(epoch)})
    with open(os.path.join(d, 'args.pickle'), 'wb') as f:
        pickle.dump(snap_args, f)


def _archive_previous_run(project_dir, keep=(), globs=RUN_OUTPUT_GLOBS):
    """Move an earlier run's outputs into project_dir/superseded/<time>/; returns that dir or None.

    Files in keep (e.g. a pretrained checkpoint inside project_dir) stay in place. globs: which
    outputs (default all of them; a resumed run moves only STALE_ATTEMPT_GLOBS).
    """
    root = Path(project_dir)
    keep = {Path(k).resolve() for k in keep if k}
    found = [p for g in globs for p in sorted(root.glob(g))
             if p.is_file() and p.resolve() not in keep]
    if not found:
        return None
    dest = root / 'superseded' / time.strftime('%Y%m%d_%H%M%S')
    for p in found:
        target = dest / p.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        p.rename(target)
    logging.warning(f"Moved {len(found)} outputs of an earlier run to {dest}")
    return dest


# ---- Checkpoint resume ----------------------------------------------------------------------
#
# Every epoch ends (after its validation, diagnostics and snapshot) with an atomic write of
# checkpoints/checkpoint.pth: weights, optimizer, the global torch / numpy / python (and CUDA)
# RNG states, the stats so far and the loop counters, tagged with a fingerprint of the config.
# A run started in a project_dir that holds such a checkpoint continues from it when
#   - resume_from_checkpoint is true (the default),
#   - no final model is there (resnet50*.pth other than resnet50_oom*.pth: a finished run's
#     rerun keeps the historical behaviour, archive and start fresh),
#   - the checkpoint loads, has a fingerprint equal to this config's (config_fingerprint), an
#     epoch <= epochs, and matches the model's parameter names and shapes.
# Otherwise (with a warning when a checkpoint was there) the earlier outputs move aside and
# training starts from epoch 0, as before. The epoch loop then starts at the saved epoch with
# the saved state, so on CPU a resumed run reproduces the uninterrupted one exactly (weights,
# losses and stats; tested). An epoch interrupted midway is redone from its start.

CHECKPOINT_NAME = os.path.join('checkpoints', 'checkpoint.pth')
HEARTBEAT_NAME = os.path.join('checkpoints', 'heartbeat')
# Outputs an interrupted attempt writes on its way out (its finally block: stats, args, an OOM
# model). A resumed run replaces them, so they move aside, while the checkpoint, the epoch
# snapshots and the plots of the earlier epochs stay.
STALE_ATTEMPT_GLOBS = ('resnet50_oom*.pth', 'args*.pickle', 'log/stats*.json')
# Config keys that do not change what is trained, so a rerun may differ in them and still
# resume: where outputs go (project_dir, config_fname), logging (wandb_*, DEBUG only switches
# wandb off, timing_log), dryrun (stops before training), epochs (no schedule depends on it:
# augmentation, the pooled interleave order and the learning rate depend on the epoch index
# only, so raising it continues the same trajectory), epoch_snapshots (which weights are also
# kept), the opt-out key itself, and keys training writes into its own args.
RESUME_FINGERPRINT_EXCLUDED_KEYS = frozenset({
    'project_dir', 'config_fname', 'wandb_name', 'wandb_username', 'DEBUG', 'dryrun', 'timing_log',
    'epochs', 'epoch_snapshots', 'resume_from_checkpoint', 'model_fname', 'model_type',
    'pooled_recordings'})


def _canonical(value):
    """Plain JSON-able form of a config value, equal across a yaml round trip (tuples -> lists,
    1 == 1.0, numpy scalars -> python, ruamel's commented maps -> dicts)."""
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return str(value)


def training_config(args):
    """The part of a config that decides what is trained (see RESUME_FINGERPRINT_EXCLUDED_KEYS)."""
    return {k: _canonical(v) for k, v in vars(args).items() if k not in RESUME_FINGERPRINT_EXCLUDED_KEYS}


def config_fingerprint(args_or_config):
    """sha256 (16 hex digits) of training_config: equal configs resume each other's checkpoints."""
    import hashlib
    cfg = args_or_config if isinstance(args_or_config, Mapping) else training_config(args_or_config)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def resume_enabled(args):
    """resume_from_checkpoint (default true): continue an unfinished run of the same config."""
    return bool(getattr(args, 'resume_from_checkpoint', True))


def final_model_files(project_dir, keep=()):
    """Final models of a finished run in project_dir (resnet50*.pth, not resnet50_oom*.pth and
    not the files in keep, e.g. a pretrained model stored there)."""
    keep = {Path(k).resolve() for k in keep if k}
    return [p for p in sorted(Path(project_dir).glob('resnet50*.pth'))
            if not p.name.startswith('resnet50_oom') and p.resolve() not in keep]


def read_checkpoint(project_dir):
    """The checkpoint dict in project_dir, or None if there is none or it does not load (warned)."""
    path = os.path.join(str(project_dir), CHECKPOINT_NAME)
    if not os.path.isfile(path):
        return None
    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        logging.warning(f"Unreadable checkpoint {path} ({e!r})")
        return None
    if not isinstance(state, Mapping) or 'epoch' not in state:
        logging.warning(f"Checkpoint {path} is not a training checkpoint")
        return None
    return state


def resumable_epoch(project_dir, epochs=None):
    """Epoch a rerun in project_dir would continue from, or None if it would start fresh.

    For callers that rerun the trial's own saved config (the sweep dispatcher), so the config
    fingerprint is not compared here; train_barlow_network compares it.
    """
    if final_model_files(project_dir):
        return None
    state = read_checkpoint(project_dir)
    if state is None or 'fingerprint' not in state:
        return None
    epoch = int(state['epoch'])
    return epoch if epochs is None or epoch <= int(epochs) else None


def _resume_state(args, fingerprint, config, model, keep=()):
    """The checkpoint this run continues from, or None to start fresh (see the rule above)."""
    if not resume_enabled(args):
        return None
    path = os.path.join(args.project_dir, CHECKPOINT_NAME)
    if not os.path.isfile(path):
        return None
    finals = final_model_files(args.project_dir, keep)
    if finals:
        print(f"{args.project_dir} holds a finished run ({finals[0].name}); starting a new run", flush=True)
        return None
    state = read_checkpoint(args.project_dir)
    why = None
    if state is None:
        why = "it does not load"
    elif 'fingerprint' not in state:
        why = "it predates checkpoint resume (no config fingerprint)"
    elif state['fingerprint'] != fingerprint:
        old = state.get('config') or {}
        diff = sorted(k for k in set(old) | set(config) if old.get(k, '<unset>') != config.get(k, '<unset>'))
        why = f"it was trained with another config (keys differing: {diff})"
    elif int(state['epoch']) > int(args.epochs):
        why = f"it is at epoch {state['epoch']}, beyond epochs={args.epochs}"
    else:
        have = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        saved = {k: tuple(v.shape) for k, v in state['model'].items()}
        if have != saved:
            why = "its weights do not fit this model"
    if why is not None:
        logging.warning(f"Not resuming from {path}: {why}. Moving it aside and training from scratch.")
        return None
    return state


def _rng_state():
    state = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    cuda = state.get('cuda')
    if cuda is not None and torch.cuda.is_available():
        if len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda)
        else:
            logging.warning(f"Checkpoint has {len(cuda)} CUDA RNG states, this job sees "
                            f"{torch.cuda.device_count()} devices; CUDA RNG not restored")


def save_checkpoint(state, path):
    """torch.save to a temporary file, then rename it over path: a kill mid-write leaves the
    previous checkpoint intact (a stale .partial at most), never a truncated one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.partial"
    with open(tmp, 'wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class _Heartbeat:
    """Touches checkpoints/heartbeat at most once a minute while a run trains, validates or
    tests, so a relaunched sweep dispatcher can tell a trial whose job still runs from one whose
    job died (optimize_hyperparameters.trial_activity_age_minutes)."""

    def __init__(self, project_dir, every_s=60.0):
        self.path = os.path.join(str(project_dir), HEARTBEAT_NAME)
        self.every_s = every_s
        self._last = None

    def beat(self):
        now = time.monotonic()
        if self._last is not None and now - self._last < self.every_s:
            return
        self._last = now
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            Path(self.path).touch()
        except OSError as e:
            logging.warning(f"Could not touch {self.path}: {e}")


def _validation_descriptor_diagnostics(model, data_module, gpu, use_position):
    """Anti-collapse health metrics on the first usable validation volume.

    Runs pre-projector fused/contextual descriptors through
    volume_descriptor_diagnostics (effective rank, off-diag correlation,
    dead-dim fraction, per-volume mean fraction, attention entropy/gate,
    position jitter sensitivity). Returns {} for legacy image-only models
    or when no validation volume has >= 2 objects. Values are plain floats
    (NaN where undefined); exceptions propagate to the caller, which logs
    a warning instead of failing the epoch.
    """
    import torch as _torch

    if not use_position or not hasattr(model, 'fused_descriptors'):
        gate = getattr(model, 'attn_gate', None)
        out = {}
        if gate is not None:
            out = {'attn_gate': float(_torch.sigmoid(gate.detach()).cpu())}
        # Standardization health on the first usable validation volume
        # (legacy path mirrors _run_forward's formatting).
        try:
            from barlow_track.utils.barlow_superglue import standardization_health
            with _torch.no_grad():
                for batch in data_module.val_dataloader():
                    vy1, vy2 = _format_vectors_on_gpu(batch[0], batch[1], gpu)
                    if vy1.shape[0] < 2:
                        continue
                    for _k, _v in standardization_health(
                            model.embed(vy1), model.embed(vy2)).items():
                        out[f"val_{_k}"] = _v
                    break
        except (RuntimeError, ValueError, IndexError, AttributeError):
            pass
        return out
    from barlow_track.utils.barlow_superglue import volume_descriptor_diagnostics
    with _torch.no_grad():
        for batch in data_module.val_dataloader():
            y1, k1 = batch[0].to(gpu), batch[2].to(gpu)
            if y1.shape[0] < 2:
                continue
            diag = volume_descriptor_diagnostics(model, y1, k1)
            out = {k: (float(v) if v == v else float('nan')) for k, v in diag.items()}
            # Standardization health on the same volume (both views, as the
            # loss sees them).
            try:
                from barlow_track.utils.barlow_superglue import standardization_health
                z1 = model.embed_with_position(y1, k1)
                z2 = model.embed_with_position(batch[1].to(gpu), batch[3].to(gpu))
                for _k, _v in standardization_health(z1, z2).items():
                    out[f"val_{_k}"] = _v
            except (RuntimeError, ValueError, IndexError, AttributeError):
                pass
            return out
    gate = getattr(model, 'attn_gate', None)
    if gate is not None:
        return {'attn_gate': float(_torch.sigmoid(gate.detach()).cpu())}
    return {}


def _run_forward(model, batch, gpu, use_position):
    """Single forward handling both legacy (y1,y2) and position batches.

    Position batches are (y1, y2, k1, k2) or (y1, y2, k1, k2, idx1, idx2);
    the idx pair aligns independent per-view dropout on the intersection.
    Returns (loss, loss_original, loss_transpose, loss_match_or_None).
    """
    if not use_position:
        y1, y2 = _format_vectors_on_gpu(batch[0], batch[1], gpu)
        out = model.forward(y1, y2)
    else:
        y1, y2, k1, k2 = batch[0], batch[1], batch[2], batch[3]
        y1, y2 = y1.to(gpu), y2.to(gpu)
        k1, k2 = k1.to(gpu), k2.to(gpu)
        idx1 = batch[4].to(gpu) if len(batch) > 4 else None
        idx2 = batch[5].to(gpu) if len(batch) > 5 else None
        out = model.forward(y1, y2, k1, k2, idx1=idx1, idx2=idx2)
    if len(out) == 4:
        return out
    loss, loss_original, loss_transpose = out
    return loss, loss_original, loss_transpose, None


def _correlation_for_plot(model, batch, gpu, use_position):
    """Feature correlation matrix for the live training plot, either pipeline.

    Returns None for degenerate (empty) batches; callers must skip None.
    An empty view (0 objects after filtering/dropout) cannot embed, and one
    bad volume must not kill the whole trial (cf. the grad-free skip in the
    training loop).
    """
    # A 1-object view also cannot: a batch-norm projector in train mode needs
    # more than one row (reaug trial crashed in the epoch-31 validation plot).
    if batch[0].numel() == 0 or batch[1].numel() == 0 or len(batch[0]) < 2 or len(batch[1]) < 2:
        return None
    if not use_position:
        y1, y2 = _format_vectors_on_gpu(batch[0], batch[1], gpu)
        return model.calculate_correlation_matrix(y1, y2)
    from barlow_track.utils.barlow_superglue import both_correlation_matrices, intersection_gather
    y1, y2, k1, k2 = batch[0], batch[1], batch[2], batch[3]
    y1, y2, k1, k2 = y1.to(gpu), y2.to(gpu), k1.to(gpu), k2.to(gpu)
    z1 = model.embed_with_position(y1, k1)
    z2 = model.embed_with_position(y2, k2)
    if len(batch) > 5:
        sel1, sel2 = intersection_gather(batch[4].to(gpu), batch[5].to(gpu))
        if sel1 is not None:
            # Views paired by object id: fewer than 2 shared ids cannot be plotted, and the
            # unpaired views (e.g. 2 vs 3 objects) must not be multiplied (reaug_seed3 on
            # Samuel 137 crashed here, discarding the trial)
            if len(sel1) < 2:
                return None
            z1, z2 = z1[sel1], z2[sel2]
    if z1.shape[0] != z2.shape[0]:
        return None
    return both_correlation_matrices(z1, z2)[0]


def _format_vectors_on_gpu(y1, y2, gpu):
    # Needs to be outside the data loader because the batch dimension isn't added yet
    y1, y2 = torch.transpose(y1, 0, 1).type('torch.FloatTensor'), torch.transpose(y2, 0, 1).type(
        'torch.FloatTensor')
    y1 = y1.to(gpu)
    y2 = y2.to(gpu)
    return y1, y2


if __name__ == "__main__":
    # Get args, which is just path to yaml file
    parser = argparse.ArgumentParser(description='Train barlow network')
    parser.add_argument('--network_args', '-p', default=None,
                        help='path to yaml file (config)')

    cli_args = parser.parse_args()
    config_fname = cli_args.network_args

    # Load the yaml file
    with open(config_fname, 'r') as f:
        cfg = YAML().load(f)

    # Generate target saving locations from yaml location
    cfg['config_fname'] = config_fname
    cfg['project_dir'] = str(Path(config_fname).parent)
    args = SimpleNamespace(**cfg)
    # Run training code
    train_barlow_network(args)
