# Load a project and data, then train a Siamese network
import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import wandb
from ruamel.yaml import YAML

from barlow_track.utils.barlow import BarlowTwins3d, load_barlow_model
from barlow_track.utils.barlow_lightning import NeuronCropImageDataModule
from barlow_track.utils.barlow_superglue import BarlowVolumeAttention, BarlowWithPosition
from barlow_track.utils.barlow_visualize import visualize_model_performance
from barlow_track.utils.siamese import ResidualEncoder3D

from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.general.utils_filenames import get_sequential_filename


def train_barlow_network(args):

    torch.manual_seed(43)

    # Load ground truth
    project_data1 = ProjectData.load_final_project_data(args.project_path, allow_hybrid_loading=True)

    print("Preparing cropped volumes...")
    target_sz = np.array([args.target_sz_z, args.target_sz_xy, args.target_sz_xy])
    use_position = getattr(args, 'use_position', False)
    use_attention = getattr(args, 'use_attention', False)
    if getattr(args, 'use_gnn', False):
        raise ValueError("use_gnn was removed: pair-only matching has no inference path. "
                         "Use use_attention (BarlowVolumeAttention) instead.")
    # Attention models need coordinates even if use_position was left false
    use_position = use_position or use_attention
    print(f"Setting up data module (position={use_position}, "
          f"frames={args.num_frames})...", flush=True)
    if use_position:
        from barlow_track.utils.volume_data import VolumeCoordsDataModule
        data_module = VolumeCoordsDataModule(
            project_data=project_data1, num_frames=args.num_frames, batch_size=1,
            train_fraction=args.train_fraction, val_fraction=args.val_fraction,
            target_sz=target_sz,
            global_args=getattr(args, 'global_augment', None),
            photometric_args=getattr(args, 'crop_photometric', None),
            position_args=getattr(args, 'position_augment', None))
    else:
        data_module = NeuronCropImageDataModule(project_data=project_data1, num_frames=args.num_frames, batch_size=1,
                                                train_fraction=args.train_fraction,
                                                val_fraction=args.val_fraction,
                                                crop_kwargs=dict(target_sz=target_sz), transform_args=args)
    data_module.setup()
    loader = data_module.train_dataloader()
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
    stats_file = get_sequential_filename(os.path.join(log_dir, 'stats.json'))
    checkpoint_file = get_sequential_filename(os.path.join(args.project_dir, 'checkpoints', 'checkpoint.pth'))
    print(f"Starting training with args: {args}. Stats in folder: {args.project_dir}")
    if args.dryrun:
        print("Dryrun, therefore stopping before actual training")
        return

    wandb_opt = dict(mode="disabled") if args.DEBUG else {}
    json_stats = []
    test_losses = None
    val_losses = None
    train_losses = None

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

    num_skipped_batches = 0
    try:
        for epoch in range(0, args.epochs):
            for step, batch in enumerate(loader, start=epoch * len(loader)):
                loss, loss_original, loss_transpose, loss_match = _run_forward(model, batch, gpu, use_position)

                # Degenerate batches (a view with <2 objects, or an empty
                # dropout intersection) yield a grad-free zero loss; skip the
                # optimizer step but keep logging so one bad volume cannot
                # kill the whole trial.
                if loss.requires_grad:
                    # adjust_learning_rate(args, optimizer, loader, step)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    num_skipped_batches += 1

                if step % args.print_freq == 0 or step == 0:
                    if args.rank == 0:
                        # Just print (train_* keys match the val_*/test_* convention)
                        stats = dict(epoch=epoch, step=step,
                                     train_loss=loss.item(), train_loss_original=loss_original.item(), train_loss_transpose=loss_transpose.item(),
                                     time=int(time.time() - start_time))
                        if loss_match is not None:
                            stats['train_loss_match'] = loss_match.item()
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
                                save_fname = os.path.join(args.project_dir, 'log', f'correlation_matrix_{step}.png')
                                fig = visualize_model_performance(c, save_fname=save_fname, vmin=-0.5, vmax=1)
                                if run is not None:
                                    run.log({"chart": fig})

            if args.rank == 0:
                # save checkpoint
                state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
                torch.save(state, checkpoint_file)
                # Calculate validation loss
                with torch.no_grad():
                    val_loss, val_loss_original, val_loss_transpose = 0, 0, 0
                    c = None
                    num_val_batches = 0
                    for val_step, batch in enumerate(data_module.val_dataloader()):
                        loss, loss_original, loss_transpose, _ = _run_forward(model, batch, gpu, use_position)
                        val_loss += loss.item()
                        val_loss_original += loss_original.item()
                        val_loss_transpose += loss_transpose.item()
                        num_val_batches += 1
                        # Plot validation embedding
                        if run is not None:
                            c_batch = _correlation_for_plot(model, batch, gpu, use_position)
                            c = c_batch if c is None else c + c_batch
                    if run is not None and c is not None and num_val_batches > 0:
                        c /= num_val_batches  # Plot the average
                        fig = visualize_model_performance(c, save_fname=None, vmin=-0.5, vmax=1)
                        run.log({"validation_chart": fig})

                val_losses = {"val_loss": val_loss, "val_loss_original": val_loss_original, "val_loss_transpose": val_loss_transpose}
                # Anti-collapse descriptor health (pre-projector): effective rank,
                # off-diag correlation, dead dims, per-volume mean fraction,
                # attention entropy/gate, position jitter sensitivity. SSL loss
                # alone cannot reject collapsed tracking descriptors, so select
                # configs by these + tracking accuracy instead.
                try:
                    _diag = _validation_descriptor_diagnostics(model, data_module, gpu, use_position)
                    val_losses.update({f"val_{k}": v for k, v in _diag.items()})
                except (RuntimeError, ValueError, AttributeError, StopIteration) as e:
                    logging.warning(f"Descriptor diagnostics failed: {e}")
                if run is not None:
                    run.log(val_losses)
                # Printing
                stats = dict(epoch=epoch, val_loss=val_loss, time=int(time.time() - start_time),
                             skipped_batches=num_skipped_batches)
                stats.update({f"val_{k}": v for k, v in val_losses.items() if k != "val_loss"
                              and "val_loss_" not in k})
                print(json.dumps(stats))
                json_stats.append(stats)

        # Calculate the final test loss
        test_loss, test_loss_original, test_loss_transpose = 0, 0, 0
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for _, batch in enumerate(data_module.test_dataloader()):
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
        print("Out of memory error, saving model")
        print(e)
    finally:
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
            # save final model (not in checkpoint dir)
            fname = get_sequential_filename(args.project_dir + '/resnet50.pth')
            torch.save(model.state_dict(), fname)
            args.model_fname = fname

        # Also save the args namespace
        fname = get_sequential_filename(args.project_dir + '/args.pickle')
        with open(fname, 'wb') as f:
            pickle.dump(args, f)

        print("Training complete")
        
    return test_losses


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
        if gate is not None:
            return {'attn_gate': float(_torch.sigmoid(gate.detach()).cpu())}
        return {}
    from barlow_track.utils.barlow_superglue import volume_descriptor_diagnostics
    with _torch.no_grad():
        for batch in data_module.val_dataloader():
            y1, k1 = batch[0].to(gpu), batch[2].to(gpu)
            if y1.shape[0] < 2:
                continue
            diag = volume_descriptor_diagnostics(model, y1, k1)
            return {k: (float(v) if v == v else float('nan')) for k, v in diag.items()}
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
    """Feature correlation matrix for the live training plot, either pipeline."""
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
        if sel1 is not None and len(sel1) >= 2:
            z1, z2 = z1[sel1], z2[sel2]
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
