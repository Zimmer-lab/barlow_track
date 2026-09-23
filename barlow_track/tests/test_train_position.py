"""End-to-end smoke tests: real training loop + position-aware inference (Steps 2-4).

Needs the Step 0 test project. Both tests train 1 epoch on 3-4 frames with a
tiny backbone, save to tmp_path, and check outputs exist and losses are finite.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest barlow_track/tests/test_train_position.py -q --assert=plain --runslow
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_step0_embed_and_augment import PROJECT_PATH, requires_project


def _smoke_args(project_dir, kind):
    return SimpleNamespace(
        project_path=PROJECT_PATH,
        project_dir=str(project_dir),
        pretrained_model_path=None,
        wandb_name=None,
        wandb_username=None,
        embedding_dim=16,
        projector='32-32',
        projector_final=8,
        backbone_kwargs=dict(num_levels=2, f_maps=2),
        target_sz_z=4,
        target_sz_xy=16,
        use_position=True,
        use_attention=kind == 'attention',
        fusion='add',
        fusion_norm='layernorm' if kind == 'attention' else 'none',
        self_layers=1,
        keypoint_encoder_layers=[16, 32],
        global_augment=dict(p_global_affine=1.0, max_degrees_z=30.0),
        crop_photometric=dict(p_blur=0.0, p_noise=0.0),
        lambd=0.0051,
        lambd_obj=0.5,
        batch_size=1,
        lr=1e-4,
        epochs=1,
        num_frames=4,
        train_fraction=0.75,
        val_fraction=0.0,
        print_freq=10000,
        rank=0,
        dryrun=False,
        DEBUG=True,
    )


def _run_smoke(tmp_path, kind):
    from barlow_track.scripts.train_barlow_clusterer import train_barlow_network

    for sub in ('checkpoints', 'log'):
        (Path(tmp_path) / sub).mkdir(parents=True, exist_ok=True)
    args = _smoke_args(tmp_path, kind)
    test_losses = train_barlow_network(args)
    assert np.isfinite(test_losses['test_loss'])
    assert (Path(tmp_path) / 'resnet50.pth').exists()
    assert (Path(tmp_path) / 'args.pickle').exists()
    assert getattr(args, 'model_type') == kind

    # Saved checkpoints reload as the right class (load_barlow_model dispatch)
    from barlow_track.utils.barlow import load_barlow_model
    _, reloaded, _ = load_barlow_model(str(Path(tmp_path) / 'resnet50.pth'))
    expected_cls = {'attention': 'BarlowVolumeAttention',
                    'position': 'BarlowWithPosition'}[kind]
    assert type(reloaded).__name__ == expected_cls
    return args


@requires_project
@pytest.mark.slow
def test_train_position_smoke(tmp_path):
    _run_smoke(tmp_path, kind='position')


@requires_project
@pytest.mark.slow
def test_train_attention_smoke(tmp_path):
    _run_smoke(tmp_path, kind='attention')


@requires_project
def test_embed_volumes_with_position(tmp_path):
    """Fresh tiny fusion model embeds real frames: shapes, finite, ids align."""
    from barlow_track.utils.barlow_superglue import BarlowWithPosition
    from barlow_track.utils.siamese import ResidualEncoder3D
    from barlow_track.utils.track_using_barlow import embed_volumes_with_position
    from wbfm.utils.projects.finished_project_data import ProjectData

    project_data = ProjectData.load_final_project_data(PROJECT_PATH, allow_hybrid_loading=True)
    args = _smoke_args(tmp_path, kind='position')
    target_sz = np.array([args.target_sz_z, args.target_sz_xy, args.target_sz_xy])
    kwargs = dict(in_channels=1, num_levels=2, f_maps=2, crop_sz=target_sz)
    model = BarlowWithPosition(args, backbone=ResidualEncoder3D, **kwargs)
    model.eval()
    gpu = torch.device('cpu')

    out, seg = embed_volumes_with_position(gpu, model, project_data, [0, 1], target_sz)
    assert set(out) == {0, 1}
    for t in (0, 1):
        assert out[t].shape == (len(seg[t]), 8)
        assert np.isfinite(out[t]).all()
