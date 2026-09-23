"""Full-pipeline tests on a COPY of the test project (never the original).

1. Position pipeline (untrained net, syntax only):
   tiny BarlowWithPosition -> embed_volumes_with_position -> WormClusterTracker
   -> track_using_global_clusterer -> tracks DataFrame.
2. Legacy regression: track_using_barlow_from_config still works end-to-end
   (DEBUG=True: 10 frames, no result files written), plus a fast check that the
   legacy checkpoint still loads as BarlowTwins3d.
3. Legacy training smoke: 1 epoch through the untouched legacy branch.

Copy location: /tmp/claude/barlow_copy_proj (2.2G, created once, reused).
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest barlow_track/tests/test_pipeline_position.py -q --assert=plain --runslow
"""
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from test_step0_embed_and_augment import requires_model, requires_project, MODEL_PATH

SRC_PROJECT = Path("/home/charles/Current_work/test_projects/barlow/worm4-2-2025-03-09")
COPY_PROJECT = Path("/tmp/claude/barlow_copy_proj")


@pytest.fixture(scope="module")
def copied_config():
    if not (COPY_PROJECT / "project_config.yaml").exists():
        shutil.copytree(SRC_PROJECT, COPY_PROJECT)
    return str(COPY_PROJECT / "project_config.yaml")


def _tiny_position_model(target_sz):
    from barlow_track.utils.barlow_superglue import BarlowWithPosition
    from barlow_track.utils.siamese import ResidualEncoder3D

    args = SimpleNamespace(
        embedding_dim=16, projector='32-32', projector_final=8,
        lambd=0.0051, lambd_obj=0.5, fusion='add', keypoint_encoder_layers=[16, 32],
    )
    kwargs = dict(in_channels=1, num_levels=2, f_maps=2, crop_sz=np.array(target_sz))
    model = BarlowWithPosition(args, backbone=ResidualEncoder3D, **kwargs)
    model.eval()
    return model


@requires_project
@pytest.mark.slow
def test_position_full_pipeline(copied_config):
    """Untrained position net -> embeddings -> cluster tracker -> tracks df."""
    from barlow_track.utils.track_using_barlow import embed_volumes_with_position
    from barlow_track.utils.utils_tracking import WormClusterTracker
    from wbfm.utils.projects.finished_project_data import ProjectData

    project_data = ProjectData.load_final_project_data(copied_config, allow_hybrid_loading=True)
    target_sz = (4, 16, 16)
    frames = [0, 1, 2, 3, 4]
    model = _tiny_position_model(target_sz)

    out, seg = embed_volumes_with_position(torch.device('cpu'), model, project_data, frames, target_sz)
    assert set(out) == set(frames)

    # Stack into tracker inputs (same schema as build_embedding_metadata)
    X, time_to_lin, lin_to_t_seg = [], defaultdict(list), {}
    i = 0
    for t in frames:
        X.append(out[t])
        for s in seg[t]:
            time_to_lin[t].append(i)
            lin_to_t_seg[i] = (t, int(s), int(s))
            i += 1
    X = np.vstack(X)

    tracker = WormClusterTracker(X, dict(time_to_lin), linear_ind_to_t_and_seg_id=lin_to_t_seg)
    df = tracker.track_using_global_clusterer()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == len(frames)


@requires_model
def test_legacy_checkpoint_still_loads_as_barlow_twins():
    from barlow_track.utils.barlow import BarlowTwins3d, load_barlow_model

    _, model, _ = load_barlow_model(MODEL_PATH)
    assert type(model).__name__ == 'BarlowTwins3d'
    assert isinstance(model, BarlowTwins3d)


@requires_project
@pytest.mark.slow
def test_legacy_train_smoke(tmp_path):
    """One epoch through the legacy (non-position) training branch."""
    from barlow_track.scripts.train_barlow_clusterer import train_barlow_network

    for sub in ('checkpoints', 'log'):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    from test_step0_embed_and_augment import PROJECT_PATH
    args = SimpleNamespace(
        project_path=PROJECT_PATH, project_dir=str(tmp_path), pretrained_model_path=None,
        wandb_name=None, wandb_username=None,
        embedding_dim=16, projector='32-32', projector_final=8,
        backbone_kwargs=dict(num_levels=2, f_maps=2),
        target_sz_z=4, target_sz_xy=16,
        p_RandomAffine_base=1.0, p_RandomBlur_base=0.0,
        p_RandomAffine_flip=0.0, p_RandomBlur=0.0, p_RandomAffine=1.0,
        p_RandomElasticDeformation=0.0, zxy_RandomElasticDeformation=[1, 3, 3],
        p_RandomNoise=0.0, std_RandomNoise=0.25, p_RandomAffine_both=None,
        lambd=0.0051, lambd_obj=0.5, batch_size=1, lr=1e-4, epochs=1,
        num_frames=4, train_fraction=0.75, val_fraction=0.0,
        print_freq=10000, rank=0, dryrun=False, DEBUG=True,
    )
    test_losses = train_barlow_network(args)
    assert np.isfinite(test_losses['test_loss'])
    assert (tmp_path / 'resnet50.pth').exists()
    assert getattr(args, 'model_type') == 'barlow'


@requires_project
@requires_model
@pytest.mark.slow
def test_legacy_track_from_config(copied_config):
    """track_using_barlow_from_config runs on the copied project (DEBUG: 10 frames)."""
    from barlow_track.utils.track_using_barlow import track_using_barlow_from_config

    track_using_barlow_from_config(copied_config, model_fname=MODEL_PATH,
                                   tracking_mode='global', DEBUG=True)
