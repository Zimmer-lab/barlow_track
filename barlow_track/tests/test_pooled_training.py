"""Pooled training: one network on several recordings (train_config `project_paths`).

Synthetic and CPU-only: fake projects stand in for wbfm ProjectData, so nothing here needs a
real project or a GPU.
"""
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import barlow_track.scripts.train_barlow_clusterer as tbc
from barlow_track.scripts.train_barlow_clusterer import interleaved_order, resolve_recordings

COLS = ['z', 'x', 'y', 'raw_segmentation_id']


class _Meta:
    def __init__(self, pts):
        self.pts = pts

    def get_all_neuron_metadata_for_single_time(self, t, as_dataframe=False):
        p = self.pts[t]
        return [list(p[:, i]) for i in range(3)] + [list(range(1, len(p) + 1))], COLS


class _Project:
    """Small volumes + centroid metadata; n_objects tells the recordings apart in a batch."""

    def __init__(self, n_objects=6, n_frames=10, shape=(12, 40, 40), seed=3):
        rng = np.random.RandomState(seed)
        self.red_data = rng.rand(n_frames, *shape).astype(np.float32)
        self.num_frames = n_frames
        self.segmentation_metadata = _Meta(
            {t: rng.uniform([2, 8, 8], [10, 32, 32], size=(n_objects, 3)) for t in range(n_frames)})
        self.final_tracks = None


# Recording -> objects per frame (no augmentation drops or adds objects in these tests)
N_OBJECTS = {'/fake/A/project_config.yaml': 5, '/fake/B/project_config.yaml': 7,
             '/fake/C/project_config.yaml': 6}
Z_OF = {'A': 2.0, 'B': 4.5}


@pytest.fixture
def fake_projects(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    load = lambda path, **kw: _Project(n_objects=N_OBJECTS.get(path, 6))
    monkeypatch.setattr(tbc, 'ProjectData', SimpleNamespace(load_final_project_data=load))


def _args(project_dir, **kw):
    """Tiny ViSNet position network; project_dir is created only when training (_train)."""
    base = dict(
        project_path=None, project_dir=str(project_dir), pretrained_model_path=None, wandb_name=None,
        wandb_username=None, embedding_dim=16, projector='32-32', projector_final=8,
        backbone_kwargs=dict(num_levels=2, f_maps=2), target_sz_z=4, target_sz_xy=16, use_position=True,
        use_attention=False, fusion='add', fusion_norm='layernorm', keypoint_encoder_layers=[16, 32],
        position_encoder='visnet', visnet_hidden=16, visnet_layers=1, visnet_rbf_rescale=True,
        visnet_z_scale=None, global_augment=dict(p_global_affine=1.0, max_degrees_z=30.0),
        crop_photometric=dict(p_blur=0.0, p_noise=0.3), position_augment=dict(jitter_std=0.5, dropout_p=0.1),
        lambd=0.0051, lambd_obj=0.5, batch_size=1, lr=1e-3, epochs=2, num_frames=6, train_fraction=0.5,
        val_fraction=0.25, print_freq=2, rank=0, dryrun=False, DEBUG=True, seed=7)
    base.update(kw)
    return SimpleNamespace(**base)


def _train(args):
    for sub in ('checkpoints', 'log'):
        (Path(args.project_dir) / sub).mkdir(parents=True, exist_ok=True)
    return tbc.train_barlow_network(args)


def _pool(*labels, **entry_kw):
    return [dict(label=l, project_path=f'/fake/{l}/project_config.yaml', visnet_z_scale=Z_OF.get(l, 1.0),
                 **entry_kw) for l in labels]


# --- interleaving ---------------------------------------------------------------------------

def test_interleaved_order_covers_every_batch_once_and_is_seeded_per_epoch():
    lengths = [3, 0, 5, 2]
    o = interleaved_order(lengths, seed=11, epoch=0)
    assert [o.count(r) for r in range(4)] == lengths
    assert o == interleaved_order(lengths, seed=11, epoch=0)
    assert o != interleaved_order(lengths, seed=11, epoch=1)
    assert o != interleaved_order(lengths, seed=12, epoch=0)
    assert o != sorted(o)  # actually interleaved, not one recording after another


def test_one_recording_is_the_plain_loader_order():
    assert interleaved_order([4], seed=1, epoch=3) == [0, 0, 0, 0]


def test_train_batches_yield_each_loaders_batches_in_its_own_order():
    recs = [SimpleNamespace(loader=['a0', 'a1', 'a2']), SimpleNamespace(loader=['b0', 'b1'])]
    out = list(tbc._train_batches(recs, seed=5, epoch=2))
    assert [r for r, _ in out] == interleaved_order([3, 2], 5, 2)
    assert [b for r, b in out if r == 0] == ['a0', 'a1', 'a2']
    assert [b for r, b in out if r == 1] == ['b0', 'b1']


# --- config ---------------------------------------------------------------------------------

def test_without_project_paths_the_single_project_is_unchanged():
    a = SimpleNamespace(project_path='/p/project_config.yaml', num_frames=50)
    assert resolve_recordings(a, 7) == [dict(label=None, project_path='/p/project_config.yaml',
                                             visnet_z_scale=None, num_frames=50, seed=7)]


def test_pooled_entries_take_num_frames_per_recording_and_get_distinct_label_seeds():
    a = _args('/unused', num_frames=64,
              project_paths=_pool('A', 'B') + [dict(project_path='/fake/C/project_config.yaml',
                                                    visnet_z_scale=1.0, num_frames=16)])
    recs = resolve_recordings(a, 7)
    assert [r['label'] for r in recs] == ['A', 'B', 'C']  # C: the project folder's name
    assert [r['num_frames'] for r in recs] == [64, 64, 16]
    assert [r['visnet_z_scale'] for r in recs] == [2.0, 4.5, 1.0]
    assert len({r['seed'] for r in recs}) == 3
    # keyed by label: reordering the list keeps each recording's data seed
    a.project_paths = list(reversed(a.project_paths))
    assert {r['label']: r['seed'] for r in resolve_recordings(a, 7)} == {r['label']: r['seed'] for r in recs}
    # a pool of one keeps the run seed (it IS the single-project run)
    a.project_paths = _pool('A')
    assert resolve_recordings(a, 7)[0]['seed'] == 7


@pytest.mark.parametrize('change, match', [
    (dict(project_path='/x/project_config.yaml'), 'not both'),
    (dict(visnet_z_scale=4.6), 'per entry'),
    (dict(use_position=False), 'position pipeline'),
    (dict(project_paths=[]), 'empty'),
    (dict(project_paths=[dict(project_path='/fake/A/p.yaml')]), 'no visnet_z_scale'),
    (dict(project_paths=[dict(project_path='/a/p.yaml', z_scale=2.0, visnet_z_scale=1.0)]), 'unknown keys'),
    (dict(project_paths=_pool('A') + _pool('A')), 'unique'),
    (dict(project_paths=['/fake/A/p.yaml']), 'mapping'),
])
def test_ambiguous_or_malformed_pools_fail_loudly(change, match):
    a = _args('/unused', project_paths=_pool('A', 'B'))
    for k, v in change.items():
        setattr(a, k, v)
    with pytest.raises(ValueError, match=match):
        resolve_recordings(a, 7)


def test_keypoint_mlp_pools_need_no_z_scale():
    a = _args('/unused', position_encoder='keypoint_mlp', project_paths=[dict(project_path='/a/p.yaml')])
    assert resolve_recordings(a, 7)[0]['visnet_z_scale'] is None


# --- training -------------------------------------------------------------------------------

def _visnet_z(model):
    from barlow_track.utils.barlow_superglue import ViSNetPositionEncoder
    zs = {m.z_scale for m in model.modules() if isinstance(m, ViSNetPositionEncoder)}
    assert len(zs) == 1
    return zs.pop()


def test_each_batch_runs_with_its_recordings_z_scale(tmp_path, fake_projects, monkeypatch):
    seen, proxy = [], []
    real_forward = tbc._run_forward

    def forward(model, batch, gpu, use_position):
        seen.append((len(batch[2]), _visnet_z(model)))
        return real_forward(model, batch, gpu, use_position)

    def matching(model, project_data, frames, *a, **kw):
        proxy.append((len(project_data.segmentation_metadata.pts[0]), _visnet_z(model), tuple(frames)))
        n = len(project_data.segmentation_metadata.pts[0])
        return {'auto': 0.1 * n, 'backbone': 0.0}, n

    import barlow_track.utils.val_matching as vm
    monkeypatch.setattr(tbc, '_run_forward', forward)
    monkeypatch.setattr(vm, 'frame_pair_matching_accuracy', matching)
    # no augmentation adds or drops objects, so a batch's row count names its recording
    args = _args(tmp_path, project_paths=_pool('A', 'B'), val_match_frames=2,
                 global_augment=dict(p_global_affine=0.0), position_augment=dict(jitter_std=0.0))
    _train(args)

    z_of_n = {5: 2.0, 7: 4.5}
    assert {n for n, _ in seen} == {5, 7}
    assert all(z == z_of_n[n] for n, z in seen), seen
    assert {(n, z) for n, z, _ in proxy} == {(5, 2.0), (7, 4.5)}

    stats = json.loads((tmp_path / 'log' / 'stats.json').read_text())
    val = [s for s in stats if 'val_loss' in s][-1]
    for label in ('A', 'B'):
        assert f'val_val_fused_eff_rank_{label}' in val
        assert f'val_mean_batch_loss_{label}' in val
    assert val['val_val_fused_eff_rank'] == pytest.approx(
        (val['val_val_fused_eff_rank_A'] + val['val_val_fused_eff_rank_B']) / 2)
    assert val['val_val_match_accuracy_auto_A'] == pytest.approx(0.5)
    assert val['val_val_match_pairs'] == 12
    assert val['val_val_match_accuracy_auto'] == pytest.approx((0.5 * 5 + 0.7 * 7) / 12)  # pair-weighted

    # Saved args: provenance per recording, and no z_scale (eval must pass the target's)
    with open(tmp_path / 'args.pickle', 'rb') as f:
        saved = pickle.load(f)
    assert saved.visnet_z_scale is None
    assert [(r['label'], r['visnet_z_scale'], r['num_frames']) for r in saved.pooled_recordings] == \
        [('A', 2.0, 6), ('B', 4.5, 6)]
    from barlow_track.utils.barlow import load_barlow_model
    _, model, _ = load_barlow_model(str(tmp_path / 'resnet50.pth'))
    assert _visnet_z(model) is None


def _weights(project_dir):
    return torch.load(os.path.join(project_dir, 'resnet50.pth'))


def test_a_pool_of_one_is_the_single_project_run(tmp_path, fake_projects):
    single = _args(tmp_path / 'single', project_path='/fake/A/project_config.yaml', visnet_z_scale=2.0)
    pooled = _args(tmp_path / 'pooled', project_paths=_pool('A'))
    loss_single = _train(single)
    loss_pooled = _train(pooled)
    assert loss_single == loss_pooled
    w1, w2 = _weights(single.project_dir), _weights(pooled.project_dir)
    assert w1.keys() == w2.keys()
    assert all(torch.equal(w1[k], w2[k]) for k in w1)


# --- dispatcher -----------------------------------------------------------------------------

def test_dispatcher_rejects_pooled_training_under_the_accuracy_objective():
    from barlow_track.scripts import optimize_hyperparameters as oh
    params = dict(project_paths=_pool('A', 'B'), use_position=True, position_encoder='visnet', num_frames=8)
    with pytest.raises(ValueError, match='objective: loss'):
        oh.check_pooled_baseline(params, accuracy_objective=True)
    assert [r['label'] for r in oh.check_pooled_baseline(params, accuracy_objective=False)] == ['A', 'B']
    assert oh.check_pooled_baseline(dict(project_path='/p.yaml', num_frames=8), accuracy_objective=True) == []


def test_pooled_grid_runs_end_to_end_through_the_dispatcher(tmp_path, monkeypatch):
    """A direct num_frames x seed grid under objective: loss: every trial's saved config keeps the
    pool and its grid values, and the training sees them per recording."""
    from barlow_track.scripts import optimize_hyperparameters as oh
    seen = []

    def train(args):
        seen.append([(r['label'], r['num_frames'], r['seed']) for r in resolve_recordings(args, args.seed)])
        return dict(test_loss=0.5)

    monkeypatch.setattr(oh, 'train_barlow_network', train)
    monkeypatch.setattr(oh.time, 'sleep', lambda s: None)
    real_executor = oh.AutoExecutor
    monkeypatch.setattr(oh, 'AutoExecutor', lambda folder, cluster: real_executor(folder=str(tmp_path / 'submitit'),
                                                                                 cluster=cluster))
    cfg = dict(vars(_args(tmp_path)), project_paths=_pool('A', 'B'), wandb_name=None)
    for k in ('project_dir',):
        cfg.pop(k)
    with open(tmp_path / 'train_config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f)
    with open(tmp_path / 'hyperparameter_search_template.yaml', 'w') as f:
        yaml.safe_dump(dict(baseline_params_path=None, objective='loss', train_hours=2, hyperparameters=[
            dict(name='num_frames', type='choice', values=[4, 8], value_type='int'),
            dict(name='seed', type='choice', values=[1, 2], value_type='int')]), f)
    oh.optimize_hyperparameters(str(tmp_path / 'hyperparameter_search_template.yaml'), run_locally=True,
                                direct_parameter_sweep=True)
    assert len(seen) == 4
    assert {tuple((l, n) for l, n, _ in s) for s in seen} == {(('A', 4), ('B', 4)), (('A', 8), ('B', 8))}
    for d in sorted(tmp_path.glob('trial_*')):
        c = yaml.safe_load((d / 'train_config.yaml').read_text())
        assert [e['label'] for e in c['project_paths']] == ['A', 'B']
        assert c['num_frames'] in (4, 8) and c['seed'] in (1, 2)
