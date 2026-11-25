"""Checkpoint resume: training continues an unfinished run of the same config, and the sweep
dispatcher resubmits a trial whose job ended unfinished into its own folder.

Synthetic and CPU-only (the fake projects of test_pooled_training.py).
"""
import json
import logging
import os
import pickle
from pathlib import Path

import pytest
import torch
import yaml

import barlow_track.scripts.train_barlow_clusterer as tbc
from barlow_track.tests.test_pooled_training import _args, _pool, _train, fake_projects  # noqa: F401
from barlow_track.utils.utils_ground_truth import check_training_finished


class _Killed(RuntimeError):
    """Stands in for submitit's walltime signal, which raises inside training."""


def _kill_at(monkeypatch, epoch, after_batches=1):
    """Make training die in `epoch` after `after_batches` batches (mid-epoch)."""
    real = tbc._train_batches

    def batches(recordings, seed, e):
        for n, item in enumerate(real(recordings, seed, e)):
            if e == epoch and n == after_batches:
                raise _Killed(f"killed in epoch {e}")
            yield item

    monkeypatch.setattr(tbc, '_train_batches', batches)


def _weights(project_dir):
    return torch.load(os.path.join(project_dir, 'resnet50.pth'))


def _stats(project_dir):
    return json.loads((Path(project_dir) / 'log' / 'stats.json').read_text())


def _loss_entries(stats):
    """Stats without the wall-clock and run-bookkeeping parts."""
    return [{k: v for k, v in s.items() if k != 'time'} for s in stats
            if 'run_name' not in s and 'resumed_from_epoch' not in s]


def _same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


SINGLE = dict(project_path='/fake/A/project_config.yaml', visnet_z_scale=2.0)
POOLED = dict(project_paths=_pool('A', 'B'))


# --- training -------------------------------------------------------------------------------

ATTENTION = dict(SINGLE, use_attention=True, self_layers=1)


@pytest.mark.parametrize('data', [SINGLE, POOLED, ATTENTION], ids=['single', 'pooled', 'attention'])
def test_resumed_run_equals_the_uninterrupted_run(tmp_path, fake_projects, monkeypatch, data):
    straight = _args(tmp_path / 'straight', epochs=3, **data)
    loss_straight = _train(straight)

    resumed = tmp_path / 'resumed'
    with monkeypatch.context() as m:
        _kill_at(m, epoch=2)
        with pytest.raises(_Killed):
            _train(_args(resumed, epochs=3, **data))
    assert not (resumed / 'resnet50.pth').exists()
    assert torch.load(resumed / 'checkpoints' / 'checkpoint.pth')['epoch'] == 2
    assert not check_training_finished(str(resumed), 2)

    loss_resumed = _train(_args(resumed, epochs=3, **data))  # a fresh namespace, as a rerun has
    assert loss_resumed == loss_straight
    assert _same_weights(_weights(resumed), _weights(straight.project_dir))
    stats = _stats(resumed)
    assert _loss_entries(stats) == _loss_entries(_stats(straight.project_dir))
    assert [s['resumed_from_epoch'] for s in stats if 'resumed_from_epoch' in s] == [2]
    assert check_training_finished(str(resumed), 2)
    # What the killed attempt wrote on its way out moved aside; the outputs keep their names
    (stale,) = (resumed / 'superseded').iterdir()
    assert sorted(str(p.relative_to(stale)) for p in stale.rglob('*') if p.is_file()) == \
        ['args.pickle', 'log/stats.json']
    with open(resumed / 'args.pickle', 'rb') as f:
        assert pickle.load(f).model_fname == str(resumed / 'resnet50.pth')
    assert not list(resumed.glob('checkpoints/*.partial'))


def test_a_run_killed_after_its_last_epoch_only_finishes(tmp_path, fake_projects, monkeypatch):
    """Killed in the test loss: the checkpoint is at epoch == epochs, so the rerun trains nothing."""
    straight = _args(tmp_path / 'straight', epochs=2, **SINGLE)
    loss_straight = _train(straight)
    resumed = tmp_path / 'resumed'
    real_forward = tbc._run_forward
    with monkeypatch.context() as m:
        def forward(model, *a):
            if not model.training:
                raise _Killed('killed in the test loss')
            return real_forward(model, *a)
        m.setattr(tbc, '_run_forward', forward)
        with pytest.raises(_Killed):
            _train(_args(resumed, epochs=2, **SINGLE))
    trained = []
    real_batches = tbc._train_batches
    monkeypatch.setattr(tbc, '_train_batches', lambda r, s, e: trained.append(e) or real_batches(r, s, e))
    assert _train(_args(resumed, epochs=2, **SINGLE)) == loss_straight
    assert trained == []
    assert _same_weights(_weights(resumed), _weights(straight.project_dir))


def test_more_epochs_continue_the_same_trajectory(tmp_path, fake_projects, monkeypatch):
    """epochs is not part of the fingerprint: a killed 2-epoch run resumed with epochs=3 is the
    3-epoch run."""
    straight = _args(tmp_path / 'straight', epochs=3, **SINGLE)
    _train(straight)
    resumed = tmp_path / 'resumed'
    with monkeypatch.context() as m:
        _kill_at(m, epoch=1)
        with pytest.raises(_Killed):
            _train(_args(resumed, epochs=2, **SINGLE))
    _train(_args(resumed, epochs=3, **SINGLE))
    assert _same_weights(_weights(resumed), _weights(straight.project_dir))


def _killed_run(tmp_path, monkeypatch, **kw):
    d = tmp_path / 'run'
    with monkeypatch.context() as m:
        _kill_at(m, epoch=1)
        with pytest.raises(_Killed):
            _train(_args(d, epochs=2, **SINGLE, **kw))
    return d


def _assert_fresh_start(d, superseded_has=('checkpoints/checkpoint.pth',), epochs=2):
    assert not any('resumed_from_epoch' in s for s in _stats(d))
    assert check_training_finished(str(d), epochs - 1)
    (old,) = (d / 'superseded').iterdir()
    for rel in superseded_has:
        assert (old / rel).exists(), rel


def test_another_config_starts_fresh(tmp_path, fake_projects, monkeypatch, caplog):
    d = _killed_run(tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING):
        _train(_args(d, epochs=2, lr=2e-3, **SINGLE))
    assert "Not resuming" in caplog.text and "'lr'" in caplog.text
    _assert_fresh_start(d)


def test_bookkeeping_keys_do_not_block_resume(tmp_path, fake_projects, monkeypatch):
    d = _killed_run(tmp_path, monkeypatch)
    _train(_args(d, epochs=2, DEBUG=False, timing_log=True, epoch_snapshots=[2], config_fname='x.yaml', **SINGLE))
    assert any('resumed_from_epoch' in s for s in _stats(d))


def test_resume_can_be_switched_off(tmp_path, fake_projects, monkeypatch):
    d = _killed_run(tmp_path, monkeypatch)
    _train(_args(d, epochs=2, resume_from_checkpoint=False, **SINGLE))
    _assert_fresh_start(d)


def test_rerunning_a_finished_run_starts_fresh(tmp_path, fake_projects):
    d = tmp_path / 'run'
    _train(_args(d, epochs=1, **SINGLE))
    _train(_args(d, epochs=1, **SINGLE))
    _assert_fresh_start(d, ('resnet50.pth', 'args.pickle', 'log/stats.json', 'checkpoints/checkpoint.pth'), epochs=1)


def test_unreadable_checkpoint_warns_and_starts_fresh(tmp_path, fake_projects, monkeypatch, caplog):
    d = _killed_run(tmp_path, monkeypatch)
    ckpt = d / 'checkpoints' / 'checkpoint.pth'
    ckpt.write_bytes(ckpt.read_bytes()[:100])  # truncated
    assert tbc.resumable_epoch(d) is None
    with caplog.at_level(logging.WARNING):
        _train(_args(d, epochs=2, **SINGLE))
    assert "Unreadable checkpoint" in caplog.text and "does not load" in caplog.text
    _assert_fresh_start(d)


def test_checkpoint_write_is_atomic(tmp_path, monkeypatch):
    path = str(tmp_path / 'checkpoints' / 'checkpoint.pth')
    tbc.save_checkpoint(dict(epoch=1, w=torch.ones(3)), path)

    def dies_mid_write(obj, f):
        f.write(b'half a checkpoint')
        raise _Killed('killed while saving')

    monkeypatch.setattr(tbc.torch, 'save', dies_mid_write)
    with pytest.raises(_Killed):
        tbc.save_checkpoint(dict(epoch=2, w=torch.zeros(3)), path)
    monkeypatch.undo()
    state = torch.load(path)
    assert state['epoch'] == 1 and torch.equal(state['w'], torch.ones(3))
    assert sorted(os.listdir(tmp_path / 'checkpoints')) == ['checkpoint.pth', 'checkpoint.pth.partial']
    tbc.save_checkpoint(dict(epoch=2), path)  # the next write replaces the leftover
    assert sorted(os.listdir(tmp_path / 'checkpoints')) == ['checkpoint.pth']


def test_fingerprint_survives_a_yaml_round_trip_and_ignores_bookkeeping():
    args = _args('/a', project_paths=_pool('A', 'B'), lr=1.0, epoch_snapshots=(1, 2))
    fp = tbc.config_fingerprint(args)
    cfg = yaml.safe_load(yaml.safe_dump({k: (list(v) if isinstance(v, tuple) else v)
                                         for k, v in vars(args).items()}))
    cfg['lr'] = 1  # yaml may write 1.0 back as 1
    cfg.update(project_dir='/b', wandb_name='w', epochs=99)
    back = _args(**cfg)
    assert tbc.config_fingerprint(back) == fp
    back.target_sz_xy = 32
    assert tbc.config_fingerprint(back) != fp


# --- dispatcher -----------------------------------------------------------------------------

from barlow_track.scripts import optimize_hyperparameters as oh  # noqa: E402
from submitit.core.utils import UncompletedJobError  # noqa: E402


class _HardKill(BaseException):
    """A job killed outright (SIGKILL, node failure): no result at all."""


class _FakeJob:
    def __init__(self, fn, args):
        self.fn, self.args = fn, args

    def done(self):
        return True

    def result(self):
        try:
            return self.fn(*self.args)
        except _HardKill as e:
            raise UncompletedJobError(f"Job has not produced any output (state: TIMEOUT): {e}")

    def cancel(self):
        pass


class _FakeExecutor:
    def __init__(self, folder=None, cluster=None):
        pass

    def update_parameters(self, **kw):
        pass

    def submit(self, fn, *args):
        return _FakeJob(fn, args)


def _fake_training(calls, plan):
    """Training stand-in: plan(trial folder name, attempt number) -> 'finish', 'signal' (submitit's
    walltime signal inside training), 'hard' (job killed outright) or 'crash'. Unfinished attempts
    leave a checkpoint one epoch further than the last, as real training does; resumed attempts
    check that the dispatcher's config fingerprints as the one the checkpoint was written with."""

    def train(args):
        d = args.project_dir
        name = os.path.basename(d)
        state = tbc.read_checkpoint(d)
        if state is not None:
            assert state['fingerprint'] == tbc.config_fingerprint(args)
        start = 0 if state is None else state['epoch']
        attempt = sum(1 for c in calls if c[0] == name)
        calls.append((name, start))
        what = plan(name, attempt)
        if what == 'finish':
            open(os.path.join(d, 'resnet50.pth'), 'wb').write(b'w')
            os.makedirs(os.path.join(d, 'log'), exist_ok=True)
            json.dump([dict(epoch=args.epochs - 1, test_loss=0.5)], open(os.path.join(d, 'log', 'stats.json'), 'w'))
            return dict(test_loss=0.5)
        tbc.save_checkpoint(dict(epoch=start + 1, fingerprint=tbc.config_fingerprint(args)),
                            os.path.join(d, tbc.CHECKPOINT_NAME))
        if what == 'signal':
            raise UncompletedJobError("Job not requeued because: timed-out and not checkpointable.")
        if what == 'hard':
            raise _HardKill('walltime')
        raise RuntimeError('crashed')

    return train


def _sweep(tmp_path, monkeypatch, plan, lrs=(0.1,), template_extra=None, grid=True, **kw):
    calls = []
    monkeypatch.setattr(oh, 'train_barlow_network', _fake_training(calls, plan))
    monkeypatch.setattr(oh, 'AutoExecutor', _FakeExecutor)
    monkeypatch.setattr(oh.time, 'sleep', lambda s: None)
    sweep = tmp_path / 'round'
    sweep.mkdir(exist_ok=True)
    yaml.safe_dump(dict(project_path='/data/p/project_config.yaml', epochs=8, seed=43, wandb_name=None),
                   open(sweep / 'train_config.yaml', 'w'))
    hp = dict(name='lr', type='choice', values=list(lrs)) if grid else dict(name='lr', type='range', bounds=[0.1, 0.5])
    yaml.safe_dump(dict(baseline_params_path=None, objective='loss', hyperparameters=[hp], **(template_extra or {})),
                   open(sweep / 'hyperparameter_search_template.yaml', 'w'))
    oh.optimize_hyperparameters(str(sweep / 'hyperparameter_search_template.yaml'), run_locally=True,
                                direct_parameter_sweep=grid, **kw)
    return calls, sweep


def _trial_dirs(sweep):
    return sorted(p.name for p in sweep.glob('trial_*'))


@pytest.mark.parametrize('kill', ['signal', 'hard'])
def test_timed_out_trial_resumes_in_its_folder_until_max_resumes(tmp_path, monkeypatch, capsys, kill):
    calls, sweep = _sweep(tmp_path, monkeypatch, lambda name, attempt: kill,
                          template_extra=dict(max_resumes=2))
    assert calls == [('trial_0', 0), ('trial_0', 1), ('trial_0', 2)]  # the first run + 2 resumes
    assert _trial_dirs(sweep) == ['trial_0']
    out = capsys.readouterr().out
    assert out.count('Resubmitting it into the same folder') == 2
    assert 'resubmitted 2 times already (max_resumes=2); marking it FAILED' in out
    assert 'Trial 0 failed' in out


def test_resumed_trials_complete(tmp_path, monkeypatch, capsys):
    calls, sweep = _sweep(tmp_path, monkeypatch, lambda name, attempt: 'signal' if attempt < 2 else 'finish',
                          lrs=(0.1, 0.2), num_parallel_jobs=2)
    assert sorted(calls) == [('trial_0', 0), ('trial_0', 1), ('trial_0', 2),
                             ('trial_1', 0), ('trial_1', 1), ('trial_1', 2)]
    assert _trial_dirs(sweep) == ['trial_0', 'trial_1']
    assert 'failed' not in capsys.readouterr().out


def test_crashed_trial_is_not_resubmitted(tmp_path, monkeypatch):
    calls, _ = _sweep(tmp_path, monkeypatch, lambda name, attempt: 'crash')
    assert calls == [('trial_0', 0)]


def test_bayesian_sweep_resumes_a_timed_out_trial_too(tmp_path, monkeypatch):
    calls, sweep = _sweep(tmp_path, monkeypatch,
                          lambda name, attempt: 'signal' if (name, attempt) == ('trial_0', 0) else 'finish',
                          grid=False, total_budget=2, num_parallel_jobs=1)
    assert calls == [('trial_0', 0), ('trial_0', 1), ('trial_1', 0)]
    assert _trial_dirs(sweep) == ['trial_0', 'trial_1']
    assert (sweep / 'best_parameters.yaml').exists()


def _age(trial_dir, minutes):
    t = os.path.getmtime(trial_dir / 'train_config.yaml') - 60 * minutes
    for p in trial_dir.rglob('*'):
        if p.is_file():
            os.utime(p, (t, t))


@pytest.mark.parametrize('busy', [False, True], ids=['dead_job', 'job_may_still_run'])
def test_relaunched_grid_resumes_an_unfinished_trial_in_place(tmp_path, monkeypatch, busy):
    # First dispatcher: lr=0.1 finishes, lr=0.2 times out and is not resubmitted (max_resumes 0)
    first, sweep = _sweep(tmp_path, monkeypatch, lambda name, attempt: 'finish' if name == 'trial_0' else 'signal',
                          lrs=(0.1, 0.2), template_extra=dict(max_resumes=0))
    assert first == [('trial_0', 0), ('trial_1', 0)]
    if not busy:
        _age(sweep / 'trial_1', 2 * oh.DEFAULT_RESUME_STALE_MINUTES)
    # Relaunch with one more grid value: lr=0.1 is skipped, lr=0.2 resumes in trial_1, and only
    # lr=0.3 becomes a new trial
    calls, _ = _sweep(tmp_path, monkeypatch, lambda name, attempt: 'finish', lrs=(0.1, 0.2, 0.3),
                      skip_completed_grid_jobs=True)
    new_dirs = sorted(set(_trial_dirs(sweep)) - {'trial_0', 'trial_1'})
    assert len(new_dirs) == 1
    assert yaml.safe_load(open(sweep / new_dirs[0] / 'train_config.yaml'))['lr'] == 0.3
    if busy:
        # Its files changed minutes ago: an earlier dispatcher's job may still write there
        assert calls == [(new_dirs[0], 0)]
    else:
        assert sorted(calls) == sorted([('trial_1', 1), (new_dirs[0], 0)])


def test_relaunch_without_a_checkpoint_reruns_in_the_trial_folder(tmp_path, monkeypatch):
    _, sweep = _sweep(tmp_path, monkeypatch, lambda name, attempt: 'signal', template_extra=dict(max_resumes=0))
    (sweep / 'trial_0' / tbc.CHECKPOINT_NAME).unlink()  # e.g. killed in its first epoch
    calls, _ = _sweep(tmp_path, monkeypatch, lambda name, attempt: 'finish', skip_completed_grid_jobs=True)
    assert calls == [('trial_0', 0)]
    assert _trial_dirs(sweep) == ['trial_0']


def test_relaunch_without_the_flag_keeps_the_old_behaviour(tmp_path, monkeypatch):
    _, sweep = _sweep(tmp_path, monkeypatch, lambda name, attempt: 'signal', template_extra=dict(max_resumes=0))
    _age(sweep / 'trial_0', 2 * oh.DEFAULT_RESUME_STALE_MINUTES)
    calls, _ = _sweep(tmp_path, monkeypatch, lambda name, attempt: 'finish')
    assert calls == [('trial_1', 0)]
    assert _trial_dirs(sweep) == ['trial_0', 'trial_1']
