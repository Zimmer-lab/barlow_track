"""Sweep objective: tracking accuracy in Ax's objective slot.

Covers the pieces of scripts/optimize_hyperparameters.py that decide WHAT Ax
optimizes and WHAT a finished trial leaves on disk:

  * failed trials and the loss/accuracy direction (a failed trial is marked
    FAILED in Ax, never scored, and Ax must not be asked to minimize an
    accuracy), plus the stop after repeated failures and the preflight,
  * that a template without `objective` behaves exactly like before,
  * that the in-sweep record tag is the one the cross-dataset benchmark
    deduplicates on, so a post-sweep benchmark skips instead of re-embedding,
  * that the eval is called with the trial's OWN project for both crops and
    ground truth, and that a trial without a checkpoint is never tracked,
  * the embedding-cache rules that make a killed sweep resumable.
"""
import json
import os
import shutil

import numpy as np
import pytest

from barlow_track.scripts import optimize_hyperparameters as oh
from barlow_track.scripts import eval_accuracy as ea
from barlow_track.utils.utils_ground_truth import read_results_by_tag, record_tag


# --------------------------------------------------------------------------
# Objective direction and failure values
# --------------------------------------------------------------------------

def test_accuracy_objective_is_a_maximization():
    """Ax is told minimize=False, and real accuracies are reported unnegated.

    Sign-flipping the score while ALSO telling Ax to maximize it would silently
    rank the worst checkpoint first, so the readable value is what goes in.
    """
    assert oh.objective_value(0.42) == pytest.approx(0.42)
    assert oh.objective_value(0.0) < oh.objective_value(1.0)


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), float('-inf'), 'oops'])
def test_unusable_scores_are_failures_not_values(bad):
    """No finite stand-in: a -1e6 outlier would flatten every real accuracy in the GP."""
    assert oh.objective_value(bad) is None
    assert oh.trial_failed({'result': oh.objective_value(bad)})


def test_trial_failed_only_for_unusable_results():
    assert not oh.trial_failed({'result': 0.3})
    assert oh.trial_failed({'result': None})
    assert oh.trial_failed(None)


def test_failed_trial_is_left_out_of_the_fit():
    """A FAILED trial contributes no data; a completed one does."""
    from ax.service.ax_client import AxClient
    client = AxClient()
    oh.create_sweep_experiment(client, [{'name': 'lr', 'type': 'range', 'bounds': [0.1, 1.0]}],
                               accuracy_objective=True)
    trials, _ = client.get_next_trials(max_trials=2)
    good, bad = sorted(trials)
    client.complete_trial(good, raw_data={'result': (0.6, 0.0)})
    oh.log_trial_failure(client, bad)
    assert client.experiment.trials[bad].status.is_failed
    df = client.experiment.fetch_data().df
    assert set(df.trial_index) == {good}


def test_grid_sweep_candidate_trials_can_be_failed():
    """Grid sweeps make trials with experiment.new_trial(), which Ax leaves CANDIDATE."""
    from ax.core.arm import Arm
    client = _ax_client()
    trial = client.experiment.new_trial()
    trial.add_arm(Arm(parameters={'lr': 0.5}))
    assert trial.status.is_candidate
    oh.log_trial_failure(client, trial.index)
    assert client.experiment.trials[trial.index].status.is_failed


def test_consecutive_failures_stop_the_sweep_and_a_success_resets_them():
    guard = oh.ConsecutiveFailureGuard(3)
    assert [guard.record(f) for f in (True, True, False, True, True)] == [False] * 5
    assert guard.record(True), "third failure in a row"


@pytest.mark.parametrize('setting', [0, None])
def test_consecutive_failure_stop_can_be_disabled_or_defaulted(setting):
    guard = oh.ConsecutiveFailureGuard(setting)
    results = [guard.record(True) for _ in range(oh.DEFAULT_MAX_CONSECUTIVE_FAILURES)]
    if setting == 0:
        assert not any(results)
    else:
        assert results[-1] and not any(results[:-1])


# --------------------------------------------------------------------------
# Preflight: config errors surface before any trial trains
# --------------------------------------------------------------------------

class _FakeProject:
    def __init__(self, finished, final):
        self._finished, self.final_tracks = finished, final

    def get_final_tracks_only_finished_neurons(self):
        return self._finished, None


def _patch_project(monkeypatch, project=None, error=None):
    from wbfm.utils.projects.finished_project_data import ProjectData

    def load(path, **kwargs):
        if error is not None:
            raise error
        return project
    monkeypatch.setattr(ProjectData, 'load_final_project_data', staticmethod(load))


def _gt_df():
    import pandas as pd
    cols = pd.MultiIndex.from_product([['neuron_001', 'neuron_002'], ['raw_segmentation_id']])
    return pd.DataFrame(np.ones((5, 2)), columns=cols)


def test_preflight_passes_for_a_project_with_ground_truth(monkeypatch):
    _patch_project(monkeypatch, _FakeProject(_gt_df(), None))
    assert '2 neurons, 5 frames' in ea.preflight_trained_eval('/p/project_config.yaml')


def test_preflight_falls_back_to_final_tracks_like_the_eval(monkeypatch):
    import pandas as pd
    _patch_project(monkeypatch, _FakeProject(pd.DataFrame(), _gt_df()))
    assert '2 neurons' in ea.preflight_trained_eval('/p/project_config.yaml')


def test_preflight_rejects_a_project_without_ground_truth(monkeypatch):
    import pandas as pd
    _patch_project(monkeypatch, _FakeProject(pd.DataFrame(), None))
    with pytest.raises(ValueError, match='no ground-truth tracks'):
        ea.preflight_trained_eval('/p/project_config.yaml')


def test_preflight_rejects_an_unloadable_project(monkeypatch):
    _patch_project(monkeypatch, error=FileNotFoundError('no such project'))
    with pytest.raises(ValueError, match='no such project'):
        ea.preflight_trained_eval('/p/project_config.yaml')


def test_preflight_nwb_source_needs_an_existing_nwb(tmp_path):
    with pytest.raises(ValueError, match='NWB file'):
        ea.preflight_trained_eval('/p/project_config.yaml', source='nwb')
    with pytest.raises(ValueError, match='not found'):
        ea.preflight_trained_eval(str(tmp_path / 'x.nwb'), source='nwb')
    nwb = tmp_path / 'x.nwb'
    nwb.write_bytes(b'')
    assert 'NWB' in ea.preflight_trained_eval(str(nwb), source='nwb')


# --------------------------------------------------------------------------
# Template parsing
# --------------------------------------------------------------------------

def test_template_without_objective_key_is_the_historical_loss_sweep():
    objective, label, kwargs = oh.resolve_objective({})
    assert objective == 'loss'
    assert label is None
    # Same eval defaults as the benchmark runner, so nothing downstream changes.
    assert kwargs['num_seeds'] == 25 and kwargs['seed'] == 0
    assert kwargs['cluster'] == 'labelprop' and kwargs['descriptor_stage'] == 'auto'
    assert kwargs['max_frames'] is None


def test_template_objective_accuracy_reads_dataset_and_knobs():
    objective, label, kwargs = oh.resolve_objective({
        'objective': 'accuracy', 'objective_dataset': 'zimmer_1128',
        'objective_num_seeds': 10, 'objective_max_frames': 60})
    assert objective == 'accuracy'
    assert label == 'zimmer_1128' and kwargs['lab'] == 'zimmer_1128'
    assert kwargs['num_seeds'] == 10 and kwargs['max_frames'] == 60


def test_objective_dataset_defaults_to_a_label_no_benchmark_knows():
    """Unset label must not silently claim a benchmark cell it never evaluated."""
    from barlow_track.scripts.multiproject_scripts import run_trials_on_all_ground_truth as bench
    objective, label, _ = oh.resolve_objective({'objective': 'accuracy'})
    assert objective == 'accuracy'
    assert label not in bench.DATASETS


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match='Unknown objective'):
        oh.resolve_objective({'objective': 'iou'})


# --------------------------------------------------------------------------
# Record tags: the contract with the cross-dataset benchmark
# --------------------------------------------------------------------------

def test_sweep_record_tag_is_the_tag_the_benchmark_skips_on(tmp_path):
    """The whole point of writing standard records: the benchmark dedups by tag."""
    from barlow_track.scripts.multiproject_scripts import run_trials_on_all_ground_truth as bench
    sweep = tmp_path / 'round_bayes_zimmer'
    sweep.mkdir()
    for trial_num in (0, 7):
        for dataset in ('zimmer_1128', 'samuel'):
            tag = oh.record_tag(str(sweep), trial_num, dataset)
            assert tag == f'round_bayes_zimmer_trial{trial_num}_{dataset}'
    # What the sweep wrote, as the benchmark reads it.
    results_jsonl = str(sweep / 'exp_results.jsonl')
    with open(results_jsonl, 'w') as f:
        for trial_num in (0, 7):
            for dataset in ('zimmer_1128', 'samuel'):
                f.write(json.dumps(dict(tag=oh.record_tag(str(sweep), trial_num, dataset),
                                        accuracy=0.5)) + '\n')
    already = bench.recorded_tags(results_jsonl)
    # Every cell the sweep evaluated is already covered -> nothing left to run.
    for trial_num in (0, 7):
        for dataset in ('zimmer_1128', 'samuel'):
            assert oh.record_tag(str(sweep), trial_num, dataset) in already


def test_record_tag_normalizes_trailing_slash():
    assert (record_tag('/a/b/sweep/', 2, 'leifer')
            == record_tag('/a/b/sweep', 2, 'leifer') == 'sweep_trial2_leifer')


# --------------------------------------------------------------------------
# The in-sweep eval: which project, which tag, which paths
# --------------------------------------------------------------------------

def _fake_trial(tmp_path, with_checkpoint=True, n=3):
    trial_dir = tmp_path / 'sweep' / f'trial_{n}'
    (trial_dir / 'log').mkdir(parents=True)
    (trial_dir / 'checkpoints').mkdir()
    if with_checkpoint:
        (trial_dir / 'resnet50.pth').write_bytes(b'not really a checkpoint')
    with open(trial_dir / 'log' / 'stats.json', 'w') as f:
        json.dump([dict(epoch=1, val_loss=0.25), dict(epoch=2, test_loss=0.5)], f)
    return trial_dir


def test_measure_tracking_accuracy_scores_the_trials_own_project(tmp_path, monkeypatch):
    trial_dir = _fake_trial(tmp_path)
    calls = {}

    def fake_eval(**kwargs):
        calls.update(kwargs)
        return dict(accuracy=0.375, n_frames=1667, total=206701, minutes=37.5)

    monkeypatch.setattr(oh, 'evaluate_trained_checkpoint', fake_eval)
    args = type('Args', (), dict(project_dir=str(trial_dir),
                                 project_path='/data/my_project/project_config.yaml'))()
    acc = oh.measure_tracking_accuracy(
        args, dict(test_loss=0.5, test_loss_transpose=0.25), str(tmp_path / 'sweep'),
        'zimmer_1128', 'accuracy',
        str(tmp_path / 'sweep' / 'exp_results.jsonl'), str(tmp_path / 'sweep' / 'emb_cache'),
        dict(lab='zimmer_1128', source='project', cluster='labelprop', num_seeds=25, seed=0,
             descriptor_stage='auto', max_frames=None, center_per_volume=False,
             l2_per_volume=False))
    assert acc == pytest.approx(0.375)
    # Crops AND ground truth come from the training project: no test dataset can
    # be reached, because project_path is the only path passed in.
    assert calls['project'] == calls['gt'] == '/data/my_project/project_config.yaml'
    assert calls['weights'] == str(trial_dir / 'resnet50.pth')
    assert calls['tag'] == f'sweep_trial3_zimmer_1128'
    # Records and embeddings go where the benchmark runner looks for them.
    assert calls['results_jsonl'] == str(tmp_path / 'sweep' / 'exp_results.jsonl')
    assert calls['emb_dir'] == str(tmp_path / 'sweep' / 'emb_cache')
    # Losses travel with the accuracy as diagnostics.
    assert calls['extra_record']['test_loss'] == pytest.approx(0.5)
    assert calls['extra_record']['val_loss'] == pytest.approx(0.25)


def test_trial_without_checkpoint_is_never_tracked(tmp_path, monkeypatch):
    """A crashed trial saves no checkpoint; tracking it would waste an hour."""
    trial_dir = _fake_trial(tmp_path, with_checkpoint=False)
    monkeypatch.setattr(oh, 'evaluate_trained_checkpoint',
                        lambda **kw: pytest.fail("must not track a trial with no checkpoint"))
    args = type('Args', (), dict(project_dir=str(trial_dir), project_path='/p/project_config.yaml'))()
    with pytest.raises(FileNotFoundError, match='did not finish training'):
        oh.measure_tracking_accuracy(args, dict(test_loss=float('inf')), str(tmp_path / 'sweep'),
                                     'zimmer_1128', 'accuracy', 'r.jsonl', 'emb',
                                     dict(lab='zimmer_1128'))


def test_nonfinite_losses_do_not_reach_the_record_as_nan(tmp_path, monkeypatch):
    """JSON has no NaN: an inf loss must be recorded as null, not poison the file."""
    trial_dir = _fake_trial(tmp_path)
    captured = {}
    monkeypatch.setattr(oh, 'evaluate_trained_checkpoint',
                        lambda **kw: captured.update(kw) or dict(accuracy=0.1, n_frames=10,
                                                                 total=10, minutes=1.0))
    args = type('Args', (), dict(project_dir=str(trial_dir), project_path='/p/project_config.yaml'))()
    oh.measure_tracking_accuracy(args, dict(test_loss=float('inf')), str(tmp_path / 'sweep'),
                                 'zimmer_1128', 'accuracy', 'r.jsonl', 'emb',
                                 dict(lab='zimmer_1128'))
    assert captured['extra_record']['test_loss'] is None
    assert captured['extra_record']['val_loss'] == pytest.approx(0.25)


def test_experiment_minimizes_loss_and_maximizes_accuracy():
    """The direction lives in the experiment, not in a flipped score."""
    from ax.service.ax_client import AxClient
    params = [{'name': 'lr', 'type': 'range', 'bounds': [0.1, 1.0]}]
    for accuracy_objective, minimize in ((False, True), (True, False)):
        client = AxClient()
        oh.create_sweep_experiment(client, params, accuracy_objective)
        objective = client.experiment.optimization_config.objective
        assert isinstance(objective.minimize, bool)
        assert objective.minimize is minimize


# --------------------------------------------------------------------------
# Attaching finished trials to Ax (the resume path)
# --------------------------------------------------------------------------

def _ax_client():
    pytest.importorskip('ax')
    from ax.service.ax_client import AxClient
    client = AxClient()
    client.create_experiment(name='test', parameters=[{'name': 'lr', 'type': 'range',
                                                        'bounds': [0.1, 1.0]}])
    return client


def test_attach_trial_returns_an_integer_index_and_keeps_the_config():
    """Ax >= 0.3 returns (parameterization, index) and read-only run_metadata."""
    client = _ax_client()
    config = {'lr': 0.5, 'project_dir': '/sweep/trial_0', 'backbone_kwargs': {'f_maps': 4}}
    index = oh._attach_trial(client, config)
    assert isinstance(index, int), "callers index the experiment with this"
    trial = client.experiment.trials[index]
    assert trial.run_metadata['project_dir'] == '/sweep/trial_0'
    assert trial.run_metadata['backbone_kwargs'] == {'f_maps': 4}
    # Still pending: the score arrives later (from the eval-only job).
    assert not str(trial.status).endswith('COMPLETED')
    client.complete_trial(index, raw_data={'result': (0.25, 0.0)})
    assert str(client.experiment.trials[index].status).endswith('COMPLETED')


def test_attach_prior_trial_completes_with_the_accuracy():
    client = _ax_client()
    index = oh.attach_prior_trial_to_ax_client(client, {'lr': 0.5}, 0.42)
    df = client.experiment.fetch_data().df
    assert float(df.loc[df.trial_index == index, 'mean'].iloc[0]) == pytest.approx(0.42)
    # get_best_parameters hands back ({metric: mean}, {metric: sem}) in Ax 0.3.
    assert oh.objective_mean(client.get_best_parameters()[1]) == pytest.approx(0.42)


def test_attach_prior_trial_rejects_nonfinite_scores():
    client = _ax_client()
    for bad in (None, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='Non-finite'):
            oh.attach_prior_trial_to_ax_client(client, {'lr': 0.5}, bad)


# --------------------------------------------------------------------------
# Tracking device (the OOM this setting exists for)
# --------------------------------------------------------------------------

def test_track_device_defaults_follow_the_embed_device():
    torch = pytest.importorskip('torch')
    args = type('A', (), dict(track_device='embed_device'))()
    assert ea.resolve_track_device(args, torch.device('cuda')) == torch.device('cuda')
    # None is the tracker's "keep it on the CPU" value.
    assert ea.resolve_track_device(args, torch.device('cpu')) is None


def test_track_device_cpu_keeps_full_video_graphs_off_the_gpu():
    """Full-video kNN graphs OOM a 12 GB GPU next to the trained model."""
    torch = pytest.importorskip('torch')
    args = type('A', (), dict(track_device='cpu'))()
    assert ea.resolve_track_device(args, torch.device('cuda')) is None


def test_track_device_cuda_falls_back_when_cuda_is_absent(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    args = type('A', (), dict(track_device='cuda'))()
    assert ea.resolve_track_device(args, torch.device('cpu')) is None


def test_sweep_defaults_objective_tracking_to_cpu():
    _, _, kwargs = oh.resolve_objective({'objective': 'accuracy'})
    assert kwargs['track_device'] == 'cpu'


# --------------------------------------------------------------------------
# Resume: reading records back, and the embedding cache
# --------------------------------------------------------------------------

def test_read_results_by_tag_keeps_the_latest_record_per_tag(tmp_path):
    path = tmp_path / 'exp_results.jsonl'
    with open(path, 'w') as f:
        f.write(json.dumps(dict(tag='sweep_trial0_zimmer_1128', accuracy=0.10)) + '\n')
        f.write(json.dumps(dict(tag='sweep_trial0_zimmer_1128', accuracy=0.42)) + '\n')
        # A kill mid-append leaves a line with no trailing newline, so the NEXT
        # append continues it and both are unparseable. Dropping them is the
        # safe direction: the reader re-runs those cells, it never misreads one.
        f.write('{"tag": "sweep_trial1_zimmer_1128", "accur')
        f.write(json.dumps(dict(tag='sweep_trial2_zimmer_1128', accuracy=0.3)) + '\n')
        f.write(json.dumps(dict(tag='sweep_trial3_zimmer_1128', accuracy=0.2)) + '\n')
    records = read_results_by_tag(str(path))
    assert set(records) == {'sweep_trial0_zimmer_1128', 'sweep_trial3_zimmer_1128'}
    assert records['sweep_trial0_zimmer_1128']['accuracy'] == pytest.approx(0.42)


    # The benchmark's skip set comes from the same reader, so the two agree.
    from barlow_track.scripts.multiproject_scripts import run_trials_on_all_ground_truth as bench
    assert bench.recorded_tags(str(path)) == set(records)


def test_read_results_by_tag_missing_file_is_empty(tmp_path):
    assert read_results_by_tag(str(tmp_path / 'nope.jsonl')) == {}


def test_truncated_embedding_cache_is_re_embedded_not_tracked(tmp_path):
    """kill -9 during np.savez leaves a partial zip; it must not be trusted."""
    emb = tmp_path / 'emb_zimmer_trained_sweep_trial0.npz'
    emb.write_bytes(b'PK\x03\x04 truncated')
    assert ea._load_cached_embeddings(str(emb)) is None


def test_embedding_cache_of_the_wrong_length_is_re_embedded(tmp_path):
    emb = str(tmp_path / 'emb.npz')
    ea._save_embeddings(emb, X=np.zeros((4, 3), dtype=np.float32),
                        time_to_lin={0: [0, 1], 1: [2, 3]}, lin_to_t_seg={}, n_frames=2)
    assert ea._load_cached_embeddings(emb, expected_n_frames=2) is not None
    # A re-run with a different --max_frames must not track the wrong frames.
    assert ea._load_cached_embeddings(emb, expected_n_frames=60) is None


def test_embeddings_are_written_atomically(tmp_path):
    """No half-written cache at emb_path: a kill leaves only a temp file."""
    emb = str(tmp_path / 'emb.npz')
    ea._save_embeddings(emb, X=np.ones((2, 2), dtype=np.float32), time_to_lin={}, lin_to_t_seg={},
                        n_frames=0)
    assert os.path.isfile(emb)
    assert not any(f.startswith('emb.npz.tmp') for f in os.listdir(str(tmp_path)))


# --------------------------------------------------------------------------
# Job sizing
# --------------------------------------------------------------------------

@pytest.mark.parametrize('minutes,expected', [
    (780, '13:00:00'),        # 65*12 hours of training budget, unchanged default
    (1500, '1-01:00:00'),     # train + a 4 h objective budget
    (150, '2:30:00'),
])
def test_slurm_duration_format(minutes, expected):
    assert oh.slurm_duration(minutes) == expected


@pytest.mark.parametrize('epochs', [25, 100, 250])
def test_loss_sweep_walltime_is_unchanged(epochs):
    """The loss objective keeps its historical {num_days}-12:00:00 job."""
    num_days = int(epochs / 100) + 1
    assert oh.slurm_duration(oh.job_budget_minutes(epochs)) == f"{num_days}-12:00:00"


@pytest.mark.parametrize('epochs', [25, 100, 250])
def test_accuracy_trial_gets_the_full_training_budget_plus_the_eval(epochs):
    """A trial that trains AND tracks must not be killed mid-training or mid-tracking."""
    loss = oh.job_budget_minutes(epochs)
    accuracy = oh.job_budget_minutes(epochs, objective_minutes=240)
    assert accuracy == loss + 240 + 60


def test_slurm_job_for_an_accuracy_sweep_of_25_epochs():
    # 1 day + 12 h of training, 4 h of eval, 1 h of headroom
    assert oh.slurm_duration(oh.job_budget_minutes(25, objective_minutes=240)) == '1-17:00:00'
