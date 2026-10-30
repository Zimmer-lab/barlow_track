"""train_barlow_clusterer output bookkeeping (synthetic, fast)."""
from barlow_track.scripts.train_barlow_clusterer import _archive_previous_run


def test_previous_run_outputs_move_aside(tmp_path):
    for rel in ['resnet50.pth', 'resnet50-1.pth', 'args.pickle', 'log/stats.json',
                'checkpoints/checkpoint.pth', 'train_config.yaml', 'log/correlation_matrix_0.png']:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(rel)
    dest = _archive_previous_run(tmp_path)
    for rel in ['resnet50.pth', 'resnet50-1.pth', 'args.pickle', 'log/stats.json', 'checkpoints/checkpoint.pth']:
        assert not (tmp_path / rel).exists()
        assert (dest / rel).read_text() == rel
    assert (tmp_path / 'train_config.yaml').exists()  # inputs and plots stay


def test_fresh_dir_and_kept_pretrained_checkpoint(tmp_path):
    assert _archive_previous_run(tmp_path) is None
    (tmp_path / 'resnet50.pth').write_text('pretrained')
    assert _archive_previous_run(tmp_path, keep=[str(tmp_path / 'resnet50.pth'), None]) is None
    assert (tmp_path / 'resnet50.pth').exists()


def test_epoch_snapshot_is_a_loadable_model_folder_and_moves_aside_on_rerun(tmp_path):
    import pickle
    from types import SimpleNamespace
    import torch
    from barlow_track.scripts.train_barlow_clusterer import _save_epoch_snapshot, epoch_snapshot_dir
    model = torch.nn.Linear(3, 2)
    args = SimpleNamespace(project_dir=str(tmp_path), epochs=30)
    _save_epoch_snapshot(model, args, 5)
    d = epoch_snapshot_dir(tmp_path, 5)
    weights = torch.load(f"{d}/resnet50.pth")
    assert torch.equal(weights['weight'], model.weight.detach())
    with open(f"{d}/args.pickle", 'rb') as f:
        snap = pickle.load(f)
    assert snap.model_fname == f"{d}/resnet50.pth" and snap.snapshot_epoch == 5 and snap.epochs == 30
    assert not hasattr(args, 'model_fname')  # the run's own args are untouched
    dest = _archive_previous_run(tmp_path)
    assert (dest / 'epoch_snapshots/epoch_005/resnet50.pth').exists()
