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
