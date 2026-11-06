"""Regression tests for the 2026-11-06 code-review fixes (shared sweep bookkeeping, CUDA fallback)."""
import pytest
import torch

from barlow_track.scripts.optimize_shared_config import SharedSweep
from barlow_track.utils.utils_label_propagation import _is_cuda_resource_error


def _sweep(max_frames):
    sw = SharedSweep.__new__(SharedSweep)
    sw.max_frames = max_frames
    return sw


def test_records_count_only_at_the_runs_eval_length():
    full, smoke = _sweep(None), _sweep(12)
    assert full.usable(dict(accuracy=0.9)) is not None            # legacy record: full video
    assert full.usable(dict(accuracy=0.9, max_frames=None)) is not None
    assert full.usable(dict(accuracy=0.9, max_frames=12)) is None  # a smoke cell never feeds a real run
    assert smoke.usable(dict(accuracy=0.9, max_frames=12)) is not None
    assert smoke.usable(dict(accuracy=0.9)) is None
    assert full.usable(None) is None


def test_lower_is_better_metric_is_not_an_objective(tmp_path):
    (tmp_path / 'shared_template.yaml').write_text(
        "tuning_labels: [a]\nobjective_metric: dup_frac\nreference_dup_frac: {a: 0.0}\n"
        "base_train_config: base.yaml\n")
    with pytest.raises(ValueError, match='dup_frac'):
        SharedSweep(str(tmp_path))


def test_cuda_resource_errors_fall_back_but_bugs_do_not():
    assert _is_cuda_resource_error(torch.cuda.OutOfMemoryError('CUDA out of memory'))
    assert _is_cuda_resource_error(RuntimeError('CUSPARSE_STATUS_ALLOC_FAILED when calling cusparseSpMM'))
    assert _is_cuda_resource_error(RuntimeError('CUDA error: an illegal memory access was encountered'))
    assert not _is_cuda_resource_error(RuntimeError('size mismatch, got 3 and 4'))
