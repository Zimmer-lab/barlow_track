import os
import tempfile

import pytest


def _writable_cache_dir(name):
    # numba/matplotlib need writable caches; HOME may be read-only (containers)
    d = os.path.join(tempfile.gettempdir(), name)
    os.makedirs(d, exist_ok=True)
    return d


os.environ.setdefault('NUMBA_CACHE_DIR', _writable_cache_dir('barlow_numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', _writable_cache_dir('barlow_mpl'))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
