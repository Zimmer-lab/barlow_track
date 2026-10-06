"""One seeding strategy for the whole pipeline.

Randomness in this repo comes from four places (python ``random``, numpy,
torch, and libraries that take an explicit ``random_state``: pynndescent,
UMAP, sklearn). Two rules cover all of them:

1. **Explicit RNGs (preferred).** Pipeline functions and classes take a
   ``seed`` / ``random_state`` argument and build *private* RNGs from it
   (``random.Random``, ``np.random.RandomState``, a library ``random_state``).
   They never read a global stream, so their output depends only on the seed
   and the input -- not on what ran before, on the DataLoader worker id, or on
   iteration order. ``derive_seed`` is the single way to split one base seed
   into independent sub-seeds (per epoch, per worker, per dataset item).

2. **Global seeding (entry points only).** Code that still reaches for the
   global streams -- the legacy crop pipeline in ``barlow_lightning``
   (``random.sample``), third-party defaults such as sklearn's
   ``TruncatedSVD`` -- is not worth an API change. Every entry point that runs
   such code calls ``seed_all(seed)`` once at startup, which makes those
   globals reproducible *for that entry point*. Keep the call at the very top
   of ``main`` and leave a comment pointing here.

``seed=None`` everywhere means "unseeded", i.e. the historical behaviour: it is
never a silent 0.
"""
import hashlib
import random

import numpy as np

# numpy RandomState seeds must fit in 32 bits; torch.manual_seed is happier
# below 2**31, so call sites that seed torch pass bits=31.
NUMPY_SEED_BITS = 32
TORCH_SEED_BITS = 31


def derive_seed(*parts, bits=NUMPY_SEED_BITS):
    """Mix parts (e.g. base seed, epoch, index, a tag like ``'torch'``) into one seed.

    Stable across runs and platforms (hashlib, not ``hash``), order-sensitive,
    and unlike ``base + epoch + worker`` it does not collide when parts are
    permuted. Parts may be ints or short strings; each is type-tagged so the
    index ``1`` and the tag ``"1"`` cannot land on the same seed. Returns an int
    in ``[0, 2**bits)``.
    """
    key = ",".join(f"i:{int(p)}" if isinstance(p, (int, np.integer)) else f"s:{p}"
                   for p in parts).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    bits = int(bits)
    if not 0 < bits <= 64:
        raise ValueError(f"bits must be in 1..64, got {bits}")
    return int.from_bytes(digest, 'big') >> (64 - bits)


def python_rng(seed):
    """A private ``random.Random`` for one call (``None`` = unseeded)."""
    return random.Random(None if seed is None else int(seed))


def numpy_rng(seed):
    """A private ``np.random.RandomState`` for one call (``None`` = unseeded)."""
    return np.random.RandomState(None if seed is None else derive_seed(seed))


def replicate_seed(base_seed, repetition):
    """Seed for replicate number ``repetition`` of one configuration.

    Hyperparameter sweeps keep one ``base_seed`` for every configuration so the
    differences between configs stay attributable to the hyperparameters, and
    only re-runs of the *same* config need a fresh seed. Replicate 0 keeps
    ``base_seed`` unchanged (a plain sweep reproduces the template seed); later
    replicates get derived seeds, which is what turns them into real
    measurements of init/data noise rather than identical jobs.
    """
    base_seed = int(base_seed)
    repetition = int(repetition)
    if repetition < 0:
        raise ValueError(f"repetition must be >= 0, got {repetition}")
    return base_seed if repetition == 0 else derive_seed(base_seed, 'rep', repetition)


def search_trial_seed(base_seed, trial_index):
    """Distinct seed per trial of a Bayesian (Ax/BoTorch) search.

    A constant seed across trials makes the objective look noise-free: the GP
    then over-trusts a configuration that happened to draw a lucky init or a
    favorable volume selection. Deriving one seed per trial (a fixed function of
    the trial index, so the whole search still reproduces) decorrelates that
    nuisance randomness from the hyperparameters being optimized.
    """
    return derive_seed(int(base_seed), 'trial', int(trial_index))


def seed_all(seed, include_cuda=False):
    """Seed python's ``random``, numpy and torch from one integer.

    Only for entry points (rule 2 in the module docstring): library code that
    reads a global stream is then reproducible, but code that *takes* a seed
    should still be given it explicitly -- ``seed_all`` is a fallback, not a
    substitute for threading the seed through.

    Each stream gets ``derive_seed(seed, <name>)`` rather than the bare
    ``seed``, so the three libraries do not start from the same integer and
    sample correlated values (same trick torch uses to seed its DataLoader
    workers). The mapping is fixed, so one ``--seed`` still reproduces a run.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(derive_seed(seed, 'numpy'))
    import torch
    torch.manual_seed(derive_seed(seed, 'torch', bits=TORCH_SEED_BITS))
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(derive_seed(seed, 'cuda', bits=TORCH_SEED_BITS))
