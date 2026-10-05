# Barlow Track Bug-Fix PR Plan

Generated after sequential local subagent reviews of architecture, tracking, data, label propagation, eval, and agglomeration modules.

Current repo state at planning time:

- Branch: `global_augmentation`
- Latest reviewed commit: `9fbeda5 Fix bug with string/int comparison`
- Already fixed: `data_loading.py` int-vs-string `tracked_segs` bug.

This directory is intentionally documentation-only. Use these files to create branches/worktrees and execute changes.

## Recommended PR Order

1. `PR01` - eval accuracy correctness and reporting denominators.
2. `PR02` - label propagation and spectral relabeling reliability.
3. `PR03` - tracking pipeline windows, resume, and final-frame handling.
4. `PR04` - data/crop/augmentation pipeline correctness.
5. `PR05` - NN architecture, model loading, and architecture tests.
6. `PR06` - deterministic seeding and reproducibility.
7. `PR07` - agglomeration post-processing latent fixes (optional/dead-path hardening).

## PR Index

| PR | Branch | Scope | Priority |
| --- | --- | --- | --- |
| PR01 | `bugfix/eval-accuracy-correctness` | Accuracy eval denominators, NaN matching, unmatched columns, CPU/GPU normalization | High |
| PR02 | `bugfix/label-propagation-reliability` | Label propagation thresholds, label 0, alignment matching, spectral top-k, single-object handling | High |
| PR03 | `bugfix/tracking-windows-resume` | Window coverage, final frame skip, resume pickle names, tracking latent traps | High |
| PR04 | `bugfix/data-crop-augmentation` | Crop padding, silent zero crops, `train_fraction`, augmentation test expectations | Medium/High |
| PR05 | `bugfix/nn-model-loading-tests` | Position fusion defaults, old checkpoint loading tests, projection shape assumptions | Medium |
| PR06 | `feature/determinism-seeding` | Seed NNDescent, Python random, UMAP, worker RNG; make `--seed` meaningful | Medium |
| PR07 | `bugfix/agglomeration-latent-paths` | Dead/latent agglomeration fixes, partition invariants, goodness metric | Low/Optional |

## Conventions

- Each PR should be reviewable without changing unrelated behavior.
- Keep API changes small and documented.
- Add or update unit tests for behavior changes, especially eval denominators and augmentation variable-N behavior.
- If a result number changes, call it out explicitly in the PR description and compare against a baseline on a known project.

## Useful Test Commands

Use the `wbfm` conda environment:

```bash
/home/charles/anaconda3/envs/wbfm/bin/python -m pytest barlow_track/tests/ -q
```

Focused checks:

```bash
/home/charles/anaconda3/envs/wbfm/bin/python -m pytest barlow_track/tests/test_barlow_position.py -q
/home/charles/anaconda3/envs/wbfm/bin/python -m pytest barlow_track/tests/test_step1_global_augment.py -q
/home/charles/anaconda3/envs/wbfm/bin/python -m pytest barlow_track/tests/test_step0_embed_and_augment.py -q
```
