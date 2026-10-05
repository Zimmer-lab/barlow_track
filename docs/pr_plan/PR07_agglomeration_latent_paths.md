# PR07 - Agglomeration Latent-Path Hardening

Suggested branch: `bugfix/agglomeration-latent-paths`

## Goal

Fix latent bugs in the agglomeration post-processing module so it can be used safely later. This PR is optional if the module is currently dead code.

## Files Likely Touched

- `barlow_track/utils/utils_agglomeration.py`
- possibly a new unit-test file for agglomeration

## Background

The agglomeration module appears mostly or fully unused in the current scripted pipeline, but it is likely relevant for custom post-processing notebooks.

The review found several correctness issues that would make experimental agglomeration results hard to trust.

## Bugs / Tasks

- [ ] Treat HDBSCAN label `0` as a real cluster.
  - Current code:
    ```python
    if labels[idx] > 0:
    ```
  - HDBSCAN uses `-1` for noise; `0` is a valid cluster label.
  - Fix:
    ```python
    if labels[idx] >= 0:
    ```

- [ ] Preserve the partition invariant.
  - Current risk: an index can be an outlier and later a kept index or member of another accepted cluster.
  - Fix: exclude `outliers` from future seed selection and acceptance logic.

- [ ] Make the goodness metric reject obvious merges of distinct neurons.
  - Current concern: coverage dominates purity, and purity only checks “at most one object per frame,” so temporally disjoint neurons can be merged.
  - Fix options:
    - score `purity * coverage`;
    - enforce a purity floor;
    - require temporal continuity or penalize large gaps.

- [ ] Clean hygiene issues.
  - Remove stray `from cProfile import label`.
  - Avoid mutable default arguments.
  - Fix variable shadowing where seed index is overwritten by loop variable.
  - Return or remove unused `conflicted_times`.

## Suggested Test Coverage

- [ ] Tiny synthetic clustering test where HDBSCAN label 0 is selected correctly.
- [ ] Test that outliers are not later reassigned.
- [ ] Test that two temporally disjoint neurons are not merged by default goodness settings.

## Out of Scope

- Wiring agglomeration into the main pipeline.
- Designing a final post-processing API.
