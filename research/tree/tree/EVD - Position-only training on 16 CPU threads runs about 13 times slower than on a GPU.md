---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a188f2-0567-7ead-a679-d5fb781f5055
status: done
priority: normal
dateCreated: 2026-10-29T19:35:24+01:00
dateModified: 2026-10-29T19:35:24+01:00
confidence: low
completedDate: 2026-10-29
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Find whether position-only (keypoints-only) networks can train in CPU-only Slurm jobs, since GPU slots are the cluster budget and CPU jobs are plentiful.

## Method

r261029-3966658: the h512 / 600-frame position-only network of long14 trained with CUDA_VISIBLE_DEVICES="" and OMP_NUM_THREADS=16 on the workstation (56 cores, load ~55-69 from other jobs), stopped after 400 steps of epoch 0.

## Results

A steady ~262 s per 100 steps, so ~1260 s per 480-step epoch, against ~94 s per epoch on the workstation GPU (long14): 13x slower. Full-video CPU-only evals on the cluster took 40-130 min per recording (benchmark_more_datasets).

## Conclusion

A position-only network of the current size trains in roughly 10-20 h of CPU time for 60 epochs, against about 1.6 h on a GPU: slow but usable on CPU-only Slurm jobs (template key train_cpu_job, 22c6f0c) when GPU slots are scarce. Image or fused networks were not measured.

## Caveats

One probe on a loaded machine; a dedicated cluster node may be up to ~2x faster. Thread scaling was not measured.
