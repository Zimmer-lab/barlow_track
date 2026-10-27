---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a1776f-cad9-72cb-b7fa-4772ea954a0a
status: open
priority: normal
dateCreated: 2026-10-26T09:59:37+01:00
dateModified: 2026-10-27T04:00:51+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Claim

The revision sweeps' Flavell scores (.62-.73, about 25 points under the preprint) were mostly evaluation artefacts: the NWB neuron-order bug (about 9 points) and scoring against the project GT rather than the NWB GT the paper used (about 4 points). Scored like for like, a typical revision network is about 9 points below the preprint network, and the best of 13 training seeds 2.8 points below.

## Rationale

The same bug costs the preprint's own network 10 points; the GT switch costs its stored tracks 4 points, and the sweep cells 2.9-4.1 points. Against the NWB GT, 13 training seeds of the best revision configuration score 0.80-0.93 (median 0.865, [[EVD - Position-pipeline Flavell accuracy spans 12 points across training seeds]]) against the preprint network's 0.952 on the same path.

## Scope

Flavell. The residual gap is real and lies in the image features; see the position-pipeline Evidence. Its size depends on the training seed: 2.8 points was trial27, the best of 13 seeds, and the earlier "10x the seed spread" compared against labelprop seeds, not training seeds. Every round_flavell* and NWB-path Flavell record from before the fix needs rescoring before it is compared with the paper.
