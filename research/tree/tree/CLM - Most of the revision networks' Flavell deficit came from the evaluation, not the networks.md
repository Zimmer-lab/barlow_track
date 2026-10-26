---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a1776f-cad9-72cb-b7fa-4772ea954a0a
status: open
priority: normal
dateCreated: 2026-10-26T09:59:37+01:00
dateModified: 2026-10-26T09:59:37+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Claim

The revision sweeps' Flavell scores (.62-.73, about 25 points under the preprint) were mostly evaluation artefacts: the NWB neuron-order bug (about 9 points) and scoring against the project GT rather than the NWB GT the paper used (about 4 points). Scored like for like, the best revision network is 2.8 points below the preprint network.

## Rationale

The same bug costs the preprint's own network 10 points; the GT switch costs its stored tracks 4 points; the best revision network moves from 0.880 (project GT) to 0.924 (NWB GT, same path as the preprint network's 0.952).

## Scope

Flavell. The residual 2.8-point gap is real (10x the seed spread) and lies in the image features; see the position-pipeline Evidence. Every round_flavell* and NWB-path Flavell record from before the fix needs rescoring before it is compared with the paper.
