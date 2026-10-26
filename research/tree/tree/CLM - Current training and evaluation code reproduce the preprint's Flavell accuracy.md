---
nodeTypeId: node_ygg_claim
nodeInstanceId: 01a1776f-c9b6-7859-8fe1-866133b0a9fb
status: done
priority: normal
dateCreated: 2026-10-26T09:59:37+01:00
dateModified: 2026-10-26T09:59:37+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
completedDate: 2026-10-26
---

## Claim

On Flavell, the code as of bc9ff96 reproduces the preprint: the preprint's saved embeddings re-track to its stored 0.968, its network scores 0.952-0.969 under the current evaluation, and its configuration retrained today reaches 0.957.

## Rationale

Three independent routes (saved embeddings, saved network, retrained network) land within 1.5 points of the paper's 96.7%, and the remaining differences are accounted for by measured choices (crop normalization, U*S SVD, GT-centred crops). The post-preprint label-propagation fixes move trained accuracy by 0.1 pt.

## Scope

Flavell only, single training seed; Zimmer, Leifer and Samuel are not re-scored. Requires the Flavell NWB fields aligned by neuron name (issue i261026-1) and the NWB GT, not the project GT.
