---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1776f-7ea0-71c4-b30d-3f389b4fdcbd
status: done
priority: normal
dateCreated: 2026-10-24T07:35
dateModified: 2026-10-26T09:59:37+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-10-26
---

## Purpose

Redo the comparison of revision networks with the preprint now that the Flavell evaluation bug and GT mismatch are known (supersedes the Evidence that found Flavell 25 points lower).

## Method

Best Flavell-trained revision network (round_flavell_more_epochs trial27; emb 512, layernorm, ViSNet add fusion, 75 epochs, 17,900 steps) rescored against the NWB GT from its cached embedding, --source project, 25 seeds, seeds 0-2 (r261024-2431943, r261024-2472680); descriptor variants backbone-only and eval-time per-frame centring (r261024-2523199). Zimmer-trained round_visnet_full trials (cross-dataset) on the aligned NWB path (r261024-2391371). Other labs from the October sweep summary, unchanged.

## Results

Flavell, like for like (project path, NWB GT): trial27 0.9271 / 0.9209 / 0.9250 (mean 0.924) vs preprint checkpoint 0.952 (gap 2.8 pt, about 10x the seed spread); trial27 recorded 0.880 against the project GT. trial27 backbone-only 0.9154, centred 0.9130, fused 0.9271. Zimmer-trained transfer to Flavell (aligned NWB path): visnet_full trial27 0.912, trial20 0.831, vs the paper network 0.969 on that path. From the sweep summary: Zimmer .949, Leifer .961, Samuel .950 vs the preprint's 97.3 and 97.4% (Samuel not in the paper).

## Conclusion

No revision network matches the preprint yet, but the gap is 1-3 points on every lab, not 25 on Flavell: the flat .62-.73 came from the NWB bug and the project GT. Position fusion adds about 1 point; the revision network's image backbone alone is 3.7 points below the preprint's, so the gap is in the image features (training pipeline or backbone-side design).

## Caveats

Zimmer, Leifer and Samuel numbers are from the git-ignored AGENTS.md summary and were not rescored here. The visnet_full transfer networks were trained on Zimmer with z 8 crops, 100 frames and 25 epochs, so they are not like for like with the Flavell-trained preprint network. Single training seed for trial27.
