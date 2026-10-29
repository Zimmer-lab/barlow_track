---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a18538-d909-7bf1-b4bc-fe52dc032eb5
status: done
priority: normal
dateCreated: 2026-10-29T02:14:17+01:00
dateModified: 2026-10-29T02:14:24+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
outcome: succeeded
completedDate: 2026-10-29
---

## Conclusion
For round_visnet_full trial 27 (trained on Zimmer), tracking with the fused position-plus-image descriptor instead of the image backbone alone raises accuracy on zimmer_1128 from 0.923 to 0.968 and on Leifer from 0.958 to 0.972, but lowers Samuel 153 from 0.949 to 0.837. The image backbone transfers to Samuel; the position and attention head does not. A Samuel-trained add-fusion network (round_bayes_samuel trial 0) is near neutral on Samuel (backbone 0.951, fused 0.943). So which descriptor stage to track with depends on the dataset, for this network, and cannot be fixed globally.

## Method
Runs r261028-3159902 and r261028-3398874; label propagation, 25 seeds; same checkpoint and ground truth at both stages. Samuel on NWB GT objects (oracle detections), Zimmer and Leifer from their projects.

## Caveats
One Zimmer-trained network and one Samuel-trained network; no seed repeats, so differences of about 1 point are not resolved (local vs cluster record for bayes0 on Samuel already differ by 0.8). Not checked whether Samuel's loss comes from the ViSNet branch or the self-attention layers. The ViSNet branches in both networks predate the RBF fix (issue i261028-2), so the position branch here sees little distance information.
