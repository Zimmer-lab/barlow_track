---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a165a4-a369-7b64-90e0-989cf2c75d13
status: open
priority: high
dateCreated: 2026-10-23T00:04:10+02:00
dateModified: 2026-10-23T00:04:10+02:00
projects:
  - "[[barlowtrack]]"
---

## Question

Does encoding the positions of neurons in a volume (keypoint MLP, ViSNet, attention across the objects of one volume) on top of the image crop raise accuracy and generalization, and which fusion (add, concat, position only) and attention depth work best?

## Context

The Discussion names position information as the clear next step (transformer architectures, SuperGlue). Since the preprint this is the main development line on global_augmentation: position encoding (430f4c2, #128), attention over single-volume embeddings (6e1b1e4, #135, #136), ViSNet position encoder (d23836a, 270aad5), Bayesian sweeps on tracking accuracy (346c26c, #145). Per the (unversioned) Oct 2026 summary, ViSNet beats the keypoint MLP everywhere, embedding dim barely matters, and attention/self-depth/fusion are still open.
