---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a165a4-a3d0-7581-9f8a-7f4378a37343
status: open
priority: normal
dateCreated: 2026-10-23T00:04:11+02:00
dateModified: 2026-10-23T00:04:11+02:00
projects:
  - "[[barlowtrack]]"
---

## Question

How much does accuracy vary across training seeds and across label-propagation seeds on fixed embeddings, and are the preprint's medians stable?

## Context

Table 2 reports a median over networks without spread or n. After the preprint, label propagation was found to be seed-fragile on weak embeddings (an unseeded random.shuffle of seed times), and docs/pr_plan/PR06 plans seeding of Python random, NNDescent, UMAP and DataLoader workers; partly done in 80a2bbd (#144, fix_determinism) and aa42e45 (derived sweep seeds), not verified against the plan.
