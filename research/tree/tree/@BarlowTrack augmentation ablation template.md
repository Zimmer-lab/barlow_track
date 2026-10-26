---
nodeTypeId: node_ygg_source
nodeInstanceId: 01a1659f-a81d-79b5-ac07-f143bb08c528
status: done
priority: normal
dateCreated: 2025-11-12
dateModified: 2025-11-12
completedDate: 2025-11-12
run: barlow_track@987eca2:barlow_track/alternative_project_templates/augmentation_ablation/hyperparameter_search_template.yaml
projects:
  - "[[barlowtrack]]"
---

## Description

One-at-a-time sweep template (used with --one_at_a_time_sweep) setting each augmentation probability to 0: p_RandomBlur_base, p_RandomAffine_both, p_RandomElasticDeformation, p_RandomAffine_flip, p_RandomNoise. The design of Fig. S1e (remove one) and, inverted, S1f (only one).

Outputs not rerun here. Cited at tag v0.1.0 (987eca2, 2025-10-29); code is identical to the last commit before the preprint (7a555d5), but the paper's runs may predate the tag, so the exact code version behind the figure is uncertain.
