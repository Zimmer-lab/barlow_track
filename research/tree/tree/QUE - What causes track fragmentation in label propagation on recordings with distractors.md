---
nodeTypeId: node_ygg_question
nodeInstanceId: 01a1b113-0277-7789-851e-9bfcedc0a35a
status: in-progress
priority: high
dateCreated: 2026-11-06T14:36:15+01:00
dateModified: 2026-11-06T14:36:15+01:00
projects:
  - "[[barlowtrack]]"
---

## Question

When a GT neuron's objects end up on a label matched to no GT neuron, is it because (1) the neuron is absent from most seed volumes, (2) the spectral sync splits the neuron into two consensus labels over time, or (3) a distractor track's label captures the neuron in some volumes? And which lever (more or better seed volumes, temporal links, re-propagation, label merging, cannot-link) reduces it?

## Context

Fragment errors (pt of GT cells, base tracker, first 100 volumes): Zimmer 1210 t36 2.4, Flavell 0330-02 t61 1.9, Leifer t57 9.7; Samuel 0 (GT objects only, no distractors). Perfect-merge bounds: Flavell .925 -> .941, Leifer .847 -> .936.

Only cause 1 responds to seed count (`seeds50`, `seeds100`, `seed_complete`). Cause 2 needs something that links time regimes (`temporal1/5`, `reprop*`, a label merge). Cause 3 gets worse with more seeds and needs cannot-link (`framenorm`), thresholds or label pruning. The earlier finding that 100 seeds are neutral on S137 says nothing here, because S137 has no fragments.

Since 2026-11-06, sweep_labelprop.py records split `err_fragment` into `err_frag_split` (the unmatched label's majority GT neuron is the object's own: a split track), `err_frag_distractor` (the label is mostly non-GT objects) and `err_frag_other`. Plan: PLAN.md section 3a.
