---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1aa86-fb11-7b16-b0eb-a407bdf67fef
status: open
priority: normal
dateCreated: 2026-11-05T08:05:35+01:00
dateModified: 2026-11-05T08:05:35+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Conclusion
The question is how much of the strict Leifer StarDist accuracy (base 0.920 over kept clicks) is GT trouble on a few neurons.

Six tracking runs were compared: base seeds 0 and 1, trials 57, 50 and 74, and fDNC, which uses positions only.
- In 3,618 of base's 6,737 errors (54%), every run holds the same object, and it is not the GT's.
- The click sits a median 1.51 µm from the GT's object on those clicks; the typical assigned click is 0.53 µm from its object. The consensus object is the one closer to the click in 27% of them.
- The consensus errors cluster in a few neurons: 023 (36% of its clicks), 072 (23%), 076 (21%), 044 (20%) and 032 (20%).

Removing neurons changes the headline as follows:
- The 2 ambiguous-click neurons (023, 058): +0.7 points (0.927).
- The 5 neurons ranked worst by GT-only warning signs: +1.1 points (0.931).
- The 5 with the most consensus errors: +1.8 points (0.938).
- 10 neurons, by either ranking: +2.4 to +3.1 points (0.944–0.950).

The ranking of networks does not change, and fDNC gains less (0.803 to 0.809–0.829).

## Method
- `neuron_removal_scores.py` (62921a6) on sd_strict_status.npz.
- Runs: saved pred_all tracks, labelprop seed 0 unless named, and fDNC swap_mirror with the 5-template vote. All are matched to the full GT with gt_valid; removed neurons' clicks are dropped after matching.
- GT-only warning signs per neuron, combined as a mean rank called suspicion:
  - fraction of clicks excluded (coincident or merged);
  - fraction of segmentation misses;
  - median click-to-object-centroid distance;
  - spikiness of the GT-tracked ratio trace, from the `_stardist_strictgt` traces.
- Consensus error: an assigned click where all six runs hold the same object, and it is not the GT label.
- Output: `~/.cache/barlow_scratch/train/leifer_reseg/neuron_removal/`:
  - neurons.csv: per neuron, the warning signs, each run's accuracy, consensus_err_frac and any_err_frac;
  - removal_scores.csv;
  - consensus_error_clicks.csv: 3,618 clicks with the NWB voxel click, the GT object and the consensus object, for manual checking.

Accuracy over kept clicks (over assigned clicks in brackets):

| Removed | n | base s0 | t74 s0 | t57 s0 | fDNC vote |
|---|---|---|---|---|---|
| none | 0 | 0.920 (0.940) | 0.923 (0.944) | 0.917 (0.937) | 0.803 (0.821) |
| ambiguous clicks: 023, 058 | 2 | 0.927 (0.948) | 0.930 (0.951) | 0.924 (0.944) | 0.810 (0.828) |
| neuron_070 (37% segmentation misses, 16% excluded) | 1 | 0.924 (0.940) | 0.927 (0.943) | 0.921 (0.937) | 0.806 (0.820) |
| GT-only suspicion, top 5 | 5 | 0.931 (0.948) | 0.934 (0.951) | 0.927 (0.945) | 0.809 (0.825) |
| GT-only suspicion, top 10 | 10 | 0.944 (0.955) | 0.947 (0.958) | 0.940 (0.951) | 0.816 (0.825) |
| consensus errors, top 5 | 5 | 0.938 (0.959) | 0.938 (0.959) | 0.934 (0.955) | 0.819 (0.837) |
| consensus errors, top 10 | 10 | 0.950 (0.969) | 0.951 (0.969) | 0.946 (0.964) | 0.829 (0.844) |
| worst traces: 023, 015, 068, 024, 058 (chosen from tracker disagreement) | 5 | 0.939 (0.957) | 0.941 (0.960) | 0.935 (0.954) | 0.817 (0.833) |

## Caveats
- **Only a manual check justifies dropping a neuron.** Removal chosen from tracker errors (consensus, worst traces) is circular: dropping the neurons a tracker fails on raises its score by construction.
- **The runs are not independent.** Five of the six are networks of one family. fDNC is the independent one, and requiring it to agree makes the consensus conservative.
- **A consensus error can also be a shared segmentation problem:** e.g. a StarDist split puts the click's object next to a fragment every tracker prefers.
- **One labelprop seed per network** except base.
