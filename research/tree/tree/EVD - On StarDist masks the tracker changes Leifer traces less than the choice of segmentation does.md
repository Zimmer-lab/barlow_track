---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1a9e9-2240-7b4b-9c23-908ffc99c7a0
status: open
priority: normal
dateCreated: 2026-11-05T05:13:10+01:00
dateModified: 2026-11-05T07:47:36+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Conclusion
On the Leifer recording, each GT neuron was compared between pairs of trace sets (Pearson r of the raw green/red ratio on frames both have; base network, labelprop seed 0, 79 neurons):

- Tracker-tracked vs GT-tracked traces on the same StarDist masks agree at a median r of 0.987 (8 neurons below 0.9).
- On NeRVE's own masks the same comparison gives 0.943 (24 neurons below 0.9).
- Switching segmentation with the GT fixed (NeRVE vs StarDist masks, both GT-tracked) gives 0.908 (33 below 0.9).
- NeRVE's GT-tracked traces vs our full StarDist + tracker pipeline give 0.881.

wbfm's processed paper traces agree less in every pair (0.975, 0.938, 0.842, 0.816), and their gap interpolation differs between projects. So on StarDist masks the tracker's errors cost the traces less than the choice of segmentation does.

The worst neurons are the same low-activity neurons in every pair, including GT vs GT across segmentations. On StarDist, their differences are short flickers to an unannotated object 3–4 µm away. On those frames it is usually the GT-tracked trace that spikes: the tracker's trace is smoother than the GT-tracked one in 73 of 79 neurons, including all 7 with r_paper below 0.8.

## Method
- Project copies made by `tracks_to_project.py` (9f14b48):
  - `_stardist_strictgt`: final tracks are the strict StarDist GT.
  - `_stardist_track_base`: base's predicted tracks, matched to the GT neurons with gt_valid on sd_strict (reproduces 0.9404). Each matched track is written at every volume it holds an object, including volumes without an assigned click, as a user would get them.
  - `_nervegt_track_base`: base's predicted tracks on the NeRVE-masks project, matched on nerve_only.
- The reference NeRVE traces are the `_nervegt` project's own (NeRVE segmentation_id GT).
- wbfm `snakemake traces` ran on each copy (r261105-2936414; it plans only extract_full_traces and traces).
- Compared with `compare_traces.py` (5c74d01): per neuron, Pearson r of the raw green/red ratio (summed mask intensities) and of `paper_traces.h5` on frames both have, and the fraction of shared frames holding the same mask.
- Check: two GTs on one segmentation (1.5 µm centroid vs mask GT) give r = 1.000 and the same object on 100% of shared frames.
- Output: `~/.cache/barlow_scratch/train/leifer_reseg/traces_cmp/compare/` (summary.json, a csv per pair, trace_agreement.png).
- Worst neurons: `inspect_trace_neurons.py` (f8f20c1), output in `traces_cmp/inspect/`, which holds `<pair>_neurons.csv`, `<pair>_worst.png` (traces with differing frames shaded) and `<pair>_r_vs_snr.png`.
  - Click distances are from the re-export NWB clicks.
  - Spikiness of a trace is the median |x_t − (x_{t−1} + x_{t+1})/2| of the ratio divided by its median, on frames both projects have.

| Reference vs test | Ratio r median (p10, min) | Ratio r below 0.9 | Paper r median | Same object (median) |
|---|---|---|---|---|
| StarDist strict GT vs StarDist tracker | 0.987 (0.900, 0.40) | 8 / 79 | 0.975 | 0.987 |
| NeRVE GT vs NeRVE tracker | 0.943 (0.725, 0.42) | 24 / 78 | 0.938 | 0.942 |
| NeRVE GT vs StarDist strict GT | 0.908 (0.776, 0.56) | 33 / 78 | 0.842 | - |
| NeRVE GT vs StarDist tracker | 0.881 (0.729, 0.43) | 46 / 78 | 0.816 | - |

Worst neurons, StarDist GT vs tracker:

| Neuron | r_paper | r_ratio | Same object | Differing runs (longest) | Click to GT / tracker object on differing frames (µm) | Spikiness GT / tracker | SNR |
|---|---|---|---|---|---|---|---|
| 023 | 0.22 | 0.79 | 0.63 | 308 (13) | 1.51 / 1.57 | 0.059 / 0.030 | 2.3 |
| 015 | 0.35 | 0.40 | 0.82 | 162 (4) | 0.76 / 3.40 | 0.071 / 0.063 | 2.0 |
| 068 | 0.41 | 0.91 | 0.89 | 97 (4) | 1.04 / 2.76 | 0.053 / 0.048 | 3.7 |
| 024 | 0.45 | 0.75 | 0.50 | 247 (27) | 0.57 / 3.46 | 0.058 / 0.022 | 3.8 |
| 058 | 0.52 | 0.82 | 0.72 | 257 (10) | 1.99 / 1.62 | 0.087 / 0.047 | 3.0 |

- **Low activity:** the worst neurons carry little activity beyond a slow drift. SNR (ratio s.d. over noise) is 2.0–3.8, against a median of 4.5 for neurons with r_paper ≥ 0.9; Spearman r_paper vs SNR is 0.43. The same neurons (023, 068, 015) are worst in the GT-vs-GT segmentation pair.
- **Swaps:** almost none are between annotated neurons; the other object is another GT neuron's on at most 5% of differing frames.
- **Two kinds of disagreement:**
  - Ambiguous clicks (023, 058): the click is about as far from the tracker's object as from the GT object, within 1 µm on 77–88% of differing frames. 023's click sits 1.19 µm from its object even where the two agree; the median over neurons is 0.53 µm.
  - Tracker flicker (015, 024, 068): the click is 0.6–1.0 µm from the GT object and the tracker's object is 2.8–3.5 µm away.
- **Spikiness:** single-volume ratio spikes in the GT-tracked traces line up with differing frames. On NeRVE masks the tracker's trace is smoother in only 58% of neurons.

## Caveats
- One network and one labelprop seed.
- Pearson r on the raw ratio, with no bleach correction or background subtraction.
- The tracker comparisons include frames where the GT has no click (the tracker's own object), so they mix tracking errors with coverage.
- Which segmentation gives the truer trace is not answered: NeRVE's masks are not ground truth either.
- A smoother trace is not proof of correct tracking. A tracker that consistently follows the wrong neighbour is smooth too. Single-volume spikes at GCaMP6s kinetics are implausible, so the spikes argue for click hopping, but no second annotator has checked those frames.
