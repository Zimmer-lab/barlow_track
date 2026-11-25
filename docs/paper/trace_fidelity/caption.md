# Trace agreement for R3.W6 and R3.Q2

## Draft figure captions

### BarlowTrack trace overlays

```latex
\caption{\textbf{\normalsize Fluorescence traces from BarlowTrack and reference tracks.}
Solid blue lines follow the reference identity and dashed orange lines follow the matched BarlowTrack prediction from the first trial~59 training replicate.
\textbf{(a)} Zimmer 1128, worm~1, with manually curated tracks.
\textbf{(b)} Leifer NeRVE worm~1, with manual clicks assigned to unambiguous StarDist masks.
\textbf{(c)} Samuel 153, with manual centroid tracks and centroid-seeded watershed masks.
\textbf{(d)} Flavell 2022-03-31-01, a GFP control with identities from the laboratory's published tracker.
For each track, fluorescence is the raw ratio of summed green to summed red intensity in the selected mask. Both tracks use the same images and segmentation. Gaps indicate missing or undefined measurements.
Gray shading marks backward crawling in \textbf{(a,b)}. Green shading marks ventral contact in \textbf{(c)}.
Each row shows the first 120~s for one neuron. The purple box marks the 8~s enlarged at right, chosen around the largest difference between distinct mask assignments in the displayed 10th-percentile example.
For each recording, two neurons were selected nearest the median and 10th percentile of full-recording Pearson $r$. Additional calcium examples were selected by reference-trace signal-to-noise ratio.
Labels give full-recording $r$, mask accuracy and coverage. Of all observed reference frames for that neuron, accuracy is the percentage assigned the reference mask and coverage is the percentage assigned any mask.}
\label{fig:barlowtrack_overlays}
```

### BarlowTrack agreement across neurons

```latex
\caption{\textbf{\normalsize Agreement between BarlowTrack and reference fluorescence traces across neurons and training replicates.}
\textbf{(a--d)} Each point represents one neuron in the first trial~59 replicate, ranked by full-recording Pearson $r$ between predicted and reference traces.
The recordings are Zimmer 1128 (manually curated tracks), Leifer NeRVE worm~1 (manual clicks on StarDist masks), Samuel 153 (manual centroid tracks) and Flavell 2022-03-31-01 (published tracker identities, GFP control), respectively.
Both traces use the raw ratio of integrated green to red fluorescence in the assigned mask. Correlations include all shared, observed frames with finite ratios, including frames with different mask assignments. Each trace must vary over at least ten paired observations.
Circles and triangles mark the median and 10th-percentile examples in Fig.~\ref{fig:barlowtrack_overlays}. The dashed line marks $r=0.9$.
\textbf{(e)} Median $r$ across neurons for each of five training replicates on the same recording. Replicates run from top to bottom within each row, with the first replicate filled.
\textbf{(f)} First-replicate median $r$, first and third quartiles, number with $r<0.9$ out of all evaluable neurons, median coverage, and numbers excluded for constant traces or insufficient overlap.
Coverage is the percentage of a neuron's observed reference frames with any predicted mask. It is undefined when the reference has no observations.}
\label{fig:barlowtrack_agreement}
```

### fDNC trace overlays

```latex
\caption{\textbf{\normalsize Fluorescence traces from fDNC and reference tracks.}
Solid blue lines follow the reference identity and dashed orange lines follow the matched track from the fDNC five-template vote.
\textbf{(a)} Zimmer 1128, worm~1, with manually curated tracks.
\textbf{(b)} Leifer NeRVE worm~1, with manual clicks assigned to unambiguous StarDist masks.
\textbf{(c)} Samuel 153, with manual centroid tracks and centroid-seeded watershed masks.
\textbf{(d)} Flavell 2022-03-31-01, a GFP control with identities from the laboratory's published tracker.
Both traces show the raw ratio of integrated green to red fluorescence in the selected mask. Gaps indicate missing or undefined measurements.
Each row shows the first 120~s for one neuron. The purple box marks the same 8~s enlarged in Fig.~\ref{fig:barlowtrack_overlays}. Gray shading marks backward crawling in \textbf{(a,b)} and green shading marks ventral contact in \textbf{(c)}.
The median and 10th-percentile examples were selected from each recording's fDNC correlation distribution. The seven additional calcium examples are the same reference neurons shown for BarlowTrack.
Labels give full-recording Pearson $r$, mask accuracy and coverage. Of all observed reference frames for that neuron, accuracy is the percentage assigned the reference mask and coverage is the percentage assigned any mask.}
\label{fig:fdnc_overlays}
```

### fDNC agreement across neurons

```latex
\caption{\textbf{\normalsize Agreement between fDNC and reference fluorescence traces across neurons.}
\textbf{(a--d)} Each point represents one neuron, ranked by full-recording Pearson $r$ between the fDNC and reference raw green/red fluorescence ratios.
The recordings are Zimmer 1128 (manually curated tracks), Leifer NeRVE worm~1 (manual clicks on StarDist masks), Samuel 153 (manual centroid tracks) and Flavell 2022-03-31-01 (published tracker identities, GFP control), respectively.
Ratios were extracted from the same images and segmentation. Each correlation uses at least ten shared, observed frames with finite ratios and variation in both traces. Frames with different mask assignments are included.
Circles and triangles mark the median and 10th-percentile examples in Fig.~\ref{fig:fdnc_overlays}. The dashed line marks $r=0.9$.
\textbf{(e)} Percentage of observed reference frames assigned the correct mask, pooled across neurons within each recording, for the deterministic fDNC five-template vote (filled) and the first BarlowTrack replicate (open). A missing prediction counts as an incorrect assignment.
\textbf{(f)} Median $r$, first and third quartiles, number with $r<0.9$ out of all evaluable neurons, median coverage, and numbers excluded for constant traces or insufficient overlap. Coverage is the percentage of a neuron's observed reference frames with any predicted mask.
The accompanying tables also report agreement after background subtraction, bleach correction and normalization to each track's median ratio.}
\label{fig:fdnc_agreement}
```

## Proposed Methods correction for approval

In the manuscript’s *Calculation of time series* subsection (`sn-article.tex`, line 711), replace the sentence “The Leifer lab datasets … did not include segmentation, and traces were not recalculated for them” with the following text.

> For Leifer NeRVE worm 1, we recalculated fluorescence traces from the green and red image volumes using the same 3D StarDist masks for the manual reference, BarlowTrack and fDNC. At each volume, the reference mask was the one containing the retained manual click and each tracker supplied the mask assigned by its matched track. For each selected mask we divided its integrated green fluorescence by its integrated red fluorescence. We also evaluated the background-subtracted, per-channel bleach-corrected ratio normalized to each track’s median. For both signals we calculated Pearson correlations on shared observed frames, including frames with different mask assignments.

## Regeneration

From the root of `barlow_track`, with the benchmark data mounted at `/lisc/data/scratch`, run

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/barlow_mpl NUMBA_CACHE_DIR=/tmp/barlow_numba_cache \
  /home/charles/anaconda3/envs/wbfm/bin/python -m barlow_track.scripts.paper_scripts.revision.trace_fidelity \
  --cache /tmp/r3trace-calc --workers 10 --out docs/paper/trace_fidelity
```

The analysis used the wbfm checkout at commit `543fb5e7`. The script builds its cache directly from the source images and segmentation if needed. Omit `--cache /tmp/r3trace-calc` to read all images again. Inputs are the `gt` and `pred_matched` keys in `round_shared_v1/tracks_v2/round_shared_v1_trial59{,r1,r2,r3,r4}_<recording>.h5`. Each predicted column retains the benchmark’s identity match across the full recording. The script verifies the saved mask-level accuracy for each replicate before scoring fluorescence.

The fDNC input directory defaults to `~/.cache/barlow_scratch/fdnc/r1m2_trial59` (override with `--fdnc-dir`). Its `paired_table_details.json` identifies the published pretrained model’s five-template vote for each recording. The benchmark chose the axis convention by highest mean match probability across four conventions. These runs used template volume 10 plus four volumes spread across the recording and rescaled positions to the fDNC reference worms. The script reads the selected vote’s saved tracks from that directory or from the benchmark’s recorded `reused_from` location. For Zimmer and Leifer it converts per-volume detection indices to mask labels using trial 59’s embedding index. It matches predicted identities over the whole video with the benchmark’s `nandist` rule, checks the reference matrix against BarlowTrack’s and requires the counts of missed and wrong assignments to reproduce the benchmark exactly. Samuel has five additional trailing fDNC frames with no reference observations. Both figures use the 1,326-volume common recording.

For the secondary processed signal, we subtracted a constant background per voxel, independently fitted an exponential bleach curve to each track and channel, and computed the corrected green/red ratio R. We normalized each track to its own median ratio R₅₀ as (R − R₅₀)/R₅₀, removed values more than five standard deviations from the mean and applied a centered three-volume mean. Before scoring, we explicitly restored gaps at missing and rejected observations. The primary raw ratio uses the integrated channel intensities directly, with no additional background subtraction, bleach fit, normalization, smoothing or outlier removal. The offsets used for the secondary signal were 75 (Zimmer, Leifer and Flavell) and 0 (Samuel, where the median intensity outside masks was zero in both channels in four checked volumes). The nominal volume rates were 3.5, 6, 1.7 and 10 Hz for Zimmer, Leifer, Flavell and Samuel, respectively. The Samuel comparison uses the 1,326 volumes in the saved trial 59 tracks. Leifer coverage is conditional on a manual click having an assigned mask. A click outside all masks supplies no reference fluorescence.

In `per_neuron.csv`, `n_ref`, `n_pred` and `n_shared` count measured masks. Both per-neuron tables include `n_correct_masks`, the number of exact frame-local reference-mask matches, and `mask_accuracy`, this count divided by `n_ref`. A neuron with no observed reference frames has undefined accuracy and coverage. The `processed_n` and `raw_n` fields count finite paired values used by each signal, and the corresponding `n_different` fields count paired frames with differing mask assignments. Status fields give the reason for an undefined correlation. A variable corrected ratio with a zero median cannot be normalized to ΔR/R₅₀. Fit fields record wbfm’s fallback to the uncorrected channel when exponential fitting fails. In `per_replicate.csv`, quartiles, threshold counts and exclusions are reported separately for each signal. Coverage fractions range from 0 to 1 in the tables. Replicate 1 is the unsuffixed trial 59 file. The fDNC tables are `fdnc_per_neuron.csv` and `fdnc_per_replicate.csv`. The latter contains a single output per recording with an empty training-seed field. Both summary tables include `mask_accuracy`, the fraction of observed reference frames assigned the correct mask. In all five BarlowTrack Samuel replicates, neuron_001 had no reference observations and neuron_067 had a varying corrected ratio with R₅₀ = 0. The latter remains evaluable in the primary raw-ratio comparison.


## Identity, behavior and signal-to-noise sources

- **Zimmer 1128.** Anatomical labels come from the recording’s `3-tracking/manual_annotation/manual_annotation.xlsx`. We used unique primary IDs with certainty ≥2 and an empty alternative-ID field. The candidate classes were RID, BAG, RME and AVA from the manuscript’s fast-worm, high-magnification and larval example panels. Backward crawling is the `Rev` label in the same recording’s `beh_annotation_manual_corrected_timeseries.csv`. We sampled the middle behavior exposure in each fluorescence volume (index 12 of 24, following the wbfm convention).
- **Leifer worm 1.** Backward crawling is code −1 in `outputFiles/heatData.mat::ethoTrack`. The first timestamp in `hasPointsTime` corresponds to the end of acquisition stack 3. All 1,516 published timestamps matched the corresponding stack end in `hiResData.mat` to the millisecond. Volumes without a behavioral annotation are unshaded. [Nguyen et al., NeRVE data dictionary](https://github.com/leiferlab/NeRVEclustering)
- **Samuel 153.** The authors’ static viewer coordinates equal (zero-based NWB XYZ + 1) × (3, 3, 8) at NWB volume 4. Ninety-five of 97 viewer positions match a single NWB centroid within 0.001 viewer units. We display anatomical names for unique position matches and omit the viewer’s anonymous Q numbers. The included centroid table contains the `neuron_id,x,y,z` fields from `dataset_id=mating_153`. The included contact intervals are contiguous runs of `Ventral.Contact=1` for dataset 153 in the deposited activity table (533 of 1,325 published volumes). We mapped published volume 1 to benchmark frame 0. As a timing check, the median correlation between position-matched reference traces and the published traces peaked at zero lag over offsets of ±6 volumes after centered smoothing. The final benchmark frame has no published behavioral label. [Susoy et al., viewer coordinates](https://storage.googleapis.com/mating-data.worm.world/annotations.json) · [Susoy et al., activity and behavior data, version 1 (CC BY 4.0)](https://data.mendeley.com/datasets/922xsfxh2g/1)
- **Example selection.** SNR uses the raw reference green/red ratio. Signal amplitude is its standard deviation after clipping values to the 5th and 95th percentiles for ranking only. Noise is 1.4826 times the median absolute difference between successive observed volumes divided by √2. Differences across gaps are omitted. Each calcium recording’s eligible examples were ranked by this ratio, independently of tracker agreement. Within Zimmer, we take one member of each of the top three eligible classes. Within Samuel we take the two highest-ranked remaining anatomically identified neurons, and within Leifer the top two remaining tracks. The plotted ratios retain their measured amplitudes. The `anatomical_id`, `reference_snr` and `example` fields in the per-neuron table document the selections.
