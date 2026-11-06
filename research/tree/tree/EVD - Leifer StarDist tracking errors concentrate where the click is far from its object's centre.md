---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a1a858-e52f-7c17-850d-f482479488ac
status: open
priority: normal
dateCreated: 2026-11-04T21:56:00+01:00
dateModified: 2026-11-04T21:56:09+01:00
confidence: medium
projects:
  - "[[barlowtrack]]"
---

## Conclusion
For `base` on the mask GT the error rate climbs steadily with the distance from the click to the centroid of the GT's object: 1.8% under 0.5 um, 3.8% at 0.5-1, 10% at 1-1.5, 24% at 1.5-2, 32% at 2-3 and 53% beyond 3 um. Cells whose click is 1.5 um or more from the GT object's centre (10% of the mask GT, nearly all of them added by the mask rule over the 1.5 um rule) hold 47% of its errors. Nearly all errors are brief jumps to an unannotated neighbouring object about 3.8 um away (5,578 of 6,983 errors; runs at most 31 frames, 41% single frames), not identity swaps between annotated neurons (222). Where the click is 1.5 um or more from the GT object's centre, the tracker's object is nearer the click in 48% of mismatches, so a large share of these "errors" are ambiguous GT, not tracker failures.

## Method
Tracks as in [[EVD - On the 1.5 um Leifer StarDist ground truth the column matching loses neurons with low coverage]]; `tracking_error_patterns.py` (05d2104) per valid GT cell: click to GT object centroid, object volume and mean intensity relative to the neuron's median, nearest other click and object; per error its kind and the predicted object's distance to the click and to the GT object. Distances in um (1.5 x 0.3226 x 0.3226), centroids from the StarDist segmentation metadata.

Mask GT, error rate by GT-only property: object volume 0.75-1.33x the neuron's median 4.3% (90% of cells), below 0.75x 21%, 1.33-2x 21%, above 2x 89% (0.1%); brightness 0.75-1.33x 5.3%, below 0.5x 52% (0.06%); nearest other object 2-3 um 12%, beyond 4 um 3%; nearest other click and objects per volume barely matter. Worst 10 neurons hold 64% of errors; no volume has more than half its cells wrong, and the 6 volumes above 20% hold 1.2% of errors.

Fixed GT-only filters (thresholds set before looking at errors), mask GT: dropping clicks more than 1 um from their object's centroid drops 22% of cells and 67% of errors (accuracy 0.9739); object volume outside 0.5-2x drops 0.4% of cells and 4% of errors (0.9408); dim objects, crowded clicks and crowded objects change nothing. Centroid GT, click more than 1 um away: 14% of cells, 44% of errors (0.9661).

## Caveats
- One network, one labelprop seed; the 1.0 um GT scored directly is running (r261104-3067468).
- "Nearer the click" compares centroid distances only, not which mask contains the click.
- Click-to-centroid distance mixes annotation habit (some neurons are clicked off-centre) with segmentation shape (large or merged masks); not separated yet.
- Filtering by click distance raises accuracy by removing hard cells as well as wrong ones; it is a sensitivity analysis, not a corrected score.
