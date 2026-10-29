---
nodeTypeId: node_ygg_evidence
nodeInstanceId: 01a188f2-0476-7487-afdc-8d1fe7a84669
status: done
priority: normal
dateCreated: 2026-10-29T19:35:24+01:00
dateModified: 2026-10-29T19:35:24+01:00
confidence: medium
completedDate: 2026-10-29
outcome: succeeded
projects:
  - "[[barlowtrack]]"
---

## Purpose

Look for a data-poor regime where the object-space (transpose) Barlow term, weighted by lambd_obj, helps a position-only network: fewer training frames, where accuracy is not saturated.

## Method

Batch 19 (r261029-3931397), local fast loop on Zimmer 2022-11-28: position-only ViSNet (RBF fix on, width 256, elastic off, dropout 0), lambd_obj {0 (feature-space Barlow only), 0.67 (default), 1.0 (object-space only)} x num_frames {30, 100, 300}, seeds 43 and 44, 15 epochs; epoch-14 frame-pair proxy (seed noise about 0.04).

## Results

| frames | lambd_obj 0 | 0.67 | 1.0 |
|---|---|---|---|
| 30 | 0.496 / 0.467 | 0.490 / 0.463 | 0.477 / 0.452 |
| 100 | 0.604 / 0.608 | 0.594 / 0.592 | 0.585 / 0.577 |
| 300 | 0.768 / 0.676 | 0.714 / 0.648 | 0.682 / 0.636 |

(seed 43 / seed 44)

## Conclusion

In position-only networks the object-space term gives no data-efficiency gain: lambd_obj 0 > 0.67 > 1.0 in every cell on both seeds, by 0.01-0.02 at 30 and 100 frames and 0.03-0.09 at 300 frames.

## Caveats

Proxy at epoch 14 only, no full-video accuracy; Zimmer only; position-only networks. Image-only networks behave differently: in batch 20 (r261029-4011812, seed 43) lambd_obj 0.67 reached a proxy of 0.617 against 0.482 at 0, in about 4x fewer epochs, so this note says nothing about the image or fused networks the term was designed for ([[EVD - Loss-term ablation where accuracy is not saturated]]).
