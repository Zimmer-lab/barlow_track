"""Are the Leifer volumes NeRVE left without detections (78, 405, 945, 1516: the 208 extra GT
clicks of the A2 re-export) corrupted, or ordinary images a detector should have handled?

For each of them and its neighbours (t-2..t+2): red intensity stats, the fraction of
saturated / zero voxels, the correlation with the previous volume (motion or corruption),
the red at the GT clicks against background (are the clicks on cells?), a per-plane
profile (missing or shifted planes), and the number of masks in the segmentation.

    python barlow_track/scripts/scratch/leifer_deep_dive/blank_volumes.py
"""
import numpy as np

NWB = ('/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/leifer_reexport/'
       'Leifer_NeRVE_Worm1_reexport_A2.nwb')
BLANK = [78, 405, 945, 1516]


def main():
    from barlow_track.scripts.eval_accuracy import _load_nwb_arrays
    from barlow_track.scripts.scratch.leifer_deep_dive.lazy_nwb import load_nwb_project
    p = load_nwb_project(NWB)
    A = _load_nwb_arrays(p)
    print(f"{'t':>5s} {'masks':>5s} {'clicks':>6s} {'mean':>6s} {'p99':>6s} {'max':>6s} {'zero%':>6s} "
          f"{'corr_prev':>9s} {'click/bg':>8s} {'click>p90%':>10s}  plane means (every 4th)")
    for b in BLANK:
        prev = None
        for t in range(b - 2, b + 3):
            if not 0 <= t < p.num_frames:
                continue
            r = np.asarray(p.red_data[t], dtype=float)
            seg = np.asarray(p.raw_segmentation[t])
            n_masks = len(np.unique(seg)) - 1
            js = np.flatnonzero(np.isfinite(A['gx'][t]))
            gi = np.round(np.stack([A['gz'][t, js], A['gx'][t, js], A['gy'][t, js]], 1)).astype(int)
            gi = np.clip(gi, 0, np.array(r.shape) - 1)
            # max red in a 3x3x1 neighbourhood of each click, against the volume's median
            vals = np.array([r[z, max(x - 1, 0):x + 2, max(y - 1, 0):y + 2].max() for z, x, y in gi])
            bg = np.median(r)
            p90 = np.percentile(r, 90)
            corr = np.corrcoef(r.ravel()[::7], prev.ravel()[::7])[0, 1] if prev is not None else np.nan
            planes = r.mean(axis=(1, 2))
            mark = ' <' if t == b else ''
            print(f"{t:5d} {n_masks:5d} {len(js):6d} {r.mean():6.0f} {np.percentile(r, 99):6.0f} {r.max():6.0f} "
                  f"{100 * (r == 0).mean():6.2f} {corr:9.3f} {np.median(vals) / bg:8.2f} "
                  f"{100 * (vals > p90).mean():10.1f}  {' '.join(f'{v:.0f}' for v in planes[::4])}{mark}")
            prev = r
        print()


if __name__ == '__main__':
    main()
