"""Write a per-frame-centred copy of a cached eval embedding (no re-embedding).

eval_accuracy --center_per_volume subtracts each frame's descriptor mean before
tracking; this applies the same _normalize_frame to a saved npz so the effect
can be measured on networks already embedded. The output is named as
eval_accuracy expects with --center_per_volume, so it is picked up by
--skip_embed (it stores no cache_key, so it is reused as a pre-key cache).

Usage (wbfm env): python center_cached_embeddings.py IN.npz OUT.npz
"""
import sys

import numpy as np

from barlow_track.scripts.eval_accuracy import _normalize_frame, _save_embeddings


def main(src, dst):
    with np.load(src, allow_pickle=True) as d:
        X = d['X'].astype(np.float64)
        time_to_lin = d['time_to_lin'].item()
        lin_to_t_seg = d['lin_to_t_seg'].item()
        n_frames = int(d['n_frames']) if 'n_frames' in d else len(time_to_lin)
    out = X.copy()
    for t, idx in time_to_lin.items():
        if len(idx):
            out[idx] = _normalize_frame(X[idx], True, False)
    _save_embeddings(dst, X=out.astype(np.float32), time_to_lin=time_to_lin,
                     lin_to_t_seg=lin_to_t_seg, n_frames=n_frames)
    print(f"{src} -> {dst}: {X.shape}, {len(time_to_lin)} frames centred", flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
