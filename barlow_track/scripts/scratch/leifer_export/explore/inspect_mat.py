"""Print a compact summary of MATLAB v5 .mat files (read-only).

Usage: python -I inspect_mat.py file.mat [file2.mat ...]
"""
import sys
import numpy as np
import scipy.io as sio


def summ(x, depth=0, maxdepth=3, name=""):
    pad = "  " * depth
    if isinstance(x, np.ndarray) and x.dtype.names:
        print(f"{pad}{name}: struct array shape={x.shape} fields={x.dtype.names}")
        if depth < maxdepth and x.size:
            el = x.flat[0]
            for fn in x.dtype.names:
                summ(el[fn], depth + 1, maxdepth, fn)
    elif isinstance(x, np.ndarray) and x.dtype == object:
        print(f"{pad}{name}: cell shape={x.shape}")
        if depth < maxdepth and x.size:
            summ(x.flat[0], depth + 1, maxdepth, "[0]")
    elif isinstance(x, np.ndarray):
        s = f"{pad}{name}: {x.dtype} shape={x.shape}"
        if x.size and np.issubdtype(x.dtype, np.number):
            xf = x.astype(float)
            fin = xf[np.isfinite(xf)]
            if fin.size:
                s += f" min={fin.min():.4g} max={fin.max():.4g} nan={np.isnan(xf).sum()}"
            s += f" first={x.ravel()[:6]}"
        elif x.dtype.kind == "U":
            s += f" = {x.ravel()[:3]}"
        print(s)
    else:
        print(f"{pad}{name}: {type(x).__name__} {x!r}"[:200])


for fn in sys.argv[1:]:
    print("=" * 10, fn)
    m = sio.loadmat(fn)
    for k, v in m.items():
        if k.startswith("__"):
            continue
        summ(v, 0, 3, k)
