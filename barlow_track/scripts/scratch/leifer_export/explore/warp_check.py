"""Read-only: warp NeRVE straightened segmentation (baseImg) back to raw coords via transformx/y/z."""
import numpy as np, scipy.io as sio
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
W = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/leifer_data/Nguyen2017NeuronTracking/worm1/'
for sidx in [100, 1001]:
    ps = sio.loadmat(W + f'outputFiles/CLstraight/pointStats{sidx:05d}.mat')['pointStats'][0, 0]
    bi = ps['baseImg']; tx = ps['transformx'].astype(int); ty = ps['transformy'].astype(int); tz = ps['transformz'].astype(int)
    rp = ps['rawPoints']; sp = ps['straightPoints']
    # MATLAB column-major labeling order: label on transposed array
    labF, n = ndi.label(np.transpose(bi, (2, 1, 0)))
    labF = np.transpose(labF, (2, 1, 0))
    cs = np.array(ndi.center_of_mass(bi, labF, range(1, n + 1)))  # (i,j,k) 0-based
    # straightPoints vs comps centroid (1-based matlab -> +1); which column order?
    for perm in [(0, 1, 2), (1, 0, 2)]:
        dd = np.abs(sp[:, list(perm)] - (cs + 1)).max(1)
        print(sidx, 'straightPoints perm', perm, 'max |diff| vs Fortran-order label centroids', np.median(dd).round(3), dd.max().round(3))
    # warp to raw: value transform* at each labeled straight voxel
    v = labF > 0
    raw = np.c_[tx[v], ty[v], tz[v]].astype(float)
    lab = labF[v]
    cr = np.array([raw[lab == k].mean(0) for k in range(1, n + 1)])
    dd = np.abs(rp - cr)
    print(sidx, 'rawPoints vs mean(transformx,y,z) of label voxels: median abs diff per axis', np.median(dd, 0).round(3), 'max', dd.max(0).round(2))
    # coverage: unique raw voxels hit vs straight voxels
    uniq = np.unique(raw.astype(int), axis=0)
    print(sidx, 'straight vox', v.sum(), 'unique raw vox hit', len(uniq), 'z frames span', tz[v].min(), tz[v].max(), 'Volume field == comp sizes?',
          np.array_equal(np.bincount(lab)[1:], ps['Volume'].ravel()))
