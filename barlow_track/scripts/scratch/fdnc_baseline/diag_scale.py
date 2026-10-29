import sys, numpy as np, torch
sys.path.insert(0, '/home/charles/Current_work/repos/barlow_track/barlow_track/scripts/scratch/fdnc_baseline')
import fdnc_track_and_score as F
from types import SimpleNamespace
torch.set_num_threads(4)
model = F.load_fdnc_model()
args = SimpleNamespace(source='project', project=sys.argv[1], gt=sys.argv[1], nwb=None, max_frames=None)
frames, df_gt, match_col, src, info = F.load_frames(args)
gt_ind = df_gt.xs('raw_neuron_ind_in_list', axis=1, level=1).values
rng = np.random.default_rng(0)
ts = sorted(rng.choice(np.arange(len(frames)), 60, replace=False))
def acc(t_tmp, t, orient, scale):
    conv = lambda x: F.to_fdnc_units(x, info['um_xy'], info['um_z'], orient) * scale
    mp = {c: r for c, r, p in F.fdnc_match(model, conv(frames[t_tmp][0]), conv(frames[t][0]))}
    good = tot = 0
    for j in range(gt_ind.shape[1]):
        a, b = gt_ind[t_tmp, j], gt_ind[t, j]
        if np.isnan(a) or np.isnan(b): continue
        tot += 1; good += int(mp.get(int(a), -1) == int(b))
    return good / max(tot, 1)
for orient in F.ORIENTS:
    for scale in [0.5, 0.7, 1.0, 1.4]:
        print(orient, scale, round(np.mean([acc(10, t, orient, scale) for t in ts]), 3), flush=True)
