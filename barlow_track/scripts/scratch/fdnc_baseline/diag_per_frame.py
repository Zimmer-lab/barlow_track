import sys, pickle, numpy as np, torch
sys.path.insert(0, '/home/charles/Current_work/repos/barlow_track/barlow_track/scripts/scratch/fdnc_baseline')
import fdnc_track_and_score as F
from fDNC.src.DNC_predict import pre_matt
torch.set_num_threads(4)
model = F.load_fdnc_model()
# 1. fDNC example: template vs test with GT labels
d = '/home/charles/Current_work/repos/fDNC_Neuron_ID/Data/Example/'
temp = pre_matt(d + 'template.data'); test = pre_matt(d + 'test.data')
m = F.fdnc_match(model, temp['pts'], test['pts'])
tl, sl = temp['label'][:, 0], test['label'][:, 0]
ok = [tl[c] == sl[r] for c, r, p in m if sl[r] >= 0 and tl[c] >= 0 ]
print('example: labelled test neurons', np.sum(sl >= 0), 'acc', np.mean(ok), len(ok))
print('example pts range', np.ptp(temp['pts'][:, :3], 0), 'std', temp['pts'][:, :3].std(0))
# 2. zimmer: per-frame accuracy vs time to template using GT identity
from types import SimpleNamespace
args = SimpleNamespace(source='project', project=sys.argv[1], gt=sys.argv[1], nwb=None, max_frames=80)
frames, df_gt, match_col, src, info = F.load_frames(args)
z = F.to_fdnc_units(frames[10][0], info['um_xy'], info['um_z'])
print('zimmer pts range', np.ptp(z, 0), 'std', z.std(0))
gt_ind = df_gt.xs('raw_neuron_ind_in_list', axis=1, level=1).values[:80]
def acc(t_tmp, t, mirror=False, swap=False):
    def conv(x):
        y = F.to_fdnc_units(x, info['um_xy'], info['um_z'], mirror)
        return y[:, [1, 0, 2]] if swap else y
    mm = F.fdnc_match(model, conv(frames[t_tmp][0]), conv(frames[t][0]))
    mp = {c: r for c, r, p in mm}
    good = tot = 0
    for j in range(gt_ind.shape[1]):
        a, b = gt_ind[t_tmp, j], gt_ind[t, j]
        if np.isnan(a) or np.isnan(b): continue
        tot += 1; good += int(mp.get(int(a), -1) == int(b))
    return good / tot
for variant in [dict(), dict(mirror=True), dict(swap=True)]:
    print(variant, 'self', acc(10, 10, **variant), [round(acc(10, t, **variant), 3) for t in [9, 11, 12, 15, 20, 30, 50, 79]])
