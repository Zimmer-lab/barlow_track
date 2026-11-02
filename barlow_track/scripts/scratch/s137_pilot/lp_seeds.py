"""Re-track a pilot checkpoint's cached embedding with other labelprop seeds / seed counts."""
import os, sys, yaml
os.environ.setdefault('WBFM_NWB_LAZY_SEGMENTATION', '1')
sys.path.insert(0, '/home/charles/Current_work/repos/barlow_track')
from barlow_track.scripts.eval_accuracy import evaluate_trained_checkpoint
from barlow_track.scripts.optimize_hyperparameters import resolve_objective
ROUND = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files/round_bayes_wk2_samuel_137'
OUT = os.path.expanduser('~/.cache/barlow_scratch/train/s137_pilot')
name = sys.argv[1]
d = os.path.join(OUT, name)
cfg = yaml.safe_load(open(os.path.join(d, 'train_config.yaml')))
_, _, ek = resolve_objective(yaml.safe_load(open(os.path.join(ROUND, 'hyperparameter_search_template.yaml'))))
for spec in sys.argv[2:]:  # "seed:num_seeds"
    s, n = map(int, spec.split(':'))
    kw = {k: v for k, v in ek.items() if k not in ('gt', 'snapshots')}
    kw.update(device='cpu', seed=s, num_seeds=n)
    rec = evaluate_trained_checkpoint(weights=os.path.join(d, 'resnet50.pth'), project=cfg['project_path'],
                                      gt=ek.get('gt') or cfg['project_path'], tag=f's137_pilot_{name}',
                                      results_jsonl=os.path.join(OUT, 'lp_seeds.jsonl'), emb_dir=os.path.join(d, 'emb'),
                                      extra_record={'variant': name, 'lp_seed': s, 'lp_num_seeds': n}, **kw)
    print(f"[{name}] lp_seed {s} num_seeds {n}: accuracy {rec['accuracy']:.4f} ({rec.get('minutes', 0):.0f} min)", flush=True)
