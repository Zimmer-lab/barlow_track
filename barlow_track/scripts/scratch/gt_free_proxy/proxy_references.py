"""GT-free reference values for the shared sweep's consistency objective (optimize_shared_config.py,
objective_metric != accuracy): per recording, the best value of the metric over every scored trial of
its per-recording round (consistency_proxy.py output). Picking the maximum uses no GT; the GT accuracy
of that trial and the round's best GT accuracy are printed only to show what the reference corresponds to.

    python barlow_track/scripts/scratch/gt_free_proxy/proxy_references.py --metric agree_mean \
        label=consistency_all_round_bayes_wk2_samuel_137.jsonl label=...
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='label=consistency_proxy output (JSON lines)')
    ap.add_argument('--metric', default='agree_mean')
    ap.add_argument('--sync', default='cpqr_refine10',
                    help="spectral-sync variant the references must share with the sweep's cells (rows written "
                         "before the field existed used 'independent_refine0')")
    ap.add_argument('--num_seeds', type=int, default=25,
                    help='label-propagation seeds (rows written before the field existed used 25)')
    args = ap.parse_args()
    refs = {}
    for kv in args.files:
        label, path = kv.split('=', 1)
        rows = [json.loads(l) for l in open(path) if l.strip()]
        rows = [r for r in rows if label in r['tag'] and r.get(args.metric) is not None
                and r.get('sync', 'independent_refine0') == args.sync
                and int(r.get('num_seeds', 25)) == args.num_seeds]
        if not rows:
            raise SystemExit(f"{label}: no rows with sync {args.sync} and {args.num_seeds} seeds in {path}")
        best = max(rows, key=lambda r: r[args.metric])
        best_acc = max(r['accuracy'] for r in rows)
        refs[label] = round(float(best[args.metric]), 4)
        print(f"# {label}: n={len(rows)}, best {args.metric} {best[args.metric]:.4f} (trial {best['trial']}, "
              f"accuracy {best['accuracy']:.4f}; round's best accuracy {best_acc:.4f})")
    print(f"reference_{args.metric}: " + json.dumps(refs))


if __name__ == '__main__':
    main()
