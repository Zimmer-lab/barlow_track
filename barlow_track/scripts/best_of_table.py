"""Best tracking accuracy per dataset for image-only, fused and position-only networks.

Reads every accuracy record we have (cluster sweep rounds' exp_results.jsonl and the local
scratch jsonl files under ~/.cache/barlow_scratch), keeps comparable ones and writes a
Markdown table, one row per dataset and evaluation condition:

- trained checkpoints only (mode trained, head_init trained), label propagation, the default
  evaluation (no eval-time per-frame centring or L2, scaled SVD, nandist GT matching), full
  videos only (no max_frames, and at least 90% of the dataset's longest scored run);
- category: fusion position_only -> position-only; no position branch (legacy pipeline or
  fusion None) -> image-only; a fused network at descriptor_stage backbone -> image-only
  (its image backbone alone, marked "backbone of fused"); otherwise fused;
- Flavell: NWB ground truth only (the project GT is a tracker output, issue i261028-3), and
  --source nwb records only if they carry nwb_arrays='by_name' (the neuron-order fix for
  issue i261026-1, 79abd66); --source project against the NWB GT is unaffected;
- Zimmer: the updated-format project copies only (the original-format ones are the same
  recordings); Samuel: GT objects only (the project segmentation row was redundant);
- one row per recording: Zimmer 2022-11-28 / 11-23 worm 11 / 12-10, the five Flavell NWBs,
  Leifer, Samuel 137 / 153 / 184-004.

Duplicate tags keep the latest record of a file. Usage (repo root, wbfm env):
  python barlow_track/scripts/best_of_table.py [--out docs/best_of_table.md]
"""
import argparse
import glob
import json
import os
from pathlib import Path

ROUNDS = '/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_revisions/config_files'
LOCAL = os.path.expanduser('~/.cache/barlow_scratch')
CATEGORIES = ('image-only', 'fused', 'position-only')


def _stem(path):
    p = str(path or '')
    if p.endswith('.nwb'):
        return os.path.basename(p)
    return os.path.basename(os.path.dirname(p)) if p.endswith('.yaml') else os.path.basename(p.rstrip('/'))


ZIMMER_RECORDINGS = {'2022_11_28': 'Zimmer 2022-11-28', '2022-11-23_worm11': 'Zimmer 2022-11-23 worm 11',
                     '2022-12-10': 'Zimmer 2022-12-10'}


# Every GT recording we can score, shown even before it has a record (the benchmark driver's
# datasets: zimmer_1123 and zimmer_1210, flavell_2022*, samuel_137, samuel_184 beyond the defaults)
ALL_ROWS = [*ZIMMER_RECORDINGS.values(), 'Flavell 2022-03-31-01 (project segmentation)',
            *(f'Flavell {d} (GT objects)' for d in ('2022-03-31-01', '2022-03-16-03', '2022-03-24-02',
                                                    '2022-03-30-01', '2022-03-30-02')),
            'Leifer NeRVE worm 1 (GT objects)', *(f'Samuel {r} (GT objects)' for r in ('137', '153', '184-004'))]


def _flavell_recording(nwb):
    # images_for_charlie (and its local copy, local_data/flavell) holds 2022-03-31-01 (its label
    # maps are named so); the others sit in forCharlie_2/<recording>/
    parent = os.path.basename(os.path.dirname(str(nwb)))
    return '2022-03-31-01' if parent in ('images_for_charlie', 'flavell') else parent


def dataset_of(rec):
    """(dataset row label, None) or (None, reason it is left out)."""
    lab, source, gt = str(rec.get('lab') or ''), rec.get('source'), str(rec.get('gt') or '')
    if lab.startswith('zimmer'):
        where = str(rec.get('project') or gt)
        if 'updated_format' not in where:
            # Same recordings as the updated-format copies, which every current run uses
            return None, 'zimmer original-format project (redundant with the updated copy)'
        for key, name in ZIMMER_RECORDINGS.items():
            if key in where:
                return name, None
        return None, f'unknown zimmer project {where}'
    if lab == 'leifer':
        return "Leifer NeRVE worm 1 (GT objects)", None
    if lab == 'flavell':
        if source == 'nwb':
            if rec.get('nwb_arrays') != 'by_name':
                return None, 'flavell nwb before the neuron-order fix'
            return f"Flavell {_flavell_recording(gt)} (GT objects)", None
        if gt.endswith('.nwb'):
            return f"Flavell {_flavell_recording(gt)} (project segmentation)", None
        return None, 'flavell project GT (tracker output)'
    if lab == 'samuel':
        if source == 'project':
            return None, 'samuel project segmentation (redundant with GT objects)'
        objects = rec.get('nwb_objects') or 'gt'
        if objects != 'gt':
            return None, f'samuel nwb_objects {objects}'
        return f"Samuel {_stem(gt).replace('.nwb', '') or '153'} (GT objects)", None
    return None, f'unknown lab {lab!r}'


def category_of(rec):
    tc = rec.get('train_config') or {}
    fusion = rec.get('fusion', tc.get('fusion'))
    use_position = tc.get('use_position', fusion is not None)
    if fusion == 'position_only':
        return 'position-only', ''
    if not use_position or fusion is None:
        return 'image-only', ''
    if rec.get('descriptor_stage') == 'backbone':
        return 'image-only', 'backbone of fused'
    return 'fused', ''


def load_records():
    files = sorted(glob.glob(f'{ROUNDS}/*/exp_results*.jsonl')) + \
        sorted(glob.glob(f'{LOCAL}/**/*.jsonl', recursive=True))
    out = []
    for f in files:
        latest = {}
        for i, line in enumerate(open(f)):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'accuracy' not in rec or rec.get('mode') != 'trained':
                continue
            latest[rec.get('tag') or f'{f}:{i}'] = rec
        out += [(f, r) for r in latest.values()]
    return out


def nondefault_eval(rec):
    """Why a record's evaluation is not the default one, or None."""
    if rec.get('cluster', 'labelprop') != 'labelprop':
        return 'not label propagation'
    if rec.get('head_init', 'trained') != 'trained':
        return 'untrained head'
    if rec.get('center_per_volume') or rec.get('l2_per_volume'):
        return 'eval-time centring or L2'
    if rec.get('svd', 'scaled') != 'scaled' or rec.get('gt_matching', 'nandist') != 'nandist':
        return 'whitened SVD or gt_valid matching'
    if rec.get('max_frames'):
        return 'truncated (max_frames)'
    return None


def best_table(records):
    best, skipped, kept = {}, {}, []
    for f, rec in records:
        acc = rec.get('accuracy')
        if acc is None or acc != acc:
            continue
        ds, why = dataset_of(rec)
        why = why or nondefault_eval(rec)
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        kept.append((f, rec, ds))
    # Longest scored run per recording, across its evaluation conditions
    recording = lambda ds: ds.split(' (')[0]
    longest = {}
    for _, rec, ds in kept:
        longest[recording(ds)] = max(longest.get(recording(ds), 0), rec.get('n_frames') or 0)
    for f, rec, ds in kept:
        acc = rec['accuracy']
        if (rec.get('n_frames') or 0) < 0.9 * longest[recording(ds)]:
            skipped['truncated (few frames)'] = skipped.get('truncated (few frames)', 0) + 1
            continue
        cat, note = category_of(rec)
        key = (ds, cat)
        if key not in best or acc > best[key][0]:
            best[key] = (acc, rec.get('tag'), note, f, rec.get('descriptor_stage'))
    return best, skipped


def render(best, skipped, n_records):
    rows = ALL_ROWS + sorted({ds for ds, _ in best} - set(ALL_ROWS))
    lines = ["# Best tracking accuracy per dataset", "",
             "Generated by `barlow_track/scripts/best_of_table.py` from every trained-checkpoint record "
             f"(label propagation) in the cluster rounds and the local scratch results ({n_records} records). "
             "Each cell is the best accuracy, the record's tag and where it came from (cluster round or local).",
             "", "| dataset | " + " | ".join(CATEGORIES) + " |", "|---|" + "---|" * len(CATEGORIES)]
    for ds in rows:
        cells = []
        for cat in CATEGORIES:
            if (ds, cat) not in best:
                cells.append('')
                continue
            acc, tag, note, f, stage = best[(ds, cat)]
            where = 'cluster' if f.startswith(ROUNDS) else 'local'
            extra = f", {note}" if note else ''
            cells.append(f"**{acc:.3f}** `{tag}` ({where}{extra})")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    lines += ["", "Left out as not comparable: " + "; ".join(f"{k} ({v})" for k, v in sorted(skipped.items())) + ".",
              "", "Caveats: GT-object rows (Leifer, Samuel NWB, Flavell NWB source) track oracle detections; project "
              "segmentation rows track every detection. Flavell GT is the Flavell lab's own tracker output. "
              "Local records exist only on the workstation's cache. The preprint networks have not been "
              "re-scored on Zimmer or Leifer yet."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/best_of_table.md')
    args = ap.parse_args()
    records = load_records()
    best, skipped = best_table(records)
    text = render(best, skipped, len(records))
    Path(args.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
