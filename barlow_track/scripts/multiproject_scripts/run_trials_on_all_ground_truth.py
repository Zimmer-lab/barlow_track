"""Evaluate trained barlow trial(s) on all hardcoded ground truth datasets.

Lightweight alternative to the full snakemake analysis pipeline: for each
(trial, dataset) it runs barlow_track/scripts/eval_accuracy.py --mode trained,
which embeds all frames with the checkpoint's own heads, tracks with label
propagation, and appends an accuracy record (tagged per trial) to a JSONL file.

Run on a GPU node (embedding is ~10-50x faster than CPU):
    python run_trials_on_all_ground_truth.py --trial_parent_dir <dir> --trials 0 1 --device cuda

All trials in the folder with a model file on all datasets:
    python run_trials_on_all_ground_truth.py --trial_parent_dir <dir> --all --device cuda

Top-K sweep winners on all datasets (default set):
    python run_trials_on_all_ground_truth.py --trial_parent_dir <sweep_dir> --top_k 3 --device cuda

Same, but one Slurm job per evaluation (keeps the local-GPU flags for --debug preview):
    python run_trials_on_all_ground_truth.py --trial_parent_dir <sweep_dir> --top_k 3 --device cuda --slurm

Smoke test first (50 frames per dataset):
    python run_trials_on_all_ground_truth.py --trial_parent_dir <dir> --trials 0 1 --device cuda --max_frames 50

Split across nodes by passing a subset of datasets, e.g. --labs zimmer_1128 zimmer_1123 zimmer_1210
"""

import argparse
import json
import os
import re
import subprocess
import sys

EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "eval_accuracy.py")

_PAPER_DATA = "/lisc/data/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/paper_data"
_PAPER_OTHER = "/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper"

# Hardcoded ground truth datasets. zimmer_* use project-frame crops from the GT
# project itself; the rest crop around GT xyz straight from the GT NWB.
DATASETS = {
    "zimmer_1128": dict(source="project",
                        project=os.path.join(_PAPER_DATA, "ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format"),
                        gt=os.path.join(_PAPER_DATA, "ZIM2165_Gcamp7b_worm1-2022_11_28_updated_format")),
    "zimmer_1123": dict(source="project",
                        project=os.path.join(_PAPER_DATA, "2022-11-23_worm11_updated_format"),
                        gt=os.path.join(_PAPER_DATA, "2022-11-23_worm11_updated_format")),
    "zimmer_1210": dict(source="project",
                        project=os.path.join(_PAPER_DATA, "ZIM2165_Gcamp7b_worm1-2022-12-10_updated_format"),
                        gt=os.path.join(_PAPER_DATA, "ZIM2165_Gcamp7b_worm1-2022-12-10_updated_format")),
    "flavell": dict(source="nwb",
                    nwb=os.path.join(_PAPER_OTHER, "flavell_data/images_for_charlie/flavell_data.nwb")),
    "leifer": dict(source="nwb",
                   nwb=os.path.join(_PAPER_OTHER, "leifer_data/Leifer_NeRVE_Worm1.nwb")),
    "samuel": dict(source="nwb",
                   nwb=os.path.join(_PAPER_OTHER, "samuel_data/153.nwb")),
}

# eval_accuracy.py addresses projects by lab; nwb-source datasets reuse the
# matching lab entry and override the NWB path.
LAB_FOR_DATASET = {"zimmer_1128": "zimmer", "zimmer_1123": "zimmer", "zimmer_1210": "zimmer",
                   "flavell": "flavell", "leifer": "leifer", "samuel": "samuel"}

# Default evaluation set: one zimmer dataset plus the three external labs.
# The other zimmer datasets remain available via --labs.
DEFAULT_LABS = ["zimmer_1128", "flavell", "leifer", "samuel"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained trials on all ground truth datasets.")
    parser.add_argument("--trial_parent_dir", required=True,
                        help="Folder with trial_N subfolders (each with resnet50.pth)")
    parser.add_argument("--trials", nargs="+", type=int, default=None,
                        help="Which trials to run, e.g. --trials 0 1 (default: all with a model file)")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Run every trial_N with a model file (same as omitting --trials; "
                             "mutually exclusive with --trials)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Take the top-K trials by test_loss (fallback: last val_loss) from the "
                             "sweep in --trial_parent_dir instead of --trials (trained mode only)")
    parser.add_argument("--labs", nargs="+", default=None, choices=list(DATASETS),
                        help="Subset of datasets (default: all)")
    parser.add_argument("--model_fname", default="resnet50.pth")
    parser.add_argument("--device", default="cuda", help="torch device for embedding")
    parser.add_argument("--results_jsonl", default=None,
                        help="Where to append results (default: <trial_parent_dir>/exp_results.jsonl)")
    parser.add_argument("--unified_json", default=None,
                        help="Write this run's records as one JSON array here "
                             "(default: <results_jsonl basename>_unified.json alongside it)")
    parser.add_argument("--emb_dir", default=None,
                        help="Embedding cache dir (default: <trial_parent_dir>/emb_cache)")
    parser.add_argument("--max_frames", type=int, default=None, help="Frame cap per dataset (smoke test)")
    parser.add_argument("--cluster", default="labelprop", choices=["labelprop", "global"])
    parser.add_argument("--mode", default="trained",
                        choices=["trained", "image", "position", "posonly", "attention", "both"],
                        help="trained: evaluate trial checkpoints (--trials/--trial_parent_dir); "
                             "otherwise run the reference untrained baselines once per dataset")
    parser.add_argument("--num_seeds", type=int, default=25)
    parser.add_argument("--descriptor_stage", default="auto",
                        choices=["auto", "backbone", "fused", "contextual", "projected"])
    parser.add_argument("--center_per_volume", action="store_true",
                        help="Subtract the per-frame descriptor mean before tracking "
                             "(re-scores existing checkpoints for mean pollution)")
    parser.add_argument("--l2_per_volume", action="store_true",
                        help="Row-wise L2-normalize descriptors before tracking")
    parser.add_argument("--debug", action="store_true",
                        help="Print commands without running them, and verify the torch --device actually loads")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Max concurrent evaluations (default: one per GPU in --gpus)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run evaluations one at a time (equivalent to --jobs 1)")
    parser.add_argument("--slurm", action="store_true",
                        help="Submit one Slurm job per evaluation instead of running locally "
                             "(--jobs/--gpus/--sequential then only affect the debug preview)")
    parser.add_argument("--slurm_time", default="01:00:00",
                        help="Slurm wall time per evaluation job")
    parser.add_argument("--slurm_cpus", type=int, default=16,
                        help="CPUs per evaluation job (tracking is CPU-bound)")
    parser.add_argument("--slurm_mem", default="32G", help="Memory per evaluation job")
    parser.add_argument("--gpus", default="auto",
                        help="GPUs to round-robin jobs over, e.g. '0,1' (default: all visible via nvidia-smi)")
    parser.add_argument("--fail_fast", dest="fail_fast", action="store_true",
                        help="stop starting new evaluations after the first failure (default)")
    parser.add_argument("--no_fail_fast", dest="fail_fast", action="store_false",
                        help="keep running remaining evaluations even if one fails")
    parser.set_defaults(fail_fast=True)
    parser.add_argument("--rerun_completed", action="store_true",
                        help="re-run (trial, dataset) tags already present in --results_jsonl "
                             "(default is to skip them, so a killed run resumes; reruns "
                             "overwrite the old records instead of appending duplicates)")
    return parser.parse_args()


def check_device(device_str):
    """Smoke-test that torch can actually use the requested device."""
    try:
        import torch
    except ImportError:
        print("WARNING: torch is not importable here; cannot verify --device")
        return False
    try:
        d = torch.device(device_str)
        if d.type == "cuda" and not torch.cuda.is_available():
            print(f"WARNING: device '{device_str}' requested but torch.cuda.is_available() is False")
            return False
        _ = (torch.ones(4, device=d) + 1).sum().item()  # real op on the device
        name = torch.cuda.get_device_name(d) if d.type == "cuda" else "cpu"
        print(f"Device check OK: '{device_str}' ({name})")
        return True
    except Exception as e:
        print(f"Device check FAILED for '{device_str}': {e}")
        return False


def resolve_trials(trial_parent_dir, trials, model_fname):
    if trials is None:
        trials = sorted(int(m.group(1)) for d in os.listdir(trial_parent_dir)
                        if os.path.isdir(os.path.join(trial_parent_dir, d))
                        for m in [re.match(r"trial_(\d+)", d)] if m)
    selected = []
    for trial_num in sorted(trials):
        model_path = os.path.join(trial_parent_dir, f"trial_{trial_num}", model_fname)
        if not os.path.isfile(model_path):
            print(f"trial_{trial_num}: no model file at {model_path}; skipping")
            continue
        selected.append(trial_num)
    return selected


def top_k_trials(trial_parent_dir, k, model_fname):
    """Trial numbers with the lowest test_loss (fallback: last val_loss).

    Only trials with a model file and a usable loss are ranked.
    """
    from barlow_track.utils.utils_ground_truth import discover_trials, extract_val_from_json
    scored = []
    for trial_num in discover_trials(trial_parent_dir):
        trial_dir = os.path.join(trial_parent_dir, f"trial_{trial_num}")
        if not os.path.isfile(os.path.join(trial_dir, model_fname)):
            print(f"trial_{trial_num}: no model file; skipping")
            continue
        val = extract_val_from_json(trial_dir, key="test_loss")
        if val is None:
            val = extract_val_from_json(trial_dir, key="val_loss")
        if val is None or val != val:  # None or NaN
            print(f"trial_{trial_num}: no usable loss; skipping")
            continue
        scored.append((float(val), trial_num))
    scored.sort()
    top = [t for _, t in scored[:k]]
    print(f"Top {k} by loss: {[(t, round(v, 6)) for v, t in scored[:k]]}")
    return top


def resolve_gpus(gpus_arg):
    """Return a list of GPU indices; 'auto' queries nvidia-smi."""
    if gpus_arg != "auto":
        return [g.strip() for g in gpus_arg.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        gpus = [line.strip() for line in out.splitlines() if line.strip()]
        if gpus:
            return gpus
    except Exception as e:
        print(f"nvidia-smi query failed ({e}); trying torch")
    try:
        import torch
        n = torch.cuda.device_count()
        if n:
            return [str(i) for i in range(n)]
    except ImportError:
        pass
    print("WARNING: no GPUs detected; falling back to a single worker")
    return ["0"]


def run_one(cmd, gpu, label):
    """Run one evaluation, optionally pinned to a single visible GPU.

    gpu=None leaves GPU selection to the environment (e.g. Slurm-assigned).
    Must stay module-level and picklable for submitit.
    """
    if gpu is None:
        env = dict(os.environ)
        where = "slurm-assigned GPU"
    else:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        where = f"gpu {gpu}"
    print(f"\n### {label} [{where}]\n{' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, env=env)
    return rc


def run_slurm(tasks, results_jsonl, args):
    """Submit one Slurm job per evaluation; poll until all finish.

    Returns a list of (trial_num, lab, exit_code) failures.
    """
    import time
    from submitit import AutoExecutor
    folder = os.path.dirname(os.path.abspath(results_jsonl))
    executor = AutoExecutor(folder=folder, cluster="slurm")
    executor.update_parameters(
        timeout_min=180,
        slurm_time=args.slurm_time,
        cpus_per_task=args.slurm_cpus,
        slurm_mem=args.slurm_mem,
        slurm_gres="gpu:1",
        slurm_job_name="barlow_benchmark",
    )
    future_to_task = {}
    for lab, trial_num, tag, cmd in tasks:
        fut = executor.submit(run_one, cmd, None, tag)
        future_to_task[fut] = (trial_num, lab, tag)
        print(f"Submitted {tag} (job {fut.job_id})", flush=True)
    failures = []
    pending = dict(future_to_task)
    while pending:
        for fut in list(pending):
            try:
                finished = fut.done()
            except Exception:
                finished = False
            if not finished:
                continue
            trial_num, lab, tag = pending.pop(fut)
            try:
                rc = fut.result()
            except Exception as e:
                print(f"### {tag} raised: {e}", flush=True)
                rc = 1
            print(f"### {tag} finished with exit code {rc}", flush=True)
            if rc != 0:
                failures.append((trial_num, lab, rc))
                if args.fail_fast:
                    print("Fail-fast: cancelling remaining jobs", flush=True)
                    for other in pending:
                        try:
                            other.cancel()
                        except Exception:
                            pass
                    pending.clear()
                    break
        if pending:
            time.sleep(30)
    return failures


def main():
    args = parse_args()
    trial_parent_dir = os.path.abspath(args.trial_parent_dir)
    parent_base = os.path.basename(trial_parent_dir.rstrip("/"))
    results_jsonl = args.results_jsonl or os.path.join(trial_parent_dir, "exp_results.jsonl")
    emb_dir = args.emb_dir or os.path.join(trial_parent_dir, "emb_cache")
    labs = args.labs or DEFAULT_LABS

    if args.top_k is not None:
        if args.mode != "trained":
            raise SystemExit("--top_k only applies to --mode trained")
        if args.trials is not None or args.run_all:
            print("WARNING: --trials/--all ignored because --top_k was given")
        trials = top_k_trials(trial_parent_dir, args.top_k, args.model_fname)
    else:
        if args.run_all and args.trials is not None:
            raise SystemExit("--all and --trials are mutually exclusive; use one or the other")
        trials = resolve_trials(trial_parent_dir, args.trials, args.model_fname)
    if args.mode == "trained" and not trials:
        raise SystemExit("No runnable trials found.")
    trial_str = trials if args.mode == "trained" else "(untrained baselines; no checkpoints)"
    print(f"Mode: {args.mode}; Trials: {trial_str}\nDatasets: {labs}\nResults: {results_jsonl}\n")

    tasks = []  # (lab, trial_num, tag, cmd)
    if args.mode == "trained":
        trial_jobs = [(lab, trial_num, f"{parent_base}_trial{trial_num}_{lab}",
                       os.path.join(trial_parent_dir, f"trial_{trial_num}", args.model_fname))
                      for trial_num in trials for lab in labs]
    else:
        # Untrained reference baselines: one run per dataset, no checkpoint.
        trial_jobs = [(lab, None, f"untrained_{args.mode}_{lab}", None) for lab in labs]
    def _recorded_tags(path):
        """Tags already in the results file ({} if none). Tolerates junk lines."""
        done = set()
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        try:
                            done.add(json.loads(line).get("tag"))
                        except (json.JSONDecodeError, AttributeError):
                            pass
        return done

    if not args.rerun_completed:
        done = _recorded_tags(results_jsonl)
        before = len(trial_jobs)
        trial_jobs = [t for t in trial_jobs if t[2] not in done]
        skipped = before - len(trial_jobs)
        if skipped:
            print(f"Skipping {skipped}/{before} evaluations already recorded "
                  f"(pass --rerun_completed to redo them); {len(trial_jobs)} remaining")
        if not trial_jobs:
            print("Nothing left to run.")
            return
    for lab, trial_num, tag, weights in trial_jobs:
            spec = DATASETS[lab]
            cmd = [sys.executable, "-u", EVAL_SCRIPT, "--lab", LAB_FOR_DATASET[lab],
                   "--source", spec["source"], "--mode", args.mode,
                   "--device", args.device,
                   "--cluster", args.cluster, "--num_seeds", str(args.num_seeds),
                   "--descriptor_stage", args.descriptor_stage,
                   "--tag", tag, "--results_jsonl", results_jsonl, "--emb_dir", emb_dir]
            if weights is not None:
                cmd += ["--weights", weights]
            if spec["source"] == "nwb":
                cmd += ["--nwb", spec["nwb"]]
            else:
                cmd += ["--project", os.path.join(spec["project"], "project_config.yaml")]
            if args.max_frames is not None:
                cmd += ["--max_frames", str(args.max_frames)]
            if args.center_per_volume:
                cmd += ["--center_per_volume"]
            if args.l2_per_volume:
                cmd += ["--l2_per_volume"]
            tasks.append((lab, trial_num, tag, cmd))
    tags = [(lab, trial_num, tag) for lab, trial_num, tag, _ in tasks]

    gpus = resolve_gpus(args.gpus)
    jobs = 1 if args.sequential else (args.jobs or len(gpus))
    print(f"GPUs: {gpus}; max concurrent jobs: {jobs}")

    if args.debug:
        for i, (lab, trial_num, tag, cmd) in enumerate(tasks):
            where = "slurm" if args.slurm else f"gpu {gpus[i % len(gpus)]}"
            print(f"\n### {tag} [{where}]\n{' '.join(cmd)}", flush=True)
        check_device(args.device)
        return

    if args.slurm:
        print(f"Submitting {len(tasks)} evaluations as Slurm jobs "
              f"({args.slurm_cpus} cpus, {args.slurm_mem}, {args.slurm_time} each)", flush=True)
        failures = run_slurm(tasks, results_jsonl, args)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        failures = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            future_to_task = {}
            for i, (lab, trial_num, tag, cmd) in enumerate(tasks):
                gpu = gpus[i % len(gpus)]
                fut = pool.submit(run_one, cmd, gpu, tag)
                future_to_task[fut] = (trial_num, lab, tag)
            for fut in as_completed(list(future_to_task)):
                if fut.cancelled():
                    continue
                trial_num, lab, tag = future_to_task[fut]
                rc = fut.result()
                print(f"### {tag} finished with exit code {rc}", flush=True)
                if rc != 0:
                    failures.append((trial_num, lab, rc))
                    if args.fail_fast:
                        print("Fail-fast: no further evaluations will be started "
                              "(running ones finish)", flush=True)
                        for other in future_to_task:
                            other.cancel()
                        break
    print(f"\n{'=' * 60}\nSummary (from {results_jsonl}):")
    print(f"{'dataset':<14}{'trial':<8}{'accuracy':<10}{'miss':<8}{'mismatch':<10}n_frames")
    try:
        with open(results_jsonl) as f:
            records = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        # Fail-fast (or a cluster-wide kill) can leave zero completed
        # evaluations; report cleanly instead of crashing on the summary.
        print(f"No results file (no evaluation completed): {results_jsonl}")
        records = []
    if records:
        # Overwrite, don't accumulate: reruns append duplicates (workers share
        # one file), so collapse to the latest record per tag in place.
        # Tagless lines are malformed; keep them untouched, never merged.
        latest = {}
        untagged = []
        for rec in records:
            if rec.get("tag") is None:
                untagged.append(rec)
            else:
                latest[rec["tag"]] = rec
        merged = list(latest.values()) + untagged
        if len(merged) != len(records):
            with open(results_jsonl, "w") as f:
                for rec in merged:
                    f.write(json.dumps(rec) + "\n")
            print(f"Deduplicated {results_jsonl}: {len(records)} -> {len(merged)} records")
            records = merged
    run_records = []
    for lab, trial_num, tag in tags:
        matches = [r for r in records if r.get("tag") == tag]
        rec = matches[-1] if matches else None
        if rec is None:
            print(f"{lab:<14}{trial_num:<8}{'MISSING':<10}")
        else:
            run_records.append(rec)
            print(f"{lab:<14}{trial_num:<8}{rec['accuracy']:<10.4f}{rec['misses']:<8d}"
                  f"{rec['mismatches']:<10d}{rec['n_frames']}")
    if failures:
        print(f"\nFAILED: {failures}")
    # Unified output: one JSON array with this run's full records (weights
    # paths + training configs included) for future display/analysis.
    unified_json = args.unified_json or (os.path.splitext(results_jsonl)[0] + "_unified.json")
    with open(unified_json, "w") as f:
        json.dump(run_records, f, indent=2)
    print(f"Unified JSON ({len(run_records)} records) written to {unified_json}")
    if failures or not run_records:
        raise SystemExit(f"Benchmark incomplete: {len(failures)} failures, "
                         f"{len(run_records)}/{len(tags)} records")


if __name__ == "__main__":
    main()
