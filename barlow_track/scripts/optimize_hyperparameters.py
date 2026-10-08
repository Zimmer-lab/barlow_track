# Use the Ax library to optimize hyperparameters
# See: https://ax.dev/tutorials/submitit.html
import argparse
import inspect
import json
import logging
import os
import re
import copy
# This script only dispatches jobs; its own BLAS thread pools must stay tiny
# or imports (scipy via ax/botorch) can exhaust threads on login nodes
# ("pthread_create failed ... Resource temporarily unavailable").
# setdefault: an explicit export in the shell still wins (e.g. for workers).
# _DISPATCH_THREAD_VARS tracks the ones WE set, so submissions can strip them
# again -- trial workers must not inherit the dispatcher cap.
_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_DISPATCH_THREAD_VARS = [v for v in _THREAD_VARS if v not in os.environ]
for _v in _DISPATCH_THREAD_VARS:
    os.environ[_v] = "1"
import time
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
import yaml  # We are only using this for reading
from ruamel.yaml import YAML
from submitit import AutoExecutor, LocalJob, DebugJob
from itertools import product
from barlow_track.scripts.train_barlow_clusterer import train_barlow_network
from barlow_track.scripts.eval_accuracy import evaluate_trained_checkpoint
try:
    from barlow_track.utils.barlow import PretrainedArchitectureMismatchError
except ImportError as e:
    raise ImportError(
        "Installed barlow_track package is stale (no PretrainedArchitectureMismatchError); "
        "it shadows your checkout. Reinstall from your checkout, e.g.: "
        "pip install --no-deps -e <path-to-barlow_track-checkout>"
    ) from e
from barlow_track.utils.utils_ground_truth import (check_training_finished, discover_trials,
                                                   extract_val_from_json, record_tag)
from barlow_track.utils.utils_seeding import replicate_seed, search_trial_seed

# Failure value handed to Ax. "More or less infinity" in the objective's own
# direction, i.e. strictly worse than any real score but still finite (BoTorch
# cannot fit NaN/inf). Kept as a named constant because three places must agree
# on it: the trial path, the resumed-trial path, and attach_prior_trial.
FAILURE_MAGNITUDE = 1e6


def objective_failure_value(objective):
    """"More or less infinity" in the objective's own direction.

    Loss is minimized, so a failure is +1e6; accuracy is maximized, so a failure
    is -1e6. Either way it is strictly worse than any real score AND finite,
    because BoTorch cannot fit NaN/inf.
    """
    return FAILURE_MAGNITUDE if objective == 'loss' else -FAILURE_MAGNITUDE


def objective_value(score, objective):
    """Raw objective value for Ax, or the failure value if the score is unusable.

    No sign flipping: the direction lives in the experiment's minimize flag, so
    the objective column stays the readable metric (a loss, or an accuracy).
    """
    try:
        if score is None or not np.isfinite(score):
            return objective_failure_value(objective)
    except TypeError:
        return objective_failure_value(objective)
    return float(score)


def slurm_duration(minutes):
    """Minutes -> a #SBATCH --time string (D-HH:MM:SS, or HH:MM:SS under a day)."""
    minutes = int(minutes)
    days, rem = divmod(minutes, 24 * 60)
    hours, rem = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{rem:02d}:00"
    return f"{hours}:{rem:02d}:00"


def checkpoint_path(trial_dir, model_fname='resnet50.pth'):
    """Final checkpoint of a trial, or None if the trial never produced one."""
    path = os.path.join(trial_dir, model_fname)
    return path if os.path.isfile(path) else None


def trial_number(trial_dir):
    """The N of trial_N. Raises if the folder is not a trial folder."""
    num = trial_number_or_none(trial_dir)
    if num is None:
        raise ValueError(f"{trial_dir!r} is not a trial_N folder")
    return num


def trial_number_or_none(trial_dir):
    """The N of trial_N, or None if the folder is not a trial folder."""
    match = re.search(r"trial_(\d+)", os.path.basename(os.path.normpath(str(trial_dir))))
    return int(match.group(1)) if match else None


def _finite_or_none(value):
    """Plain float, or None for missing/NaN/inf values (JSON has no NaN)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def objective_mean(mean_and_variance):
    """The objective's mean out of whatever shape Ax returned.

    Ax versions disagree: some return ({'result': mean}, {'result': sem}),
    others a bare (mean, sem). The sweep has a single objective, so take the
    first value either way and refuse to guess beyond that.
    """
    mean = mean_and_variance[0] if mean_and_variance else None
    if isinstance(mean, dict):
        mean = next(iter(mean.values()), None)
    return _finite_or_none(mean)


def read_accuracy_records(results_jsonl):
    """tag -> latest record, from a sweep's shared results jsonl.

    Latest wins, mirroring the benchmark driver's dedup: a rerun overwrites a
    tag rather than accumulating duplicates. Junk lines (a kill mid-write) are
    skipped instead of failing the whole resume.
    """
    latest = {}
    if not os.path.isfile(results_jsonl):
        return latest
    with open(results_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tag = rec.get('tag') if isinstance(rec, dict) else None
            if tag is not None:
                latest[tag] = rec
    return latest


def create_sweep_experiment(ax_client, parameters, accuracy_objective):
    """Create the Ax experiment with the objective's direction attached.

    Loss is minimized, accuracy is maximized. The direction lives here and
    nowhere else -- scores are reported unnegated, so the objective column in
    the sweep log stays the readable metric (a loss, or an accuracy in [0, 1]).
    """
    ax_client.create_experiment(
        name="my_experiment",
        # Deep-copy: create_experiment() mutates these dicts in place (e.g.
        # single-value choices become fixed parameters), which would corrupt
        # the direct/one-at-a-time sweep logic below that reads the originals.
        parameters=copy.deepcopy(parameters),
        objectives={"result": ObjectiveProperties(minimize=not accuracy_objective)},
    )


def resolve_objective(hyperparameter_args):
    """Read the template's objective settings (see the template header).

    Returns (objective, dataset_label, eval_kwargs). 'loss' keeps the historical
    behaviour exactly; 'accuracy' scores each trial by full-video tracking on
    the trial's own training project. Dataset labels and every knob of the eval
    are template keys, but the PROJECT is never one of them: it is always the
    trial's own project_path.
    """
    objective = (hyperparameter_args.get('objective') or 'loss').strip().lower()
    if objective not in ('loss', 'accuracy'):
        raise ValueError(f"Unknown objective {objective!r}; use 'loss' or 'accuracy'")
    accuracy_objective = objective == 'accuracy'
    # The label recorded for this cell. Set it to the benchmark dataset name
    # whose project IS this training project (e.g. zimmer_1128) so a later
    # cross-dataset benchmark recognizes the cell and skips re-embedding it.
    dataset_label = hyperparameter_args.get('objective_dataset') or None
    if accuracy_objective and dataset_label is None:
        dataset_label = 'train'
        logging.warning(
            "objective_dataset is not set in the template: recording these evaluations under "
            "label 'train'. That is not a benchmark dataset name, so the post-sweep benchmark "
            "will NOT skip this cell (it will re-embed). Set objective_dataset to the benchmark "
            "dataset whose project equals train_config.yaml project_path to make it skip.")
    eval_kwargs = dict(
        lab=dataset_label,
        source=hyperparameter_args.get('objective_source', 'project'),
        cluster=hyperparameter_args.get('objective_cluster', 'labelprop'),
        num_seeds=int(hyperparameter_args.get('objective_num_seeds', 25)),
        seed=int(hyperparameter_args.get('objective_seed', 0)),
        descriptor_stage=hyperparameter_args.get('objective_descriptor_stage', 'auto'),
        max_frames=hyperparameter_args.get('objective_max_frames'),
        center_per_volume=bool(hyperparameter_args.get('objective_center_per_volume', False)),
        l2_per_volume=bool(hyperparameter_args.get('objective_l2_per_volume', False)),
        # Embedding runs on the job's GPU; label propagation runs on the CPU.
        # Full-video graphs (100k-200k nodes) allocate dense (N x classes)
        # matrices per propagation step, which OOMs a 12 GB GPU next to the
        # trained model -- and that would waste a whole trial.
        track_device=hyperparameter_args.get('objective_track_device', 'cpu'),
    )
    if eval_kwargs['max_frames'] is not None:
        eval_kwargs['max_frames'] = int(eval_kwargs['max_frames'])
        logging.warning(f"objective_max_frames={eval_kwargs['max_frames']}: the objective is "
                        f"measured on that many frames only. Fine for smoke tests, NOT comparable "
                        f"to a full-video evaluation.")
    return objective, dataset_label, eval_kwargs


def measure_tracking_accuracy(args, test_losses, parent_folder, dataset_label, objective,
                              results_jsonl, emb_dir, eval_kwargs):
    """Full-video tracking accuracy of one trial's checkpoint, on its own project.

    This is the whole `objective: accuracy` body, at module level so it can be
    exercised without a GPU or a dataset. `args` is the trial's training args
    namespace; only project_dir, project_path and the losses are read.
    """
    ckpt = checkpoint_path(args.project_dir)
    if ckpt is None:
        # train_barlow_network deliberately saves no checkpoint when a trial
        # crashes, so this is a failed trial: no accuracy, and no point
        # spending an hour tracking a checkpoint that does not exist.
        raise FileNotFoundError(
            f"No checkpoint in {args.project_dir}; refusing to measure tracking accuracy "
            f"for a trial that did not finish training")
    tag = record_tag(parent_folder, trial_number(args.project_dir), dataset_label)
    # Keep the losses as per-trial diagnostics next to the accuracy: they
    # remain in log/stats.json, and go into the record so one jsonl answers
    # "which checkpoint both trains well and tracks well".
    extra = dict(objective=objective, trial_dir=os.path.abspath(args.project_dir))
    if isinstance(test_losses, dict):
        extra['test_loss'] = _finite_or_none(test_losses.get('test_loss'))
        extra['test_loss_original'] = _finite_or_none(test_losses.get('test_loss_original'))
        extra['test_loss_transpose'] = _finite_or_none(test_losses.get('test_loss_transpose'))
    extra['val_loss'] = _finite_or_none(extract_val_from_json(args.project_dir, key='val_loss'))
    # The trained model is a local of train_barlow_network and is gone by now;
    # release its GPU memory before the eval loads the checkpoint back.
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception as e:
        logging.warning(f"Could not release GPU memory before the objective eval: {e}")
    print(f"[{os.path.basename(str(args.project_dir))}] scoring tracking accuracy of {ckpt} "
          f"on its own training project (this takes tens of minutes)", flush=True)
    record = evaluate_trained_checkpoint(
        weights=ckpt,
        # Same project as training, for both crops and ground truth: the trial
        # must be scored on the data it was trained on and on nothing else.
        project=args.project_path, gt=args.project_path,
        tag=tag, results_jsonl=results_jsonl, emb_dir=emb_dir,
        extra_record=extra, **eval_kwargs)
    print(f"[{os.path.basename(str(args.project_dir))}] tracking accuracy "
          f"{record['accuracy']:.4f} ({record['n_frames']} frames, {record['total']} GT "
          f"detections, {record['minutes']:.1f} min); losses kept as diagnostics", flush=True)
    return record['accuracy']


def _to_yaml_safe(obj):
    """Recursively convert numpy types (and tuples) to plain python types for ruamel.yaml."""
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_yaml_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _ax_params(ax_client, full_params):
    # keep only parameters Ax knows about
    return {k: v for k, v in full_params.items() if k in ax_client.experiment.search_space.parameters}


def _attach_trial(ax_client, full_params):
    """Register a trial Ax knows nothing about yet, and stash its full config.

    Used for both trials that already have a score (completed earlier) and
    trials that only need their accuracy evaluation finished (their training
    survived a kill, their checkpoint is on disk). run_metadata keeps the whole
    train_config, so the trial shows what actually ran.

    Ax 0.3 changed two things this has to absorb: attach_trial returns
    (parameterization, index) instead of a bare index, and run_metadata became
    a read-only property fed by a `run_metadata=` kwarg (older versions needed
    an attribute assignment). Both shapes are handled here so the call sites
    only see an integer index.
    """
    ax_params = _ax_params(ax_client, full_params)
    supports_metadata = 'run_metadata' in inspect.signature(ax_client.attach_trial).parameters
    attached = ax_client.attach_trial(
        parameters=ax_params, run_metadata=dict(full_params) if supports_metadata else None)
    trial_index = attached[1] if isinstance(attached, tuple) else attached
    if not supports_metadata:
        ax_client.experiment.trials[trial_index].update_run_metadata(dict(full_params))
    return trial_index


def attach_prior_trial_to_ax_client(ax_client, full_params, result):
    # BoTorch GP cannot fit NaN/inf/None; fail loudly here so callers can skip
    mean = result[0] if isinstance(result, tuple) else result
    try:
        is_finite = bool(np.isfinite(mean))
    except TypeError:
        is_finite = False
    if not is_finite:
        raise ValueError(f"Non-finite prior trial result {mean!r}; skipping")
    trial_index = _attach_trial(ax_client, full_params)
    if not isinstance(result, tuple):
        result = (result, 0.0)  # Fake error bars
    ax_client.complete_trial(trial_index, raw_data=result)
    return trial_index



def optimize_hyperparameters(hyperparameter_path, run_locally=False, num_parallel_jobs=None, 
                             direct_parameter_sweep=False, one_at_a_time_sweep=False, repetitions=1, 
                             job_name=None, DEBUG=False, total_budget=None):
    """
    Parameters
    ---------------------
    hyperparameter_path - Path to the yaml file, usually named hyperparameter_search_template.yaml
    run_locally - Instead of the cluster, via slurm
    num_parallel_jobs
    direct_parameter_sweep - Generate runs as all combinations of parameters instead of optimizing (e.g. for grid search)
    one_at_a_time_sweep - Generate runs by using only one of the hyperparameter changes at a time; the rest are defaults (e.g. for ablations)
    job_name - SLURM job name
    total_budget - Number of trials in a Bayesian sweep (default 30, or 5 in DEBUG).
                   Grid/one-at-a-time sweeps ignore it: their size is the grid's size.

    """
    if DEBUG:
        run_locally = True
    if hyperparameter_path is None:
        raise ValueError("Please provide a hyperparameter template path")
    if os.path.isdir(hyperparameter_path):
        # Convenience: point at a sweep folder and pick up its template.
        hyperparameter_path = os.path.join(hyperparameter_path, 'hyperparameter_search_template.yaml')
    with open(hyperparameter_path, 'r') as f:
        hyperparameter_args = yaml.safe_load(f)

    # Full training script

    # Set up baseline parameters; load from template yaml file
    fname = hyperparameter_args['baseline_params_path']
    if fname is None:
        # Then assume there is a template in the same folder as the hyperparameter_path
        fname = os.path.join(os.path.dirname(hyperparameter_path), 'train_config.yaml')
    logging.info(f"Loading baseline parameters from {fname}")
    with open(fname, 'r') as f:
        baseline_params = yaml.safe_load(f)
    if DEBUG:
        experiment_parent_folder = '/lisc/data/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/hyperparameter_search_debug'
        baseline_params['wandb_name'] = 'barlow-hyperparameter-search-debug'
        baseline_params['num_frames'] = 20
        baseline_params['epochs'] = 2
        baseline_params['print_freq'] = 10
    else:
        experiment_parent_folder = Path(hyperparameter_path).parent
        if run_locally and baseline_params['wandb_name'] is None:
            baseline_params['wandb_name'] = 'barlow-hyperparameter-search-local'

    # Seed policy for the search (see barlow_track/utils/utils_seeding.py): the
    # baseline train_config.yaml carries one `seed`, and each trial gets a seed
    # derived from it (see the submission loop). The derived value is written
    # into the trial's own train_config.yaml, so any single trial reproduces.
    base_seed = int(baseline_params.get('seed', 43))

    # ---- Objective -------------------------------------------------------
    # 'loss' (default, unchanged): Ax minimizes the trial's test loss.
    # 'accuracy': Ax MAXIMIZES full-video tracking accuracy, measured on the
    # trial's own training project inside the same Slurm job that trained it.
    # The dataset is never a choice here -- it is always args.project_path, the
    # exact project the trial trained on, so train and objective cannot drift.
    objective, dataset_label, objective_kwargs = resolve_objective(hyperparameter_args)
    accuracy_objective = objective == 'accuracy'
    if accuracy_objective and not baseline_params.get('project_path'):
        raise ValueError("objective: accuracy needs the trial's project_path: the objective is "
                         "tracking accuracy on the trial's own training project "
                         "(train_config.yaml project_path is empty)")
    # Records/embeddings land where the standard benchmark runner looks for
    # them, with the tag scheme it deduplicates on (utils_ground_truth.record_tag).
    results_jsonl = os.path.join(str(experiment_parent_folder), 'exp_results.jsonl')
    emb_dir = os.path.join(str(experiment_parent_folder), 'emb_cache')

    def evaluate(parameters, train=True):
        """Train one trial and score it. Runs inside the trial's own Slurm job.

        With objective: accuracy the score comes from full-video tracking on
        args.project_path -- the same project the trial trained on -- using the
        checkpoint this job just wrote. train=False re-runs only that scoring
        step (used when a kill lost the evaluation but not the training).
        """
        # Add the baseline parameters
        args = SimpleNamespace(**parameters)
        try:
            test_losses = train_barlow_network(args) if train else None
            if accuracy_objective:
                score = measure_tracking_accuracy(
                    args, test_losses, experiment_parent_folder, dataset_label, objective,
                    results_jsonl, emb_dir, objective_kwargs)
            else:
                score = test_losses['test_loss'] if isinstance(test_losses, dict) else None
        except PretrainedArchitectureMismatchError:
            # Systematic config error affecting every trial; fail fast instead of
            # scoring the failure value and letting Ax optimize noise.
            raise
        except Exception as e:
            logging.exception(f"Encountered error with trial; quitting gracefully: {e}")
            score = None
        return {"result": objective_value(score, objective)}

    # Set up the Ax client
    ax_client = AxClient(enforce_sequential_optimization=DEBUG)
    # Read parameters from yaml file
    parameters = list(hyperparameter_args['hyperparameters'])
    for param in parameters:
        # Silence Ax UserWarning: `is_ordered` defaulting for ChoiceParameter.
        # Explicitly preserve Ax's default (True for int) so existing searches are unaffected.
        # NOTE: do not pass `sort_values` here; this Ax version's parameter_from_json
        # rejects it (ValueError: Unexpected keys). Template values are pre-sorted,
        # so the default is fine.
        if param.get('type') == 'choice':
            if 'sort_values' in param:
                # Rejected by this Ax version (parameter_from_json); drop it.
                logging.warning("Ignoring unsupported 'sort_values' for parameter "
                                f"{param.get('name')!r}; remove it from the yaml")
                param.pop('sort_values')
            if param.get('value_type') == 'int':
                param.setdefault('is_ordered', True)
            else:
                param.setdefault('is_ordered', False)
    create_sweep_experiment(ax_client, parameters, accuracy_objective)

    # See if there are any previously run trials, and load them
    prior_trials = discover_trials(experiment_parent_folder)
    # Trials that finished training but have no accuracy record yet: their
    # checkpoint survives, so only the tracking eval has to run again (the
    # embedding cache in emb_cache makes that cheap). Keyed by Ax trial index.
    pending_evals = {}
    if len(prior_trials) > 0:
        logging.info(f"Discovered {len(prior_trials)} prior trials, loading...")
        accuracy_by_tag = read_accuracy_records(results_jsonl) if accuracy_objective else {}
        for trial_num in prior_trials:
            trial_name = f"trial_{trial_num}"
            trial_path = os.path.join(experiment_parent_folder, trial_name)
            network_config_path = os.path.join(trial_path, "train_config.yaml")

            try:
                with open(network_config_path, "r") as f:
                    config = yaml.safe_load(f)

                if not check_training_finished(trial_path, int(config['epochs']) - 1):
                    print(f"Prior trial {trial_name}: training was not finished; skipping")
                    continue
                if accuracy_objective:
                    tag = record_tag(experiment_parent_folder, trial_num, dataset_label)
                    record = accuracy_by_tag.get(tag)
                    if record is not None:
                        try:
                            attach_prior_trial_to_ax_client(ax_client, config, record['accuracy'])
                            continue
                        except (ValueError, KeyError) as e:
                            print(f"Prior trial {trial_name}: unusable accuracy record; skipping ({e})",
                                  flush=True)
                            continue
                    if checkpoint_path(trial_path) is None:
                        print(f"Prior trial {trial_name}: trained but no checkpoint, and no "
                              f"accuracy record; skipping", flush=True)
                        continue
                    # Trained, scored nothing yet: attach the trial and re-run
                    # only the eval (its embeddings are already cached).
                    pending_evals[_attach_trial(ax_client, config)] = config
                    print(f"Prior trial {trial_name}: training finished but no {tag} record; "
                          f"will re-run only the tracking accuracy eval", flush=True)
                    continue
                loss = extract_val_from_json(trial_path, key="test_loss")
                if loss is None:
                    print(f"Prior trial {trial_name}: no test_loss found; skipping")
                    continue
                try:
                    attach_prior_trial_to_ax_client(ax_client, config, loss)
                except ValueError as e:
                    print(f"Prior trial {trial_name}: skipping ({e})")
                    continue

            except FileNotFoundError:
                print(f"{trial_name}: train_config.yaml not found.")
                continue

    # Set up SubmitIt
    # Log folder and cluster. Specify cluster='local' or cluster='debug' to run the jobs locally during development.
    # When we're are ready for deployment, switch to cluster='slurm'
    if run_locally:
        executor = AutoExecutor(folder="/tmp/submitit_runs", cluster='debug')
    else:
        # Can't use /tmp/submitit_runs because the cluster can't access it
        # https://github.com/facebookincubator/submitit/blob/main/docs/tips.md
        executor = AutoExecutor(folder=experiment_parent_folder, cluster='slurm')
        logging.info(f"Running experiments in folder: {experiment_parent_folder}")

    # About 100 epochs per day
    num_days = int(baseline_params['epochs'] / 100) + 1
    train_budget_min = 65 * 12 * num_days
    # A trial now also has to track the whole video, which costs tens of minutes
    # to hours of mostly-CPU work AFTER training. Size the job for train + eval
    # and keep headroom, or long trials get killed mid-eval (and lose the
    # tracking work, though the checkpoint and embedding cache survive).
    objective_minutes = int(hyperparameter_args.get('objective_minutes') or 0) if accuracy_objective else 0
    if accuracy_objective and not objective_minutes:
        objective_minutes = 240
        logging.warning("objective_minutes not set in the template; assuming 240 min for the "
                        "full-video tracking eval. Measure one trial and set it explicitly "
                        "(leifer-scale videos need hours, not minutes).")
    eval_budget_min = objective_minutes + 60 if accuracy_objective else 0
    executor.update_parameters(timeout_min=train_budget_min + eval_budget_min)
    if not run_locally:
        if accuracy_objective:
            executor.update_parameters(slurm_time=slurm_duration(train_budget_min + eval_budget_min))
        else:
            # Unchanged default sizing for the loss objective.
            executor.update_parameters(slurm_time=f"{num_days}-12:00:00")
        executor.update_parameters(cpus_per_task=8)
        executor.update_parameters(slurm_mem="128G")
        executor.update_parameters(slurm_job_name=job_name if job_name is not None else "barlow_hyperparameter_search")
        executor.update_parameters(slurm_gres="gpu:1")
        executor.update_parameters(slurm_constraint="l40s|a30|t4|l4")
        executor.update_parameters(slurm_additional_parameters={"no-requeue": True})  # bash equivalent (no-arg flag): #SBATCH --no-requeue

    if direct_parameter_sweep:
        # Manually define all the trials as all combinations
        all_param_lists = []
        for param in parameters:
            if param['type'] == 'choice':
                assert 'values' in param, "For direct parameter sweep, the parameter must have a list of values"
                # List of lists, which will be combined into a grid
                all_param_lists.append(param['values'])
            else:
                raise ValueError(f"For direct parameter sweep, all parameters must be of type 'choice'; got {param['type']} for parameter {param['name']}")

        # Make a grid of all combinations as a dict of parameter name to value
        all_combinations = list(product(*all_param_lists))
        all_combinations = [{parameters[i]['name']: v for i, v in enumerate(comb)} for comb in all_combinations]
        print(f"Running a direct parameter sweep with {len(all_combinations)} combinations")
    elif one_at_a_time_sweep:
        # Define all trials as a sweep of one parameter at a time
        all_combinations = []
        for i, param in enumerate(parameters):
            if param['type'] == 'choice': 
                assert 'values' in param, "For one-at-a-time parameter sweep, the parameter must have a list of values or a single value"
                for v in param['values']:
                    all_combinations.append({param['name']: v})
            elif param['type'] == 'fixed':  # fixed is just a choice with one value
                # If it has been auto-converted to fixed, it won't have 'values', but just 'value'
                assert 'value' in param, "For one-at-a-time parameter sweep, the parameter must have a list of values or a single value"
                all_combinations.append({param['name']: param['value']})
            else:
                raise ValueError(f"For one-at-a-time parameter sweep, all parameters must be of type 'choice'; got {param['type']} for parameter {param['name']}")
        print(f"Running a one-at-a-time parameter sweep with {len(all_combinations)} combinations")
    else:
        # Ax owns the proposal logic and the budget; --total_budget only caps
        # how many trials a single invocation is allowed to run.
        total_budget = (5 if DEBUG else 30) if total_budget is None else int(total_budget)
        if total_budget <= 0:
            raise ValueError(f"total_budget must be positive, got {total_budget}")

    if direct_parameter_sweep or one_at_a_time_sweep:
        # Directly duplicate planned jobs. Remember how many configs there are
        # before duplicating: job i is replicate i // n_unique_configs of config
        # i % n_unique_configs, and the replicate index drives the seed policy.
        n_unique_configs = len(all_combinations)
        all_combinations = all_combinations * repetitions
        total_budget = len(all_combinations)
        # Audit trail: the exact grid being run (names AND values). Miswired
        # name/value pairings here have silently mistargeted sweeps before.
        print(f"Direct sweep plan ({total_budget} jobs):", flush=True)
        for _i, _c in enumerate(all_combinations):
            print(f"  job {_i}: " + ", ".join(f"{k}={v!r}" for k, v in _c.items()), flush=True)

    if num_parallel_jobs is None:
        num_parallel_jobs = 1 if (DEBUG or run_locally) else 10
    else:
        num_parallel_jobs = int(num_parallel_jobs)

    if accuracy_objective:
        print(f"Objective: {objective} (Ax MAXIMIZES it) on each trial's own training project "
              f"({baseline_params['project_path']}), label '{dataset_label}'.", flush=True)
        print(f"Each trial = training + a full-video tracking eval of {objective_minutes} min budget; "
              f"results in {results_jsonl}, embeddings cached in {emb_dir}.", flush=True)
        print(f"Tracking is CPU-bound: {num_parallel_jobs} parallel trials means that many "
              f"CPU-saturating tracking stages at once. Lower --num_parallel_jobs (and/or "
              f"cpus_per_task) if the node is oversubscribed.", flush=True)

    jobs = []
    submitted_jobs = 0
    trial_offset = 0
    start_time = time.time()

    # Run until all the jobs have finished and our budget is used up.
    while submitted_jobs < total_budget or jobs or pending_evals:
        for job, trial_index in jobs[:]:
            # Poll if any jobs completed
            # Local and debug jobs don't run until .result() is called.
            if job.done() or type(job) in [LocalJob, DebugJob]:
                # The log file isn't being produced, so print the stdout instead
                if type(job) in [LocalJob, DebugJob]:
                    print(f"Running trial {trial_index} inline ({type(job).__name__})...", flush=True)
                try:
                    result = job.result()
                    ax_client.complete_trial(trial_index=trial_index, raw_data=result)
                except ValueError as e:
                    if direct_parameter_sweep or one_at_a_time_sweep:
                        # We are manually managing the trials, so this is expected
                        print(f"Encountered error in finishing trial {trial_index}, this is expected but may be fixable; {e}")
                    else:
                        raise e
                except RuntimeError as e:
                    print(f"Encountered Error, trial {trial_index} may need to be rerun: {e}")

                jobs.remove((job, trial_index))
                # Display the current and completed trials
                print(exp_to_df(ax_client.experiment))
                
        # Schedule new jobs if there is availablity
        # Resumed evals go first: they are cheap (checkpoint + cached
        # embeddings already exist) and a fresh sweep must not start scoring
        # new trials while an older trial's score is still missing.
        n_pending = min(len(pending_evals), max(0, num_parallel_jobs - len(jobs)))
        free_slots = max(0, num_parallel_jobs - len(jobs)) - n_pending
        trial_seeds = {}
        if direct_parameter_sweep or one_at_a_time_sweep:
            # Get a new trial manually, without using the AxClient's internal logic (it can't do a grid search)
            # Use the submitted_jobs index as the start point of the next batch of trials

            trial_index_to_param = {}
            for i in range(submitted_jobs, submitted_jobs + free_slots):
                if i >= total_budget:
                    break
                trial = ax_client.experiment.new_trial()
                parameters = all_combinations[i]
                trial_index_to_param[trial.index] = parameters
                # Grid/ablation sweeps: every config shares the base seed, so the
                # differences between configs stay attributable to the
                # hyperparameter. Only a *replicate* of a config (the
                # --repetitions duplicates) gets a fresh seed; before this, N
                # repetitions were literally the same job, which made Ax see
                # zero variance where there should be measurement noise.
                trial_seeds[trial.index] = replicate_seed(base_seed, i // n_unique_configs)
        else:
            trial_index_to_param, _ = ax_client.get_next_trials(
                max_trials=min(free_slots, total_budget - submitted_jobs))
            # Bayesian search: one distinct seed per trial. A seed fixed across
            # trials makes each observation look noise-free, so the GP
            # over-trusts a config that drew a lucky init/volume selection.
            for trial_index in trial_index_to_param:
                trial_seeds[trial_index] = search_trial_seed(base_seed, trial_index)

        # Eval-only jobs first; their trial folders and configs already exist,
        # so they are submitted with the saved config verbatim and train=False.
        submissions = [(trial_index, params, True)
                       for trial_index, params in list(pending_evals.items())[:n_pending]]
        for trial_index, _, _ in submissions:
            del pending_evals[trial_index]
        submissions += [(trial_index, params, False)
                        for trial_index, params in trial_index_to_param.items()]

        for trial_index, parameters, eval_only in submissions:
            if not eval_only:
                # Make a new folder in the parent folder
                # Find a unique folder name by incrementing trial_offset if needed
                while True:
                    this_folder = os.path.join(experiment_parent_folder, f"trial_{trial_index + trial_offset}")
                    if not os.path.exists(this_folder):
                        break
                    logging.warning(f"Found folder {this_folder}; assuming these are old trials with the same settings, and using an index offset")
                    trial_offset += 1
                
                logging.info(f"Making parameter files for trial {trial_index} in folder {this_folder} with index offset {trial_offset}")
                os.makedirs(this_folder, exist_ok=False)
                os.makedirs(os.path.join(this_folder, 'log'), exist_ok=False)
                os.makedirs(os.path.join(this_folder, 'checkpoints'), exist_ok=False)
                parameters['project_dir'] = this_folder
                # Add the baseline parameters, and save in this folder. The seed is
                # set BEFORE the dump: the saved train_config.yaml is the config
                # that actually ran, so a single trial can be re-run standalone with
                # train_barlow_clusterer.py -p <trial>/train_config.yaml. A sweep
                # that samples `seed` itself wins over the policy.
                sampled = parameters
                parameters = {**baseline_params, **sampled}
                if 'seed' not in sampled:
                    parameters['seed'] = trial_seeds.get(trial_index, base_seed)
                YAML().dump(parameters, open(os.path.join(this_folder, 'train_config.yaml'), 'w'))
            # Actually submit. Strip the dispatcher-only thread caps first: slurm
            # jobs capture this process's environment at submission, and trial
            # workers must choose their own threading (or the user's export).
            for _v in _DISPATCH_THREAD_VARS:
                os.environ.pop(_v, None)
            job = executor.submit(evaluate, parameters, not eval_only)
            for _v in _DISPATCH_THREAD_VARS:
                os.environ[_v] = "1"
            if not eval_only:
                submitted_jobs += 1
            jobs.append((job, trial_index))
            what = "eval-only job" if eval_only else f"{submitted_jobs}/{total_budget} submitted"
            print(f"Submitted trial {trial_index} ({type(job).__name__}): {what}", flush=True)
            time.sleep(1)

        # Report status BEFORE sleeping, so the log never looks stalled.
        # Local/debug jobs run inline, so poll fast; slurm jobs are slow.
        print(f"Time={time.time()-start_time}. Checking status of {len(jobs)} jobs; {submitted_jobs}/{total_budget} submitted",
              flush=True)
        time.sleep(10 if (DEBUG or run_locally) else 10*60)

    out = ax_client.get_best_parameters()
    if out is None:
        # No trial produced usable data (all failed or were cancelled).
        # The per-trial logs have the real errors; do not write a bogus file.
        print("No completed trials with data; cannot determine best parameters. "
              "Check the trial log files for errors.", flush=True)
        return
    if len(out) == 4:
        best_parameters, mean_and_variance, best_trial_index, best_trial_name = out
    elif len(out) == 2:
        # older versions of Ax return only two values
        best_parameters, mean_and_variance = out
        best_trial_index, best_trial_name = None, None
    else:
        best_parameters, mean_and_variance, best_trial_index, best_trial_name = None, None, None, None
        logging.warning(f"Could not unpack best parameters from AxClient.get_best_parameters(); got {out}")
    
    print(f'Best set of parameters: {best_parameters}')
    print(f'Mean objective value: {mean_and_variance}')
    best_accuracy = objective_mean(mean_and_variance) if accuracy_objective else None
    if accuracy_objective:
        print(f'Best tracking accuracy: {best_accuracy} (objective: maximize)')
    # The covariance is only meaningful when multiple objectives are present.
    # render(ax_client.get_contour_plot())

    # Copy the best parameters and index to a file
    best_params_path = os.path.join(experiment_parent_folder, 'best_parameters.yaml')
    with open(best_params_path, 'w') as f:
        best_parameters = _to_yaml_safe(best_parameters or {})
        best_parameters['best_trial_index'] = _to_yaml_safe(best_trial_index)
        best_parameters['best_trial_name'] = _to_yaml_safe(best_trial_name)
        best_parameters['mean_and_variance'] = _to_yaml_safe(mean_and_variance)
        best_parameters['objective'] = objective
        if accuracy_objective:
            # The objective column is a maximization, so the mean Ax reports IS
            # the accuracy; record it under its own name too.
            best_parameters['best_accuracy'] = best_accuracy
        YAML().dump(best_parameters, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameter_template_path', '-p', default=None)
    parser.add_argument('--run_locally', action='store_true')
    parser.add_argument('--num_parallel_jobs', default=None)
    parser.add_argument('--total_budget', default=None,
                        help='number of trials to run in a Bayesian (Ax-driven) sweep '
                             '(default 30; ignored for direct/one-at-a-time sweeps, whose size '
                             'is the size of the grid). Lower it for a quick end-to-end check.')
    parser.add_argument('--direct_parameter_sweep', action='store_true')
    parser.add_argument('--one_at_a_time_sweep', action='store_true')
    parser.add_argument('--repetitions', default=1)
    parser.add_argument('--job_name', default=None)
    parser.add_argument('--DEBUG', action='store_true')

    args = parser.parse_args()
    hyperparameter_template_path = args.hyperparameter_template_path
    run_locally = args.run_locally
    num_parallel_jobs = args.num_parallel_jobs
    direct_parameter_sweep = args.direct_parameter_sweep
    one_at_a_time_sweep = args.one_at_a_time_sweep
    job_name = args.job_name
    repetitions = int(args.repetitions)
    DEBUG = args.DEBUG
    total_budget = int(args.total_budget) if args.total_budget is not None else None

    optimize_hyperparameters(hyperparameter_template_path, run_locally, num_parallel_jobs, 
         direct_parameter_sweep, one_at_a_time_sweep, repetitions, job_name, DEBUG=DEBUG,
         total_budget=total_budget)
