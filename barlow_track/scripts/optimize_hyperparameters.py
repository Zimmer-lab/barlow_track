# Use the Ax library to optimize hyperparameters
# See: https://ax.dev/tutorials/submitit.html
import argparse
import logging
import os
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
try:
    from barlow_track.utils.barlow import PretrainedArchitectureMismatchError
except ImportError as e:
    raise ImportError(
        "Installed barlow_track package is stale (no PretrainedArchitectureMismatchError); "
        "it shadows your checkout. Reinstall from your checkout, e.g.: "
        "pip install --no-deps -e <path-to-barlow_track-checkout>"
    ) from e
from barlow_track.utils.utils_ground_truth import check_training_finished, discover_trials, extract_val_from_json
from barlow_track.utils.utils_seeding import replicate_seed, search_trial_seed


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


def attach_prior_trial_to_ax_client(ax_client, full_params, result):
    # keep only parameters Ax knows about
    ax_params = {k: v for k, v in full_params.items() if k in ax_client.experiment.search_space.parameters}
    # BoTorch GP cannot fit NaN/inf/None; fail loudly here so callers can skip
    mean = result[0] if isinstance(result, tuple) else result
    try:
        is_finite = bool(np.isfinite(mean))
    except TypeError:
        is_finite = False
    if not is_finite:
        raise ValueError(f"Non-finite prior trial result {mean!r}; skipping")
    trial_index = ax_client.attach_trial(parameters=ax_params)
    ax_client.experiment.trials[trial_index].run_metadata = full_params
    if not isinstance(result, tuple):
        result = (result, 0.0)  # Fake error bars
    ax_client.complete_trial(trial_index, raw_data=result)
    return trial_index



def optimize_hyperparameters(hyperparameter_path, run_locally=False, num_parallel_jobs=None, 
                             direct_parameter_sweep=False, one_at_a_time_sweep=False, repetitions=1, 
                             job_name=None, DEBUG=False):
    """
    Parameters
    ---------------------
    hyperparameter_path - Path to the yaml file, usually named hyperparameter_search_template.yaml
    run_locally - Instead of the cluster, via slurm
    num_parallel_jobs
    direct_parameter_sweep - Generate runs as all combinations of parameters instead of optimizing (e.g. for grid search)
    one_at_a_time_sweep - Generate runs by using only one of the hyperparameter changes at a time; the rest are defaults (e.g. for ablations)
    job_name - SLURM job name

    """
    if DEBUG:
        run_locally = True
    if hyperparameter_path is None:
        raise ValueError("Please provide a hyperparameter template path")
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

    def evaluate(parameters):
        # Add the baseline parameters
        args = SimpleNamespace(**parameters)
        try:
            test_losses = train_barlow_network(args)
            result = test_losses['test_loss'] if isinstance(test_losses, dict) else 1e6
        except PretrainedArchitectureMismatchError:
            # Systematic config error affecting every trial; fail fast instead of
            # scoring 1e6 and letting Ax optimize noise.
            raise
        except Exception as e:
            logging.exception(f"Encountered error with trial; quitting gracefully: {e}")
            result = 1e6
        try:
            if result is None or not np.isfinite(result):
                result = 1e6  # More or less infinity; Ax/BoTorch cannot fit NaN/inf
        except TypeError:
            result = 1e6
        return {"result": float(result)}

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
    ax_client.create_experiment(
        name="my_experiment",
        parameters=parameters,
        objectives={"result": ObjectiveProperties(minimize=True)},
    )

    # See if there are any previously run trials, and load them
    prior_trials = discover_trials(experiment_parent_folder)
    if len(prior_trials) > 0:
        logging.info(f"Discovered {len(prior_trials)} prior trials, loading...")
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
    executor.update_parameters(timeout_min=65 * 12 * num_days)
    if not run_locally:
        executor.update_parameters(slurm_time=f"{num_days}-12:00:00")
        executor.update_parameters(cpus_per_task=8)
        executor.update_parameters(slurm_mem="128G")
        executor.update_parameters(slurm_job_name=job_name if job_name is not None else "barlow_hyperparameter_search")
        executor.update_parameters(slurm_gres="gpu:1")
        executor.update_parameters(slurm_constraint="l40s|a30|t4|l4")
        executor.update_parameters(slurm_additional_parameters={"no-requeue": True})  # bash equivalent (no-arg flag): #SBATCH --no-requeue

    if direct_parameter_sweep:
        # Manually define all the trials as all combinations
        for param in parameters:
            all_param_lists = []
            if param['type'] == 'choice':
                assert 'values' in param, "For direct parameter sweep, the parameter must have a list of values"
                # List of lists, which will be combined into a grid
                all_param_lists.append(param['values'])
            else:
                raise ValueError("For direct parameter sweep, all parameters must be of type 'choice'")
        
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
        total_budget = 5 if DEBUG else 30

    if direct_parameter_sweep or one_at_a_time_sweep:
        # Directly duplicate planned jobs. Remember how many configs there are
        # before duplicating: job i is replicate i // n_unique_configs of config
        # i % n_unique_configs, and the replicate index drives the seed policy.
        n_unique_configs = len(all_combinations)
        all_combinations = all_combinations * repetitions
        total_budget = len(all_combinations)

    if num_parallel_jobs is None:
        num_parallel_jobs = 1 if (DEBUG or run_locally) else 10
    else:
        num_parallel_jobs = int(num_parallel_jobs)

    jobs = []
    submitted_jobs = 0
    trial_offset = 0
    start_time = time.time()

    # Run until all the jobs have finished and our budget is used up.
    while submitted_jobs < total_budget or jobs:
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
        trial_seeds = {}
        if direct_parameter_sweep or one_at_a_time_sweep:
            # Get a new trial manually, without using the AxClient's internal logic (it can't do a grid search)
            # Use the submitted_jobs index as the start point of the next batch of trials

            trial_index_to_param = {}
            for i in range(submitted_jobs, submitted_jobs + num_parallel_jobs - len(jobs)):
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
                max_trials=min(num_parallel_jobs - len(jobs), total_budget - submitted_jobs))
            # Bayesian search: one distinct seed per trial. A seed fixed across
            # trials makes each observation look noise-free, so the GP
            # over-trusts a config that drew a lucky init/volume selection.
            for trial_index in trial_index_to_param:
                trial_seeds[trial_index] = search_trial_seed(base_seed, trial_index)
        
        for trial_index, parameters in trial_index_to_param.items():
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
            job = executor.submit(evaluate, parameters)
            for _v in _DISPATCH_THREAD_VARS:
                os.environ[_v] = "1"
            submitted_jobs += 1
            jobs.append((job, trial_index))
            print(f"Submitted trial {trial_index} ({type(job).__name__}); "
                  f"{submitted_jobs}/{total_budget} submitted", flush=True)
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
    # The covariance is only meaningful when multiple objectives are present.
    # render(ax_client.get_contour_plot())

    # Copy the best parameters and index to a file
    best_params_path = os.path.join(experiment_parent_folder, 'best_parameters.yaml')
    with open(best_params_path, 'w') as f:
        best_parameters = _to_yaml_safe(best_parameters or {})
        best_parameters['best_trial_index'] = _to_yaml_safe(best_trial_index)
        best_parameters['best_trial_name'] = _to_yaml_safe(best_trial_name)
        best_parameters['mean_and_variance'] = _to_yaml_safe(mean_and_variance)
        YAML().dump(best_parameters, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameter_template_path', '-p', default=None)
    parser.add_argument('--run_locally', action='store_true')
    parser.add_argument('--num_parallel_jobs', default=None)
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

    optimize_hyperparameters(hyperparameter_template_path, run_locally, num_parallel_jobs, 
         direct_parameter_sweep, one_at_a_time_sweep, repetitions, job_name, DEBUG=DEBUG)
