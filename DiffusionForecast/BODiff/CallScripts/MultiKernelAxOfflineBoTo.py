# Created on 23/09/2022 by Lewis Dickson

"""

NOTE: see here for more information on the kernel choices:
https://docs.gpytorch.ai/en/stable/kernels.html

Main function running the BO loop
- var_dict : (dict) the dict contianing the variables to optimise, the scaling, bound and starting values (created through the function optimiser_funcs.create_var_dict(var_names, start_vals, var_scales, var_bounds) )
N_rand : (int) number of random searches to be performed on the parmater space to perform before optimisation loop

---------------
Related files
---------------
-> csv - handles the tracking of requested, real and observation results, best found results, etc
-> flag_file - a file which the optimiser checks after it has made a request to allow the time for loading the experimental results or waiting on simulations to complete
"""

"""
TODO:
- currently running multiple acquisition functions - change these to be an argument so that they can be changed in a loop and compared
- Add input/output between this script and the main call script which controlls the loading of the expeirmental data etc
"""
# updated on 03/11/2022 by Lewis Dickson

"""
Code based on BoTorch tutorial from:
https://botorch.org/tutorials/custom_botorch_model_in_ax
"""

# Next cell sets up a decorator solely to speed up the testing of the notebook. You can safely ignore this cell and the use of the decorator throughout the tutorial.

#============================================================================
# Imports
#============================================================================


# --- General Imports --- #
import sys
import os
from threading import Event

# --- Required BO modules --- #
import torch
from contextlib import contextmanager
from ax.utils.testing.mock import fast_botorch_optimize_context_manager
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.datasets import SupervisedDataset
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, RQKernel, LinearKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.test_functions import Branin
from ax.utils.notebook.plotting import render
import plotly.io as pio
import pandas as pd
import torch
from ax import (
    Data,
    Experiment,
    Metric,
    Objective,
    OptimizationConfig,
    ParameterType,
    RangeParameter,
    Runner,
    SearchSpace,
)
from botorch.test_functions import Branin
from ax.modelbridge.registry import Models
import numpy as np
from ax.plot.trace import optimization_trace_single_method
from ax.storage.registry_bundle import RegistryBundle
from ax.storage.json_store.save import save_experiment
# from ax import json_save
from ax.storage.json_store.save import save_experiment as json_save

#============================================================================
# Functions
#============================================================================

####### Class Definitions #######

# --- Implementing the custom model --- #
class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class RBFKernel_SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class RQKernel_SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RQKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class LinearKernel_SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=LinearKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class MaternKernel_1_over_2_SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=LinearKernel(nu=0.5, ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class MaternKernel_3_over_2_SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=LinearKernel(nu=1.5, ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class MaternKernel_5_over_2_SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=LinearKernel(nu=2.5, ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata

####### Function Definitions #######

# Search space defines the parameters, their types, and acceptable values.
def create_search_space(var_dict):
    name_list = var_dict['name'] # holds variable names
    bound_list = var_dict['bounds'] # bounds [[var1_lower, var1_upper],[var2_lower, var2_upper],..,etc]
    lower_bound_list = [l[0] for l in bound_list]
    upper_bound_list = [l[1] for l in bound_list]

    parameter_list = [RangeParameter(name=name_val, parameter_type=ParameterType.FLOAT, lower=lower_bound_val, upper=upper_bound_val) for name_val, lower_bound_val, upper_bound_val in zip(name_list, lower_bound_list, upper_bound_list)]

    out_search_space = SearchSpace(parameters= parameter_list)
    return out_search_space

def get_ret_val(optimiser_csv_path):
    dF_csv = pd.read_csv(optimiser_csv_path)
    df_last_row = dF_csv.iloc[-1]
    eparam_eval = df_last_row['yVal']
    eparam_eval_err = df_last_row['yVal_err']
    return eparam_eval, eparam_eval_err

# Input e.g : parameters = {'x1': 10.0, 'x2': 15.0}
# Returns e.g : evaluate(parameters) = {'branin': (117.71708679199219, nan)}
def evaluate(parameters, optimiser_csv_path, returned_merit_event_obj, written_request_event_obj,trial_index):

    # Create a dF with the parameters
    param_dF = pd.DataFrame(parameters, index=[trial_index])

    # Append the dF to the optimiser csv
    current_dF = pd.read_csv(optimiser_csv_path)
    updated_dF = pd.concat([current_dF,param_dF], join='outer')
    updated_dF.to_csv(optimiser_csv_path, mode='w', header=True)
    written_request_event_obj.set()

    """
    Here we pass the requested parameters to the main script to find and evaluate them
    """

    # Wait for flag that result value has been updated
    print(f'Waiting for merit return')
    flag = returned_merit_event_obj.wait(timeout=None)
    print(f'Merit returned')
    # 10 second wait timeout, can also be set to None if the process of returning value is long or unknown
    eparam_eval, eparam_eval_err = get_ret_val(optimiser_csv_path)
    ret_dict = {'test_objective' : (eparam_eval, eparam_eval_err)}
    returned_merit_event_obj.clear()
    return ret_dict

# Creates a list of dicts in the format required by the Ax experiemet function call
# NOTE : It is crucial to use floats for the bounds, i.e., 0.0 rather than 0.
# Otherwise, the parameter would be inferred as an integer range.
def var_dict_to_Ax_exp_form(var_dict):
    name_list = var_dict['name'] # holds variable names
    bound_list = var_dict['bounds'] # bounds [[var1_lower, var1_upper],[var2_lower, var2_upper],..,etc]
    lower_bound_list = [l[0] for l in bound_list]
    upper_bound_list = [l[1] for l in bound_list]
    out_LoD = [{"name" : name_val, "type" : "range", "bounds" : [float(lower_bound_val), float(upper_bound_val)]} for name_val, lower_bound_val, upper_bound_val in zip(name_list, lower_bound_list, upper_bound_list)]
    return out_LoD

def save_experiment_func(exp_to_save, ax_exp_objs_data_pkl_path, optimiser_csv_path, AcquisitionMetric, SimpleCustomGP):
    # --- Creating correct output path --- #
    # Making sure that the pkl name hase a matching time stamp to the csv data
    csv_fname = os.path.basename(optimiser_csv_path)
    csv_timestamp = csv_fname.split('_')[-1]
    csv_timestamp = csv_timestamp.split('.')[0]

    # bundle = RegistryBundle(
    #     metric_clss={AcquisitionMetric: None, SimpleCustomGP: None},
    #     runner_clss={MyRunner: None})

    bundle = RegistryBundle(
    metric_clss={BoothMetric: None, L2NormMetric: None, Hartmann6Metric: None},
    runner_clss={MyRunner: None}
    )

    full_ax_pkl_path = f'{ax_exp_objs_data_pkl_path}//{csv_timestamp}_ax_exp_obj.json'
    save_experiment(exp_to_save, full_ax_pkl_path)#, encoder_registry = bundle.encoder_registry)




#============================================================================
# Main
#============================================================================

"""
Implementing the custom model
For this code, we implement a very simple gpytorch Exact GP Model that uses an RBF kernel (with ARD) and infers a (homoskedastic) noise level.

Model definition is straightforward - here we implement a gpytorch `ExactGP` that also inherits from `GPyTorchModel` -- this adds all the api calls that botorch expects in its various modules.

*Note:* botorch also allows implementing other custom models as long as they follow the minimal `Model` API. For more information, please see the [Model Documentation](../docs/models).
"""

"""
- var_dict : (dict) the dict contianing the variables to optimise, the scaling, bound and starting values (created through the function optimiser_funcs.create_var_dict(var_names, start_vals, var_scales, var_bounds) )
N_rand : (int) number of random searches to be performed on the parmater space to perform before optimisation loop
N_trials : (int) number of searches to perform during the optimisation after the random search has finished
optimiser_csv_path : (path string) path to the output csv for tracking progress of optimisation
csv_headers : (list of strings) headers used in the optimiser csv file
maximise_bool : (bool) if True will maximise the merit function otherwise False will minimise it
"""

def run_optimiser(var_dict, N_rand, N_trials, optimiser_csv_path, csv_headers, maximise_bool, returned_merit_event_obj, written_request_event_obj, error_event_obj, ax_exp_objs_data_pkl_path, kernel_choice):

    N_full = N_rand + N_trials # required if using the ax api training method
    use_developer_options = True # default is False - Developer gives more detailed control but also more compelex to implement and follow, Developer also seems to be faster
    if not use_developer_options:
        # --- Instantiate a BoTorchModel in Ax --- #
        """
        A `BoTorchModel` in Ax encapsulates both the surrogate (commonly referred to as `Model` in BoTorch) and an acquisition function. Here, we will only specify the custom surrogate and let Ax choose the default acquisition function.

        Most models should work with the base `Surrogate` in Ax, except for BoTorch `ModelListGP`, which works with `ListSurrogate`.
        Note that the `Model` (e.g., the `SimpleCustomGP`) must implement `construct_inputs`, as this is used to construct the inputs required for instantiating a `Model` instance from the experiment data.

        In case the `Model` requires a complex set of arguments that cannot be constructed using a `construct_inputs` method, one can initialize the `model` and supply it via `Surrogate.from_botorch(model=model, mll_class=<Optional>)`, replacing the `Surrogate(...)` below.
        """

        ax_model = BoTorchModel(
            surrogate=Surrogate(
                # The model class to use
                botorch_model_class=SimpleCustomGP,
                # Optional, MLL class with which to optimize model parameters
                # mll_class=ExactMarginalLogLikelihood,
                # Optional, dictionary of keyword arguments to model constructor
                # model_options={}
            ),
            # Optional, acquisition function class to use - see custom acquisition tutorial
            # botorch_acqf_class=qExpectedImprovement,
        )


        # --- Using the custom model in Ax to optimize the function --- #
        gs = GenerationStrategy(
            steps=[
                # Quasi-random initialization step
                GenerationStep(
                    model = Models.SOBOL,
                    num_trials = N_rand,  # How many trials should be produced from this generation step
                ),
                # Bayesian optimization step using the custom acquisition function
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials = N_trials,  #-1 for no limitation on how many trials should be produced from this step
                    # For `BOTORCH_MODULAR`, we pass in kwargs to specify what surrogate or acquisition function to use.
                    model_kwargs = {
                        "surrogate": Surrogate(SimpleCustomGP),
                    },
                ),
            ]
        )


        # --- Setting up the experiment --- #
        """
        In order to use the `GenerationStrategy` we just created, we will pass it into the `AxClient`.
        """

        # Initialize the client - AxClient offers a convenient API to control the experiment
        ax_client = AxClient(generation_strategy=gs)
        # Setup the experiment
        ax_exp_LoD = var_dict_to_Ax_exp_form(var_dict)
        ax_client.create_experiment(
            name="test_experiment",
            parameters=ax_exp_LoD,
            objectives={
                "test_objective": ObjectiveProperties(minimize = not maximise_bool),
            },
        )

        # --- Running the BO with Service API --- #

        rand_loop_counter = 0
        for i in range(N_full):
            parameters, trial_index = ax_client.get_next_trial()
            print(f'parameters = {parameters}')
            # Local evaluation here can be replaced with deployment to external system.
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters, optimiser_csv_path, returned_merit_event_obj, written_request_event_obj, trial_index))
            rand_loop_counter += 1

        print(f'rand_loop_counter = {rand_loop_counter}')


        return


        # --- Viewing the evaluated trials --- #
        # ax_client.get_trials_data_frame()
        #
        # parameters, values = ax_client.get_best_parameters()
        # print(f"Best parameters: {parameters}")
        # print(f"Corresponding mean: {values[0]}, covariance: {values[1]}")
        #
        # # --- Plotting the response surface and optimization progress --- #
        #
        # # render(ax_client.get_contour_plot())
        #
        # best_parameters, values = ax_client.get_best_parameters()
        # best_parameters, values[0]

        # render(ax_client.get_optimization_trace())


    if use_developer_options:
        # --- Optimization with the Developer API --- #
        """
        A detailed tutorial on the Service API can be found [here](https://ax.dev/tutorials/gpei_hartmann_developer.html).

        Set up the Experiment in Ax

        We need 3 inputs for an Ax `Experiment`:
        - A search space to optimize over;
        - An optimization config specifiying the objective / metrics to optimize, and optional outcome constraints;
        - A runner that handles the deployment of trials. For a synthetic optimization problem, such as here, this only returns simple metadata about the trial.
        """

        # Search space defines the parameters, their types, and acceptable values.
        search_space = create_search_space(var_dict)

        # Define the surrogate based on the kernel choice
        if 'MaternKernel_1_over_2' in kernel_choice:
            GP_kernel_choice = MaternKernel_1_over_2_SimpleCustomGP

        elif 'MaternKernel_3_over_2' in kernel_choice:
            GP_kernel_choice = MaternKernel_3_over_2_SimpleCustomGP

        elif 'MaternKernel_5_over_2' in kernel_choice:
            GP_kernel_choice = MaternKernel_5_over_2_SimpleCustomGP

        elif 'RBFKernel' in kernel_choice:
            GP_kernel_choice = RBFKernel_SimpleCustomGP

        elif 'RQKernel' in kernel_choice:
            GP_kernel_choice = RQKernel_SimpleCustomGP

        elif 'LinearKernel' in kernel_choice:
            GP_kernel_choice = LinearKernel_SimpleCustomGP

        # the metric is a wrapper that structures the function output.
        class AcquisitionMetric(Metric):
            def fetch_trial_data(self, trial):
                records = []
                for arm_name, arm in trial.arms_by_name.items():
                    params = arm.parameters
                    trial_index = trial.index
                    print(f'BO Requested params = {params}')
                    eval_dict = evaluate(params, optimiser_csv_path, returned_merit_event_obj, written_request_event_obj,trial_index)
                    eparam_eval = eval_dict['test_objective'][0]
                    eparam_eval_err = eval_dict['test_objective'][1]

                    records.append(
                        {
                            "arm_name": arm_name,
                            "metric_name": self.name,
                            "trial_index": trial_index,
                            "mean": eparam_eval,
                            "sem": float("nan")#eparam_eval_err,  # SEM (observation noise) - NaN indicates unknown
                        }
                    )
                return Data(df=pd.DataFrame.from_records(records))

        optimization_config = OptimizationConfig(
            objective=Objective(
                metric=AcquisitionMetric(name="acquisition_metric", lower_is_better= not maximise_bool),
                minimize= not maximise_bool,  # This is optional since we specified `lower_is_better=True`
            )
        )

        exp = Experiment(
            name="acquisition_experiment",
            search_space=search_space,
            optimization_config=optimization_config,
            runner=MyRunner(),
            )

        # --- Run the BO loop via Developer API --- #
        """
        First, we use the Sobol generator to create N_rand (quasi-) random initial point in the search space. Ax controls objective evaluations via `Trial`s.
        - We generate a `Trial` using a generator run, e.g., `Sobol` below. A `Trial` specifies relevant metadata as well as the parameters to be evaluated. At this point, the `Trial` is at the `CANDIDATE` stage.
        - We run the `Trial` using `Trial.run()`. In our example, this serves to mark the `Trial` as `RUNNING`. In an advanced application, this can be used to dispatch the `Trial` for evaluation on a remote server.
        - Once the `Trial` is done running, we mark it as `COMPLETED`. This tells the `Experiment` that it can fetch the `Trial` data.

        A `Trial` supports evaluation of a single parameterization. For parallel evaluations, see [`BatchTrial`](https://ax.dev/docs/core.html#trial-vs-batch-trial).
        """

        sobol = Models.SOBOL(exp.search_space)

        for i in range(N_rand):
            trial = exp.new_trial(generator_run=sobol.gen(1))
            trial.run()
            trial.mark_completed()

        """Once the initial (quasi-) random stage is completed, we can use our `SimpleCustomGP` with the default acquisition function chosen by `Ax` to run the BO loop. """

        for i in range(N_trials):
            try: # This is the part of the script that can fail due to fitting but since we have already updated all values we can just exit and use as much of the data as was saved
                model_bridge = Models.BOTORCH_MODULAR(
                    experiment=exp,
                    data=exp.fetch_data(),
                    surrogate = Surrogate(GP_kernel_choice)
                    # surrogate=Surrogate(SimpleCustomGP),
                )
                # Error arrises here due to excessive nans in the gradient
                trial = exp.new_trial(generator_run=model_bridge.gen(1))
                trial.run()
                trial.mark_completed()
            except:
                written_request_event_obj.set()
                error_event_obj.set()
                return exp

        # --- View the trials attached to the `Experiment` --- #
        exp.trials

        # --- View the evaluation data about these trials --- #
        exp.fetch_data().df

        # csv_fname = os.path.basename(optimiser_csv_path)
        # csv_timestamp = csv_fname.split('_')[-1]
        # csv_timestamp = csv_timestamp.split('.')[0]
        #
        # bundle = RegistryBundle(
        #     metric_clss={AcquisitionMetric: None, SimpleCustomGP: None},
        #     runner_clss={MyRunner: None})

        # from ax.storage.metric_registry import register_metric

        # register_metric(metric_cls=AcquisitionMetric)
        # register_metric(metric_cls=MetricB)

        # bundle = RegistryBundle(
        # metric_clss={SimpleCustomGP: None},
        # runner_clss={MyRunner: None}
        # )

        # full_ax_pkl_path = f'{ax_exp_objs_data_pkl_path}//{csv_timestamp}_ax_exp_obj.json'
        # # save_experiment(exp_to_save, full_ax_pkl_path)
        # save_experiment(exp, full_ax_pkl_path)


        # # Saving experiment
        # save_experiment_func(exp, ax_exp_objs_data_pkl_path, optimiser_csv_path, AcquisitionMetric, SimpleCustomGP)

        # from ax.storage.sqa_store.db import init_engine_and_session_factory, get_engine, create_all_tables
        # from ax.storage.sqa_store.load import load_experiment
        # from ax.storage.sqa_store.save import save_experiment
        #
        # init_engine_and_session_factory(url='sqlite:    ///foo3.db')
        #
        # engine = get_engine()
        # create_all_tables(engine)
        #
        # from ax.storage.sqa_store.sqa_config import SQAConfig
        #
        # exp.name = "new"
        #
        # sqa_config = SQAConfig(
        #     json_encoder_registry=bundle.encoder_registry,
        #     json_decoder_registry=bundle.decoder_registry,
        #     metric_registry=bundle.metric_registry,
        #     runner_registry=bundle.runner_registry,
        # )
        #
        # save_experiment(exp, config=sqa_config)

        return exp


# --- Plot results --- #
# We can use Ax utilities for plotting the results.
# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple
# optimization runs, so we wrap out best objectives array in another array.
# objective_means = np.array([[trial.objective_mean for trial in exp.trials.values()]])
# if maximise_bool:
#     y = np.maximum.accumulate(objective_means, axis=1)
# else:
#     y=np.minimum.accumulate(objective_means, axis=1)
# best_objective_plot = optimization_trace_single_method(
#     y=y)
# render(best_objective_plot)
