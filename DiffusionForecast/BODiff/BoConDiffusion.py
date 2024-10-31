# Created on 23/10/2024 by Lewis Dickson 

"""
Main call script for running Bayesian optimisation 
for hpyer parameter optimisation for time-series 
Diffusion forecasting. 

The general layout is applicable to any BO for ML
where have included:
- Automated GPU selection dictated by computation
or VRAM avaliability 
- Automated tuning of parameters 
- Launching, monitoring, and stopping training 
- Tracking of optimisation process to dictate
optimisationend point 

The overall structure of the code is:

Main: BoConDiffusion
<
 Caller of optimiser 
    <
    -BO Process
    >

 Controlls threading of Diffusion processes
    <
    -Using Alistair Brash's Time series 
    Diffusion model
    >
 
 Caller of optimisation monitoring 
    <
    Tracking progress
    Resetting optimisation process if stuck
    >
>


"""


#===========================================
# User Settings 
#===========================================

# --- Input Data Options --- # 
dataset = r"1 - Savanna Preserve"
data_set_str = "1"     
"""
    "1 - Savanna Preserve"
    "2 - Clean Urban Air"
    "3 - Resilient Fields"
"""

# dataset = "2 - Clean Urban Air"
# dataset = "3 - Resilient Fields"

# Use get_least_utilized_gpu or get_least_memory_utilized_gpu to auto launch on the optimum GPU, if auto_select = False, default_GPU is set

# --- Analysis Options --- #
auto_select_GPU_options = {'auto_select': True, # if True finds least utilised GPU in terms of utilisation or memory
                           'compute_or_memory':'memory', # 'compute' or 'memory' to set deciscion method
                           'default_GPU':3,
                           'avoid_GPU':0 # [int] static option to avoid certain GPUs - -1 for no avoids 
                           }


# Setting Sample size for forecasting
sample_size_days = 21 # Days
forecast_size_days = 7 # Days

#========================================#
# --- Bayesian Optimisations Options --- #

run_header, shot_header = 'Run', 'Shot' #header as string of run column and shot column
matchstr_header = 'MatchStr' # match str header used for extracting date, run and shot info

# optimisation_dict = {
#     'learning_rate': {
#                   'start_val' : 1e-3,
#                   'var_scale' : 1e-4,
#                   'var_bounds': [1e-5,1]
# }

"""
Add as many sub-dicts as parameters that you want to optimise
optimisation_dict = {
    '<parameter to optimise>': {
        'start_val' : (float), # initial value to use for BO search
        'var_scale' : (float), # typical scaling factor for search steps
        'var_bounds': [(float), (float)] # hard lower and upport bounds for search
        'continous' : (Bool) # if continous values can be used or need discreet
    },
}
"""

optimisation_dict = {
    'T': {
        'start_val' : 200, # initial value to use for BO search
        'var_scale' : 10, # typical scaling factor for search steps
        'var_bounds': [1e-5,1] # hard lower and upport bounds for search
    },

    "beta_0": {
        'start_val' : 0.0001,
        'var_scale' : 1e-3,
        'var_bounds': [1e-5,1e-2],
    },
    
    "beta_T": {
        'start_val' : 0.02,
        'var_scale' : 0.01,
        'var_bounds': [1e-4,1],
    }
}


y_val_header = 'yVal' # WARNING: these should be automated to change with the variable that is being optimised
y_val_err_header = 'yVal_err' # WARNING: these should be automated to change with the variable that is being optimised

## -----  Global Optimisation search options ----- ##
kernel_choice_list = ['MaternKernel_1_over_2','MaternKernel_3_over_2','MaternKernel_5_over_2', 'RBFKernel', 'LinearKernel', 'RQKernel'] # note for MeternKernel the number is the smoothness parameter: https://docs.gpytorch.ai/en/stable/kernels.html#maternkernel
use_BO = True # If True runs bayesian optimisation otherwise runs random search - if false will only complete <num_rand_searches> searches
# print('WARNING MAKE SURE TO CHANGE AFTER TESTING')
num_consequitve_searches = 1 # how many loops of the full optimisation to perform for statistics

# Intra-optimisation options
num_rand_searches = 10 # (int) number of random searches to begin building prior
num_trials = 190 # (int) number of BO iterations to complete
max_iterations = num_rand_searches + num_trials

# BO-Hyperparameteres 
kernel_choice = 'RBFKernel'

# Merit function to evaluate
maximise_bool = False # if true maximises mertit function if false minimises
fix these 
requested_e_params = ['total_charge_key', 'total_beam_energy_key']
merit_func_expression = 'total_charge_key*total_beam_energy_key'


#===========================================
# Imports 
#===========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import shutil
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as L
import pandas as pd
import numpy as np
import torch
import threading
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys

from DiffusionStuff.utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams
from DiffusionStuff.utils.util import find_max_epoch, print_size, calc_diffusion_hyperparams, training_loss_replace
from DiffusionStuff.utils.util import get_mask_bm_forecasting, create_3d_array_rollingwindow

import json
from DiffusionStuff.imputers.SSSDS4Imputer import SSSDS4Imputer
from sklearn.metrics import mean_squared_error 
from statistics import mean

# --- Training & Eval Funcs --- #  
from CallScripts import DiffusionFuncs as DF
from CallScripts import BDUtils

# --- Optimiser Funcs --- # 
from CallScripts import OptimiserFuncs as optim
from threading import Thread
from importlib import reload
from CallScripts import shotfinder as SF # file containing the functions for loading matching parmater shots
from CallScripts import loading_saving_functions as iofuncs # file for output scripts and functions
from CallScripts import MultiKernelAxOfflineBoTo
from CallScripts import optimiser_funcsV2 as optimiser_funcs

MultiKernelAxOfflineBoTo = reload(MultiKernelAxOfflineBoTo)
iofuncs = reload(iofuncs)
optimiser_funcs = reload(optimiser_funcs)
SF = reload(SF)

#===========================================
# Functions
#===========================================

def load_data(data_set_str):
    """
    "1 - Savanna Preserve"
    "2 - Clean Urban Air"
    "3 - Resilient Fields"
    """

    values_dict = {
    "1":"1 - Savanna Preserve",
    "2":"2 - Clean Urban Air",
    "3":"3 - Resilient Fields",
    }

    valid_values = ["1", "2", "3"]
    assert data_set_str in valid_values, f"Value '{data_set_str}' is not in the allowed set {valid_values}"

    # # Loading Training data
    X_train = pd.read_csv("Data/" + values_dict[data_set_str]  + rf"/{int(data_set_str)}_X_train.csv")
    y_train = pd.read_csv("Data/" + values_dict[data_set_str]  + rf"/{int(data_set_str)}_y_train.csv")

    # Loading Test Data 
    X_test = pd.read_csv(r"Data/" + values_dict[data_set_str] + rf"/{int(data_set_str)}_X_test.csv")
    y_test = pd.read_csv("Data/" + values_dict[data_set_str]  + rf"/{int(data_set_str)}_y_test.csv")

    return X_test, X_train, y_test, y_train

def preprocess_dF(dF):
    """
    Add whatever pre-processing you need to do to the 
    dataframes here
    """

    dF = dF.drop(columns=['Unnamed: 0', 'location_id'])
    dF["date"] = pd.to_datetime(dF["date"])
    return dF

# Class definition of a thread allowing for a return value to be given upon joining
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

############################################
# Main
############################################

#------------------------------------------
# Loading the experimental data
#------------------------------------------

# Load the training and test data 
X_test, X_train, y_test, y_train = load_data(data_set_str)

# Clean input data
X_test = preprocess_dF(X_test)
X_train = preprocess_dF(X_train)
y_test = preprocess_dF(y_test)
y_train = preprocess_dF(y_train)

# Append test and merged training data 
merged_train = pd.merge(y_train, X_train, on='date', how='inner')
merged_test = pd.merge(y_test, X_test, on='date', how='inner')
merged_all = pd.concat([merged_train, merged_test], axis=0)

#------------------------------------------
# Creating output paths 
#------------------------------------------

# Handling server saving

if BDUtils.get_host_name()=='cava':
    base_output_dir = r'/mnt/hdd/ldickson/'
    
elif BDUtils.get_host_name()=='papa': # Put whatever path here you like 
    pass

# ... 

else:
    base_output_dir = r'./'


# Checking if the test is run to use random serach or BO
if not use_BO: # restated to make sure that num_trials is always zero when random searching
    num_trials = 0
    batch_folder_name = 'RandBatch'
else:
    batch_folder_name = 'BOBatch'


# Renaming paths depending on which server is used
iterative_output_folder = '/BOcDiff/TestDIRName/' # NOTE: make this automatically change between runs i.e run time, params, idk?? 
main_out_dir = rf'{base_output_dir}/{iterative_output_folder}/'
modelout_dir = rf'{main_out_dir}/model/' # where to dump split frames depending on local testing or cava server
optimiser_dir = rf'{main_out_dir}/optimiser/'
BOObj_pickle_dump_dir = rf'{main_out_dir}/BOObj_pickle_dump/' 
config_dir = rf'{main_out_dir}/configs/'

# Prepping output paths 
BDUtils.ifdir_doesntexist_created_nested(main_out_dir, silence_warnings=True)
BDUtils.ifdir_doesntexist_created_nested(modelout_dir, silence_warnings=True)
BDUtils.ifdir_doesntexist_created_nested(optimiser_dir, silence_warnings=True)

#------------------------------------------
# Creating/loading the optimisation tracker
#------------------------------------------

# Explicitly extracting for ease of file name creation
var_names = [*optimisation_dict]
start_vals = [optimisation_dict[key]['start_val'] for key in var_names]
var_scales = [optimisation_dict[key]['var_scale'] for key in var_names]
var_bounds = [optimisation_dict[key]['var_bounds'] for key in var_names]

#------------------------------------------
# Loading default configs for non-optimised variables
#------------------------------------------

with open("DiffusionStuff/configs/conditionalForecastV2.json") as f:
    data = f.read()

default_config = json.loads(data)

#------------------------------------------
# Input training data prep
#------------------------------------------

# Extracting array of merged data from sample size and focast window with moving window
sample_size = sample_size_days*24
forecast_size = forecast_size_days*24
sample_size = sample_size+forecast_size
merged_all_data = create_3d_array_rollingwindow(merged_all, sample_size, forecast_size)

# Splitting all data into training, testing and validation 
percent_idx = int(np.shape(merged_all_data)[0]*0.8) # 80%
train = merged_all_data[:percent_idx]
test = merged_all_data[percent_idx:]

scaler = StandardScaler().fit(train.reshape(-1, train.shape[-1]))
train_scaled = scaler.transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)
test_scaled = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)

print("train_scaled.shape", train_scaled.shape)
print("test_scaled.shape", test_scaled.shape)


############################################
# Beginning optimisation loop
############################################

# Counts the previous number of batches and creates a new folder with an iterative name
checked_batch_folder_name = iofuncs.create_batch_folder_number(optimiser_dir, batch_folder_name)
optimiser_dir_single_batch = f'{optimiser_dir}//{checked_batch_folder_name}//'
BDUtils.ifdir_doesntexist_created_nested(optimiser_dir_single_batch, silence_warnings=True)

# Create initial empty .csv for the optimiser, returns header list
csvfilename = 'BO_optimsier.csv'
full_optimiser_csv_path, headers = iofuncs.multikernel_created_optim_csv(csvfilename, optimiser_dir_single_batch, var_names, y_val_header, y_val_err_header, matchstr_header, kernel_choice)
print(f'optimiser_csv_path = {full_optimiser_csv_path}')

# Creates the metadata for each optimisation run
iofuncs.multi_kernel_create_optim_meta_data_readme(optimiser_dir, 
                                                var_names,
                                                start_vals,   
                                                var_scales,
                                                var_bounds,
                                                y_val_header,
                                                y_val_err_header,
                                                use_BO,
                                                maximise_bool,
                                                num_consequitve_searches,
                                                num_rand_searches,
                                                num_trials, 
                                                max_iterations, 
                                                requested_e_params, 
                                                merit_func_expression, 
                                                checked_batch_folder_name, 
                                                kernel_choice)

# NOTE: everything after here is looped 
optim_completed_counter = 0
while optim_completed_counter < num_consequitve_searches:

    #------------------------------------------------------
    # Creating a meta data file for post-analysis reference
    #------------------------------------------------------

    # Creating a dictionary to hold the variable data
    var_dict, NDIMS = optimiser_funcs.create_var_dict(var_names, start_vals, var_scales, var_bounds)

    # Creating event object for threads
    returned_merit_event_obj = threading.Event()
    written_request_event_obj = threading.Event()
    error_event_obj = threading.Event() # case where GP fit fails and we need to abandon and move onto next loop

    returned_merit_event_obj.clear() # Setting event to False
    written_request_event_obj.clear() # Setting event to False
    error_event_obj.clear() # Setting event to False

    # Creating thread to run optimiser script indefinetly
    optimiser_thread = ThreadWithReturnValue(
        target = MultiKernelAxOfflineBoTo.run_optimiser, 
            args = (
                var_dict, 
                num_rand_searches, 
                num_trials, 
                full_optimiser_csv_path, 
                headers, 
                maximise_bool, 
                returned_merit_event_obj,
                written_request_event_obj,
                error_event_obj, 
                BOObj_pickle_dump_dir, 
                kernel_choice,)
        )
    optimiser_thread.start() # Starting child thread


    # ---- Autosetting GPU ---- # 
    # Setting the optimum GPU 
    GPU_number, GPU_utilisation = optim.auto_set_GPU(auto_select_GPU_options)
    print(f'\nAutoset GPU: {GPU_number}')
    print(f'{GPU_number=}')

    ############################################
    # Finding the requested data from the optimiser code
    ############################################
    i = 0

    while i < max_iterations:
        print(f'\n\nIteration: {i}\n\n')

        ############################################
        # Error checking and inter-thread timing
        ############################################

        # Checking if there has been an error in the GP fit
        if error_event_obj.is_set():
            break

        # Wait for request parameters to be written
        written_request_flag = written_request_event_obj.wait(timeout=None)
        written_request_event_obj.clear() # reset flag to False
        
        # Checking if there has been an error in the GP fit
        if error_event_obj.is_set():
            break

    # ---- Begin config managment ---- # 
    

    # global train_config
    train_config = config["train_config"]  # training parameters

    gen_config = config['gen_config']

    # global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    # global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters
    print(diffusion_config)

    # global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    # global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config'] 

    print(model_config)

    DF.training(**train_config, 
                train_scaled = train_scaled,
                gpu_number=GPU_number,
                trainset_config = trainset_config,
                diffusion_config=diffusion_config, 
                diffusion_hyperparams= diffusion_hyperparams,
                model_config = model_config
                )
