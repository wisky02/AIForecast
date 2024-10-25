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

dataset = r"1 - Savanna Preserve"
# dataset = "2 - Clean Urban Air"
# dataset = "3 - Resilient Fields"

# Use get_least_utilized_gpu or get_least_memory_utilized_gpu to auto launch on the optimum GPU, if auto_select = False, default_GPU is set
auto_select_GPU_options = {'auto_select': True, # if True finds least utilised GPU in terms of utilisation or memory
                           'compute_or_memory':'memory', # 'compute' or 'memory' to set deciscion method
                           'default_GPU':3,
                           'avoid_GPU':0 # [int] static option to avoid certain GPUs - -1 for no avoids 
                           }


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
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys


from DiffusionStuff.utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams
from DiffusionStuff.utils.util import find_max_epoch, print_size, calc_diffusion_hyperparams, training_loss_replace
from DiffusionStuff.utils.util import get_mask_bm_forecasting, create_3d_array_rollingwindow

# sys.path.append('../')
import json
from DiffusionStuff.imputers.SSSDS4Imputer import SSSDS4Imputer
from sklearn.metrics import mean_squared_error 
from statistics import mean

# --- Training & Eval Funcs --- #  
from CallScripts import DiffusionFuncs as DF

# --- Optimiser Funcs --- # 
from CallScripts import OptimiserFuncs as optim



#===========================================
# Functions
#===========================================







#===========================================
# Main
#===========================================

# Training Data Switch 
# training_data_switch()
# TODO: add a training data_switch to clean up and allow for control from the BoConMain


# Loading Test Data 
if(dataset == "1 - Savanna Preserve"):
    X_test = pd.read_csv(r"Data/" + dataset + r"/1_X_test.csv")
elif(dataset == "2 - Clean Urban Air"):
    X_test = pd.read_csv(r"Data/" + dataset + r"/2_X_test.csv")
else:
    X_test = pd.read_csv(r"Data/" + dataset + r"/3_X_test.csv")

# # Loading Training data
if(dataset == "1 - Savanna Preserve"):
    X_train = pd.read_csv("Data/" + dataset + "/1_X_train.csv")
elif(dataset == "2 - Clean Urban Air"):
    X_train = pd.read_csv("Data/" + dataset + "/2_X_train.csv")
else:
    X_train = pd.read_csv("Data/" + dataset + "/3_X_train.csv")
    

if(dataset == "1 - Savanna Preserve"):
    y_train = pd.read_csv("Data/" + dataset + "/1_y_train.csv")
elif(dataset == "2 - Clean Urban Air"):
    y_train = pd.read_csv("Data/" + dataset + "/2_y_train.csv")
else:
    y_train = pd.read_csv("Data/" + dataset + "/3_y_train.csv")

if(dataset == "1 - Savanna Preserve"):
    y_test = pd.read_csv("Data/" + dataset + "/1_y_test.csv")
elif(dataset == "2 - Clean Urban Air"):
    y_test = pd.read_csv("Data/" + dataset + "/2_y_test.csv")
else:
    y_test = pd.read_csv("Data/" + dataset + "/3_y_test.csv")

# ---- Autosetting GPU ---- # 
# Setting the optimum GPU 
GPU_number, GPU_utilisation = optim.auto_set_GPU(auto_select_GPU_options)
print(f'\nAutoset GPU: {GPU_number}')
print(f'{GPU_number=}')

# Dataframe Cleaning: training data
X_train = X_train.drop(columns=['Unnamed: 0', 'location_id'])
X_train["date"] = pd.to_datetime(X_train["date"])

y_train = y_train.drop(columns=['Unnamed: 0',  'location_id'])
y_train["date"] = pd.to_datetime(y_train["date"])

merged_train = pd.merge(y_train, X_train, on='date', how='inner')

# Print the merged DataFrame
print(f'{merged_train}')

# Dataframe Cleaning: test data
X_test = X_test.drop(columns=['Unnamed: 0', 'location_id'])
X_test["date"] = pd.to_datetime(X_test["date"])

y_test = y_test.drop(columns=['Unnamed: 0',  'location_id'])
y_test["date"] = pd.to_datetime(y_test["date"])

merged_test = pd.merge(y_test, X_test, on='date', how='inner')
print(f'{merged_test}')

# Append test and merged training data 
merged_all = pd.concat([merged_train, merged_test], axis=0)

# file_path = "Data/1 - Savanna Preserve/1_y_train.csv"
sample_size_days = 21
sample_size = sample_size_days*24
forecast_size_days = 7
forecast_size = forecast_size_days*24

sample_size = sample_size+forecast_size

# Extracting array of merged data from sample size and focast window with moving window
merged_all_data = create_3d_array_rollingwindow(merged_all, sample_size, forecast_size)

# Splitting all data into training, testing and validation 
percent_idx = int(np.shape(merged_all_data)[0]*0.8) # 80%
train = merged_all_data[:percent_idx]
test = merged_all_data[percent_idx:]

# ---- Begin config managment ---- # 
scaler = StandardScaler().fit(train.reshape(-1, train.shape[-1]))
train_scaled = scaler.transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)
test_scaled = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)

print("train_scaled.shape", train_scaled.shape)
print("test_scaled.shape", test_scaled.shape)


with open("DiffusionStuff/configs/conditionalForecastV2.json") as f:
    data = f.read()

config = json.loads(data)

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
