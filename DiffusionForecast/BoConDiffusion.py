# Created on 23/10/2024 by Lewis Dickson 

#===========================================
# User Settings 
#===========================================

dataset = r"1 - Savanna Preserve"
# dataset = "2 - Clean Urban Air"
# dataset = "3 - Resilient Fields"


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
from DiffusionStuff.utils.util import get_mask_bm_forecasting

# sys.path.append('../')
import json
from DiffusionStuff.imputers.SSSDS4Imputer import SSSDS4Imputer
from sklearn.metrics import mean_squared_error 
from statistics import mean

#===========================================
# Functions
#===========================================

def create_3d_array(data, 
                    sample_size_days # 
                    ):

    """
    asdf
    """

    # Remove unnecessary columns
    data = data.drop(columns=['date', 'location_id', 'Unnamed: 0'], errors='ignore')
    
    # Calculate the number of samples and trim based on rounding
    num_samples = len(data) // sample_size_days
    
    # Trim the data if the sample size doesn't divide evenly into the length of the dataset
    trimmed_length = num_samples * sample_size_days
    data = data.iloc[-trimmed_length:]
    
    # Reshape the data into the desired 3D array
    data_array = data.values.reshape(num_samples, sample_size_days, -1)
    
    return data_array

# TODO: Create shifting window sampling with extra var forecasting window

# def create_3d_array_rollingwindow(data, 
#                     sample_size,
#                     forecast_window_length 
#                     ):

#     """
#     asdf
#     """
#     import math

#     # Remove unnecessary columns
#     data = data.drop(columns=['date', 'location_id', 'Unnamed: 0'], errors='ignore')
    
#     # Calculate the number of samples and trim based on rounding
#     num_samples = (len(data) - (sample_size-forecast_window_length))// forecast_window_length + 1

#     overhang = math.modf((len(data) - (sample_size-forecast_window_length))/ 
#     forecast_window_length)[0]

#     overhang_trim_idx = int(overhang*forecast_window_length)
#     print("overhang_trim_idx", overhang_trim_idx)
#     # Trim the data if the sample size doesn't divide evenly into the length of the dataset
#     # trimmed_idx = (num_samples - sample_size) // forecast_window_length 
#     # overshoot = len(data) - (num_samples*forecast_window_length + (sample_size-forecast_window_length))

#     # trimmed_idx = sample_size + overshoot
    
#     trimmed_idx = overhang_trim_idx

#     print("trimmed index", trimmed_idx)

#     data = data.iloc[trimmed_idx:]

#     print("val", len(data)/num_samples)
#     data_array = np.zeros((num_samples, sample_size, len(data.columns)))
#     for idx in range(num_samples):
        
#         print(data.iloc[idx*forecast_window_length: 
#                         idx*forecast_window_length + sample_size].shape)

#         data_array[idx, :, :] = data.iloc[idx*forecast_window_length: 
#                                 idx*forecast_window_length + sample_size]
#                                           #reshape(1, sample_size, -1)
#     # Extracting rolling window of values 
#     # data_array = [ d]

#     return data_array




def create_3d_array_rollingwindow(data, 
                                  sample_size,
                                  forecast_window_length):
    """
    Splits the data into rolling windows of specified size, trimming the start
    so the values at the end are included in the windows.
    """
    debug = False
    # Remove unnecessary columns
    data = data.drop(columns=['date', 'location_id', 'Unnamed: 0'], errors='ignore')
    
    # Calculate the number of samples (rolling windows)
    num_samples = (len(data) - (sample_size - forecast_window_length)) // forecast_window_length

    # Calculate the number of rows to trim at the beginning
    overhang = (len(data) - (num_samples * forecast_window_length + (sample_size - forecast_window_length)))

    if overhang > 0:
        # Trim from the start, so the end values are preserved
        trimmed_idx = overhang
        if debug:
            print("overhang_trim_idx:", overhang, " | trimmed_idx (rows to drop at the start):", trimmed_idx)
        data = data.iloc[trimmed_idx:]
    else:
        print("No overhang to trim.")

    # Update num_samples based on the trimmed data
    num_samples = (len(data) - (sample_size - forecast_window_length)) // forecast_window_length

    # Initialize the 3D array: (num_samples, sample_size, number_of_features)
    data_array = np.zeros((num_samples, sample_size, data.shape[1]))

    # Populate the 3D array with rolling window slices
    for idx in range(num_samples):
        start_idx = idx * forecast_window_length
        end_idx = start_idx + sample_size
        
        if debug:
            print(f"Processing window {idx+1}: Start={start_idx}, End={end_idx}, Shape={data.iloc[start_idx:end_idx].shape}")
        
        # Fill each window into the 3D array
        data_array[idx, :, :] = data.iloc[start_idx:end_idx].values


    return data_array

def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate, # <optimise>
          use_model,
          only_generate_missing, # N/A
          missing_k, 
          batch_size, 
          forecast_window, # points to forecast
          forecast_cols, # idx of columns we want to forecast - defauly [0,1] 
          dummy_columns_for_forecast_window # idx of conditional cols, hidden during forecast - i.e day of week CAN be seen - sets to zero
          ):
    
    """
    To optimise: 
    - learning_rate : O(1e-4)
    - "T" : number of diffusion steps O(10)
    - "beta_0" : amount of noise at start of linear scheduel O(0.01)
    - "beta_T" : amount of noise at end of linear scheduel O(0.01)
    """


    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"]
                                              )
    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()


    net = SSSDS4Imputer(**model_config).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    training_data = train_scaled
    
    num_groups = training_data.shape[0] // batch_size
    training_data = np.split(training_data, num_groups, 0)
    
    training_data = np.array(training_data)
    print(training_data.shape)
    training_data = torch.from_numpy(training_data).float().cuda()
    print('Data loaded')

    
    
    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1: # TODO: can alter for early stopping
        for batch in training_data: # 40 N, 65 K, 200 L  (batch into the model)
                # batch = torch.from_numpy(batch)
                
                #have one mask for generation and one mask for loss so we optimise on the correct data but dont leak answers to the model
                batch = batch.to('cuda:0').float()
                transposed_mask = get_mask_bm_forecasting(batch[0,:,:], forecast_window, forecast_cols)
                
   
                # Replacing conditionals with dummy values to block forecasting on conditionals
                for col in dummy_columns_for_forecast_window:
                    batch[:, -forecast_window:, col] = 0

                # Repeating mask over each sample
                mask = transposed_mask.permute(1, 0)
                mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
                
                # Invert mask for loss function calculation
                loss_mask = ~mask.bool()
        
                batch = batch.permute(0, 2, 1)   
    
                assert batch.size() == mask.size() == loss_mask.size()

                # back-propagation
                optimizer.zero_grad()
                X = batch, batch, mask, loss_mask
                loss = training_loss_replace(net, nn.MSELoss(), X, diffusion_hyperparams,
                                    only_generate_missing=only_generate_missing)
                
                loss.backward()
                optimizer.step()

                if n_iter % iters_per_logging == 0:
                    print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

                # save checkpoint
                if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                    checkpoint_name = '{}.pkl'.format(n_iter)
                    torch.save({'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(output_directory, checkpoint_name))
                    print('model at iteration %s is saved' % n_iter)

                n_iter += 1


#===========================================
# Main
#===========================================

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
train_config = config["train_config"]  # training parameters

gen_config = config['gen_config']

global trainset_config
trainset_config = config["trainset_config"]  # to load trainset

global diffusion_config
diffusion_config = config["diffusion_config"]  # basic hyperparameters

global diffusion_hyperparams
diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

global model_config
if train_config['use_model'] == 0:
    model_config = config['wavenet_config']
elif train_config['use_model'] == 1:
    model_config = config['sashimi_config']
elif train_config['use_model'] == 2:
    model_config = config['wavenet_config'] 
    