import os
import numpy as np
import torch
import random
from itertools import permutations


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size, GPU_number):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda(GPU_number)


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in, GPU_number):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda(GPU_number)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling(net, 
             size, 
             diffusion_hyperparams, 
             cond, 
             mask, 
             GPU_number,
             only_generate_missing=0, 
             guidance_weight=0,
             ):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size, GPU_number) 

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            if only_generate_missing == 1:
                x = x * (1 - mask).float() + (cond * mask.float())
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda(GPU_number)  # use the corresponding reverse step
            epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            # update x_{t-1} to \mu_\theta(x_t)
            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * std_normal(size, GPU_number)  # add the variance term to x_{t-1}

    return x


def training_loss(net, loss_fn, X, diffusion_hyperparams, only_generate_missing=1):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda(GPU_number)  # randomly sample diffusion steps from 1~T

    z = std_normal(audio.shape)
    if only_generate_missing == 1:
        z = audio * mask.float() + z * (1 - mask).float()
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
        1 - Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    epsilon_theta = net(
        (transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta

    if only_generate_missing == 1:
        return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
    elif only_generate_missing == 0:
        return loss_fn(epsilon_theta, z)
    
    

def training_loss_replace(net, 
                          loss_fn, 
                          X, 
                          diffusion_hyperparams, 
                          GPU_number,
                          only_generate_missing=1,
                          ):
    """
    Same as training lost except we replace the missing values with the conditional values
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda(GPU_number)  # randomly sample diffusion steps from 1~T

    z = std_normal(audio.shape, GPU_number)

    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
        1 - Alpha_bar[diffusion_steps]) * z
    
    noisy_target = (1 - mask.float()) * transformed_X
    cond_target = (mask.float() * cond)

    new_transformed_X = noisy_target + cond_target

    # compute x_t from q(x_t|x_0)
    epsilon_theta = net(
        (new_transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta

    if only_generate_missing == 1:
        return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
    elif only_generate_missing == 0:
        return loss_fn(epsilon_theta, z)
    
    
    
# def training_loss_replace_focus(net, loss_fn, X, diffusion_hyperparams, only_generate_missing=1):
#     """
#     Same as training lost except we replace the missing values with the conditional values
#     """

#     _dh = diffusion_hyperparams
#     T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

#     audio = X[0]
#     cond = X[1]
#     mask = X[2]
#     loss_mask = X[3]

#     B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
#     diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda(GPU_number)  # randomly sample diffusion steps from 1~T

#     z = std_normal(audio.shape)

#     transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
#         1 - Alpha_bar[diffusion_steps]) * z
    
#     noisy_target = (1 - mask.float()) * transformed_X
#     cond_target = (mask.float() * cond)

#     new_transformed_X = noisy_target + cond_target

#     # compute x_t from q(x_t|x_0)
#     epsilon_theta = net(
#         (new_transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta

#     if only_generate_missing == 1:
#         return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
#     elif only_generate_missing == 0:
#         return loss_fn(epsilon_theta, z)
    
    
    
   




def get_mask_rm(sample, k):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))  # lenght of series indexes
    for channel in range(mask.shape[1]):
        perm = torch.randperm(len(length_index))
        idx = perm[0:k]
        mask[:, channel][idx] = 0

    return mask

def get_rm_range(sample, ratio_min, ratio_max):
    #get random value between k_min and k_max
    return 0
    


def get_mask_mnr(sample, k):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


def get_mask_bm(sample, k):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask

def get_mask_pm(sample, k_min, k_max):
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    for channel in range(mask.shape[1]):
        current_idx = 0
        while current_idx < len(length_index):
            k_length = torch.randint(k_min, k_max + 1, (1,)).item()
            end_idx = current_idx + 1 + k_length
            if end_idx <= len(length_index):
                mask[current_idx + 1:end_idx, channel] = 0
                current_idx = end_idx
            else:
                mask[current_idx + 1:, channel] = 0
                break
    return mask



import torch

def get_mask_bm_forecasting(sample, k, columns=None):
    """Get mask of same segments (black-out missing) across channels for forecasting.
    The final k points are removed. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers. If columns is provided, only those columns will have the mask applied."""

    mask = torch.ones(sample.shape)
    
    # Determine the range to be masked
    start_index = mask.shape[0] - k
    end_index = mask.shape[0]

    # If columns are not provided, apply mask to all columns
    if columns is None:
        columns = range(mask.shape[1])

    # Apply the mask to the specified columns
    for channel in columns:
        mask[start_index:end_index, channel] = 0

    return mask

import torch
import random

def get_mask_pm_options(sample, options):
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    for channel in range(mask.shape[1]):
        current_idx = 0
        while current_idx < len(length_index):
            selected_option = random.choice(options)
            k_length = selected_option
            end_idx = current_idx + 1 + k_length
            if end_idx <= len(length_index):
                mask[current_idx + 1:end_idx, channel] = 0
                current_idx = end_idx
            else:
                mask[current_idx + 1:, channel] = 0
                break
    return mask



import torch
import random

# def get_mask_pm_multi_options(sample, options_list):
#     mask = torch.ones(sample.shape)
#     length_index = torch.tensor(range(mask.shape[0]))
    
#     for channel, options in enumerate(options_list):
#         current_idx = 0
#         while current_idx < len(length_index):
#             selected_option = random.choice(options)
#             k_length = selected_option
#             end_idx = current_idx + k_length
#             if end_idx <= len(length_index):
#                 mask[current_idx:end_idx, channel] = 0
#                 current_idx = end_idx
#             else:
#                 mask[current_idx:, channel] = 0
#                 break
                
#     return mask



def get_mask_pm_multi_options(sample, options_list):
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
        
    for channel, options in enumerate(options_list):
        current_idx = 0
        while current_idx < len(length_index):
            selected_option = random.choice(options)
            k_length = selected_option
            end_idx = current_idx + 1 + k_length
            if end_idx <= len(length_index):
                mask[current_idx + 1:end_idx, channel] = 0
                current_idx = end_idx
            else:
                mask[current_idx + 1:, channel] = 0
                break
                
    return mask









def p_norm_fn(f, x, p, w):
    n = f.size(0)
    min_error = float('inf')
    
    for perm in permutations(range(n)):
        permuted_f = torch.zeros_like(f)
        for i, j in enumerate(perm):
            if abs(i - j) <= w:
                permuted_f[j] = f[i]
        error = torch.norm(permuted_f - x, p=p)
        if error < min_error:
            min_error = error
    
    return min_error




def training_loss_pnorm(net, p_norm_fn, X, diffusion_hyperparams, GPU_number, p=4, w=3, only_generate_missing=1):

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda(GPU_number)  # randomly sample diffusion steps from 1~T

    z = std_normal(audio.shape, GPU_number)

    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
        1 - Alpha_bar[diffusion_steps]) * z
    
    noisy_target = (1 - mask.float()) * transformed_X
    cond_target = (mask.float() * cond)

    new_transformed_X = noisy_target + cond_target

    # compute x_t from q(x_t|x_0)
    epsilon_theta = net(
        (new_transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta

    if only_generate_missing == 1:
        # Calculate adjusted p-norm error
        error = p_norm_fn(epsilon_theta[loss_mask], z[loss_mask], p, w)
    elif only_generate_missing == 0:
        # Calculate adjusted p-norm error for entire tensor
        error = p_norm_fn(epsilon_theta, z, p, w)

    return error

import numpy as np

import numpy as np

import torch

def p_norm_error(f, a, p=4):

    # Ensure f and a have the same shape
    if f.shape != a.shape:
        raise ValueError("Tensors f and a must have the same shape")

    # Calculate the p-norm error measure
    p_norm = torch.sum(torch.abs(f - a) ** p) ** (1 / p)

    return p_norm



    
def training_loss_euclid(net, euclidean_distance, X, diffusion_hyperparams, p=4, only_generate_missing=1):

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda(GPU_number)  # randomly sample diffusion steps from 1~T

    z = std_normal(audio.shape)

    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
        1 - Alpha_bar[diffusion_steps]) * z
    
    noisy_target = (1 - mask.float()) * transformed_X
    cond_target = (mask.float() * cond)

    new_transformed_X = noisy_target + cond_target

    # compute x_t from q(x_t|x_0)
    epsilon_theta = net(
        (new_transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta

    if only_generate_missing == 1:
        # Calculate adjusted p-norm error
        error = p_norm_error(epsilon_theta[loss_mask], z[loss_mask], p)
    elif only_generate_missing == 0:
        # Calculate adjusted p-norm error for entire tensor
        error = p_norm_error(epsilon_theta, z, p)

    return error


def create_3d_array(data, 
                    sample_size_days # 
                    ):

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