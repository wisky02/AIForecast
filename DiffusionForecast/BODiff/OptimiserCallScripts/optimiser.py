"""
Basic optimisation script for arbitrary number of variables.

Reads an input CSV file and diagnositc(s) of interest
Outputs the same CSV file with suggested coordinates appended.
"""

from gp_opt import BasicOptimiser
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from time import sleep
import numpy as np
import os
import pandas as pd
import time
from var_dict import *

if __name__ == "__main__":
    #Set up optimiser
    N_rand = 10

    length_scale = []
    length_scale_bounds=[]
    for n in range(0,NDIMS):
        length_scale.append(1)
        length_scale_bounds.append([0.1,5])



    chosen_kernel = Matern() + WhiteKernel(noise_level_bounds = (1e-30,1e5))
    BO = BasicOptimiser(NDIMS, kernel=chosen_kernel, sample_scale=1, scale=var_dict['scale'],
            bounds=var_dict['bounds'], maximise_effort=100,fit_white_noise=False, xi='auto', explore = True)

    current_shot = 1

    while(True):
        time.sleep(1.)
        if current_shot >1:
            #Read reult of previous shot
            df = pd.read_csv(path_to_csv)
            prev_xtest = df[var_dict['name']].iloc[-1].values

            print('Telling optimiser:', prev_xtest,df['yVal'].iloc[-1] )
            BO.tell(prev_xtest, df['yVal'].iloc[-1], df['yVal_err'].iloc[-1])

            _, best_val = BO.optimum()
            print('Predicted optimum:',_, best_val)


        else:
            df = pd.DataFrame({})
            BO.xi_calc = np.nan

        x_test = [] #next coordinates to measure
        if current_shot <= N_rand:
            # generate random coordinates, clipped by bounds
            for i, var in enumerate(var_dict['name']):
                val_raw = var_dict['start'][i] +np.random.normal(0,var_dict['scale'][i] )
                bound = var_dict['bounds'][i]
                val = np.clip(val_raw,bound[0], bound[1] )
                x_test.append(val)

        else:
            #Ask optimiser where to measure next
            x_test =BO.ask(1e-50)
            print('x_test = ', x_test)

        #New row for the dataframe
        newshot = {'shot_num': current_shot}

        for i, var in enumerate(var_dict['name']):
            newshot[var] = x_test[i]
        newshot['yVal'] = np.nan
        newshot['yVal_err'] = np.nan

        newshot['xi'] = BO.xi_calc

        #Save dataframe
        df = df.append(newshot, ignore_index = True)
        df.to_csv(path_to_csv)

        current_shot+=1
        orig_time = os.path.getmtime(path_to_csv)
        print('=====================================')
        while(orig_time == os.path.getmtime(path_to_csv)):
            time.sleep(0.5)
