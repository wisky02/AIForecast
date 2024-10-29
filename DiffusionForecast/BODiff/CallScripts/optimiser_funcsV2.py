# Created on 15/10/2021 by Lewis Dickson

# This script loads and calls the required functions from the different optimiser scripts

# Library imports
import sys
# sys.path.append(r'../OptimiserFuncs/OptimiserV2/scripts//') # path to the functions blackbox, gp_opt etc
from importlib import reload
from OptimiserCallScripts.gp_opt import BasicOptimiser
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from OptimiserCallScripts.BlackBox import Ndim_multinorm, Ndim_multinorm_multipeak
from time import sleep
import numpy as np
import os
import pandas as pd
import time
import signal

# Function imports
# import BlackBox
# import optimiser
# import var_dict
# import take_shot
#
# BlackBox = reload(BlackBox)
# optimiser = reload(optimiser)
# var_dict = reload(var_dict)
# take_shot = reload(take_shot)

## Create the var_dict with an arbitrary number of variables
## ---->>>> Required layouts for the input variables
# var_names = ['jetx', 'gratsep', ..., 'dazzler_4']
# var_bounds = [[x1_lower, x1_upper],...,[xn_lower, xn_upper]]
# start_vals = [x1_startval,...,xn_startval]
# var_scales = [x1_scale, ... , xn_scale]

# Short function to ask the user if they are SURE they want to exit and allow the optimiser thread to be joined and close
def exit_gracefully(signum, frame):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)

    try:
        if input("\nReally quit? (y/n)> ").lower().startswith('y'):
            sys.exit(1)

    except KeyboardInterrupt:
        print("Ok ok, quitting")
        sys.exit(1)

    # restore the exit gracefully handler here
    signal.signal(signal.SIGINT, exit_gracefully)


def create_var_dict(var_names, start_vals, var_scales, var_bounds):
    # Create dictionary class for saving run dict
    class auto_dict(dict):
        # __init__ function
        def __init__(self):
            self = dict()
        # Function to add key:value
        def add(self, key, value):
            self[key] = value

    var_dict = auto_dict()
    var_dict.add('name',var_names)
    var_dict.add('start',start_vals)
    var_dict.add('scale',var_scales)
    var_dict.add('bounds',var_bounds)

    # Calculating the number of variables to optimise
    NDIMS = len(var_dict['name'])
    return(var_dict,NDIMS)

# begins initialising and running the optimiser with the variables which are defined
def initial_run_optimiser(var_dict, NDIMS, path_to_csv, N_rand_selected):
    #Set up optimiser

    chosen_kernel = Matern() + WhiteKernel(noise_level_bounds = (1e-30,1e5))
    BO = BasicOptimiser(NDIMS, kernel=chosen_kernel, sample_scale=1, scale=var_dict['scale'],
            bounds=var_dict['bounds'], maximise_effort=100,fit_white_noise=True, xi='auto', explore = True)

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
        if current_shot <= N_rand_selected:
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

#  Does the same as initial_run_optimiser but in a class
class RunOptimiser:
    """
    Class to operate the optimiser

    * Must initialise with var_dict.
    * Can also initialise with the number of random steps before the
      optimisation begins.
    * Can specify total number of steps to optimize for, beyond which further
      coordinate requests will return the optimum coordinates.
    * Can specify a custom kernel (default is Matern).

    Methods:
    * ask() - get the next coordinates to measure
    * tell(X, y, y_err) - tell the optimiser the result (y +- y_err) obtained
                          at coordinate X.
    * info() - returns tuple of useful parameters
    """
    def __init__(self, var_dict, N_rand = 3, N_opt = 1e5, kernel = False,
                    shots_per_setting = 1, xi = False, normalise_Y =True):
        self.var_dict = var_dict
        self.N_rand = N_rand
        self.N_opt = N_opt
        self.shots_per_setting = shots_per_setting
        self.normalise_Y = normalise_Y

        if kernel == False:
            chosen_kernel = Matern()# + WhiteKernel() #noise_level_bounds = (1e-5,1e5))
        else:
            chosen_kernel = kernel
        NDIMS = len(self.var_dict['scale']) #could be any variable of them
        self.BO = BasicOptimiser(NDIMS, kernel=chosen_kernel, sample_scale=1, scale=var_dict['scale'],
                bounds=var_dict['bounds'], maximise_effort=100,fit_white_noise=False,
                xi=xi, explore = True, normalise_Y = True)

        self.current_shot = 1

    def tell(self, X, y, y_err):
        self.BO.tell(X, y, y_err)
        #best_coord, best_val = self.BO.optimum()
        #print('Predicted optimum:',best_coord, best_val)
        self.current_shot += 1

    def ask(self):
        var_dict = self.var_dict
        # returns next coordinates to measure
        x_test = [] #next coordinates to measure
        if self.current_shot%self.shots_per_setting == 0 or self.current_shot==1:
            if self.current_shot <= self.N_rand:
                # generate random coordinates, clipped by bounds
                for i, var in enumerate(var_dict['name']):
                    val_raw = var_dict['start'][i] +np.random.normal(0,var_dict['scale'][i] )
                    bound = var_dict['bounds'][i]
                    val = np.clip(val_raw,bound[0], bound[1] )
                    x_test.append(val)
            elif self.current_shot <= self.N_opt:
                #Ask optimiser where to measure next
                x_test =self.BO.ask(1e-50)
            else:
                best_coord, best_val = self.BO.optimum()
                x_test = best_coord
        else:
            x_test = self.new_x_test
        #return coordinates as a dict
        return_dict = {}
        for i, variable in enumerate(self.var_dict['name']):
            return_dict[variable] = x_test[i]
        self.new_x_test = x_test
        self.new_return_dict = return_dict
        return return_dict, x_test
    def info(self):
        #returns some metrics from the optimiser
        best_coord, best_val = self.BO.optimum()
        xi = self.BO.xi_calc
        return best_coord, best_val, xi
    def get_df(self):
        df = pd.DataFrame(self.BO.x_samples,
                            columns = self.var_dict['name'],
                            )
        df['yVal'] =  self.BO.y_samples_unnorm
        df['yVal_err'] = self.BO.y_err_samples_unnorm
        df['Shot'] = range(1, len(df)+1)
        df = df.set_index('Shot')
        return df

class RunScanner:
    """
    Class to facilitate parameter scans

    * Initialise with lists: variables, starts, ends, and N_steps: these must all be the same length
    * Generates a dataframe of coordinates (accessible via the atribute shot_df)
    * The variables are scanned in reverse order
    * Method 'ask' returns the next coordinates (returned as a dictionary)

    """
    def __init__(self, variables, starts, ends, N_steps, N_shots_per_step):
        self.variables = variables
        self.NDIMS= len(variables)
        # generate coordinate dictionary
        l = []
        for i, var in enumerate(variables):
            a = np.linspace(starts[i], ends[i], N_steps[i])
            a_full = np.sort(np.concatenate([a for j in range(N_shots_per_step[i])]))
            print(a_full)
            if starts[i] < ends[i]:
                l.append(a_full)
            else:
                l.append(a_full[::-1])
        coords = np.vstack([i.ravel() for i in np.meshgrid(*l, indexing = 'ij')])
        coords_dict = {}
        for i, var in enumerate(variables):
            coords_dict[var] = coords[i]


        # make a dataframe for easy access
        self.shot_df = pd.DataFrame(coords_dict)
        self.shot_df['Shot'] = [i+1 for i in range(len(coords_dict[var]))]
        self.shot_df = self.shot_df[['Shot', *variables]]
        self.shot_df.set_index('Shot', inplace = True)
        self.current_shot = 1
        self.Nshots = len(self.shot_df) +1
        self.finished_flag = False

    def ask(self, shot=False):
        # can either ask for the next shot, or request a specific shot
        shot_flag = 1
        if shot == False:
            shot_flag = 0
            shot = self.current_shot
        if shot > self.Nshots:
            shot = self.Nshots
            self.finished_flag = True
        row = self.shot_df.loc[shot]
        if not shot_flag:
            self.current_shot +=1
        return row.to_dict(), row.to_list()

    def tell(self,X = np.nan, y = np.nan, y_err= np.nan, shot = False):
        if self.finished_flag:
            return
        #Optional tell method for adding
        if shot == False:
            shot = self.current_shot

        self.shot_df.at[shot, 'yVal'] = y
        self.shot_df.at[shot, 'yVal_err'] = y_err
    def info(self):
        return

    def get_df(self):
        return self.shot_df













def take_shot(var_dict,path_to_csv):
    print('Iniitialising multi peak Blackbox function')
    means = [np.zeros(len(var_dict['name'])),
                3*np.ones(len(var_dict['name']))]
    print('means', means)
    cov_mats = [2*np.ones(len(var_dict['name']))
                , 2*np.ones(len(var_dict['name']))]

    amplitudes = [0.5,1]
    noise = 1

    BB = Ndim_multinorm_multipeak(means, cov_mats, amplitudes, noise = noise)

    #assert os.path.exists(path_to_csv)

    while(True):
        #input("Press Enter to continue...")
        df = pd.read_csv(path_to_csv)
        next_coords = df[var_dict['name']].iloc[-1].values

        #take shot
        result = BB.ask(next_coords)

        #Error bodge
        if noise == 0:
            result_err = result*0.1
        else:
            result_err = noise *0.01
        print('Result = ', result, '+-',result_err )

        #Attach result of shot
        df['yVal'].iat[-1]  = result
        df['yVal_err'].iat[-1] = result_err
        _ = var_dict['name']
        columns = _.copy()
        columns.insert(0, 'shot_num')
        columns.append('yVal')
        columns.append('yVal_err')

        df = df[columns]

        #Save
        df.to_csv(path_to_csv, index = False)

        #watch file for changes
        orig_time = os.path.getmtime(path_to_csv)
        while(orig_time == os.path.getmtime(path_to_csv)):
            time.sleep(1.)
            print(os.path.getmtime(path_to_csv))
            print(orig_time)
