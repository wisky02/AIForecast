# Created on 24/09/2021 by Lewis Dickson

# This script handles the saving functions for the electron data and for conversion of raw data into dictionaries etc

#=============================================================================
# Imports
#=============================================================================

import numpy as np
import pandas as pd
import pickle
import os
import time
from PIL import Image
import matplotlib.pyplot as plt

#=============================================================================
# Saving Functions
#=============================================================================

# Create dictionary class for saving run dict
class my_dictionary(dict):
    # __init__ function
    def __init__(self):
        self = dict()
    # Function to add key:value
    def add(self, key, value):
        self[key] = value

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# Create a folder name with a new numbering index to seperate each batch
def create_batch_folder_number(optimiser_csv_path, batch_folder_name):
    all_dirs = get_immediate_subdirectories(optimiser_csv_path)
    batch_num_list = []
    for ddir in all_dirs:
        print(ddir)
        if batch_folder_name in ddir:
            batch_num = ddir.split('_')[-1]
            batch_num_list.append(int(batch_num))

    if len(batch_num_list) > 0:
        new_batch_num = np.max(batch_num_list) + 1
    else:
        new_batch_num = 0
    ret_name = f'{batch_folder_name}_{new_batch_num}'
    return ret_name


# Creates a .csv for the optimiser where the labels are always in the same order
def multikernel_created_optim_csv(csvfilename, csv_path, exp_vars, yVal_str, yVal_err_str, matchstr_header, kernel_choice):
    headers = my_dictionary()
    # headers.add('shot_num',[])
    headers.add(matchstr_header, [])
    # if 'CellFocPos' in exp_vars:
    #     headers.add('CellFocPos',[])
    # if 'GasPressure_LogFile' in exp_vars:
    #     headers.add('GasPressure_LogFile',[])
    # if 'compressor_sep' in exp_vars:
    #     headers.add('compressor_sep',[])
    for var in exp_vars:
        headers.add(var,[])
    headers.add(yVal_str,[])
    headers.add(yVal_err_str,[])

    # NOTE: etc for the other variables to mainitain a readable order
    # Creating data frame to then create .csv WARNING currently produces .xslx
    dataF = pd.DataFrame.from_dict(headers)
    # write_to = pd.ExcelWriter(csv_path, engine = 'openpyxl', mode='w') # create and xlsxwriter
    # Defining first file
    only_file_name = csvfilename.split('.')[0]
    datetimenow = time.strftime("%Y%m%d-%H%M%S")
    file_name = only_file_name + '_'+ datetimenow + '.csv'
    full_path = csv_path + file_name
    if os.path.exists(full_path):
        _,_,counter = file_name.split('_')
        int_counter = int(counter.split('.')[0])
        file_name = f'shot_data_{str(int_counter + 1)}_{kernel_choice}.csv'
        csv_path =  full_path = csv_path + '\\' + file_name
    else: # case where the file doesn't exist so we set the file as the first iteration
        fname = os.path.basename(full_path)
        f_path = os.path.dirname(full_path)
        fname_split = fname.split('.')[0]
        extension = fname.split('.')[1]
        csv_path = f'{f_path}//{fname_splitp}_{kernel_choice}.{extension}'
        
    full_path = csv_path
    print(full_path)
    dataF.to_csv(full_path, index=False)
    # write_to.save() #close and output file
    return(full_path, headers)

def create_optim_meta_data_readme(optimiser_csv_path,var_names,start_vals,   var_scales,var_bounds,y_val_header,y_val_err_header,use_BO,maximise_bool,num_consequitve_searches,num_rand_searches,num_trials, max_iterations, requested_e_params, merit_func_expression,batch_folder_name):
    with open(f'{optimiser_csv_path}//optimisation_settings.txt', "w") as dump_file:
        dump_file.writelines('#Dump file for optimiser settings\n\n')
        # Variable definition dump
        dump_file.writelines('#Variable Definitons\n')
        dump_file.writelines(f'var_names = {var_names} # name of variables to optimise\n')
        dump_file.writelines(f'start_vals = {start_vals} # values to start the optimisation at - not used in current implementation (start with random search)\n')
        dump_file.writelines(f'var_scales = {var_scales} # Scaling currently left to optimiser\n')
        dump_file.writelines(f'var_bounds = {var_bounds} # min and maximum values for each variable (list of list)\n')
        dump_file.writelines(f'y_val_header = {y_val_header} # column header representing merit function return\n')
        dump_file.writelines(f'y_val_err_header = {y_val_err_header} # column header representing error on merit function return\n\n')
        # Optimiser definition dump
        dump_file.writelines('# Optimiser Definitions\n')
        dump_file.writelines(f'use_BO = {use_BO} # If True runs bayesian optimisation otherwise runs random search - if false will only complete <num_rand_searches> searches\n')
        dump_file.writelines(f'maximise_bool = {maximise_bool} # If true maximises mertit function if false minimises\n')
        dump_file.writelines(f'num_consequitve_searches = {num_consequitve_searches} # How many loops of the full optimisation to perform for statistics\n')
        dump_file.writelines(f'num_rand_searches = {num_rand_searches} # (int) number of random searches to begin building prior\n')
        dump_file.writelines(f'num_trials = {num_trials} # (int) number of reruns to average over\n')
        dump_file.writelines(f'max_iterations = {max_iterations} # (int) number of reruns to average over\n')
        dump_file.writelines(f'requested_e_params = {requested_e_params} # Loaded electron parameters for the merit function\n')
        dump_file.writelines(f'merit_func_expression = {merit_func_expression} # Merit function definition evaluated with eval()\n\n')
        dump_file.writelines(f'Further Metadata\n')
        dump_file.writelines(f'batch_folder_name = {batch_folder_name} # Merit function definition evaluated with eval()\n')

def multi_kernel_create_optim_meta_data_readme(optimiser_csv_path,var_names,start_vals,   var_scales,var_bounds,y_val_header,y_val_err_header,use_BO,maximise_bool,num_consequitve_searches, num_rand_searches, num_trials, max_iterations, requested_e_params, merit_func_expression, batch_folder_name, kernel_choice):
    # orig_dump_path =  f'{optimiser_csv_path}//optimisation_settings.txt'
    # if os.path.isfile(orig_dump_path):
    #
    #     dump_path =
    # else:
    #     dump_path = orig_dump_path
    dump_path = f'{optimiser_csv_path}//optimisation_settings_{kernel_choice}.txt'
    with open(f'{dump_path}', "w") as dump_file:
        dump_file.writelines('#Dump file for optimiser settings\n\n')
        # Variable definition dump
        dump_file.writelines('#Variable Definitons\n')
        dump_file.writelines(f'var_names = {var_names} # name of variables to optimise\n')
        dump_file.writelines(f'start_vals = {start_vals} # values to start the optimisation at - not used in current implementation (start with random search)\n')
        dump_file.writelines(f'var_scales = {var_scales} # Scaling currently left to optimiser\n')
        dump_file.writelines(f'var_bounds = {var_bounds} # min and maximum values for each variable (list of list)\n')
        dump_file.writelines(f'y_val_header = {y_val_header} # column header representing merit function return\n')
        dump_file.writelines(f'y_val_err_header = {y_val_err_header} # column header representing error on merit function return\n\n')
        # Optimiser definition dump
        dump_file.writelines('# Optimiser Definitions\n')
        dump_file.writelines(f'use_BO = {use_BO} # If True runs bayesian optimisation otherwise runs random search - if false will only complete <num_rand_searches> searches\n')
        dump_file.writelines(f'kernel_choice = {kernel_choice} # Defines the kernel used for the Gaussian Process\n')
        dump_file.writelines(f'maximise_bool = {maximise_bool} # If true maximises mertit function if false minimises\n')
        dump_file.writelines(f'num_consequitve_searches = {num_consequitve_searches} # How many loops of the full optimisation to perform for statistics\n')
        dump_file.writelines(f'num_rand_searches = {num_rand_searches} # (int) number of random searches to begin building prior\n')
        dump_file.writelines(f'num_trials = {num_trials} # (int) number of reruns to average over\n')
        dump_file.writelines(f'max_iterations = {max_iterations} # (int) number of reruns to average over\n')
        dump_file.writelines(f'requested_e_params = {requested_e_params} # Loaded electron parameters for the merit function\n')
        dump_file.writelines(f'merit_func_expression = {merit_func_expression} # Merit function definition evaluated with eval()\n\n')
        dump_file.writelines(f'Further Metadata\n')
        dump_file.writelines(f'batch_folder_name = {batch_folder_name} # Merit function definition evaluated with eval()\n')

# Function to write a value to the last line of a pandas dF
def write_val_to_dF_last_line(dF,val,col_name):
    dF.iloc[-1, dF.columns.get_loc(col_name)] = val
    return(dF)

# Saving function for the y and y_err into the optimisation .csv
def save_val_to_optim_csv(val, col_name, optim_csv_path):
    optim_dF = pd.read_csv(optim_csv_path)
    optim_dF = write_val_to_dF_last_line(optim_dF,val,col_name) #adding val to dataframe
    unnamed_cols = [col for col in optim_dF.columns if 'Unnamed' in col] # Removing the padded column 'Unnamed' that is added by default when creating a pandas dataframe to .csv
    for col_head in unnamed_cols:
        del optim_dF[col_head]
    optim_dF.to_csv(optim_csv_path, index=False) #sending dF to csv
    return

def update_optimsier_to_real_vals(optim_path, real_vals_dict):
    # Autodetecting file type
    file_ext = (os.path.basename(optim_path)).split('.')[-1]
    if 'csv' in file_ext:
        dF = pd.read_csv(optim_path)
    elif 'xlsx' in file_ext or 'xls' in file_ext:
        dF = pd.read_excel(optim_path)
    else:
        raise ValueError('Invalid file type for optmiser file')

    for kkey in [*real_vals_dict]: # updating final row as this is the value that has just been written (NOTE: need to check if this actually improves the maxima finding or if it )
        # print(f'before update = {dF.iloc[-1, dF.columns.get_loc(kkey)]}')
        dF.iloc[-1, dF.columns.get_loc(kkey)] = real_vals_dict[kkey]
        # print(f'after update = {dF.iloc[-1, dF.columns.get_loc(kkey)]}')
    return

#=============================================================================
# Loading Functions
#=============================================================================

def load_analysed_img_txt(ppath):
    with open(ppath) as f:
        lines = f.readlines()
        shot_list = [line.rstrip() for line in lines]
    # WARNING: if memory error instead use:
    # shot_list = []
    # while open(ppath) as f:
    #     for line in file:
    #         shot_list.append(line.rstrip())
    return shot_list

# Loads the .txt calibration file and outputs lists contraining the interpolated variabls
def load_calibrationfile(calib_path, skip_rows_int):
    print(calib_path)
    grid = np.loadtxt(calib_path,skiprows = skip_rows_int, delimiter =' ')
    num_entries = len(grid[:,1])
    pos = np.zeros(num_entries)
    energy = np.zeros(num_entries)
    path_length = np.zeros(num_entries)
    angle_to_lanex = np.zeros(num_entries)
    for i in range(num_entries):
        pos[i] = grid[i,0]
        energy[i] = grid[i,1]
        path_length[i] = grid[i,2]
        angle_to_lanex[i] = grid[i,3]
    return(pos, energy, path_length, angle_to_lanex)


# Outputs lanex as a numpy array
def load_lanex_to_array(ppath,use_plt_imread):
    ## NOTE: applies any array type conversion necessary HERE!
    if use_plt_imread: # using matplotlib for 16 bit tifs
        loaded_lanx = plt.imread(ppath) # creates uint16 np array
    else: #using PIL library for 16 bit png
        loaded_lanx = np.asarray(Image.open(ppath))
    return(loaded_lanx)

# Finds the file creation time
def find_file_creation_time(fpath):
    file_time = datetime.datetime.fromtimestamp(Path(fpath).stat().st_ctime)
    return(file_time)

def extract_data_time(ppath):
    file_time = find_file_creation_time(ppath)
    return(file_time)
