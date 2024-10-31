# Created on 27/09/2021 by Lewis Dickson

# This script finds data as close as possible to the requested data for the optimiser.
# If there is more than one data point at a request position a random path is chosen from the valid shots

# ----- Overview ----- #
# 1) Fed requested values of experimental parameters from the optimiser
# 2) Finds shots with matching variables
# 3) Choses one of avaliable shots at random
# 4) Returns path to lanex image corresponding to this shot
# ----- Overview ----- #

#=============================================================================
# Imports
#=============================================================================

import pandas as pd
import numpy as np
import sys
import shutil
import os
from importlib import reload
from CallScripts import LDfunctionspy3 as funcs
funcs = reload(funcs)
import matplotlib.pyplot as plt

#=============================================================================
# Funcs
#=============================================================================

# dictionary class with easy add key:value pair feature
class my_dictionary(dict):
    # __init__ function
    def __init__(self):
        self = dict()
    # Function to add key:value
    def add(self, key, value):
        self[key] = value

# Loads the log file with the shot data into a pandas data frame
def load_logfile(ppath):
    filename = funcs.only_file_with_ext(ppath)
    ext = filename.split('.')[1]
    if 'xlsx' in ext:
        print(f'Shot Data Excel log file detected: loading accordingly')
        dF_log = pd.read_excel(ppath)
    elif 'csv' in ext:
        print(f'Shot Data csv log file detected: loading accordingly')
        dF_log = pd.read_csv(ppath)
    return(dF_log)

# Loads the shot list and experimental parameters into a pandas dataframe for easier searching etc
def df_from_shotlist(ppath):
    # Checks if the file is excel or csv and loads to pandas dataframe accordingly
    filename = funcs.only_file_with_ext(ppath)
    ext = filename.split('.')[1]
    if 'xlsx' in ext:
        print(f'Excel file detected: loading accordingly')
        dF = pd.read_excel(ppath)
    elif 'csv' in ext:
        print(f'csv file detected: loading accordingly')
        dF = pd.read_csv(ppath)
    else:
        raise ValueError('shot file must be in .csv or .xlsx format - please verify file type')
    return(dF)

# Reduces a dF down to a smaller dF by taking only the columns with names in the var_list
def reduce_dF(dF,var_list,run_header,shot_header):
    copy_varlist = var_list.copy()
    copy_varlist.append(run_header)
    copy_varlist.append(shot_header)
    new_dF = dF[copy_varlist].copy()
    return new_dF

# Same as reduce_dF but reduces to the match_str column which contains all date, run and shot info
def reduce_dF_matchstr(dF, var_list ,matchstr_header):
    copy_varlist = var_list.copy()
    copy_varlist.append(matchstr_header)
    new_dF = dF[copy_varlist].copy()
    return new_dF

# Loads the excel file which holds all the experimental parmaeters and returns a dictionary
def load_shotdata(ppath):
    pass

# dF.to_dict returns a dictionary of dictionaries so that all of the entries are enumerated by their row position in the pandas dataframe. We are only interessted in the final entry however so we find all the keys and take the greatest value and then index this value.
def last_dict_of_dict_val(dict_of_dicts):
    new_dict = my_dictionary()
    top_level_keys = funcs.getkeys(dict_of_dicts)
    for key in top_level_keys:
        dict_list = dict_of_dicts[key]
        keys = funcs.getkeys(dict_list)
        max_key = np.max(keys)
        max_entry = dict_list[max_key] #converting all values to float
        if isinstance(max_entry,str) and ',' in max_entry: #removing ','
            max_entry = max_entry.replace(',','')
        max_entry = float(max_entry)
        new_dict.add(key,max_entry)
    return(new_dict)

# Loads the latest data from the optimiser CSV and returns a dictionary with all the requested variables and their desired values
def load_demanded_vars(an_optimser_csv_path):
    dF_optim = pd.read_csv(an_optimser_csv_path)
    dF_lastline = dF_optim.iloc[-1:]
    raw_dF_dict = dF_lastline.to_dict() #produces dict of dict with seperate keys for each row and the top level key for the header name
    requested_dict = last_dict_of_dict_val(raw_dF_dict)
    return(requested_dict, dF_optim)

# This function finds the closest value in terms of percentage difference between all the requested variables and the avaliable data. This is done since the scaling on the variables can be very different (i.e jetx and compressor seperation)
def find_closest_values():
    pass

def find_valid_shot(shot_info_dict, request_dict):
    request_keys = funcs.getkeys(request_dict)
    for variable_name in request_keys:
        request_val = request_dict[variable_name]

# To find the nearest value we find the one with the smallest relative percentage since each variable scales differently
def nearest_val_perc(dF, target_val):
    dF_diff = np.abs(dF - target_val) #need abs to make sure the we find the closest to zero
    dF_ret = np.abs(dF_diff*100/dF) # calculating percentage difference - absolute again incase any parameters are negative such as focal position
    return(dF_ret)

# This function handles the dFs and selects the shot which should be loaded based on the optimiser requested values
import matplotlib.pyplot as plt
def find_matching_shot(shot_info_dict, request_dict, exp_params, matchstr_header):
    dF_lists = []
    print(request_dict)
    for key in exp_params:
        key_dF = shot_info_dict[key]
        request_val = request_dict[key]
        diff_dF = nearest_val_perc(key_dF,request_val)
        dF_lists.append(diff_dF)
    dF_sumlists = np.zeros(len(diff_dF))
    for list in dF_lists[:]:
        dF_sumlists = dF_sumlists + list
    selected_shot_idx = dF_sumlists.argmin()
    print(f'selected_shot_idx = {selected_shot_idx}')
    selected_shot = shot_info_dict.iloc[[selected_shot_idx]][matchstr_header].values[0]
    return(selected_shot)

# Takes the output of find_matching_shot and creates a path to the data
def create_shot_path(espec_data_path, rrun, sshot):
    load_path = espec_data_path + str(rrun).zfill(4) +'_'+ str(sshot).zfill(4) + '_Lanex.tif'
    return(load_path)

def copy_lanex_data(lanex_file, lanex_copy_loc):
    shutil.copy2(lanex_file, lanex_copy_loc)
    return

# Returns a dict with the electron parameters for a given matchstr
def extract_shot_vals(dF, matchstr_header, matchstr, requested_e_params):
    out_dict = {}
    for requested_eparam in requested_e_params:
        print(matchstr)
        # eparam = dF.loc[dF[matchstr_header] == matchstr, requested_eparam]
        df_eparam = dF.loc[dF[matchstr_header] == matchstr][requested_eparam].values[0]
        out_dict[requested_eparam] = df_eparam
    return out_dict

def get_real_vals(shot_dF, selected_matchstr, matchstr_header, var_names):
    out_dict = {}
    for var_name in var_names:
        # eparam = dF.loc[dF[matchstr_header] == matchstr, requested_eparam]
        df_expparam = shot_dF.loc[shot_dF[matchstr_header] == selected_matchstr][var_name].values[0]
        out_dict[var_name] = df_expparam
    return out_dict







#
