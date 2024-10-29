# Created on 11/02/2020 by Lewis Dickson - ITFIP - LPGP

# Functions are for python 3

# An mixture of useful funcitons and what they do. Many stolen things in here so if anyone ever reads this thank you Chris Underwood and most of stackcexchange
#
# To use these functions use:
# # --------- My Functions --------- #
# import sys
# sys.path.append(r'C:\Users\lewis\Documents\scripts\useful scripts') #or corresponding path to function file
# import LDfunctionspy3 as funcs
# from importlib import reload
# funcs = reload(funcs)
# # --------- My Functions --------- #

import os
import datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import warnings
import glob
import pickle
#===============================================================================
# dictionary actions
#===============================================================================

# loads and returns infromation from .json file
def loadDictionary(fname):
    with open(fname) as f:
        data = json.load(f)
    return data

def saveDictionary(fname, dict_to_save):
    import json
    with open(fname, 'w') as file:
         file.write(json.dumps(dict_to_save, default=default)) # use `json.loads` to do the reverse


#===============================================================================

# returns the dictionary keys as a full_list
def getkeys(mydict):
    return[*mydict]

# Produces a list of lists for all of the groups of entries in a dictionary
def dict_to_lists(ddict):
    output = list(map(list, (ele for ele in ddict.values())))
    return(output)

#===============================================================================
def set_key_val(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]

#===============================================================================

# returns the number of values (vv) for all of the keys (kk) within a dict
def dict_numvals_for_keys(ddict):
    kk = []
    vv = []
    for k,v in ddict.items():
        kk.append(k)
        vv.append(len(list(filter(None, v))))
    return(kk, vv)

#===============================================================================

# find the key for a given value. note, will return all matches with a warning if greater than one
def find_key_for_val(dict,match):
    key_name = []
    for key, val in mydict.items():
        if val == match:
            key_name.append(key)
    if len(key_name) > 1:
        raise_warning("There are more than one matching keys!")
    return(key_name)
#===============================================================================
# Data Loading
#===============================================================================


def loadmatfile(matpath):
    import scipy.io as spio
    load_mat = spio.loadmat(matpath)
    return(load_mat)

#loads a variable from a matfile
def loadvarmat(matpath, varname):
    import scipy.io as spio
    load_mat = spio.loadmat(matpath)
    var = load_mat[varname]
    return var

def make_pickle(data, ppath):
    with open(ppath, 'wb') as handle:
        pickle.dump(data, handle)#, protocol=pickle.HIGHEST_PROTOCOL)
    return

# unpickles a file
def read_pickle(pickle_path):
    with open(pickle_path, 'rb') as handle:
        b = pickle.load(handle)
    return(b)

#===============================================================================
# Excel/.csv functions
#===============================================================================

#creates a pandas data frame from an excel file - var1/2 must be strings
def excel_to_DF(excel_file_path, var1_nam, var2_nam): #data starts from B2
    import pandas as pd
    excel_data = pd.read_excel(excel_file_path)
    excel_DF = pd.DataFrame(excel_data)
    dir_DF = excel_DF.loc[:,var1_nam]
    relpath_DF = excel_DF.loc[:,var2_nam]
    return(dir_DF, relpath_DF)

def excel_to_dict(excel_file_path, *column_names): #general loading of excel to a dictionary
    import pandas as pd
# -----------------------------
    # Create dictionary class
    class my_dictionary(dict):

        # __init__ function
        def __init__(self):
            self = dict()

        # Function to add key:value
        def add(self, key, value):
            self[key] = value
# -----------------------------
    excel_data = pd.read_excel(excel_file_path)
    excel_DF = pd.DataFrame(excel_data)
    var_dict = my_dictionary()
    for name in column_names:
        var_DF = excel_DF.loc[:,name]
        var_dict.add(str(name),var_DF)
    return(var_dict)

# Produces pandas dataframe from an .csv/.xls file data by colum number selection
def excel_pick_by_colnum_DF(excel_path, col_nums):
    excel_data = pd.read_excel(excel_path)
    excel_DF
    for col in col_nums:
        pass

#===============================================================================
# txt file functions
#===============================================================================

# Loads only certain columns from a .txt format data table
def load_txt(text_path, *col_names):
    grid = np.genfromtxt(text_path, names = True) #loadtxt(full_path,skiprows = 1, delimiter =' ')
    pass

def np_load_txt(file_path):
    import numpy as np
    array = np.loadtxt(file_path)
    return(array)

def nptextload(path,delimiter_string):
    data = np.genfromtxt(path, delimiter = delimiter_string)
    return(data)

#===============================================================================
# Data Frame Functions
#===============================================================================

# Returns the column headers as a list
def dataF_getheaders(dataframe):
    if isinstance(dataframe, pd.DataFrame):
        header_list = dataframe.columns.values.tolist()
    else:
        raise Exception('{} is NOT a Pandas\' dataframe'.format(dataframe))
    return(header_list)

# returns the number of rows for a given dataframe
def dataF_rows(df):
    num_rows = len(df.index)
    return(num_rows)

# Extracts the columns where there is a matching condition in multiple columns
# def extract_multimatch_df(df,keynames_and_keys): #keynames_and_keys should be a dictionary where the keys are the key names and the values are the keys for the data frame i.e keynames_and_keys = {'Data_Run':first_val_to_match,'Data_Shot':second_val_to_match}
#     # Extracting key names and keys
#     key_names = getkeys(keynames_and_keys)
#     # Check the data type in the df
#     truth = []
#     vals = []
#     for name in key_names:
#         # istrue = all(x.is_integer() for x in df.name)
#         # istrue_int = (df[name] % 1  == 0).all()
#         # if istrue_int: #converting values to ints for the comparison
#         #     vals.append(int(keynames_and_keys[name]))
#         # else: #leaving the values as floats
#         # vals.append(int(keynames_and_keys[name]))
#         # print(key_names)
#         print(keynames_and_keys[name])
#         istrue = df[name] == keynames_and_keys[name]
#         truth.append(istrue)
#     if all(truth):
#
#     print(keynames_and_keys)
#     print(df['Run'])
#     new_df = pd.DataFrame(map(lambda k: df[k]==keynames_and_keys[k], keynames_and_keys)).all()
#         # print(df[key_names].fillna(0).astype(int))
#     # new_df = df[(df[key_names].fillna(0).astype(int) == vals).all(1)]
#     return(new_df)


#===============================================================================
# Data Handling
#===============================================================================

# Uses dictionary's rule on single non-duplicate keys to remove duplicate entries
def remove_duplicates_list(list_data):
    return(list(dict.fromkeys(list_data)))

# Finds the indexs of matching values in a list - can then be used to build an idx dict
def value_finder(data_set, match_value):
    return [i for i, x in enumerate(data_set) if x == match_value]

# e.g usage:
#idx_dict = {} # a dictionary to hold the idxs for each slice
# data = data_dict['data']
# unique_data = funcs.remove_duplicates_list(data)
# for i in unique_data:
#     idx_list = value_finder(data, i)
#     idx_dict[str(int(i))] = idx_list

# Returns true if a variable has been localy defined

# def is_local(variable_name):
#     return(if variable_name in locals()) #checking if the value has been assigned

# remove nan from list
def remove_nan_list(data_list):
    cleaned_list = [x for x in data_list if str(x) != 'nan']
    return(cleaned_list)

# Locate the indexes where the nans are
def index_nan(nan_arr):
    nan_arr = np.argwhere(np.isnan(nan_arr))
    return(nan_arr)


#===============================================================================
# Value Handling
#===============================================================================

# rounds a given value, x, to n significant figures
def round_to_n(x, n):
    if n < 1:
        raise ValueError("number of significant digits must be >= 1")
    # Use %e format to get the n most significant digits, as a string.
    format = "%." + str(n-1) + "e"
    as_string = format % x
    return float(as_string)

# rounds a list of data to nearest integer value then returns list
def intround(data):
    intdata = []
    for val in data:
        intval = int(round(val))
        intdata.append(intval)
    return(intdata)

def add_lists(*lists):
    from operator import add
    k = 0
    for list in lists:
        if k == 0:
            return_list = list
        else:
            return_list = (map(add,return_list,list))
    return(return_list)

# NUMERICAL LIST ONLY returns a merged no duplicates,ordered list
def merge_list_nodups(list1,list2):
    resulting_list = list(list1)
    resulting_list.extend(x for x in list2 if x not in resulting_list)
    #ordering list
    resulting_list.sort()
    return(resulting_list)

# Remove duplicate values from a list
def remove_duplicates(list1):
    no_duplicates = list(set(list))
    return(no_duplicates)

#===============================================================================
# Image Analysis
#===============================================================================

# Calculates centroid of a square array
def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def remove_background():
    pass

# sets any negative values to zero. more efficient for large arrays - NOT WORKING
def large_arr_set_neg_zero(arr):
     arr*=(arr<0)
     return(arr)

#more efficient for small arrays
def small_arr_set_neg_zero(arr):
    arr = arr.clip(min=0)
    return(arr)
#===============================================================================
# Data Analysis
#===============================================================================

# Uses 1st degree polynomial fitting of polyfit and interpolates at points of interest
def lineofbestfit(x_data, y_data):
    import matplotlib.pyplot as plt
    #unique allows for unsorted or duplicated x_data
    plt.plot(np.unique(x_data), np.poly1d(np.polyfit(x_data, y_data, 1))(np.unique(x_data)))

def fit_gaussian():
    pass

#===============================================================================
# File handling
#===============================================================================

# OLD VERSION
# finds all files in a given folder and all subfolders that match a filetype and keyword
# def open_folders_and_find(main_dir, file_type,keyword):
#     import os
#     currentpy_path = os.getcwd()
#     os.chdir(main_dir)
#     dir_paths = [] #main directory path for each file
#     rel_paths = [] #relative path after using chdir in octave
#     full_paths = []
#     for root, dirs, files in os.walk(".",topdown = True):
#         for name in files:
#             if (keyword and file_type) in name:
#                 drop_it, real_root = root.split('.') #removes the '.' from the root path
#                 dir_paths.append(str(main_dir))
#                 rel_paths.append(str(real_root) + str(name))
#     os.chdir(currentpy_path) #returns to original working directory
#     return(dir_paths, rel_paths)

#KEEPING ORIGINAL WHILST EDITTING AS THIS IS A V. IMPORTANT FUNCTION I USE MANY TIMES

def open_folders_and_find(main_dir, file_type,keyword):
    import os
    import numpy as np
    currentpy_path = os.getcwd()
    os.chdir(main_dir)
    dir_paths = [] #main directory path for each file
    rel_paths = [] #relative path after using chdir in octave
    for root, dirs, files in os.walk(".",topdown = True):
        for name in files:
            if (keyword and file_type) in name:
                drop_it, real_root = root.split('.') #removes the '.' from the root path
                dir_paths.append(str(main_dir))
                rel_paths.append(str(real_root) + '\\' + str(name))
    os.chdir(currentpy_path) #returns to original working directory
    return(dir_paths, rel_paths)

def open_folders_and_find_new(main_dir, file_type,keyword):
    import os
    import numpy as np
    currentpy_path = os.getcwd()
    os.chdir(main_dir)
    dir_paths = [] #main directory path for each file
    rel_paths = [] #relative path after using chdir in octave
    full_paths = []
    for rroot, dirs, files in os.walk(".",topdown = True):
        for name in glob.glob('{kkwrd}*{ftype}'.format(kkwrd = keyword, ftype = file_type)):
            if (keyword and file_type) in name:
                drop_it, real_root = rroot.split('.') #removes the '.' from the root path
                dir_paths.append(str(main_dir))
                rel_paths.append(str(real_root) + '\\' + str(name))
                full_paths.append(str(main_dir) + str(real_root) + '\\' + str(name))
    os.chdir(currentpy_path) #returns to original working directory
    return(full_paths,dir_paths, rel_paths)

def open_folders_deep_find(ppath, filetype, keyword, exclude_keyword):
    files = glob.glob(ppath + '/**/*{kkwrd}*{ftype}'.format(kkwrd = keyword, ftype = filetype), recursive = True)
    if exclude_keyword == '':
        return(files)
    else:
        return_files = []
        for file in files:
            base_name = os.path.basename(file)
            if exclude_keyword not in base_name:
                return_files.append(file)
        return(return_files)


def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return

def find_file_creation_time(fpath):
    file_time = datetime.datetime.fromtimestamp(Path(fpath).stat().st_ctime)
    return(file_time)

#===============================================================================

# returns only the file name from a path
def only_file_name(path):
    base = os.path.basename(path)
    file_name = os.path.splitext(base)[0]
    return(file_name)

def only_file_with_ext(path):
    base = os.path.basename(path)
    return(base)

def only_path(path):
    only_file_path = os.path.dirname(os.path.abspath(path))
    return(only_file_path)

#===============================================================================

# concantenates two string arrays elementwise
def concant_twolist(str_list1, str_list2):
    str_list = [i + j for i, j in zip(str_list1,str_list2)]
    return str_list

#===============================================================================
# String Functions
#===============================================================================

def remove_symbol(symbol,sstring):
    sstring = sstring.replace(str(symbol), '')
    return(sstring)

#===============================================================================
# Error Checking / Terminal interaction
#===============================================================================

# avoids having to use {}.format(x) for quick printing statements
def ppv(str_text,var_value): # Print Python Value
    print(str_text , end = '')
    print(var_value)
    return

#===============================================================================
# Error handeling
#===============================================================================

def raise_warning(str_text):
    warnings.warn(str_text)
    return()

#===============================================================================
# Plotting Functions
#===============================================================================

# data_number_arr must be the strings to label, the data_x ect are for the positioning of the labels
def number_scatter(data_number_arr, data_x, data_y):
    for i in range(len(data_x)):
        plt.annotate(str(data_number_arr[i]), (data_x[i],data_y[i]), textcoords="offset points", xytext=(0,10),ha='center')
    return

# same as above but in 3d requires the text function
def number_scatter3D(data_number_arr, data_x, data_y, data_z, axis_object):
    for i in range(len(data_number_arr)): #plot each point + it's index as text above
        axis_object.text(data_x[i],data_y[i],data_z[i],  '%s' % (str(data_number_arr[i])), size=20, zorder=1, color='k')
    return

# combines legend labels when using multiple axis
def combine_legend(axis_label,*lines):
    lns = []
    for line in lines:
        lns = lns + line
    labs = [l.get_label() for l in lns]
    axis_label.legend(lns, labs)

#===============================================================================
# Error calculation
#===============================================================================

def calc_standerror(data):
    from scipy.stats import sem
    return(sem(data))

#===============================================================================
# Array handling
#===============================================================================
# adapted from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def findnearestval_sorteddata(data, value):
    idx = np.searchsorted(data, value, side="left")
    if idx > 0 and (idx == len(data) or mt.fabs(value - data[idx-1]) < mt.fabs(value - data[idx])):
        return data[idx-1] # taking care of issues if value doesn't fit
    else:
        return data[idx]

def strlist_to_floatlist(llist):
    float_list = []
    for i in llist:
        float_list.appen(float(i))
    return(float_list)
