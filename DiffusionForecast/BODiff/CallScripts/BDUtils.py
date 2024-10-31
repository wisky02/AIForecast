# Created on 06/09/2024 by Lewis Dickson 

"""
Function and wrapper script for cleaner code 
"""

#===========================================================
# Imports 
#===========================================================

import functools
import pickle
from pathlib import Path 
import os 
import warnings
import logging
from glob import glob
import subprocess
import shutil

#===========================================================
# Functions 
#===========================================================

def get_user_confirmation(prompt):
    """
    Prompts the user for a yes or no response and returns a boolean value.

    Args:
        prompt (str): The prompt message to display to the user.

    Returns:
        bool: True if the user inputs 'y' or 'Y', False if 'n' or 'N'.
    """
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")


def ifdir_doesntexist_created_nested(ppath, silence_warnings = False):
    """
    This function checks if the directory of a given file path exists. If it does not exist, it creates the directory and any necessary parent directories.

    Parameters:
    ppath (str): The file path for which the directory needs to be checked and created.

    Returns:
    None. The function does not return any value, but it creates the directory if it does not exist.
    """
    dir_path = os.path.dirname(ppath)
    if not os.path.exists(dir_path):
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    elif os.path.exists(dir_path) and not silence_warnings:
        logging.warning(f'Folder at {ppath} exists. Silence these warnings with silence_warnings=True.')
        ret_user_confirmation = get_user_confirmation('Do you want to overwrite this data?')
        if not ret_user_confirmation:
            raise Exception('User selected to not overwrite this data. Script stopped.')
    return 


# Always pickling so wrapper for quick-pickline output of functions  
def quick_pickle():
    """
    Decorator factory to pickle the output of a function.
    Returns:
        pickle_decorator (function): A decorator that adds pickling functionality
        to a function's output.

    Example Usage:
    @quick_pickle() # !!! note that the decorator factory requires calling as a function
    def add_func(a, b):
        return a + b

    # Example usage
    add_func(10, 2, outfpath='add_result.pkl')

    """
    def pickle_decorator(func):
        def pickle_wrapper(*args, outfpath=None, **kwargs):
            """
            Wrapper function that executes the original function and pickles its output.
            Args:
                *args: Positional arguments to pass to the original function.
                outfpath (str, optional): The file path where the output should be pickled.
                                          If None, the output is not saved.
                **kwargs: Keyword arguments to pass to the original function.
            Returns:
                The output of the original function.
            """
            # Call the original function with provided arguments
            value = func(*args, **kwargs)           
            # If a file path is provided, pickle the output
            if outfpath:
                with open(outfpath, 'wb') as handle:
                    pickle.dump(value, handle)
            return value
        return pickle_wrapper
    return pickle_decorator

def make_pickle(data, ppath):
    with open(ppath, 'wb') as handle:
        pickle.dump(data, handle)#, protocol=pickle.HIGHEST_PROTOCOL)
    return

# unpickles a file
def read_pickle(pickle_path):
    with open(pickle_path, 'rb') as handle:
        b = pickle.load(handle)
    return(b)


def get_frame_paths(fdir, keyword = ''):
    
    # Check that folder exists
    if not os.path.exists(fdir):
        raise FileNotFoundError(f'The folder {fdir} does not exist.')

    # search for all files with image extensions and optional keyword1
    # frame_paths = glob(f'{fdir}/*.png') + glob(f'{fdir}/*.jpeg') + glob(f'{fdir}/*.jpg') 
    frame_paths = glob(f'{fdir}/*{keyword}*.png') + glob(f'{fdir}/*{keyword}*.jpeg') + glob(f'{fdir}/*{keyword}*.jpg') 
    return frame_paths

def print_inputs(func):
    def wrapper(*args, **kwargs):
        # Print the function name and arguments
        print(f"Inputs to {func.__name__}:")
        for i, arg in enumerate(args):
            print(f"arg[{i}]: {arg}")
        for key, value in kwargs.items():
            print(f"kwarg[{key}]: {key} = {value}")
        
        # Call the original function
        return func(*args, **kwargs)
    
    return wrapper

def cam_data_outpath_creator(out_dir, video_file):
    # Creating full output path from video sample name 
    cam_name = os.path.basename(video_file).split('_')[0]
    date_time = os.path.basename(video_file).split('_')[1].split('.')[0]
    out_bbox_path = f'{out_dir}/{cam_name}/{date_time}/'
    return out_bbox_path, cam_name, date_time

def open_folders_walk_find(ppath, ftype, keyword, exclude_keyword=''):
    """
    Searches a directory tree for files with a specific extension and a specific keyword.

    Args:
        ppath (str): The path to search.
        ftype (str): The file extension to look for.
        keyword (str): The keyword to search for in the file names.
        exclude_keyword (str, optional): A keyword to exclude from the search. Defaults to ''.

    Returns:
        List[str]: A list of file paths that match the search criteria.

    Raises:
        Exception: If no files are found.
    """

    return_files = []

    for root, _, files in os.walk(ppath):
        for file in files:
            if file.endswith(ftype) and (keyword in file) and (not exclude_keyword or exclude_keyword not in file): # using implicit true of non-supplied exclude keyword to speed up search if not supplied
                    return_files.append(os.path.join(root, file))

    if len(return_files) == 0:
        raise Exception("No files found in function: open_folders_deep_find \nCheck path, keyword, filetype, and exclusion keyword")

    return return_files

def get_host_name():
    import socket
    return socket.gethostname()

def get_least_utilized_gpu(avoid_GPU_number):

    """
    Returns the gpu_index of the GPU with the lowest utilisation in terms of compute
    # Example usage:
    gpu_index, utilization = get_least_utilized_gpu()
    print(f"Least utilized GPU is {gpu_index} with {utilization}% utilization.")
    """

    # Run the nvidia-smi command and capture the output
    command = ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    
    # Parse the output and find the least utilized GPU
    gpu_utilization = []
    for line in result.stdout.strip().split('\n'):
        gpu_index, utilization = line.split(',')
        if int(gpu_index) != avoid_GPU_number:
            gpu_utilization.append((int(gpu_index), int(utilization)))

    # Sort by utilization and return the least utilized GPU index
    least_utilized_gpu = min(gpu_utilization, key=lambda x: x[1])
    return least_utilized_gpu


def get_least_memory_utilized_gpu(avoid_GPU_number):

    """
    Returns the gpu_index of the GPU with the lowest utilisation in terms of memory usage

    # Example usage:
    gpu_index, memory_utilization = get_least_memory_utilized_gpu()
    print(f"Least utilized GPU in terms of memory is {gpu_index} with {memory_utilization}% memory utilization.")
    """

    # Run the nvidia-smi command and capture the memory utilization output
    command = ["nvidia-smi", "--query-gpu=index,utilization.memory", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    
    # Parse the output and find the least utilized GPU in terms of memory usage
    gpu_memory_utilization = []
    for line in result.stdout.strip().split('\n'):
        gpu_index, memory_utilization = line.split(',')
        if int(gpu_index) != avoid_GPU_number:
            gpu_memory_utilization.append((int(gpu_index), int(memory_utilization)))

    print(f'{gpu_memory_utilization=}')

    # Sort by memory utilization and return the least utilized GPU index
    least_memory_utilized_gpu = min(gpu_memory_utilization, key=lambda x: x[1])
    return least_memory_utilized_gpu


def auto_set_GPU(auto_select_GPU_options):
    """
    Takes a dict input like: 
    auto_select_GPU_options = {'auto_select': True, 'compute_or_memory':'compute', 'default_GPU':0} 

    """
    avoid_GPU_number = auto_select_GPU_options['avoid_GPU']

    if auto_select_GPU_options['auto_select']:
        
        set_method = auto_select_GPU_options['compute_or_memory']

        if set_method=='compute':
            set_GPU = get_least_utilized_gpu(avoid_GPU_number)
        elif set_method=='memory':
            set_GPU = get_least_memory_utilized_gpu(avoid_GPU_number)
        else:
            raise ValueError(f'{auto_select_GPU_options["compute_or_memory"]} is an invalid choice. Please select either \'Compute\' or \'memory\' ')
    else:
        set_GPU =auto_select_GPU_options['default_GPU'], None
    return set_GPU


def break_vidfile_name_up(vid_name):
    """
    Example name: 5R_2023-05-13T12-45-00.mp4
    naming convetion:
    <cam_name>_<yyyy>-<mm>-<dd>T<hh>-<mm>-<ss>.mp3
    """
    basename = os.path.basename(vid_name)
    cam = basename.split('_')[0]
    date_time = basename.split('_')[1].split('.')[0]
    return cam, date_time

def vram_limited_data_specifier(gpu_index,
                                example_img, 
                                data_IDs, # list of IDs to split into managable sub-groups
                                model_size = 5000, #in mb
                                use_default = False
                                ):
    import subprocess

    # Default Settings 
    default_value = 1000

    # Check that input is ordered 
    data_IDs.sort()

    def get_vram_info(gpu_index=0):
        # Run the nvidia-smi command to get memory information for the specified GPU
        result = subprocess.run(
            ['nvidia-smi', f'--id={gpu_index}', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to run nvidia-smi for GPU {gpu_index}: {result.stderr}")

        # Parse the output
        total_memory, used_memory, free_memory = map(int, result.stdout.strip().split(', '))

        return {
            'GPU Index': gpu_index,
            'Total VRAM (MB)': total_memory,
            'Used VRAM (MB)': used_memory,
            'Available VRAM (MB)': free_memory
        }

    # Get the size of the image in memory
    def get_image_size_in_mb(image):
        # Get the shape of the image
        height, width, channels = image.shape
        
        # Calculate size in bytes
        size_bytes = height * width * channels
        
        # Convert bytes to megabytes (1 MB = 1024 * 1024 bytes)
        size_mb = size_bytes / (1024 * 1024)
        
        return size_mb

    if not use_default:
        vram_dict = get_vram_info(gpu_index)
        img_mb = get_image_size_in_mb(example_img)
        packed_images = int((vram_dict['Available VRAM (MB)']-model_size)/img_mb)
        print(f'*slaps roof of GPU* You can fit {packed_images} images in here.')
        

    else:
        packed_images = default_value

    # Using modified range to make sure the last value is always included
    range_values = list(range(data_IDs[0], data_IDs[-1], packed_images))
    if data_IDs[-1] not in range_values:
        range_values.append(data_IDs[-1])
    
    # --- Creating groups --- # 
    # Initialize list of lists
    list_of_lists = []
    current_list = []

    # Iterate through the numbers
    for number in data_IDs:
        current_list.append(number)  # Add the current number to the current inner list
        if number in range_values[1:]:  # Check if the number is in range_values
            # If we reach a value in range_values and current_list is not empty, append it to list_of_lists
            list_of_lists.append(current_list)  # Append the current list to list_of_lists
            current_list = []  # Reset current_list for the next group

    return list_of_lists


def create_temp_data_dir(renamed_path, frame_start, frame_end, temp_dir_loc):
    from shutil import copy2
    from tqdm import tqdm
    """
    !!!NOTE WELL that if the folder already exists it is WIPED!!! 
    
    Moves data temporarily to a folder - note that if the folder already exists it is WIPED 
    before filling  
    
    Uses naming convetion of <frame_number>.jpg
    """
    print(f'Copying temp data to {temp_dir_loc}')
    for file in tqdm(os.listdir(renamed_path)):
        extension = os.path.splitext(file)[-1]
        frame_num = int(os.path.splitext(file)[0])
        if extension in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
            if frame_start <= frame_num <= frame_end:
                copy2(f'{renamed_path}/{file}', f'{temp_dir_loc}/{file}')

    return 

def delete_folder_contents(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            # print(f'try to remove {item_path}')
            os.unlink(item_path)  # Remove the file or symlink