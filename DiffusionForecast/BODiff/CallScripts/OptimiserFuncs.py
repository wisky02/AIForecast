# Created on 25/10/2024 by Lewis Dickson 

#========================================
import subprocess

#========================================

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