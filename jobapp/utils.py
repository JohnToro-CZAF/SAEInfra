# app/utils/gpu_utils.py

import subprocess
import re
import psutil

def get_gpu_memory():
    """
    Returns a list of dictionaries containing GPU ID and available memory in MB.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            gpu_id, mem_free = re.findall(r'\d+', line)
            gpu_info.append({
                'gpu_id': int(gpu_id),
                'memory_free': int(mem_free)  # in MB
            })
        return gpu_info
    except subprocess.CalledProcessError as e:
        print(f"Error fetching GPU info: {e.stderr}")
        return []

def get_cpu_usage():
    """
    Returns the current CPU usage percentage.
    """
    return psutil.cpu_percent(interval=1)

def get_ram_usage():
    """
    Returns the current RAM usage percentage.
    """
    return psutil.virtual_memory().percent

def get_system_resources():
    """
    Returns a dictionary containing GPU, CPU, and RAM usage.
    """
    return {
        'gpus': get_gpu_memory(),
        'cpu': get_cpu_usage(),
        'ram': get_ram_usage()
    }

if __name__ == "__main__":
    print(get_gpu_memory())