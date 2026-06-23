import torch
import pynvml

def print_gpu_diagnostics():
    print("="*10 + " PYTORCH & CUDA INFO " + "="*10)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("No CUDA-capable GPU detected by PyTorch.")
        return

    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    device_capability = torch.cuda.get_device_capability(current_device)
    
    print(f"Current Device ID: {current_device}")
    print(f"Device Name:       {device_name}")
    print(f"Compute Capability: {device_capability[0]}.{device_capability[1]}")
    
    # PyTorch Memory Stats
    print("\n" + "="*10 + " PYTORCH MEMORY USAGE " + "="*10)
    # Convert bytes to Gigabytes for readability
    bytes_to_gb = 1024 ** 3
    allocated = torch.cuda.memory_allocated(current_device) / bytes_to_gb
    cached = torch.cuda.memory_reserved(current_device) / bytes_to_gb
    
    print(f"Allocated Memory: {allocated:.2f} GB (Currently used by PyTorch tensors)")
    print(f"Reserved Memory:  {cached:.2f} GB (Cached by PyTorch allocator)")

    # Advanced NVML Stats (Hardware-level details)
    print("\n" + "="*10 + " HARDWARE & POWER INFO (NVML) " + "="*10)
    try:
        pynvml.nvmlInit()
        # Get the handle for the current GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(current_device)
        
        # Exact Driver and NVML version
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode('utf-8')
        print(f"Driver Version:    {driver_version}")
        
        # Precise Hardware Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"Total GPU Memory:  {mem_info.total / bytes_to_gb:.2f} GB")
        print(f"Used GPU Memory:   {mem_info.used / bytes_to_gb:.2f} GB")
        print(f"Free GPU Memory:   {mem_info.free / bytes_to_gb:.2f} GB")
        
        # Temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        print(f"GPU Temperature:   {temp}°C")
        
        # Power Consumption
        # NVML returns power in milliwatts, convert to Watts
        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        try:
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            power_limit_str = f"{power_limit:.2f} W"
        except pynvml.NVMLError:
            power_limit_str = "N/A"
            
        print(f"Current Power Draw: {power_draw:.2f} W")
        print(f"Power Limit Max:    {power_limit_str}")
        
        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        print(f"GPU Utilization:   {util.gpu}%")
        print(f"Memory Controller: {util.memory}%")

    except pynvml.NVMLError as e:
        print(f"Could not fetch NVML metrics: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

if __name__ == "__main__":
    print_gpu_diagnostics()