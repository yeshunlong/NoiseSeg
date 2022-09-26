import pynvml # run 'pip install pynvml' to get this lib. 
import torch
import numpy as np

def get_best_gpu(): # return gpu(torch.device) with largest free memory.
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    deviceMemory = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    return torch.device("cuda:%d"%(best_device_index))

if __name__=="__main__":
    print(get_best_gpu())