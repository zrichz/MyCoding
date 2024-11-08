# attempt to load and view a .safetensors file (eg. a LoRA)

import torch
from safetensors import safe_open
from safetensors.torch import save_file
import itertools

st_filename = "Loras\cyberhelmet.safetensors"

tensors = {}
with safe_open(st_filename, framework="pt", device="cpu") as f: # framework="pt" for PyTorch
   for key in f.keys():
       tensors[key] = f.get_tensor(key)

print('length of tensors: ',len(tensors))
print('length of keys: ',len(tensors.keys()))
#print('keys: ',tensors.keys())


# Get the first few tensors
first_few_tensors = dict(itertools.islice(tensors.items(), 80))

# Print them
for key, tensor in first_few_tensors.items():
    print('Key: ',key, end = ' ')
    #print('Tensor: ',tensor.detach())
    print('Shape: ',tensor.shape)
    #print('Type: ',tensor.dtype)


#sum all the learnable parameters in the tensors
total_params = 0
for key, tensor in tensors.items():
    total_params += tensor.numel()
print('Total parameters: ',total_params)
print('Total parameters in MB: ',total_params*4/(1024*1024)) # 4 bytes per float32
#round the number of total parameters to the nearest 1000
total_params = round(total_params, -3)
print('Total parameters rounded to the nearest 1000: ',total_params)