import torch
import numpy as np
from inspect import isfunction

def contains_nan(x):
    if x.requires_grad:
        x = x.detach()
    if np.isnan(torch.min(x.cpu())):
        return True
    else:
        return False

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr