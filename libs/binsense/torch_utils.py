from typing import List, Union, Optional
from torch import Tensor

import numpy as np

import torch

def to_tensor(
    x: Union[np.ndarray, np.array, List], 
    dtype: torch.dtype =torch.float32) -> Tensor:
    return torch.as_tensor(x, dtype=dtype)

def to_float_tensor(x: Union[np.ndarray, np.array, List]) -> Tensor:
    return to_tensor(x)

def to_int_tensor(x: Union[np.ndarray, np.array, List]) -> Tensor:
    return to_tensor(x, dtype=torch.int64)

def to_long_tensor(x: Union[np.ndarray, np.array, List]) -> Tensor:
    return to_tensor(x, dtype=torch.long)

def empty_tensor(dtype: torch.dtype =torch.float32) -> Tensor:
    t = torch.Tensor()
    return t.to(dtype)

def empty_float_tensor() -> Tensor:
    return empty_tensor(torch.float32)

def empty_int_tensor() -> Tensor:
    return empty_tensor(torch.int64)

def empty_long_tensor() -> Tensor:
    return empty_tensor(torch.long)