import torch

import utilities.utils as utils


def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err= target - input_tensor
    squared_err = err *err
    mean = torch.mean(squared_err)
    return mean
