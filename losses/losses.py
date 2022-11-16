import torch

import utilities.utils as utils


def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err= target - input_tensor # (y-yi)
    squared_err = err *err #take the square of the err
    mean = torch.mean(squared_err)  #mean value
    return mean
