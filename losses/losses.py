import torch

import utilities.utils as utils
#https://vitalflux.com/mean-squared-error-vs-cross-entropy-loss-function/

def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err= target - input_tensor # (y-yi)
    squared_err = err *err #take the square of the err
    mean = torch.mean(squared_err)  #mean value
    return mean
