import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        # During training: normalize using batch statistics, then update running stats
        if training:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)

            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)

            running_mean = (1-momentum)*running_mean + momentum*batch_mean
            running_var = (1-momentum)*running_var + momentum*batch_var

        # During inference: normalize using running stats (no batch stats needed)
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        # Apply affine transform: y = gamma * x_hat + beta
        y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        return (np.round(y, 4).tolist(), np.round(running_mean, 4).tolist(), np.round(running_var, 4).tolist())
