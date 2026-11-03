import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        x = np.array(x)
        gamma = np.array(gamma)
        # Normalize x, then scale by gamma
        rms = np.sqrt(np.sum(x**2)/len(x) + eps)
        out = gamma * (x / rms)
        # Return result rounded to 4 decimal places as a list
        return np.round(out, 4).tolist()
