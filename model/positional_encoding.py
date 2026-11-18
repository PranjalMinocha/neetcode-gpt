import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        
        # Hint: Use np.arange() to create position and dimension index vectors,
        pos = np.arange(seq_len).reshape(-1, 1)
        dim = np.arange(d_model).reshape(1, -1)
        # then compute all values at once with broadcasting (no loops needed).
        PE = pos/np.pow(10000, (dim - dim%2)/d_model)
        print(PE.shape)
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        PE[:, ::2] = np.sin(PE[:, ::2])
        PE[:, 1::2] = np.cos(PE[:, 1::2])
        # Round to 5 decimal places.
        return np.round(PE, decimals=5)
        
