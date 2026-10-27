import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        np_x = np.array(x)
        np_W1 = np.array(W1)
        np_b1 = np.array(b1)
        np_W2 = np.array(W2)
        np_b2 = np.array(b2)
        np_y_true = np.array(y_true)

        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        z1 = np.dot(np_x, np_W1.T) + np_b1
        a1 = np.maximum(z1, 0)

        y_hat = np.dot(a1, np_W2.T) + np_b2
        # Loss: MSE = mean((predictions - y_true)^2)
        loss = np.mean((y_hat - np_y_true)**2)

        N = np_y_true.shape[0]
        db2 = 2*(y_hat - np_y_true)/N 
        dW2 = np.dot(db2.reshape((-1, 1)), a1.reshape((1, -1)))
        
        da1 = np.dot(db2, np_W2)

        db1 = da1*(z1>0)
        dW1 = np.dot(db1.reshape((-1, 1)), np_x.reshape((1, -1)))

        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        return {
            "loss": np.round(loss, 4),
            "dW1": np.round(dW1, 4),
            "db1": np.round(db1, 4),
            "dW2": np.round(dW2, 4),
            "db2": np.round(db2, 4)
        }
