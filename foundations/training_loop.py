import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        print(X.shape)

        n_samples, n_features = X.shape
        w, b = np.zeros(n_features), 0

        for epoch in range(epochs):
            y_hat = X @ w + b
            db = 2*np.sum(y_hat - y)/n_samples
            dw = 2*(X.T @ (y_hat - y))/n_samples
            b = b - lr*db
            w = w - lr*dw

        return (np.round(w, 5), round(b, 5))
