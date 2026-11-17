import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_frac = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    dead_comp = (x > 0).sum(axis=0)
                    frac = (dead_comp == 0).sum()/len(dead_comp)
                    dead_frac.append(round(frac.item(), 4))
        return dead_frac

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        leaky_relu, reinit, reduce_lr = False, (dead_fractions[0] > 0.3), (dead_fractions[-1] > 0.1)
        
        prev = -1
        for frac in dead_fractions:
            if frac > 0.5:
                leaky_relu = True
            if prev >= frac:
                reduce_lr = False
            
            prev = frac

        if leaky_relu:
            return "use_leaky_relu"
        elif reinit:
            return "reinitialize"
        elif reduce_lr:
            return "reduce_learning_rate"
        return "healthy"
