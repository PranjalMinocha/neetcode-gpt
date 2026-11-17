import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        activation_stats = []

        with torch.no_grad():
            for layer in model.children():
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    dead_comp = (x > 0).sum(axis=0)
                    d = {
                        "mean": x.mean().item(),
                        "std": x.std().item(),
                        "dead_fraction": ((dead_comp == 0).sum()/len(dead_comp)).item()
                    }
                    activation_stats.append(d)

        return activation_stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.

        model.zero_grad()

        out = model(x)
        loss = nn.MSELoss()(out, y)

        loss.backward()

        gradient_stats = []

        for layer in model.children():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                d = {
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "norm": torch.norm(grad).item()
                }
                gradient_stats.append(d)

        return gradient_stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        dead_neurons, expl_grads_norm, van_grads_norm, expl_grads, van_grads = False, False, False, False, False
        
        for activation_stat, grad_stat in zip(activation_stats, gradient_stats):
            if activation_stat["dead_fraction"] > 0.5:
                dead_neurons = True

            if grad_stat["norm"] > 1000:
                expl_grads_norm = True
            elif grad_stat["norm"] < 1e-5:
                van_grads_norm = True

            if activation_stat["std"] < 0.1:
                van_grads = True
            elif activation_stat["std"] > 10:
                expl_grads = True

        if dead_neurons:
            return "dead_neurons"
        elif expl_grads_norm:
            return "exploding_gradients"
        elif van_grads_norm:
            return "vanishing_gradients"
        elif van_grads:
            return "vanishing_gradients"
        elif expl_grads:
            return "exploding_gradients"
        return "healthy"
