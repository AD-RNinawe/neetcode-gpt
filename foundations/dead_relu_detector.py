import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        df=[]
        with torch.no_grad():
            for module in model.children():
                x=module(x)
                if isinstance(module,nn.ReLU):
                    dead=(x==0).all(dim=0).float().mean().item()
                    df.append(round(dead,4))
        return df

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        df=dead_fractions
        if len(df)==0:
            return 'healthy'
        max_frac=max(df)
        if max_frac>0.5:
            return 'use_leaky_relu'
        if df[0]>0.3:
            return 'reinitialize'
        if len(df)>=2:
            increasing=all(
                df[i]<df[i+1]
                for i in range(len(df)-1)
            )
            if increasing and df[-1]>0.1:
                return 'reduce_learning_rate'
        if max_frac<0.1:
            return 'healthy'
        return 'healthy'
