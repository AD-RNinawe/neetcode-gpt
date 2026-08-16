import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embed=nn.Embedding(vocabulary_size,16)
        self.proj=nn.Linear(16,1)
        self.sig=nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        embs=self.embed(x)
        avg=torch.mean(embs,dim=1)
        projected=self.proj(avg)
        return torch.round(self.sig(projected),decimals=4)
