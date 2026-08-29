import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        if kv_cache is None:
            kv_cache = KVCache()

        # x: [batch, seq_len, model_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Append the new K/V to the cache.
        full_k, full_v = kv_cache.update(k, v)

        # [batch, new_seq_len, total_seq_len]
        scores = (
            q @ full_k.transpose(-2, -1)
        ) * (self.q_proj.out_features ** -0.5)

        # Causal mask.
        new_len = q.size(1)
        total_len = full_k.size(1)

        # Query positions correspond to the final `new_len`
        # positions in the full sequence.
        query_positions = torch.arange(
            total_len - new_len,
            total_len,
            device=x.device
        )

        key_positions = torch.arange(
            total_len,
            device=x.device
        )

        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)

        scores = scores.masked_fill(
            causal_mask.unsqueeze(0),
            float("-inf")
        )

        weights = torch.softmax(scores, dim=-1)

        output = weights @ full_v

        return torch.round(output,decimals=4), kv_cache
