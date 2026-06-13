import math

import torch
from torch import nn
import torch.nn.functional as F

from .kv_cache import KVCache


def apply_rope(x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    _, _, t, d = x.shape
    if d % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    half = d // 2
    positions = torch.arange(start_pos, start_pos + t, device=x.device, dtype=x.dtype)
    freqs = torch.arange(half, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (10000 ** (freqs / half))
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, rope: bool = False):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.rope = rope
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        if self.rope:
            q = apply_rope(q, start_pos=start_pos)
            k = apply_rope(k, start_pos=start_pos)

        if kv_cache is not None:
            k, v = kv_cache.append(k, v)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if kv_cache is None:
            att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
        else:
            q_pos = torch.arange(start_pos, start_pos + t, device=x.device)[:, None]
            k_pos = torch.arange(k.size(2), device=x.device)[None, :]
            mask = k_pos <= q_pos
            att = att.masked_fill(~mask.view(1, 1, t, k.size(2)), float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))
