from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .kv_cache import KVCache, empty_cache


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    norm: str = "layernorm"
    mlp: str = "gelu"
    position: str = "learned"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm: {kind}")


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.kind = config.mlp
        if self.kind == "gelu":
            self.net = nn.Sequential(
                nn.Linear(config.n_embd, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.n_embd),
                nn.Dropout(config.dropout),
            )
        elif self.kind == "swiglu":
            self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
            self.w2 = nn.Linear(config.n_embd, hidden, bias=False)
            self.w3 = nn.Linear(hidden, config.n_embd, bias=False)
            self.dropout = nn.Dropout(config.dropout)
        else:
            raise ValueError(f"Unknown MLP: {self.kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "gelu":
            return self.net(x)
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = make_norm(config.norm, config.n_embd)
        self.attn = CausalSelfAttention(
            config.n_embd,
            config.n_head,
            config.block_size,
            config.dropout,
            rope=config.position == "rope",
        )
        self.ln_2 = make_norm(config.norm, config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = None
        if config.position == "learned":
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        elif config.position != "rope":
            raise ValueError("position must be 'learned' or 'rope'")

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = make_norm(config.norm, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: list[KVCache] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.shape
        if kv_cache is None and t > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        x = self.token_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(start_pos, start_pos + t, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        for layer, block in enumerate(self.blocks):
            cache = kv_cache[layer] if kv_cache is not None else None
            x = block(x, kv_cache=cache, start_pos=start_pos)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, use_cache: bool = True) -> torch.Tensor:
        self.eval()
        cache_enabled = use_cache and self.config.position == "rope"
        cache = empty_cache(self.config.n_layer) if cache_enabled else None
        if cache_enabled and idx.size(1) > 1:
            self(idx[:, :-1], kv_cache=cache, start_pos=0)

        for _ in range(max_new_tokens):
            if cache_enabled:
                idx_cond = idx[:, -1:]
                start_pos = idx.size(1) - 1
                logits, _ = self(idx_cond, kv_cache=cache, start_pos=start_pos)
            else:
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
