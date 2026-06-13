from dataclasses import dataclass

import torch


@dataclass
class KVCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None

    def append(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key is None:
            self.key = key
            self.value = value
        else:
            self.key = torch.cat([self.key, key], dim=2)
            self.value = torch.cat([self.value, value], dim=2)
        return self.key, self.value


def empty_cache(n_layer: int) -> list[KVCache]:
    return [KVCache() for _ in range(n_layer)]
