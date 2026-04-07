from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config
from .allocations import num_allocations


class AllocationNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        K = num_allocations(cfg)
        in_dim = cfg.num_agents * cfg.num_items   # 3*4=12
        h, d = cfg.hidden, cfg.depth
        layers: list[nn.Module] = []
        for i in range(d):
            layers += [nn.Linear(in_dim if i == 0 else h, h), nn.ReLU()]
        layers.append(nn.Linear(h, K))
        self.net = nn.Sequential(*layers)

    def forward(self, v: torch.Tensor, mask: torch.Tensor | None = None,
                temperature: float | None = None) -> torch.Tensor:
        B = v.shape[0]
        tau = temperature if temperature is not None else self.cfg.temperature
        logits = self.net(v.reshape(B, -1))
        if mask is not None:
            logits = logits + (1.0 - mask) * (-1e9)
        return F.softmax(logits / max(tau, 1e-6), dim=-1)

    @torch.no_grad()
    def argmax_alloc(self, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = v.shape[0]
        logits = self.net(v.reshape(B, -1))
        if mask is not None:
            logits = logits + (1.0 - mask) * (-1e9)
        return logits.argmax(dim=-1)
