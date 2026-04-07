"""Neural mechanism: MLP → distribution over K=8 allocations.

Input:  v ∈ [0,1]^{2×3}  (flattened to R^6, no synergy parameter)
Output: probability distribution over K=8 allocations

IR and PE constraints are enforced by masking logits before softmax.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .allocations import K


class AllocationNet(nn.Module):
    """MLP mechanism for 2-agent 3-item exchange economy."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        in_dim = cfg.num_agents * cfg.num_items   # 6
        h = cfg.hidden
        d = cfg.depth

        layers: list[nn.Module] = []
        for i in range(d):
            layers += [nn.Linear(in_dim if i == 0 else h, h), nn.ReLU()]
        layers.append(nn.Linear(h, K))   # 8 logits
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """
        Args:
            v:           [B, 2, 3]
            mask:        [B, K]  float (1=feasible, 0=infeasible); applied before softmax
            temperature: softmax temperature (default: cfg.temperature)

        Returns:
            probs: [B, K]
        """
        B = v.shape[0]
        tau = temperature if temperature is not None else self.cfg.temperature

        logits = self.net(v.reshape(B, -1))   # [B, K]

        if mask is not None:
            logits = logits + (1.0 - mask) * (-1e9)

        return F.softmax(logits / max(tau, 1e-6), dim=-1)   # [B, K]

    @torch.no_grad()
    def argmax_alloc(
        self,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic allocation: [B] allocation indices."""
        B = v.shape[0]
        logits = self.net(v.reshape(B, -1))
        if mask is not None:
            logits = logits + (1.0 - mask) * (-1e9)
        return logits.argmax(dim=-1)   # [B]
