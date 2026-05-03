from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config
from .allocations import num_allocations


class AllocationNet(nn.Module):
    """MLP that maps ordinal marginal preferences to allocation probabilities.

    Input encoding:
      marginal_rank [B, A, m]  (int in {0,...,R-1})
      -> one-hot per (agent, item): [B, A, m, R]
      -> flatten: [B, A*m*R]

    For trichotomous domain (R=3, A=3, m=4): input dim = 36.
    Output: softmax over K=81 allocations (masked by IR+PE+Balanced).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        K      = num_allocations(cfg)
        R      = cfg.num_ranks
        in_dim = cfg.num_agents * cfg.num_items * R   # e.g. 3*4*3 = 36
        h, d   = cfg.hidden, cfg.depth

        layers: list[nn.Module] = []
        for i in range(d):
            layers += [nn.Linear(in_dim if i == 0 else h, h), nn.ReLU()]
        layers.append(nn.Linear(h, K))
        self.net = nn.Sequential(*layers)

    # ── encoding ──────────────────────────────────────────────────────────

    def encode(self, marginal_rank: torch.Tensor) -> torch.Tensor:
        """[B, A, m] int -> [B, A*m*R] float (one-hot)."""
        R  = self.cfg.num_ranks
        oh = F.one_hot(marginal_rank.clamp(0, R - 1), num_classes=R).float()
        return oh.reshape(marginal_rank.shape[0], -1)

    # ── forward (soft, for training) ──────────────────────────────────────

    def forward(
        self,
        marginal_rank: torch.Tensor,
        mask: torch.Tensor | None = None,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """Return softmax probabilities over K allocations.

        Args:
            marginal_rank: [B, A, m] int
            mask:          [B, K]    float (1=feasible, 0=infeasible)
            temperature:   softmax temperature (default: cfg.temperature)

        Returns:
            probs: [B, K]
        """
        tau    = temperature if temperature is not None else self.cfg.temperature
        logits = self.net(self.encode(marginal_rank))
        if mask is not None:
            logits = logits + (1.0 - mask) * (-1e9)
        return F.softmax(logits / max(tau, 1e-6), dim=-1)

    # ── argmax (deterministic, for evaluation) ────────────────────────────

    @torch.no_grad()
    def argmax_alloc(
        self,
        marginal_rank: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return argmax allocation index [B]."""
        logits = self.net(self.encode(marginal_rank))
        if mask is not None:
            logits = logits + (1.0 - mask) * (-1e9)
        return logits.argmax(dim=-1)
