"""MLP mechanism: reported valuations → distribution over K allocations.

Input:  v_report [B, A, m]  (endowment is NOT part of input; only used for mask)
Output: probs    [B, K]     (after IR+PE masking → softmax)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .allocations import AllocationIndex


class AllocationNet(nn.Module):
    def __init__(self, cfg: Config, aidx: AllocationIndex):
        super().__init__()
        self.cfg = cfg
        self.aidx = aidx

        A, m = cfg.num_agents, cfg.num_items
        K = aidx.num_allocations

        # Input: v_report flattened (A*m) + endow_mask flattened (A*m)
        in_dim = A * m + A * m

        h = cfg.hidden
        layers: list[nn.Module] = []
        for layer_i in range(cfg.depth):
            layers.append(nn.Linear(in_dim if layer_i == 0 else h, h))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        layers.append(nn.Linear(h, K))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        v_report: torch.Tensor,
        endow_idx: torch.Tensor,
        temperature: float | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning soft probs [B, K].

        Args:
          v_report:    [B, A, m]
          endow_idx:   [B]
          temperature: softmax temperature (default: cfg.temperature)
          mask:        [B, K] 1.0 = valid, 0.0 = masked out
        """
        if temperature is None:
            temperature = getattr(self.cfg, "temperature", 1.0)

        B = v_report.shape[0]

        # Encode endowment as per-agent item mask
        endow_mask = self.aidx.allocation_to_agent_masks(endow_idx)  # [B, A, m]
        endow_flat = endow_mask.reshape(B, -1).to(dtype=v_report.dtype)

        x = torch.cat(
            [v_report.reshape(B, -1), endow_flat],
            dim=1,
        )
        logits = self.net(x)

        if mask is not None:
            logits = logits + (1.0 - mask) * (-1e9)

        tau = max(float(temperature), 1e-6)
        y_soft = F.softmax(logits / tau, dim=-1)
        return y_soft

    @torch.no_grad()
    def predict_argmax(
        self,
        v_report: torch.Tensor,
        endow_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic allocation index [B]."""
        B = v_report.shape[0]
        endow_mask = self.aidx.allocation_to_agent_masks(endow_idx)
        endow_flat = endow_mask.reshape(B, -1).to(dtype=v_report.dtype)
        x = torch.cat([v_report.reshape(B, -1), endow_flat], dim=1)
        logits = self.net(x)
        if mask is not None:
            logits = logits + (1.0 - mask) * (-1e9)
        return torch.argmax(logits, dim=-1)

    @torch.no_grad()
    def predict_onehot(
        self,
        v_report: torch.Tensor,
        endow_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic one-hot distribution [B, K]."""
        idx = self.predict_argmax(v_report, endow_idx, mask=mask)
        K = self.aidx.num_allocations
        onehot = torch.zeros((idx.shape[0], K), device=idx.device, dtype=torch.float32)
        onehot.scatter_(1, idx.view(-1, 1), 1.0)
        return onehot
