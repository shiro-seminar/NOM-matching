"""Data generation for ordinal preference setting.

A training profile is:
  - marginal_rank [B, A, m]: rank of each item for each agent
                             (0 = most preferred, num_ranks-1 = least preferred)
  - endow_idx [B]:           balanced endowment (each agent >= 1 item)
  - S [B, A, K]:             score matrix; S[b,a,k] = -rank_sum of agent a's
                             bundle under allocation k.
                             Higher S = more preferred.  Consistent with any
                             responsive preference over the trichotomous domain.
"""
from __future__ import annotations
import torch
from .config import Config
from .allocations import score_matrix, random_endowment


def sample_batch(cfg: Config) -> dict[str, torch.Tensor]:
    device = torch.device(cfg.device)
    B, A, m, R = cfg.batch_size, cfg.num_agents, cfg.num_items, cfg.num_ranks

    marginal_rank = torch.randint(0, R, (B, A, m), device=device)
    endow_idx     = random_endowment(cfg, B, device)
    S             = score_matrix(cfg, marginal_rank)

    return {"marginal_rank": marginal_rank, "endow_idx": endow_idx, "S": S}
