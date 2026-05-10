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
from .allocations import score_matrix, random_endowment, build_all_allocs


def sample_batch(cfg: Config) -> dict[str, torch.Tensor]:
    """Sample a training batch with correct trichotomous domain constraints.

    Trichotomous (Manjunath-Westkamp 2025): epsilon(3)=0, i.e. owned items
    must be in class 1 or 2 (rank 0 or 1). Unowned items can be in any class.
    """
    device = torch.device(cfg.device)
    B, A, m, R = cfg.batch_size, cfg.num_agents, cfg.num_items, cfg.num_ranks

    endow_idx   = random_endowment(cfg, B, device)
    allocs      = build_all_allocs(cfg)
    endow_alloc = allocs[endow_idx.cpu()].to(device)   # [B, m]

    marginal_rank = torch.zeros(B, A, m, dtype=torch.long, device=device)
    for a in range(A):
        for j in range(m):
            owned_mask = (endow_alloc[:, j] == a)          # [B]
            r_owned    = torch.randint(0, R - 1, (B,), device=device)  # {0,..,R-2}
            r_unowned  = torch.randint(0, R,     (B,), device=device)  # {0,..,R-1}
            marginal_rank[:, a, j] = torch.where(owned_mask, r_owned, r_unowned)

    S = score_matrix(cfg, marginal_rank)
    return {"marginal_rank": marginal_rank, "endow_idx": endow_idx, "S": S}
