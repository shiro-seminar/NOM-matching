"""Domain-aware data generation for ordinal preference experiments."""
from __future__ import annotations

import torch
from .config import Config
from .domains import DomainSpec, DOMAINS
from .allocations import build_all_allocs, score_matrix, random_endowment


def sample_domain_marginal_rank(
    cfg: Config,
    domain: DomainSpec,
    endow_idx: torch.Tensor,   # [B]
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample [B, A, m] marginal ranks consistent with domain constraints.

    For each batch element b and agent a:
      - items a owns (allocs[endow_idx[b], j] == a): sample from domain.owned_ranks
      - items a does not own: sample from domain.unowned_ranks

    For the strict domain: each agent's ranks are a random permutation of 0..m-1.
    """
    B, A, m = endow_idx.shape[0], cfg.num_agents, cfg.num_items

    allocs_all  = build_all_allocs(cfg)                        # [K, m], CPU
    endow_alloc = allocs_all[endow_idx.cpu()]                  # [B, m], CPU
    endow_flat  = endow_alloc.to(device)                       # [B, m]

    mr = torch.zeros(B, A, m, dtype=torch.long, device=device)

    if domain.strict:
        noise = torch.rand(B, A, m, device=device)
        mr = noise.argsort(dim=-1)
        return mr

    owned_t   = torch.tensor(domain.owned_ranks,   dtype=torch.long, device=device)
    unowned_t = torch.tensor(domain.unowned_ranks, dtype=torch.long, device=device)
    n_own, n_unown = len(owned_t), len(unowned_t)

    for a in range(A):
        for j in range(m):
            owned_mask = (endow_flat[:, j] == a)              # [B]
            idx_own    = torch.randint(0, n_own,   (B,), device=device)
            idx_unown  = torch.randint(0, n_unown, (B,), device=device)
            mr[:, a, j] = torch.where(owned_mask, owned_t[idx_own], unowned_t[idx_unown])

    return mr


def sample_batch(cfg: Config) -> dict[str, torch.Tensor]:
    """Sample a training batch using domain-consistent marginal ranks."""
    domain = DOMAINS[cfg.domain]
    device = torch.device(cfg.device)
    B      = cfg.batch_size

    endow_idx     = random_endowment(cfg, B, device)
    marginal_rank = sample_domain_marginal_rank(cfg, domain, endow_idx, device)
    S             = score_matrix(cfg, marginal_rank)

    return {"marginal_rank": marginal_rank, "endow_idx": endow_idx, "S": S}


def sample_domain_mr_flat(
    cfg: Config,
    domain: DomainSpec,
    endow_idx: torch.Tensor,   # [B]
    N: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample [B*N, A, m] domain-consistent ranks (for NOM inner loop).

    Each of the N samples per batch element respects the domain constraints
    given the fixed endow_idx.
    """
    B, A, m = endow_idx.shape[0], cfg.num_agents, cfg.num_items

    allocs_all  = build_all_allocs(cfg)
    endow_alloc = allocs_all[endow_idx.cpu()]                    # [B, m]
    endow_exp   = endow_alloc.unsqueeze(1).expand(B, N, m)       # [B, N, m]
    endow_flat  = endow_exp.reshape(B * N, m).to(device)         # [BN, m]

    BN = B * N
    mr = torch.zeros(BN, A, m, dtype=torch.long, device=device)

    if domain.strict:
        noise = torch.rand(BN, A, m, device=device)
        return noise.argsort(dim=-1)

    owned_t   = torch.tensor(domain.owned_ranks,   dtype=torch.long, device=device)
    unowned_t = torch.tensor(domain.unowned_ranks, dtype=torch.long, device=device)
    n_own, n_unown = len(owned_t), len(unowned_t)

    for a in range(A):
        for j in range(m):
            owned_mask = (endow_flat[:, j] == a)
            idx_own    = torch.randint(0, n_own,   (BN,), device=device)
            idx_unown  = torch.randint(0, n_unown, (BN,), device=device)
            mr[:, a, j] = torch.where(owned_mask, owned_t[idx_own], unowned_t[idx_unown])

    return mr
