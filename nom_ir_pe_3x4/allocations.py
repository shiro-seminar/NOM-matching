"""General allocation space for A agents, m items.

K = A^m deterministic allocations.
Encoding: allocs[k, j] = (k // A^j) % A  (which agent gets item j)

For 3 agents, 4 items: K = 81.
"""
from __future__ import annotations

import torch
from .config import Config


def num_allocations(cfg: Config) -> int:
    return cfg.num_agents ** cfg.num_items


def build_all_allocs(cfg: Config) -> torch.Tensor:
    """Return [K, m] long tensor: allocs[k, j] = agent receiving item j."""
    A, m = cfg.num_agents, cfg.num_items
    K = A ** m
    powers = torch.tensor([A ** j for j in range(m)], dtype=torch.long)  # [m]
    k_idx = torch.arange(K, dtype=torch.long)                             # [K]
    allocs = (k_idx.unsqueeze(1) // powers.unsqueeze(0)) % A             # [K, m]
    return allocs


def agent_masks_all(cfg: Config, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return [K, A, m] binary float: masks[k, i, j] = 1 iff agent i gets item j."""
    A, m = cfg.num_agents, cfg.num_items
    allocs = build_all_allocs(cfg).to(device)    # [K, m]
    K = allocs.shape[0]
    masks = torch.zeros(K, A, m, dtype=torch.float32, device=device)
    for i in range(A):
        masks[:, i, :] = (allocs == i).float()
    return masks                                  # [K, A, m]


def all_utilities(cfg: Config, v: torch.Tensor) -> torch.Tensor:
    """Compute utility of every allocation for every agent (additive, no synergy).

    Args:
        v:  [B, A, m]

    Returns:
        U:  [B, A, K]
    """
    device = v.device
    masks = agent_masks_all(cfg, device)          # [K, A, m]
    # U[b,i,k] = sum_j masks[k,i,j] * v[b,i,j]
    v_exp = v.unsqueeze(2)                        # [B, A, 1, m]
    m_exp = masks.permute(1, 0, 2).unsqueeze(0)  # [1, A, K, m]
    U = (v_exp * m_exp).sum(dim=-1)              # [B, A, K]
    return U


def random_endowment(cfg: Config, batch_size: int,
                     device: torch.device | str = "cpu") -> torch.Tensor:
    """Return [B] random endowment indices in [0, K)."""
    K = num_allocations(cfg)
    return torch.randint(0, K, (batch_size,), device=device)


def endowment_utilities(U: torch.Tensor, endow_idx: torch.Tensor) -> torch.Tensor:
    """Outside-option utility for each agent.

    Args:
        U:         [B, A, K]
        endow_idx: [B]

    Returns:
        u0: [B, A]
    """
    B, A, _ = U.shape
    idx = endow_idx.view(B, 1, 1).expand(B, A, 1)
    return U.gather(2, idx).squeeze(2)


def pareto_mask(U: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Binary mask of Pareto-efficient allocations.

    Uses chunked computation to avoid OOM for large K (e.g. K=81).

    Args:
        U:     [B, A, K]
        chunk: number of K rows to process at once

    Returns:
        mask: [B, K]  float (1=PE, 0=dominated)
    """
    B, A, K = U.shape
    device = U.device
    Ut = U.permute(0, 2, 1)              # [B, K, A]

    is_dominated = torch.zeros(B, K, dtype=torch.bool, device=device)

    # Process in chunks of j (candidate dominators)
    for j_start in range(0, K, chunk):
        j_end = min(j_start + chunk, K)
        Ut_j = Ut[:, j_start:j_end, :]  # [B, chunk, A]

        # diff[b, i, j_chunk, a] = U(j) - U(i)
        diff = Ut_j.unsqueeze(1) - Ut.unsqueeze(2)   # [B, K, chunk, A]

        weakly_better   = (diff >= -1e-8).all(dim=-1)  # [B, K, chunk]
        strictly_better = (diff >  1e-8).any(dim=-1)   # [B, K, chunk]
        dominates = weakly_better & strictly_better     # [B, K, chunk]  "j dom i"

        # Exclude self-domination
        i_idx = torch.arange(K, device=device)
        j_idx = torch.arange(j_start, j_end, device=device)
        self_mask = (i_idx.unsqueeze(1) == j_idx.unsqueeze(0))  # [K, chunk]
        dominates = dominates & ~self_mask.unsqueeze(0)

        # i is dominated if any j in this chunk dominates it
        is_dominated = is_dominated | dominates.any(dim=2)

    return (~is_dominated).float()


def ir_mask(U: torch.Tensor, endow_idx: torch.Tensor) -> torch.Tensor:
    """Binary mask of individually rational allocations.

    Returns:
        mask: [B, K]  float
    """
    u0 = endowment_utilities(U, endow_idx)   # [B, A]
    diff = U - u0.unsqueeze(2)               # [B, A, K]
    return (diff >= -1e-8).all(dim=1).float()


def balanced_mask(
    cfg: Config,
    endow_idx: torch.Tensor,   # [B]
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Binary mask of balanced allocations.

    Allocation k is balanced w.r.t. endowment e iff
      |{j : allocs[k, j] == i}| == |{j : allocs[e, j] == i}|  for all i ∈ I.

    Returns:
        mask: [B, K] float
    """
    A, m = cfg.num_agents, cfg.num_items
    allocs = build_all_allocs(cfg).to(device)      # [K, m]
    K = allocs.shape[0]
    B = endow_idx.shape[0]

    # Number of items each agent receives in every allocation: [K, A]
    counts = torch.stack(
        [(allocs == i).sum(dim=1) for i in range(A)], dim=1
    ).float()

    endow_counts = counts[endow_idx.to(device)]                    # [B, A]
    counts_exp   = counts.unsqueeze(0).expand(B, -1, -1)          # [B, K, A]
    endow_exp    = endow_counts.unsqueeze(1).expand(-1, K, -1)    # [B, K, A]

    return (counts_exp == endow_exp).all(dim=-1).float()           # [B, K]


def ir_pe_mask(cfg: Config, U: torch.Tensor, endow_idx: torch.Tensor) -> torch.Tensor:
    """IR ∩ PE ∩ Balanced mask. Falls back to endowment-only when empty."""
    m = ir_mask(U, endow_idx) * pareto_mask(U) * balanced_mask(cfg, endow_idx, U.device)

    empty = (m.sum(dim=1) < 0.5)
    if empty.any():
        endow_onehot = torch.zeros_like(m)
        endow_onehot.scatter_(1, endow_idx.view(-1, 1), 1.0)
        m = torch.where(empty.unsqueeze(1), endow_onehot, m)
    return m
