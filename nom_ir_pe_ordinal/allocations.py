"""Allocation space for ordinal preferences.

K = A^m deterministic allocations (same encoding as nom_ir_pe_3x4).
Encoding: allocs[k, j] = (k // A^j) % A  (which agent gets item j)

Score matrix S[B, A, K]:
  S[b, a, k] = -sum_{j : allocs[k,j]==a} marginal_rank[b, a, j]

  Higher score = more preferred bundle.  This is consistent with ANY
  responsive preference: replacing a high-rank item with a lower-rank
  item strictly increases the score, which is exactly the one-step
  improvement condition in responsiveness.
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
    powers = torch.tensor([A ** j for j in range(m)], dtype=torch.long)
    k_idx  = torch.arange(K, dtype=torch.long)
    return (k_idx.unsqueeze(1) // powers.unsqueeze(0)) % A   # [K, m]


def agent_masks_all(cfg: Config, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return [K, A, m] binary float: masks[k, i, j] = 1 iff agent i gets item j."""
    A, m = cfg.num_agents, cfg.num_items
    allocs = build_all_allocs(cfg).to(device)
    K = allocs.shape[0]
    masks = torch.zeros(K, A, m, dtype=torch.float32, device=device)
    for i in range(A):
        masks[:, i, :] = (allocs == i).float()
    return masks   # [K, A, m]


def score_matrix(cfg: Config, marginal_rank: torch.Tensor) -> torch.Tensor:
    """Compute ordinal score matrix from marginal ranks.

    Args:
        marginal_rank: [B, A, m]  int, rank of each item for each agent
                       (0 = most preferred, cfg.num_ranks-1 = least preferred)

    Returns:
        S: [B, A, K]  float, S[b,a,k] = -sum_j mask[k,a,j]*rank[b,a,j]
           Higher S = more preferred bundle.
    """
    device = marginal_rank.device
    masks = agent_masks_all(cfg, device)          # [K, A, m]
    # S[b,a,k] = -einsum_{j} masks[k,a,j] * rank[b,a,j]
    # = -(masks[K,A,m] @ rank[B,A,m].T)  rearranged
    r = marginal_rank.float()                     # [B, A, m]
    # Equivalent to all_utilities but negated: score = -rank_sum
    r_exp  = r.unsqueeze(2)                       # [B, A, 1, m]
    m_exp  = masks.permute(1, 0, 2).unsqueeze(0) # [1, A, K, m]
    S = -(r_exp * m_exp).sum(dim=-1)             # [B, A, K]
    return S


def random_endowment(cfg: Config, batch_size: int,
                     device: torch.device | str = "cpu") -> torch.Tensor:
    """Return [B] endowment indices where every agent holds >= 1 item.

    For 3 agents / 4 items: the 36 valid endowments all follow the
    (2,1,1) count pattern.
    """
    A = cfg.num_agents
    allocs = build_all_allocs(cfg)
    counts = torch.stack([(allocs == i).sum(dim=1) for i in range(A)], dim=1)
    valid  = counts.min(dim=1).values >= 1
    valid_idx = valid.nonzero(as_tuple=True)[0]
    sampled = torch.randint(0, len(valid_idx), (batch_size,))
    return valid_idx[sampled].to(device)


def endowment_scores(S: torch.Tensor, endow_idx: torch.Tensor) -> torch.Tensor:
    """Score of each agent's endowment bundle.

    Args:
        S:         [B, A, K]
        endow_idx: [B]

    Returns:
        s0: [B, A]
    """
    B, A, _ = S.shape
    idx = endow_idx.view(B, 1, 1).expand(B, A, 1)
    return S.gather(2, idx).squeeze(2)


def pareto_mask(S: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Binary mask of Pareto-efficient allocations.

    Args:
        S:     [B, A, K]  score matrix (higher = more preferred)
        chunk: chunked computation to avoid OOM

    Returns:
        mask: [B, K] float (1=PE, 0=dominated)
    """
    B, A, K = S.shape
    device   = S.device
    St       = S.permute(0, 2, 1)   # [B, K, A]
    is_dominated = torch.zeros(B, K, dtype=torch.bool, device=device)

    for j_start in range(0, K, chunk):
        j_end  = min(j_start + chunk, K)
        St_j   = St[:, j_start:j_end, :]          # [B, chunk, A]
        diff   = St_j.unsqueeze(1) - St.unsqueeze(2)  # [B, K, chunk, A]
        weakly_better   = (diff >= -1e-8).all(dim=-1)
        strictly_better = (diff >  1e-8).any(dim=-1)
        dominates = weakly_better & strictly_better

        i_idx = torch.arange(K, device=device)
        j_idx = torch.arange(j_start, j_end, device=device)
        dominates = dominates & ~(i_idx.unsqueeze(1) == j_idx.unsqueeze(0)).unsqueeze(0)
        is_dominated = is_dominated | dominates.any(dim=2)

    return (~is_dominated).float()


def ir_mask(S: torch.Tensor, endow_idx: torch.Tensor) -> torch.Tensor:
    """Binary mask of individually rational allocations (S >= endowment score).

    Returns:
        mask: [B, K] float
    """
    s0   = endowment_scores(S, endow_idx)   # [B, A]
    diff = S - s0.unsqueeze(2)              # [B, A, K]
    return (diff >= -1e-8).all(dim=1).float()


def balanced_mask(
    cfg: Config,
    endow_idx: torch.Tensor,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Binary mask of balanced allocations (same item-count per agent as endowment).

    Returns:
        mask: [B, K] float
    """
    A, m = cfg.num_agents, cfg.num_items
    allocs = build_all_allocs(cfg).to(device)
    K = allocs.shape[0]
    B = endow_idx.shape[0]

    counts = torch.stack(
        [(allocs == i).sum(dim=1) for i in range(A)], dim=1
    ).float()   # [K, A]

    endow_counts = counts[endow_idx.to(device)]                  # [B, A]
    counts_exp   = counts.unsqueeze(0).expand(B, -1, -1)        # [B, K, A]
    endow_exp    = endow_counts.unsqueeze(1).expand(-1, K, -1)  # [B, K, A]
    return (counts_exp == endow_exp).all(dim=-1).float()         # [B, K]


def ir_pe_mask(cfg: Config, S: torch.Tensor, endow_idx: torch.Tensor) -> torch.Tensor:
    """IR ∩ PE ∩ Balanced mask.  Falls back to endowment-only when empty."""
    m = ir_mask(S, endow_idx) * pareto_mask(S) * balanced_mask(cfg, endow_idx, S.device)
    empty = (m.sum(dim=1) < 0.5)
    if empty.any():
        endow_onehot = torch.zeros_like(m)
        endow_onehot.scatter_(1, endow_idx.view(-1, 1), 1.0)
        m = torch.where(empty.unsqueeze(1), endow_onehot, m)
    return m
