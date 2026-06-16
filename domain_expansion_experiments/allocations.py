"""Allocation space utilities (domain-independent).

K = A^m deterministic allocations.
Encoding: allocs[k, j] = (k // A^j) % A  (which agent gets item j)

Score matrix S[B, A, K]:
  S[b, a, k] = -sum_{j : allocs[k,j]==a} marginal_rank[b, a, j]
  Higher score = more preferred bundle, consistent with any responsive preference.

FOSD (First-Order Stochastic Dominance) bundle comparison:
  For same-size bundles, X FOSD-weakly dominates Y iff
  sorted(X)[pos] <= sorted(Y)[pos] for all positions.
  (Rank 0 = best; lower rank number = better item.)
  This is equivalent to "X is weakly preferred under ALL responsive extensions."
  See fsd_ir_mask / fsd_pareto_mask / fsd_ir_pe_mask for unambiguous IR/PE.
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
    """Compute score matrix from marginal ranks.

    Args:
        marginal_rank: [B, A, m] int, rank of each item for each agent

    Returns:
        S: [B, A, K] float, S[b,a,k] = -sum_j mask[k,a,j]*rank[b,a,j]
    """
    device = marginal_rank.device
    masks  = agent_masks_all(cfg, device)           # [K, A, m]
    r      = marginal_rank.float()                  # [B, A, m]
    r_exp  = r.unsqueeze(2)                         # [B, A, 1, m]
    m_exp  = masks.permute(1, 0, 2).unsqueeze(0)   # [1, A, K, m]
    return -(r_exp * m_exp).sum(dim=-1)             # [B, A, K]


def random_endowment(cfg: Config, batch_size: int,
                     device: torch.device | str = "cpu") -> torch.Tensor:
    """Return [B] endowment indices where every agent holds >= 1 item."""
    A      = cfg.num_agents
    allocs = build_all_allocs(cfg)
    counts = torch.stack([(allocs == i).sum(dim=1) for i in range(A)], dim=1)
    valid  = counts.min(dim=1).values >= 1
    valid_idx = valid.nonzero(as_tuple=True)[0]
    sampled   = torch.randint(0, len(valid_idx), (batch_size,))
    return valid_idx[sampled].to(device)


def endowment_scores(S: torch.Tensor, endow_idx: torch.Tensor) -> torch.Tensor:
    """Score of each agent's endowment bundle: [B, A]."""
    B, A, _ = S.shape
    idx = endow_idx.view(B, 1, 1).expand(B, A, 1)
    return S.gather(2, idx).squeeze(2)


def pareto_mask(S: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Binary mask of Pareto-efficient allocations [B, K]."""
    B, A, K = S.shape
    device  = S.device
    St      = S.permute(0, 2, 1)   # [B, K, A]
    is_dominated = torch.zeros(B, K, dtype=torch.bool, device=device)

    for j_start in range(0, K, chunk):
        j_end = min(j_start + chunk, K)
        St_j  = St[:, j_start:j_end, :]
        diff  = St_j.unsqueeze(1) - St.unsqueeze(2)
        weakly_better   = (diff >= -1e-8).all(dim=-1)
        strictly_better = (diff >  1e-8).any(dim=-1)
        dominates = weakly_better & strictly_better
        i_idx = torch.arange(K, device=device)
        j_idx = torch.arange(j_start, j_end, device=device)
        dominates = dominates & ~(i_idx.unsqueeze(1) == j_idx.unsqueeze(0)).unsqueeze(0)
        is_dominated = is_dominated | dominates.any(dim=2)

    return (~is_dominated).float()


def ir_mask(S: torch.Tensor, endow_idx: torch.Tensor) -> torch.Tensor:
    """Binary mask of individually rational allocations [B, K]."""
    s0   = endowment_scores(S, endow_idx)
    diff = S - s0.unsqueeze(2)
    return (diff >= -1e-8).all(dim=1).float()


def balanced_mask(cfg: Config, endow_idx: torch.Tensor,
                  device: torch.device | str = "cpu") -> torch.Tensor:
    """Binary mask of balanced allocations (same item-count as endowment) [B, K]."""
    A, m   = cfg.num_agents, cfg.num_items
    allocs = build_all_allocs(cfg).to(device)
    K      = allocs.shape[0]
    B      = endow_idx.shape[0]
    counts = torch.stack(
        [(allocs == i).sum(dim=1) for i in range(A)], dim=1
    ).float()                                              # [K, A]
    endow_counts = counts[endow_idx.to(device)]            # [B, A]
    counts_exp   = counts.unsqueeze(0).expand(B, -1, -1)  # [B, K, A]
    endow_exp    = endow_counts.unsqueeze(1).expand(-1, K, -1)
    return (counts_exp == endow_exp).all(dim=-1).float()   # [B, K]


# ---------------------------------------------------------------------------
# FOSD-based unambiguous IR / PE / IR+PE masks
# ---------------------------------------------------------------------------

def bundle_sorted_ranks(cfg: Config, marginal_rank: torch.Tensor) -> torch.Tensor:
    """Sorted bundle rank vectors for all (batch, allocation, agent) triples.

    Returns [B, K, A, m] float tensor.
    For agent a in allocation k:
      - Items a receives: their true rank value (lower rank = better, 0 = best)
      - Items a does NOT receive: padded with cfg.num_ranks (sentinel, always sorts last)
    Sorted ascending within the last dim (best item first).

    FOSD-weak dominance of allocation k over k' for agent a:
        result[b, k, a, :] <= result[b, k', a, :]  elementwise
    REQUIRES: same-size bundles (guaranteed by balanced constraint).
    """
    device = marginal_rank.device
    B, A, m = marginal_rank.shape
    R = float(cfg.num_ranks)   # sentinel: larger than any valid rank, sorts last

    masks = agent_masks_all(cfg, device)              # [K, A, m] binary
    mr    = marginal_rank.float()                     # [B, A, m]

    # [B, K, A, m]: owned items keep their rank; unowned get sentinel R
    mr_exp    = mr.unsqueeze(1).expand(B, masks.shape[0], A, m)      # [B, K, A, m]
    masks_exp = masks.unsqueeze(0).expand(B, masks.shape[0], A, m)   # [B, K, A, m]
    bundle    = mr_exp * masks_exp + R * (1.0 - masks_exp)           # [B, K, A, m]

    # Sort ascending: best (lowest rank) first; sentinel always last
    sorted_ranks, _ = torch.sort(bundle, dim=-1)
    return sorted_ranks   # [B, K, A, m]


def fsd_ir_mask(cfg: Config, marginal_rank: torch.Tensor,
                endow_idx: torch.Tensor) -> torch.Tensor:
    """FOSD-based IR mask [B, K].

    Allocation k is IR for agent a iff a's bundle in k FOSD-weakly dominates
    a's endowment bundle, i.e., sorted(alloc_ranks)[pos] <= sorted(endow_ranks)[pos]
    for every position pos.

    Lower rank number = better item (rank 0 is best).
    Assumes balanced allocations (same bundle size as endowment per agent).
    Returns 1.0 where IR holds for ALL agents, 0.0 otherwise.
    """
    B = marginal_rank.shape[0]
    sorted_all = bundle_sorted_ranks(cfg, marginal_rank)   # [B, K, A, m]
    K, A, m_   = sorted_all.shape[1], sorted_all.shape[2], sorted_all.shape[3]

    # Endowment sorted ranks: gather along K-dim, then broadcast
    endow_sorted = sorted_all.gather(
        1, endow_idx.view(B, 1, 1, 1).expand(B, 1, A, m_)
    )                                                       # [B, 1, A, m]
    endow_sorted = endow_sorted.expand(B, K, A, m_)        # [B, K, A, m]

    # k is IR for agent a iff allocation bundle ranks <= endowment ranks ∀ positions
    # (lower rank = better, so <= means "at least as good")
    weak   = (sorted_all <= endow_sorted + 1e-8).all(dim=-1)   # [B, K, A]
    return weak.all(dim=-1).float()                             # [B, K]


def unamb_pe_mask(cfg: Config, marginal_rank: torch.Tensor,
                  feasible_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Unambiguous Pareto-efficiency mask [B, K].

    Allocation k is unambiguously PE iff no feasible k' satisfies BOTH:
      (1) ∃j: k's bundle does NOT FOSD-weakly dominate k'(j)
          (some agent could be made better off under k')
      (2) ∀i: k's bundle does NOT FOSD-strictly dominate k'(i)
          (no agent is made unambiguously worse off under k')

    Manjunath-Westkamp (2025): unambiguous PE ⊆ additive PE ⊆ FOSD-undominated.

    feasible_mask [B, K]: only these allocations can act as improvers.
    Returns 1.0 where unambiguously PE, 0.0 where improved.
    """
    sorted_all = bundle_sorted_ranks(cfg, marginal_rank)   # [B, K, A, m]
    B, K, A, m_ = sorted_all.shape
    device = sorted_all.device

    if feasible_mask is None:
        feasible_mask = torch.ones(B, K, dtype=torch.float32, device=device)

    # diff[b, k, k', a, pos] = sorted[b,k,a,pos] - sorted[b,k',a,pos]
    x    = sorted_all.unsqueeze(2)    # [B, K, 1, A, m]
    y    = sorted_all.unsqueeze(1)    # [B, 1, K, A, m]
    diff = x - y                       # [B, K, K, A, m]: sorted[k] - sorted[k']

    # k(a) ⪰_FOSD k'(a): sorted[k,a] ≤ sorted[k',a] all pos → diff ≤ 0
    weak_dom   = (diff <= 1e-8).all(dim=-1)              # [B, K, K, A]
    # k(a) ≻_FOSD k'(a): weak AND some pos diff < 0
    strict_dom = weak_dom & (diff < -1e-8).any(dim=-1)   # [B, K, K, A]

    # k' unambiguously improves k:
    cond1 = (~weak_dom).any(dim=-1)     # ∃j: k NOT weakly-dom k' for j  [B, K, K]
    cond2 = (~strict_dom).all(dim=-1)   # ∀i: k NOT strictly-dom k' for i
    improves = cond1 & cond2

    eye = torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)   # [1, K, K]
    feasible_exp = feasible_mask.bool().unsqueeze(1)                    # [B, 1, K]
    improves = improves & ~eye & feasible_exp                           # [B, K, K]

    is_improved = improves.any(dim=2)   # [B, K]
    return (~is_improved).float()


def unamb_ir_pe_mask(cfg: Config, marginal_rank: torch.Tensor,
                     endow_idx: torch.Tensor) -> torch.Tensor:
    """Unambiguous IR ∩ PE ∩ Balanced mask.

    Combines:
      fsd_ir_mask  : unambiguous individual rationality (FOSD vs endowment)
      unamb_pe_mask: unambiguous Pareto efficiency (within balanced allocations)
      balanced_mask: same item-count as endowment

    Falls back to endowment-only if the intersection is empty.
    """
    bal      = balanced_mask(cfg, endow_idx, marginal_rank.device)   # [B, K]
    unamb_ir = fsd_ir_mask(cfg, marginal_rank, endow_idx)             # [B, K]
    unamb_pe = unamb_pe_mask(cfg, marginal_rank, feasible_mask=bal)   # [B, K]

    m_mask = unamb_ir * unamb_pe * bal
    empty  = (m_mask.sum(dim=1) < 0.5)
    if empty.any():
        endow_oh = torch.zeros_like(m_mask)
        endow_oh.scatter_(1, endow_idx.view(-1, 1), 1.0)
        m_mask = torch.where(empty.unsqueeze(1), endow_oh, m_mask)
    return m_mask


def ir_pe_mask(cfg: Config, S: torch.Tensor, endow_idx: torch.Tensor) -> torch.Tensor:
    """IR ∩ PE ∩ Balanced mask. Falls back to endowment-only when empty."""
    m     = ir_mask(S, endow_idx) * pareto_mask(S) * balanced_mask(cfg, endow_idx, S.device)
    empty = (m.sum(dim=1) < 0.5)
    if empty.any():
        endow_oh = torch.zeros_like(m)
        endow_oh.scatter_(1, endow_idx.view(-1, 1), 1.0)
        m = torch.where(empty.unsqueeze(1), endow_oh, m)
    return m
