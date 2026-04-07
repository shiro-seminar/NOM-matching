"""Allocation space for 2 agents, 3 items.

K = 2^3 = 8 deterministic allocations.
Each allocation k is encoded as a 3-bit binary number:
  bit j = 0  →  item j assigned to agent 0
  bit j = 1  →  item j assigned to agent 1

Layout of all_allocs tensor (K=8, m=3):
  k=0: [0,0,0]   k=1: [1,0,0]   k=2: [0,1,0]   k=3: [1,1,0]
  k=4: [0,0,1]   k=5: [1,0,1]   k=6: [0,1,1]   k=7: [1,1,1]
  (column j = which agent gets item j)

Pure additive utilities:
  u_i(k) = sum_{j: alloc[k,j]==i} v[i,j]
"""
from __future__ import annotations

import torch


NUM_AGENTS = 2
NUM_ITEMS  = 3
K          = 2 ** NUM_ITEMS   # 8


# ── Pre-computed allocation table (K, m) ────────────────────────────────────

def _build_all_allocs() -> torch.Tensor:
    """Return [K, m] long tensor: allocs[k, j] ∈ {0,1} = agent receiving item j."""
    allocs = torch.zeros(K, NUM_ITEMS, dtype=torch.long)
    for k in range(K):
        for j in range(NUM_ITEMS):
            allocs[k, j] = (k >> j) & 1
    return allocs


# Module-level constant (CPU). Move to device as needed.
ALL_ALLOCS: torch.Tensor = _build_all_allocs()   # [8, 3]


# ── Agent-item masks ─────────────────────────────────────────────────────────

def agent_masks(device: torch.device | str = "cpu") -> torch.Tensor:
    """Return [K, A=2, m=3] binary float: masks[k, i, j] = 1 iff agent i gets item j under alloc k."""
    A = NUM_AGENTS
    allocs = ALL_ALLOCS.to(device)               # [K, m]
    masks = torch.zeros(K, A, NUM_ITEMS, dtype=torch.float32, device=device)
    for i in range(A):
        masks[:, i, :] = (allocs == i).float()
    return masks                                  # [8, 2, 3]


# ── Utilities for all allocations ────────────────────────────────────────────

def all_utilities(v: torch.Tensor) -> torch.Tensor:
    """Compute utility of every allocation for every agent.

    Args:
        v:  [B, A=2, m=3]  item valuations (additive, no synergy)

    Returns:
        U:  [B, A=2, K=8]  U[b, i, k] = sum_j masks[k,i,j] * v[b,i,j]
    """
    device = v.device
    masks = agent_masks(device)   # [K, A, m]
    # U[b,i,k] = sum_j masks[k,i,j] * v[b,i,j]
    # Reshape for broadcast:  v: [B,A,1,m]  masks: [1,A,K,m] (after transpose)
    v_exp = v.unsqueeze(2)                           # [B, A, 1, m]
    m_exp = masks.permute(1, 0, 2).unsqueeze(0)     # [1, A, K, m]
    U = (v_exp * m_exp).sum(dim=-1)                  # [B, A, K]
    return U


# ── Endowment helpers ────────────────────────────────────────────────────────

def random_endowment(batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return [B] random endowment allocation indices in [0, K)."""
    return torch.randint(0, K, (batch_size,), device=device)


def endowment_utilities(U: torch.Tensor, endow_idx: torch.Tensor) -> torch.Tensor:
    """Outside-option utility for each agent.

    Args:
        U:         [B, A, K]
        endow_idx: [B]

    Returns:
        u0:  [B, A]
    """
    B, A, _ = U.shape
    idx = endow_idx.view(B, 1, 1).expand(B, A, 1)
    return U.gather(2, idx).squeeze(2)   # [B, A]


# ── Pareto efficiency ────────────────────────────────────────────────────────

def pareto_mask(U: torch.Tensor) -> torch.Tensor:
    """Return binary mask of Pareto-efficient (non-dominated) allocations.

    An allocation k is PE iff no other allocation k' weakly Pareto-dominates it
    (i.e. ∀i u_i(k') ≥ u_i(k)  AND  ∃i u_i(k') > u_i(k)).

    Args:
        U:  [B, A, K]

    Returns:
        mask: [B, K]  float, 1.0 = PE, 0.0 = dominated
    """
    B, A, Kk = U.shape
    # U: [B, K, A]  (reorder for easier broadcasting)
    Ut = U.permute(0, 2, 1)           # [B, K, A]

    # For each pair (k, k'): does k' dominate k?
    # diff[b, k', k, a] = U[b,a,k'] - U[b,a,k]
    # k' dom k  ⟺  all(diff ≥ 0) and any(diff > 0)
    Ut_k  = Ut.unsqueeze(1)           # [B, 1,  K, A]  = candidate dominator k'
    Ut_kp = Ut.unsqueeze(2)           # [B, K,  1, A]  = allocation being tested k

    diff = Ut_k - Ut_kp               # [B, K', K, A]  diff[b, k', k, a] = u(k') - u(k)

    weakly_better = (diff >= -1e-8).all(dim=-1)    # [B, K', K]
    strictly_better = (diff > 1e-8).any(dim=-1)    # [B, K', K]
    dominates = weakly_better & strictly_better    # [B, K', K]  dominates[b, k', k] = True iff k' ≻ k

    # k is dominated if any k' ≠ k dominates it
    # Exclude self-domination (k'=k diagonal)
    eye = torch.eye(Kk, dtype=torch.bool, device=U.device).unsqueeze(0)   # [1, K, K]
    dominates = dominates & ~eye                   # [B, K', K]

    # dominates[b,i,j] = "j dominates i"
    # k is dominated iff exists j s.t. j dominates k = dominates[b, k, j] for some j
    is_dominated = dominates.any(dim=2)            # [B, K]  is_dominated[b,k] = exists j: j dominates k
    pe = (~is_dominated).float()                   # [B, K]  1.0 = PE
    return pe


# ── IR mask ──────────────────────────────────────────────────────────────────

def ir_mask(U: torch.Tensor, endow_idx: torch.Tensor) -> torch.Tensor:
    """Return binary mask of individually rational allocations.

    IR: every agent weakly prefers the allocation to their endowment.

    Args:
        U:         [B, A, K]
        endow_idx: [B]

    Returns:
        mask: [B, K]  float, 1.0 = IR
    """
    u0 = endowment_utilities(U, endow_idx)   # [B, A]
    # diff[b, a, k] = U[b,a,k] - u0[b,a]
    diff = U - u0.unsqueeze(2)               # [B, A, K]
    feasible = (diff >= -1e-8).all(dim=1)    # [B, K]
    return feasible.float()


# ── Combined IR ∩ PE mask ────────────────────────────────────────────────────

def ir_pe_mask(U: torch.Tensor, endow_idx: torch.Tensor) -> torch.Tensor:
    """Return mask of allocations that are both IR and PE.

    When no IR∩PE allocation exists (endowment is not PE-improvable with IR),
    falls back to allowing only the endowment itself.

    Returns:
        mask: [B, K]  float
    """
    m = ir_mask(U, endow_idx) * pareto_mask(U)   # [B, K]

    # Fallback: where mask is all-zero, allow the endowment (trivially IR)
    empty = (m.sum(dim=1) < 0.5)                 # [B]
    if empty.any():
        endow_onehot = torch.zeros_like(m)
        endow_onehot.scatter_(1, endow_idx.view(-1, 1), 1.0)
        m = torch.where(empty.unsqueeze(1), endow_onehot, m)

    return m
