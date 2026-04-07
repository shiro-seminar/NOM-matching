"""Allocation space management and Pareto-efficiency judgment.

An allocation assigns each of M items to one of N agents.
Total allocations K = N^M.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class AllocationIndex:
    """Manages allocation space for N agents and M items."""
    num_agents: int
    num_items: int

    @property
    def num_allocations(self) -> int:
        return self.num_agents ** self.num_items

    # ---- Enumeration ----

    def all_allocations_tensor(self) -> torch.Tensor:
        """All K allocations as [K, m] (each row assigns items to agents 0..A-1)."""
        A, m = self.num_agents, self.num_items
        K = self.num_allocations
        allocs = torch.zeros((K, m), dtype=torch.long)
        for k in range(K):
            val = k
            for j in range(m):
                allocs[k, j] = val % A
                val //= A
        return allocs

    # ---- Index ↔ Mask conversions ----

    def allocation_to_agent_masks(self, alloc_idx: torch.Tensor) -> torch.Tensor:
        """[B] allocation indices → [B, A, m] binary masks."""
        device = alloc_idx.device
        B = alloc_idx.shape[0]
        A, m = self.num_agents, self.num_items
        all_allocs = self.all_allocations_tensor().to(device)  # [K, m]
        allocs = all_allocs[alloc_idx.long()]  # [B, m]
        masks = torch.zeros((B, A, m), dtype=torch.float32, device=device)
        for i in range(A):
            masks[:, i, :] = (allocs == i).float()
        return masks

    def agent_masks_to_allocation(self, masks: torch.Tensor) -> torch.Tensor:
        """[B, A, m] binary masks → [B] allocation indices."""
        device = masks.device
        A, m = self.num_agents, self.num_items
        allocs = masks.argmax(dim=1)  # [B, m]
        powers = torch.tensor([A ** j for j in range(m)], device=device, dtype=torch.long)
        return (allocs * powers).sum(dim=1)

    # ---- Endowment sampling ----

    def random_endowment_no_disposal(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Random endowment: each item → uniform random agent.  Returns [B]."""
        A, m = self.num_agents, self.num_items
        allocs = torch.randint(0, A, (batch_size, m), device=device)
        powers = torch.tensor([A ** j for j in range(m)], device=device, dtype=torch.long)
        return (allocs * powers).sum(dim=1)


# ---- Balanced allocations ----

def balanced_mask(
    aidx: AllocationIndex,
    endow_idx: torch.Tensor,   # [B]
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Binary mask of balanced allocations.

    Allocation k is balanced w.r.t. endowment e iff
      |{j : allocs[k, j] == i}| == |{j : allocs[e, j] == i}|  for all i ∈ I.

    Returns:
        mask: [B, K] float
    """
    A, m = aidx.num_agents, aidx.num_items
    allocs = aidx.all_allocations_tensor().to(device)  # [K, m]
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


# ---- Pareto efficiency ----

def pareto_mask(U: torch.Tensor) -> torch.Tensor:
    """Identify Pareto-efficient allocations.

    Args:
        U: [B, A, K]  utility of each agent under each of K allocations.

    Returns:
        pe_mask: [B, K]  float mask, 1.0 = Pareto-efficient, 0.0 = dominated.

    An allocation k is Pareto-efficient iff no other allocation k' satisfies:
      U[:, i, k'] >= U[:, i, k] for all i  AND  strict for at least one i.

    Implementation: for each k, check against all k' (vectorised over B).
    Complexity O(B * K^2 * A) — fine for small K (e.g. 81).
    """
    B, A, K = U.shape
    # U_k: [B, A, K, 1],  U_k': [B, A, 1, K]
    U_k = U.unsqueeze(3)   # [B, A, K, 1]   allocation under scrutiny
    U_kp = U.unsqueeze(2)  # [B, A, 1, K]   candidate dominator

    # weakly_better[b, k, k'] = True iff k' >= k for ALL agents
    weakly = (U_kp >= U_k - 1e-7).all(dim=1)        # [B, K, K]
    # strictly_better[b, k, k'] = True iff k' > k for SOME agent
    strictly = (U_kp > U_k + 1e-7).any(dim=1)        # [B, K, K]

    # dominated[b, k] = exists k' that Pareto-dominates k
    dominated = (weakly & strictly).any(dim=2)         # [B, K]
    return (~dominated).float()                        # [B, K]
