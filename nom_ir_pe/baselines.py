"""Baseline mechanisms for comparison.

1. Max-Welfare IRPE  — welfare ceiling (no SP/NOM)
2. TTC Bundle Exchange — SP (theory)
3. Greedy IRPE Swap   — no SP guarantee
4. Random IRPE        — random among IRPE allocations
"""
from __future__ import annotations
import torch
from .allocations import AllocationIndex, pareto_mask
from .config import Config
from .data_gen import compute_all_utilities, outside_option_utils
from .losses import compute_ir_mask, compute_irpe_mask


# ============================================================
#  1. Max-Welfare IRPE
# ============================================================

@torch.no_grad()
def max_welfare_irpe(
    cfg: Config,
    aidx: AllocationIndex,
    U_true: torch.Tensor,        # [B, A, K]
    endow_idx: torch.Tensor,     # [B]
) -> torch.Tensor:
    """Pick the IRPE allocation that maximises total welfare. Returns [B]."""
    mask = compute_irpe_mask(cfg, aidx, U_true, endow_idx)   # [B, K]
    welfare = U_true.sum(dim=1)  # [B, K]
    welfare_masked = welfare + (1.0 - mask) * (-1e9)
    return welfare_masked.argmax(dim=1)  # [B]


# ============================================================
#  2. TTC Bundle Exchange
# ============================================================

@torch.no_grad()
def ttc_bundle_exchange(
    cfg: Config,
    aidx: AllocationIndex,
    U_true: torch.Tensor,        # [B, A, K]
    endow_idx: torch.Tensor,     # [B]
) -> torch.Tensor:
    """TTC with bundle exchange for N agents. Returns [B]."""
    device = U_true.device
    B = U_true.shape[0]
    A = cfg.num_agents

    all_allocs = aidx.all_allocations_tensor().to(device)
    current_allocs = all_allocs[endow_idx.long()]  # [B, m]
    powers = torch.tensor(
        [A ** p for p in range(cfg.num_items)], device=device, dtype=torch.long,
    )

    # util_matrix[b, i, j] = utility of agent i receiving agent j's bundle
    util_matrix = torch.zeros((B, A, A), device=device)
    for i in range(A):
        for j in range(A):
            if i == j:
                util_matrix[:, i, j] = U_true[torch.arange(B), i, endow_idx.long()]
            else:
                swapped = current_allocs.clone()
                mask_i = (current_allocs == i)
                mask_j = (current_allocs == j)
                swapped[mask_i] = j
                swapped[mask_j] = i
                swap_idx = (swapped * powers).sum(dim=1)
                util_matrix[:, i, j] = U_true[torch.arange(B), i, swap_idx]

    # Run TTC per sample (CPU loop — fine for eval)
    assigned = torch.full((B, A), -1, dtype=torch.long, device=device)
    available = torch.ones((B, A), dtype=torch.bool, device=device)

    for _ in range(A):
        preferences = torch.full((B, A), -1, dtype=torch.long, device=device)
        for b in range(B):
            for i in range(A):
                if assigned[b, i] >= 0:
                    continue
                best_j, best_u = -1, float('-inf')
                for j in range(A):
                    if available[b, j]:
                        u = util_matrix[b, i, j].item()
                        if u > best_u:
                            best_u = u
                            best_j = j
                preferences[b, i] = best_j

        for b in range(B):
            visited = [False] * A
            for start in range(A):
                if assigned[b, start] >= 0 or visited[start]:
                    continue
                path = []
                cur = start
                while cur >= 0 and cur not in path and not visited[cur]:
                    if assigned[b, cur] >= 0:
                        break
                    path.append(cur)
                    cur = preferences[b, cur].item()
                if cur in path:
                    cycle_start = path.index(cur)
                    cycle = path[cycle_start:]
                    for agent in cycle:
                        assigned[b, agent] = preferences[b, agent].item()
                        visited[agent] = True
                    for agent in cycle:
                        available[b, preferences[b, agent].item()] = False

        if (assigned >= 0).all():
            break

    # Fallback: keep own bundle
    for b in range(B):
        for i in range(A):
            if assigned[b, i] < 0:
                assigned[b, i] = i

    # Construct final allocation
    final_allocs = torch.zeros((B, cfg.num_items), dtype=torch.long, device=device)
    for b in range(B):
        for i in range(A):
            source = assigned[b, i].item()
            source_mask = (current_allocs[b] == source)
            final_allocs[b, source_mask] = i

    return (final_allocs * powers).sum(dim=1)


# ============================================================
#  3. Greedy IRPE Swap
# ============================================================

@torch.no_grad()
def greedy_irpe_swap(
    cfg: Config,
    aidx: AllocationIndex,
    U_true: torch.Tensor,        # [B, A, K]
    endow_idx: torch.Tensor,     # [B]
    max_iter: int = 10,
) -> torch.Tensor:
    """Greedy iterative swap starting from endowment, IR-constrained. Returns [B]."""
    device = U_true.device
    B = U_true.shape[0]
    A, m = cfg.num_agents, cfg.num_items

    current_idx = endow_idx.clone()
    all_allocs = aidx.all_allocations_tensor().to(device)
    powers = torch.tensor([A ** j for j in range(m)], device=device, dtype=torch.long)

    # Outside option (endowment utilities)
    outside_u = outside_option_utils(cfg, aidx, U_true, endow_idx)  # [B, A]

    def get_utils(idx):
        idx_exp = idx.view(B, 1, 1).expand(B, A, 1)
        return U_true.gather(2, idx_exp).squeeze(2)

    curr_utils = get_utils(current_idx)

    for _ in range(max_iter):
        best_idx = current_idx.clone()
        best_gain = torch.zeros(B, device=device)
        improved_any = torch.zeros(B, dtype=torch.bool, device=device)
        curr_alloc = all_allocs[current_idx.long()]

        # Pairwise swaps
        for i in range(A):
            for k in range(i + 1, A):
                for j1 in range(m):
                    for j2 in range(m):
                        if j1 == j2:
                            continue
                        i_owns = (curr_alloc[:, j1] == i)
                        k_owns = (curr_alloc[:, j2] == k)
                        valid_swap = i_owns & k_owns
                        if not valid_swap.any():
                            continue

                        cand = curr_alloc.clone()
                        cand[:, j1] = k
                        cand[:, j2] = i
                        cand_idx = (cand * powers).sum(dim=1)
                        cand_utils = get_utils(cand_idx)

                        # IR: must be >= outside option for all
                        ir_ok = (cand_utils >= outside_u - 1e-5).all(dim=1)
                        diffs = cand_utils - curr_utils
                        pareto = (diffs >= -1e-5).all(dim=1) & (diffs > 1e-5).any(dim=1)

                        valid = valid_swap & pareto & ir_ok
                        gain = diffs.sum(dim=1)
                        update = valid & (gain > best_gain)
                        best_idx = torch.where(update, cand_idx, best_idx)
                        best_gain = torch.where(update, gain, best_gain)
                        improved_any = improved_any | update

        # Single-item transfers
        for j in range(m):
            for src in range(A):
                for dst in range(A):
                    if src == dst:
                        continue
                    owns = (curr_alloc[:, j] == src)
                    if not owns.any():
                        continue
                    cand = curr_alloc.clone()
                    cand[:, j] = dst
                    cand_idx = (cand * powers).sum(dim=1)
                    cand_utils = get_utils(cand_idx)
                    ir_ok = (cand_utils >= outside_u - 1e-5).all(dim=1)
                    diffs = cand_utils - curr_utils
                    pareto = (diffs >= -1e-5).all(dim=1) & (diffs > 1e-5).any(dim=1)
                    valid = owns & pareto & ir_ok
                    gain = diffs.sum(dim=1)
                    update = valid & (gain > best_gain)
                    best_idx = torch.where(update, cand_idx, best_idx)
                    best_gain = torch.where(update, gain, best_gain)
                    improved_any = improved_any | update

        current_idx = best_idx
        curr_utils = get_utils(current_idx)
        if not improved_any.any():
            break

    return current_idx


# ============================================================
#  4. Random IRPE
# ============================================================

@torch.no_grad()
def random_irpe(
    cfg: Config,
    aidx: AllocationIndex,
    U_true: torch.Tensor,        # [B, A, K]
    endow_idx: torch.Tensor,     # [B]
) -> torch.Tensor:
    """Uniformly random among IRPE-feasible allocations. Returns [B]."""
    mask = compute_irpe_mask(cfg, aidx, U_true, endow_idx)  # [B, K]
    # Sample from valid allocations using Gumbel trick
    noise = -torch.log(-torch.log(torch.rand_like(mask) + 1e-10) + 1e-10)
    scores = noise + (1.0 - mask) * (-1e9)
    return scores.argmax(dim=1)
