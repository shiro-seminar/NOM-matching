"""Type / endowment sampling and utility computation (pure additive)."""
from __future__ import annotations
import torch
from .allocations import AllocationIndex
from .config import Config


def sample_types(cfg: Config, aidx: AllocationIndex, batch_size: int) -> dict:
    """Sample true types and endowments for N agents.

    Returns dict with:
      v_true:    [B, A, m]
      endow_idx: [B]
    """
    device = torch.device(cfg.device)
    B, A, m = batch_size, cfg.num_agents, cfg.num_items

    v_true = torch.empty((B, A, m), device=device).uniform_(cfg.v_min, cfg.v_max)
    endow_idx = aidx.random_endowment_no_disposal(B, device)

    return {
        "v_true": v_true,
        "endow_idx": endow_idx,
    }


def compute_all_utilities(
    v: torch.Tensor,
    aidx: AllocationIndex,
) -> torch.Tensor:
    """Compute utilities for every allocation for each agent (pure additive).

    Args:
      v: [B, A, m]  item valuations

    Returns:
      U: [B, A, K], where U[:, i, k] = u_i(allocation k) = sum_{j assigned to i} v_{i,j}
    """
    device = v.device
    B, A, m = v.shape
    K = aidx.num_allocations

    all_allocs = aidx.all_allocations_tensor().to(device)  # [K, m]

    # agent_masks[k, i, j] = 1 if allocation k gives item j to agent i
    agent_masks = torch.zeros((K, A, m), dtype=torch.float32, device=device)
    for i in range(A):
        agent_masks[:, i, :] = (all_allocs == i).float()

    # v: [B, A, m] -> [B, A, 1, m],  masks: [K, A, m] -> [1, A, K, m]
    v_exp = v.unsqueeze(2)                                # [B, A, 1, m]
    masks_exp = agent_masks.unsqueeze(0).permute(0, 2, 1, 3)  # [1, A, K, m]

    # Additive utility: sum of v * mask
    U = (v_exp * masks_exp).sum(dim=-1)  # [B, A, K]
    return U


def outside_option_utils(
    cfg: Config,
    aidx: AllocationIndex,
    U_true: torch.Tensor,
    endow_idx: torch.Tensor,
) -> torch.Tensor:
    """Outside option utility for each agent given endowment.

    Args:
        U_true:    [B, A, K]
        endow_idx: [B]

    Returns:
        outside: [B, A]
    """
    B = endow_idx.shape[0]
    A = cfg.num_agents
    endow_idx_exp = endow_idx.view(B, 1, 1).expand(B, A, 1)
    outside = U_true.gather(2, endow_idx_exp).squeeze(2)  # [B, A]
    return outside
