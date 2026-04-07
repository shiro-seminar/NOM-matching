"""Data generation for 2-agent 3-item pure-additive exchange economy."""
from __future__ import annotations

import torch

from .config import Config
from .allocations import all_utilities, random_endowment


def sample_batch(cfg: Config) -> dict[str, torch.Tensor]:
    """Sample a batch of valuation profiles and endowments.

    Returns:
        v:         [B, 2, 3]   valuations ~ U[v_min, v_max]
        endow_idx: [B]         endowment allocation index in [0, 8)
        U:         [B, 2, 8]   utilities for every allocation
    """
    device = torch.device(cfg.device)
    B = cfg.batch_size

    v = torch.empty(B, cfg.num_agents, cfg.num_items, device=device).uniform_(
        cfg.v_min, cfg.v_max
    )
    endow_idx = random_endowment(B, device)
    U = all_utilities(v)

    return {"v": v, "endow_idx": endow_idx, "U": U}


def sample_opponents(
    cfg: Config,
    B: int,
    S: int,
    device: torch.device | str,
    agent_idx: int,
) -> torch.Tensor:
    """Sample S opponent profiles for one agent.

    Returns:
        v_opp: [B, S, m]  opponent's valuations (the other agent)
    """
    m = cfg.num_items
    v_opp = torch.empty(B, S, m, device=device).uniform_(cfg.v_min, cfg.v_max)
    return v_opp


def sample_misreports(
    cfg: Config,
    v_i: torch.Tensor,
    M: int,
) -> torch.Tensor:
    """Sample M misreport candidates for agent i near their true value.

    Strategy: random points in [v_min, v_max]^m (uniform over whole type space,
    not just perturbations, to avoid missing profitable deviations).

    Args:
        v_i:  [B, m]

    Returns:
        v_mis: [B, M, m]
    """
    B, m = v_i.shape
    device = v_i.device
    v_mis = torch.empty(B, M, m, device=device).uniform_(cfg.v_min, cfg.v_max)
    return v_mis
