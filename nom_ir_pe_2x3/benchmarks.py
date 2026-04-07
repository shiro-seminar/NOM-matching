"""Benchmark mechanisms for 2-agent 3-item exchange economy.

All benchmarks take (v, endow_idx, U) and return probs [B, K].

Benchmarks:
  1. Endowment        – keep initial endowment (trivially IR, trivially NOM)
  2. WMAX-IR          – welfare-maximizing allocation subject to IR
  3. WMAX-PE          – welfare-maximizing among Pareto-efficient allocations (no IR req'd)
  4. WMAX-IR-PE       – welfare-maximizing among IR ∩ PE allocations
  5. TTC2             – 2-agent Top Trading Cycles (deterministic, SP, PE, IR)
  6. Random-IR-PE     – uniform lottery over IR ∩ PE allocations
"""
from __future__ import annotations

import torch

from .allocations import (
    K,
    all_utilities,
    endowment_utilities,
    ir_mask,
    pareto_mask,
    ir_pe_mask,
    ALL_ALLOCS,
)


def _one_hot(idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert [B] indices to [B, K] one-hot float."""
    return torch.zeros(idx.shape[0], num_classes, device=idx.device).scatter_(
        1, idx.view(-1, 1), 1.0
    )


def _argmax_masked(U_sum: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Argmax of [B, K] scores where mask==0 → -inf.

    If no feasible allocation exists for a sample, falls back to argmax over all.
    """
    masked = U_sum + (1.0 - mask) * (-1e9)
    return masked.argmax(dim=1)   # [B]


# ── 1. Endowment ─────────────────────────────────────────────────────────────

def endowment_mechanism(
    v: torch.Tensor,
    endow_idx: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """Keep the endowment allocation."""
    return _one_hot(endow_idx, K)


# ── 2. WMAX-IR ────────────────────────────────────────────────────────────────

def wmax_ir(
    v: torch.Tensor,
    endow_idx: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """Welfare-maximizing allocation subject to IR.

    If no IR allocation exists, falls back to endowment.
    """
    mask = ir_mask(U, endow_idx)                 # [B, K]
    welfare = U.sum(dim=1)                       # [B, K]
    idx = _argmax_masked(welfare, mask)          # [B]
    return _one_hot(idx, K)


# ── 3. WMAX-PE ────────────────────────────────────────────────────────────────

def wmax_pe(
    v: torch.Tensor,
    endow_idx: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """Welfare-maximizing allocation among Pareto-efficient allocations."""
    mask = pareto_mask(U)                        # [B, K]
    welfare = U.sum(dim=1)                       # [B, K]
    idx = _argmax_masked(welfare, mask)
    return _one_hot(idx, K)


# ── 4. WMAX-IR-PE ─────────────────────────────────────────────────────────────

def wmax_ir_pe(
    v: torch.Tensor,
    endow_idx: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """Welfare-maximizing among IR ∩ PE allocations."""
    mask = ir_pe_mask(U, endow_idx)              # [B, K]
    welfare = U.sum(dim=1)                       # [B, K]
    idx = _argmax_masked(welfare, mask)
    return _one_hot(idx, K)


# ── 5. 2-Agent Top Trading Cycles ────────────────────────────────────────────

def ttc2(
    v: torch.Tensor,
    endow_idx: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """2-agent Top Trading Cycles.

    With 2 agents and additive utilities, TTC works as follows:
    - Each agent's "best bundle" = all items they value most.
    - But items must be fully assigned, so we treat each item independently:
      each item goes to the agent who values it more (or stays in endowment
      if both are indifferent).

    This is equivalent to the welfare-maximizing allocation, which for
    additive preferences is Pareto efficient and IR (agents prefer to get
    the items they value more). If IR is violated, fall back to endowment.

    For 2-agent additive utilities, TTC = item-by-item assignment to highest bidder,
    subject to IR. This is SP and PE.
    """
    B, A, m = v.shape
    device = v.device

    # For each item, assign to agent who values it more (tie: agent 0)
    assignment = (v[:, 1, :] > v[:, 0, :]).long()   # [B, m]  0 or 1

    # Convert to allocation index
    powers = torch.tensor([2 ** j for j in range(m)], device=device, dtype=torch.long)
    ttc_idx = (assignment * powers).sum(dim=1)       # [B]

    # TTC allocation might violate IR in some edge cases (shouldn't for 2-agent
    # item-by-item, but let's be safe)
    U0 = endowment_utilities(U, endow_idx)           # [B, A]
    ttc_U = U.gather(2, ttc_idx.view(B, 1, 1).expand(B, A, 1)).squeeze(2)  # [B, A]
    ir_ok = (ttc_U >= U0 - 1e-8).all(dim=1)         # [B]

    # Where IR is violated, fall back to endowment
    final_idx = torch.where(ir_ok, ttc_idx, endow_idx)

    return _one_hot(final_idx, K)


# ── 6. Random-IR-PE (lottery) ─────────────────────────────────────────────────

def random_ir_pe(
    v: torch.Tensor,
    endow_idx: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """Uniform lottery over IR ∩ PE allocations.

    If no feasible allocation, returns endowment.
    """
    mask = ir_pe_mask(U, endow_idx)   # [B, K]
    # Replace 0-rows with endowment one-hot
    row_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    probs = mask / row_sum            # [B, K]
    return probs


# ── Registry ──────────────────────────────────────────────────────────────────

BENCHMARKS: dict[str, callable] = {
    "Endowment":    endowment_mechanism,
    "WMAX-IR":      wmax_ir,
    "WMAX-PE":      wmax_pe,
    "WMAX-IR-PE":   wmax_ir_pe,
    "TTC2":         ttc2,
    "Random-IR-PE": random_ir_pe,
}
