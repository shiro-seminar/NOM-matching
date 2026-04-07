"""Benchmark mechanisms for A-agent M-item economy."""
from __future__ import annotations
import torch
from .config import Config
from .allocations import (
    all_utilities, build_all_allocs,
    endowment_utilities, ir_mask, pareto_mask, ir_pe_mask,
    num_allocations,
)


def _one_hot(idx: torch.Tensor, K: int) -> torch.Tensor:
    return torch.zeros(idx.shape[0], K, device=idx.device).scatter_(1, idx.view(-1, 1), 1.0)


def _argmax_masked(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (scores + (1.0 - mask) * (-1e9)).argmax(dim=1)


def endowment_mechanism(cfg, v, endow_idx, U) -> torch.Tensor:
    return _one_hot(endow_idx, num_allocations(cfg))


def wmax_ir(cfg, v, endow_idx, U) -> torch.Tensor:
    mask = ir_mask(U, endow_idx)
    return _one_hot(_argmax_masked(U.sum(1), mask), num_allocations(cfg))


def wmax_pe(cfg, v, endow_idx, U) -> torch.Tensor:
    mask = pareto_mask(U)
    return _one_hot(_argmax_masked(U.sum(1), mask), num_allocations(cfg))


def wmax_ir_pe(cfg, v, endow_idx, U) -> torch.Tensor:
    mask = ir_pe_mask(cfg, U, endow_idx)
    return _one_hot(_argmax_masked(U.sum(1), mask), num_allocations(cfg))


def random_ir_pe(cfg, v, endow_idx, U) -> torch.Tensor:
    mask = ir_pe_mask(cfg, U, endow_idx)
    return mask / mask.sum(1, keepdim=True).clamp(min=1.0)


def ttc_generalized(cfg: Config, v: torch.Tensor,
                    endow_idx: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Item-by-item assignment: each item → argmax agent.
    IR違反なら endowment fallback（TTC2と同じ発想の一般化）。
    """
    B, A, m = v.shape
    device = v.device
    K = num_allocations(cfg)

    assignment = v.argmax(dim=1)   # [B, m]  各財を最高評価エージェントに
    powers = torch.tensor([A ** j for j in range(m)], device=device, dtype=torch.long)
    ttc_idx = (assignment * powers).sum(1)   # [B]

    u0 = endowment_utilities(U, endow_idx)
    ttc_U = torch.stack([U[b, :, ttc_idx[b]] for b in range(B)])  # [B, A]
    ir_ok = (ttc_U >= u0 - 1e-8).all(1)

    final_idx = torch.where(ir_ok, ttc_idx, endow_idx)
    return _one_hot(final_idx, K)


BENCHMARKS = {
    "Endowment":    endowment_mechanism,
    "WMAX-IR":      wmax_ir,
    "WMAX-PE":      wmax_pe,
    "WMAX-IR-PE":   wmax_ir_pe,
    "TTC-general":  ttc_generalized,
    "Random-IR-PE": random_ir_pe,
}
