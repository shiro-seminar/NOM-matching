"""Benchmark mechanisms for ordinal preference setting."""
from __future__ import annotations
import torch
from .config import Config
from .allocations import (
    score_matrix, build_all_allocs,
    endowment_scores, ir_mask, pareto_mask, ir_pe_mask,
    num_allocations,
)


def _one_hot(idx: torch.Tensor, K: int) -> torch.Tensor:
    return torch.zeros(idx.shape[0], K, device=idx.device).scatter_(1, idx.view(-1, 1), 1.0)


def _argmax_masked(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (scores + (1.0 - mask) * (-1e9)).argmax(dim=1)


def endowment_mechanism(cfg, marginal_rank, endow_idx, S) -> torch.Tensor:
    """Always stay at endowment (trivially IR)."""
    return _one_hot(endow_idx, num_allocations(cfg))


def wmax_score_ir(cfg, marginal_rank, endow_idx, S) -> torch.Tensor:
    """Maximize total score subject to IR."""
    mask = ir_mask(S, endow_idx)
    return _one_hot(_argmax_masked(S.sum(1), mask), num_allocations(cfg))


def wmax_score_pe(cfg, marginal_rank, endow_idx, S) -> torch.Tensor:
    """Maximize total score subject to PE."""
    mask = pareto_mask(S)
    return _one_hot(_argmax_masked(S.sum(1), mask), num_allocations(cfg))


def wmax_score_ir_pe(cfg, marginal_rank, endow_idx, S) -> torch.Tensor:
    """Maximize total score subject to IR + PE + Balanced (ordinal WMAX)."""
    mask = ir_pe_mask(cfg, S, endow_idx)
    return _one_hot(_argmax_masked(S.sum(1), mask), num_allocations(cfg))


def random_ir_pe(cfg, marginal_rank, endow_idx, S) -> torch.Tensor:
    """Uniform random allocation in IR + PE + Balanced set."""
    mask = ir_pe_mask(cfg, S, endow_idx)
    return mask / mask.sum(1, keepdim=True).clamp(min=1.0)


def ttc_ordinal(cfg: Config, marginal_rank: torch.Tensor,
                endow_idx: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Top-Trading Cycles (ordinal): each item -> agent with lowest rank (most preferred).

    Falls back to endowment if IR violated.
    """
    B, A, m = marginal_rank.shape
    device = marginal_rank.device
    K = num_allocations(cfg)

    # Each item -> agent with smallest rank (ties broken by agent index)
    assignment = marginal_rank.argmin(dim=1)   # [B, m]  (lower rank = better)
    powers = torch.tensor([A ** j for j in range(m)], device=device, dtype=torch.long)
    ttc_idx = (assignment * powers).sum(1)     # [B]

    s0      = endowment_scores(S, endow_idx)   # [B, A]
    ttc_S   = torch.stack([S[b, :, ttc_idx[b]] for b in range(B)])  # [B, A]
    ir_ok   = (ttc_S >= s0 - 1e-8).all(1)

    final_idx = torch.where(ir_ok, ttc_idx, endow_idx)
    return _one_hot(final_idx, K)


BENCHMARKS = {
    "Endowment":       endowment_mechanism,
    "WMAX-IR":         wmax_score_ir,
    "WMAX-PE":         wmax_score_pe,
    "WMAX-IR-PE":      wmax_score_ir_pe,
    "TTC-ordinal":     ttc_ordinal,
    "Random-IR-PE":    random_ir_pe,
}
