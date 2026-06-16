"""Benchmark mechanisms (domain-independent)."""
from __future__ import annotations
import torch
from .config import Config
from .allocations import (
    build_all_allocs, endowment_scores, ir_mask, pareto_mask,
    ir_pe_mask, num_allocations, agent_masks_all, balanced_mask, fsd_ir_mask,
)


def _one_hot(idx: torch.Tensor, K: int) -> torch.Tensor:
    return torch.zeros(idx.shape[0], K, device=idx.device).scatter_(1, idx.view(-1, 1), 1.0)


def _argmax_masked(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (scores + (1.0 - mask) * (-1e9)).argmax(dim=1)


def endowment_mechanism(cfg, marginal_rank, endow_idx, S) -> torch.Tensor:
    return _one_hot(endow_idx, num_allocations(cfg))


def wmax_score_ir(cfg, marginal_rank, endow_idx, S) -> torch.Tensor:
    mask = ir_mask(S, endow_idx)
    return _one_hot(_argmax_masked(S.sum(1), mask), num_allocations(cfg))


def wmax_score_pe(cfg, marginal_rank, endow_idx, S) -> torch.Tensor:
    mask = pareto_mask(S)
    return _one_hot(_argmax_masked(S.sum(1), mask), num_allocations(cfg))


def wmax_score_ir_pe(cfg, marginal_rank, endow_idx, S) -> torch.Tensor:
    mask = ir_pe_mask(cfg, S, endow_idx)
    return _one_hot(_argmax_masked(S.sum(1), mask), num_allocations(cfg))


def random_ir_pe(cfg, marginal_rank, endow_idx, S) -> torch.Tensor:
    mask = ir_pe_mask(cfg, S, endow_idx)
    return mask / mask.sum(1, keepdim=True).clamp(min=1.0)


def ttc_ordinal(cfg: Config, marginal_rank: torch.Tensor,
                endow_idx: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Top-Trading Cycles (ordinal): each item -> agent with lowest rank."""
    B, A, m = marginal_rank.shape
    device  = marginal_rank.device
    K       = num_allocations(cfg)

    assignment = marginal_rank.argmin(dim=1)   # [B, m]
    powers  = torch.tensor([A ** j for j in range(m)], device=device, dtype=torch.long)
    ttc_idx = (assignment * powers).sum(1)     # [B]

    s0     = endowment_scores(S, endow_idx)
    ttc_S  = torch.stack([S[b, :, ttc_idx[b]] for b in range(B)])
    ir_ok  = (ttc_S >= s0 - 1e-8).all(1)
    final_idx = torch.where(ir_ok, ttc_idx, endow_idx)
    return _one_hot(final_idx, K)


def priority_mechanism(cfg: Config, marginal_rank: torch.Tensor,
                       endow_idx: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Priority mechanism phi^IP (Manjunath-Westkamp 2025).

    Restricted to component-wise-IR (FOSD-IR) & balanced allocations.
    Staged, class-by-class refinement: for class c = 0, 1, ..., R-1
    (most-preferred class first) and agent a = 0, 1, ..., A-1 (priority order),
    keep only the surviving allocations that maximize agent a's count of
    class-c items received. Class 0 (the "A" / attractive set) is fully
    resolved across all agents before class 1 ("B") is consulted, etc.
    Remaining ties are broken by lowest allocation index.
    """
    _, A, _ = marginal_rank.shape
    device  = marginal_rank.device
    K = num_allocations(cfg)
    R = cfg.num_ranks

    masks = agent_masks_all(cfg, device)              # [K, A, m]
    bal     = balanced_mask(cfg, endow_idx, device)    # [B, K]
    irmask  = fsd_ir_mask(cfg, marginal_rank, endow_idx)  # [B, K]
    feasible = bal * irmask                            # [B, K]

    mr = marginal_rank.float()                         # [B, A, m]

    for c in range(R):
        for a in range(A):
            owned = masks[:, a, :]                                 # [K, m]
            is_c  = (mr[:, a, :].unsqueeze(1) == c).float()        # [B, 1, m]
            cnt   = (is_c * owned.unsqueeze(0)).sum(-1)            # [B, K]
            score = cnt + (1.0 - feasible) * (-1e9)
            max_score = score.max(dim=1, keepdim=True).values
            keep = (score >= max_score - 1e-6).float()
            feasible = feasible * keep

    idx = torch.arange(K, device=device).float().unsqueeze(0)
    tiebreak = feasible * (-idx) + (1.0 - feasible) * (-1e9)
    chosen = tiebreak.argmax(dim=1)
    return _one_hot(chosen, K)


BENCHMARKS = {
    "Endowment":    endowment_mechanism,
    "WMAX-IR":      wmax_score_ir,
    "WMAX-PE":      wmax_score_pe,
    "WMAX-IR-PE":   wmax_score_ir_pe,
    "TTC-ordinal":  ttc_ordinal,
    "Random-IR-PE": random_ir_pe,
    "Priority-IP":  priority_mechanism,
}
