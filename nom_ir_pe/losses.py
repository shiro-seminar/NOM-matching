"""Loss functions: NOM loss, SP loss, IR/PE masks, augmented loss.

NOM (Obviously Not Manipulable):
  For each agent i, for each misreport v_i':
    BC(truth) >= BC(lie)  AND  WC(truth) >= WC(lie)
  where BC/WC are best/worst-case utility over opponent types.
  Violation = max_{v_i'} [ ReLU(BC_lie - BC_truth) + ReLU(WC_lie - WC_truth) ]

SP (Strategy-Proof):
  Standard dominant-strategy regret: max_{v_i'} ReLU(EU_lie - EU_truth).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Config
from .allocations import AllocationIndex, pareto_mask
from .allocations import balanced_mask
from .data_gen import compute_all_utilities, outside_option_utils


# ============================================================
#  IR + PE mask
# ============================================================

def compute_ir_mask(
    cfg: Config,
    aidx: AllocationIndex,
    U: torch.Tensor,        # [B, A, K]
    endow_idx: torch.Tensor,  # [B]
) -> torch.Tensor:
    """[B, K] mask: 1.0 if allocation is individually rational for all agents."""
    outside_u = outside_option_utils(cfg, aidx, U, endow_idx)  # [B, A]
    diff = U - outside_u.unsqueeze(2)  # [B, A, K]
    feasible = (diff >= -1e-5).all(dim=1)  # [B, K]
    return feasible.float()


def compute_irpe_mask(
    cfg: Config,
    aidx: AllocationIndex,
    U: torch.Tensor,         # [B, A, K]
    endow_idx: torch.Tensor,  # [B]
) -> torch.Tensor:
    """[B, K] mask: 1.0 if allocation is IR, Pareto-efficient, and balanced."""
    ir   = compute_ir_mask(cfg, aidx, U, endow_idx)        # [B, K]
    pe   = pareto_mask(U)                                   # [B, K]
    bal  = balanced_mask(aidx, endow_idx, U.device)         # [B, K]
    combined = ir * pe * bal                                # [B, K]
    # Fall back to endowment (always IR and balanced) if no valid allocation
    has_valid = combined.sum(dim=1, keepdim=True) > 0       # [B, 1]
    endow_onehot = torch.zeros_like(combined)
    endow_onehot.scatter_(1, endow_idx.view(-1, 1), 1.0)
    return torch.where(has_valid, combined, endow_onehot)


# ============================================================
#  Utility helpers
# ============================================================

def expected_utilities_from_probs(
    probs: torch.Tensor,   # [B, K]
    U: torch.Tensor,       # [B, A, K]
) -> torch.Tensor:
    """Expected utility for each agent. Returns [B, A]."""
    return torch.einsum('bk,bak->ba', probs, U)


# ============================================================
#  SP loss (dominant-strategy regret)
# ============================================================

def sp_loss_sampled(
    cfg: Config,
    aidx: AllocationIndex,
    mech,           # callable: (v_report, endow_idx, mask=) -> probs [B, K]
    v_true: torch.Tensor,      # [B, A, m]
    U_true: torch.Tensor,      # [B, A, K]
    endow_idx: torch.Tensor,   # [B]
) -> torch.Tensor:
    """Sampled dominant-strategy regret (SP violation). Returns scalar."""
    device = v_true.device
    B, A, m = v_true.shape
    M = int(cfg.misreport_samples)

    # Truthful probs and expected utilities
    mask_true = compute_irpe_mask(cfg, aidx, U_true, endow_idx)
    probs_true = mech(v_true, endow_idx, mask=mask_true)
    EU_true = expected_utilities_from_probs(probs_true, U_true)  # [B, A]

    endow_rep = endow_idx.unsqueeze(1).expand(B, M).reshape(B * M)

    regrets = []
    for i in range(A):
        # Sample misreports for agent i
        v_noise = (2 * torch.rand((B, M, m), device=device) - 1) * cfg.misreport_noise_v
        v_mis_i = torch.clamp(
            v_true[:, i, :].unsqueeze(1) + v_noise,
            cfg.v_min, cfg.v_max,
        )  # [B, M, m]

        # Build full report: replace agent i's report
        v_rep = v_true.unsqueeze(1).expand(B, M, A, m).clone()
        v_rep[:, :, i, :] = v_mis_i

        v_rep_f = v_rep.reshape(B * M, A, m)

        # Compute mask for misreported types
        U_mis_rep = compute_all_utilities(v_rep_f, aidx)
        mask_mis = compute_irpe_mask(cfg, aidx, U_mis_rep, endow_rep)

        probs_mis = mech(v_rep_f, endow_rep, mask=mask_mis).reshape(B, M, -1)

        # Evaluate under TRUE preferences
        EU_mis_all = expected_utilities_from_probs(
            probs_mis.reshape(B * M, -1),
            U_true.repeat_interleave(M, dim=0),
        ).reshape(B, M, A)

        EU_mis_i = EU_mis_all[:, :, i]  # [B, M]
        gain = EU_mis_i - EU_true[:, i].unsqueeze(1)
        regret = torch.relu(gain).max(dim=1).values  # [B]
        regrets.append(regret)

    return torch.stack(regrets, dim=1).mean()


# ============================================================
#  NOM loss
# ============================================================

def nom_loss_sampled(
    cfg: Config,
    aidx: AllocationIndex,
    mech,           # callable: (v_report, endow_idx, mask=) -> probs [B, K]
    v_true: torch.Tensor,      # [B, A, m]
    U_true: torch.Tensor,      # [B, A, K]
    endow_idx: torch.Tensor,   # [B]
) -> torch.Tensor:
    """NOM violation loss (sampled). Returns scalar.

    For each agent i:
      1. Sample S opponent-type profiles v_{-i}^{(s)}
      2. For truthful report v_i and each opponent sample:
           compute u_i under mechanism → get best-case (BC) and worst-case (WC)
      3. For each misreport v_i', repeat step 2 → get BC' and WC'
      4. violation_i = max_{v_i'} [ ReLU(BC' - BC) + ReLU(WC' - WC) ]
    """
    device = v_true.device
    B, A, m = v_true.shape
    S = int(cfg.nom_opponent_samples)
    M_nom = int(cfg.nom_misreport_samples)

    violations = []
    for i in range(A):
        # --- Sample S opponent-type profiles ---
        # v_opp[b, s, A, m]: for each batch, S opponent profiles
        # Agent i keeps their true type; others are resampled
        v_opp = torch.empty((B, S, A, m), device=device).uniform_(cfg.v_min, cfg.v_max)
        v_opp[:, :, i, :] = v_true[:, i, :].unsqueeze(1).expand(B, S, m)

        # Flatten: [B*S, A, m]
        v_opp_f = v_opp.reshape(B * S, A, m)
        endow_opp = endow_idx.unsqueeze(1).expand(B, S).reshape(B * S)

        # Compute IRPE mask for each opponent profile
        U_opp = compute_all_utilities(v_opp_f, aidx)   # [B*S, A, K]
        mask_opp = compute_irpe_mask(cfg, aidx, U_opp, endow_opp)

        # --- Truthful: agent i reports v_i ---
        probs_truth = mech(v_opp_f, endow_opp, mask=mask_opp)  # [B*S, K]

        # Utility for agent i under TRUE preferences
        # U_true for agent i: use v_true[:, i, :] to evaluate
        # Need to compute u_i for each allocation for agent i (constant across opponents)
        # U_i_true[b, k] = sum_j v_true[b,i,j] * mask_{i,k,j}
        all_allocs = aidx.all_allocations_tensor().to(device)  # [K, m]
        K = aidx.num_allocations
        agent_i_masks = (all_allocs == i).float()  # [K, m]
        U_i_true = (v_true[:, i, :].unsqueeze(1) * agent_i_masks.unsqueeze(0)).sum(dim=-1)  # [B, K]

        # Expected utility for agent i under truthful report for each opponent sample
        U_i_true_rep = U_i_true.unsqueeze(1).expand(B, S, K).reshape(B * S, K)  # [B*S, K]
        eu_truth = (probs_truth * U_i_true_rep).sum(dim=-1).reshape(B, S)  # [B, S]

        # Best-case and worst-case over opponent profiles
        bc_truth = eu_truth.max(dim=1).values   # [B]
        wc_truth = eu_truth.min(dim=1).values   # [B]

        # --- Misreports: sample M misreports for agent i ---
        v_noise = (2 * torch.rand((B, M_nom, m), device=device) - 1) * cfg.misreport_noise_v
        v_mis_i = torch.clamp(
            v_true[:, i, :].unsqueeze(1) + v_noise,
            cfg.v_min, cfg.v_max,
        )  # [B, M_nom, m]

        max_bc_gain = torch.zeros(B, device=device)
        max_wc_gain = torch.zeros(B, device=device)

        for mi in range(M_nom):
            # Replace agent i's report in each opponent profile
            v_mis_opp = v_opp.clone()  # [B, S, A, m]
            v_mis_opp[:, :, i, :] = v_mis_i[:, mi, :].unsqueeze(1).expand(B, S, m)
            v_mis_opp_f = v_mis_opp.reshape(B * S, A, m)

            # Recompute IRPE mask for misreported types
            U_mis_opp = compute_all_utilities(v_mis_opp_f, aidx)
            mask_mis_opp = compute_irpe_mask(cfg, aidx, U_mis_opp, endow_opp)

            probs_lie = mech(v_mis_opp_f, endow_opp, mask=mask_mis_opp)  # [B*S, K]

            # Utility for agent i under TRUE preferences (same U_i_true)
            eu_lie = (probs_lie * U_i_true_rep).sum(dim=-1).reshape(B, S)  # [B, S]

            bc_lie = eu_lie.max(dim=1).values   # [B]
            wc_lie = eu_lie.min(dim=1).values   # [B]

            bc_gain = torch.relu(bc_lie - bc_truth)
            wc_gain = torch.relu(wc_lie - wc_truth)

            max_bc_gain = torch.max(max_bc_gain, bc_gain)
            max_wc_gain = torch.max(max_wc_gain, wc_gain)

        # NOM violation for agent i: sum of BC and WC violations
        viol_i = max_bc_gain + max_wc_gain  # [B]
        violations.append(viol_i)

    return torch.stack(violations, dim=1).mean()


# ============================================================
#  Augmented Lagrangian loss
# ============================================================

def augmented_loss(
    cfg: Config,
    aidx: AllocationIndex,
    net,
    v_true: torch.Tensor,      # [B, A, m]
    U_true: torch.Tensor,      # [B, A, K]
    endow_idx: torch.Tensor,   # [B]
) -> tuple[torch.Tensor, dict]:
    """Augmented Lagrangian: -welfare + λ·constraint + ρ/2·constraint².

    constraint = NOM or SP depending on cfg.loss_type.
    IR and PE are enforced by masking (hard constraint).
    """
    mask = compute_irpe_mask(cfg, aidx, U_true, endow_idx)
    probs = net(v_true, endow_idx, temperature=cfg.temperature, mask=mask)

    EU = expected_utilities_from_probs(probs, U_true)  # [B, A]
    welfare = EU.sum(dim=1).mean()

    # Constraint violation
    if cfg.loss_type == "nom":
        constraint = nom_loss_sampled(cfg, aidx, net, v_true, U_true, endow_idx)
    else:
        constraint = sp_loss_sampled(cfg, aidx, net, v_true, U_true, endow_idx)

    loss = (
        -welfare
        + cfg.lambda_constraint * constraint
        + (cfg.rho / 2) * constraint * constraint
    )

    stats = {
        "welfare": float(welfare.detach().cpu()),
        "constraint": float(constraint.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    return loss, stats
