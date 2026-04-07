"""Losses for NOM + IR + PE mechanism learning.

NOM Definition (Troyan & Morrill 2020):
  A report v_i' is an "obvious manipulation" for agent i at v_i iff:
    BC(v_i', f) > BC(v_i, f)  OR  WC(v_i', f) > WC(v_i, f)
  where BC = best-case utility over all opponent reports,
        WC = worst-case utility over all opponent reports.

  A mechanism is NOM iff no agent has an obvious manipulation.

NOM Loss (sampling-based approximation):
  For each agent i, sample S opponent profiles {v_{-i}^s}.
  Approximate:
    BC_truth(v_i)     = max_s u_i(f(v_i,  v_{-i}^s))
    WC_truth(v_i)     = min_s u_i(f(v_i,  v_{-i}^s))
    BC_lie(v_i, v_i') = max_s u_i(f(v_i', v_{-i}^s))
    WC_lie(v_i, v_i') = min_s u_i(f(v_i', v_{-i}^s))

  A misreport v_i' is an obvious manipulation iff:
    BC_lie > BC_truth  AND  WC_lie > WC_truth       ← Troyan & Morrill condition

  NOM_loss_i = max_{v_i'} min(
      ReLU(BC_lie - BC_truth),
      ReLU(WC_lie - WC_truth)
  )
  (positive only when BOTH best-case AND worst-case improve)

IR and PE are enforced as hard constraints via masking (not via a soft loss).
"""
from __future__ import annotations

import torch

from .config import Config
from .allocations import all_utilities, ir_pe_mask
from .model import AllocationNet


# ── Utility helpers ──────────────────────────────────────────────────────────

def _expected_utility(probs: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """EU = sum_k probs[k] * U[:, i, k].

    Args:
        probs: [B, K]
        U:     [B, A, K]

    Returns:
        EU: [B, A]
    """
    return torch.einsum("bk,bak->ba", probs, U)


# ── NOM loss ─────────────────────────────────────────────────────────────────

def nom_loss(
    cfg: Config,
    net: AllocationNet,
    v: torch.Tensor,        # [B, 2, 3]  true valuations
    endow_idx: torch.Tensor,  # [B]
    S: int | None = None,
    M: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute NOM loss (sampling-based).

    Returns:
        nom:         scalar, mean NOM violation across agents and batch
        max_regret:  [B, A] max obvious-manipulation gain per sample per agent
    """
    if S is None:
        S = cfg.S
    if M is None:
        M = cfg.M

    B, A, m = v.shape
    device = v.device

    # ── Sample S opponent profiles ──────────────────────────────────────────
    # v_opp: [B, S, A, m]  full profiles including agent i (to be replaced)
    v_opp = torch.empty(B, S, A, m, device=device).uniform_(cfg.v_min, cfg.v_max)

    # ── Sample M misreport candidates for each agent ────────────────────────
    # v_mis: [B, M, m]  (shared template; we replace per-agent below)
    v_mis = torch.empty(B, M, m, device=device).uniform_(cfg.v_min, cfg.v_max)

    all_violations = []

    for i in range(A):
        j = 1 - i   # opponent agent index (A=2)

        # Build S opponent profiles: replace agent i's value with true v_i,
        # keep agent j's value as the sampled opponent
        # v_opp_ij: [B, S, A, m]
        v_opp_i = v_opp.clone()
        v_opp_i[:, :, i, :] = v[:, i, :].unsqueeze(1).expand(B, S, m)
        # v_opp_i[:, s, j, :] = v_opp[:, s, j, :]  (already random)

        # ── Flatten to (B*S, A, m) and run mechanism with IR∩PE mask ───────
        v_opp_flat = v_opp_i.reshape(B * S, A, m)
        U_opp_flat = all_utilities(v_opp_flat)                       # [B*S, A, 8]
        endow_rep  = endow_idx.unsqueeze(1).expand(B, S).reshape(B * S)
        mask_opp   = ir_pe_mask(U_opp_flat, endow_rep)               # [B*S, 8]
        probs_opp  = net(v_opp_flat, mask=mask_opp)                  # [B*S, 8]

        # Utility for agent i under truthful report
        # u_truth[b*S+s] = EU_i(f(v_i, v_{-i}^s))
        U_true_i = U_opp_flat[:, i, :]                               # [B*S, 8]
        u_truth_flat = (probs_opp * U_true_i).sum(dim=-1)            # [B*S]
        u_truth = u_truth_flat.reshape(B, S)                         # [B, S]

        BC_truth = u_truth.max(dim=1).values    # [B]
        WC_truth = u_truth.min(dim=1).values    # [B]

        # ── For each misreport v_i', compute BC_lie and WC_lie ──────────────
        # v_mis[:, :, :]: [B, M, m]
        # For each m-report, reuse same opponent samples

        # Expand v_mis over S: [B, M, S, m]
        v_mis_exp = v_mis.unsqueeze(2).expand(B, M, S, m)

        # Build mis-profiles: agent i uses misreport, agent j uses v_opp
        # Shape: [B, M, S, A, m]
        v_mis_full = v_opp.unsqueeze(1).expand(B, M, S, A, m).clone()
        v_mis_full[:, :, :, i, :] = v_mis_exp   # replace agent i with misreport

        # Flatten to (B*M*S, A, m)
        v_mis_flat = v_mis_full.reshape(B * M * S, A, m)
        U_mis_flat = all_utilities(v_mis_flat)                        # [BMS, A, 8]
        endow_rep2 = endow_idx.view(B, 1, 1).expand(B, M, S).reshape(B * M * S)
        mask_mis   = ir_pe_mask(U_mis_flat, endow_rep2)               # [BMS, 8]

        # Mechanism outputs under misreport
        probs_mis  = net(v_mis_flat, mask=mask_mis)                   # [BMS, 8]

        # Utility for agent i under MISREPORT profile, evaluated at TRUE values
        # True utility for agent i: use v[:, i, :] valuations
        # Rebuild U with true v_i but opponent v_j
        v_eval = v_mis_flat.clone()
        v_eval[:, i, :] = v[:, i, :].unsqueeze(1).unsqueeze(1) \
                           .expand(B, M, S, m).reshape(B * M * S, m)
        U_eval = all_utilities(v_eval)                                # [BMS, A, 8]
        U_eval_i = U_eval[:, i, :]                                    # [BMS, 8]
        u_lie_flat = (probs_mis * U_eval_i).sum(dim=-1)              # [BMS]
        u_lie = u_lie_flat.reshape(B, M, S)                          # [B, M, S]

        BC_lie = u_lie.max(dim=2).values    # [B, M]
        WC_lie = u_lie.min(dim=2).values    # [B, M]

        # NOM violation: misreport is "obvious" iff BOTH BC and WC improve
        bc_gain = torch.relu(BC_lie - BC_truth.unsqueeze(1))  # [B, M]
        wc_gain = torch.relu(WC_lie - WC_truth.unsqueeze(1))  # [B, M]
        obvious_gain = torch.min(bc_gain, wc_gain)             # [B, M]

        max_obvious = obvious_gain.max(dim=1).values           # [B]
        all_violations.append(max_obvious)

    # Stack: [B, A]
    violations = torch.stack(all_violations, dim=1)   # [B, A]
    return violations.mean(), violations


# ── Welfare ──────────────────────────────────────────────────────────────────

def welfare_loss(
    probs: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """Negative mean social welfare.

    Args:
        probs: [B, K]
        U:     [B, A, K]

    Returns:
        scalar  (-welfare, to be minimized)
    """
    EU = _expected_utility(probs, U)          # [B, A]
    welfare = EU.sum(dim=1).mean()
    return -welfare


# ── Augmented Lagrangian objective ───────────────────────────────────────────

def augmented_objective(
    cfg: Config,
    net: AllocationNet,
    v: torch.Tensor,
    endow_idx: torch.Tensor,
    U: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Full training objective.

    IR + PE are enforced via mask (hard constraint).
    NOM is penalized via Augmented Lagrangian.

    Returns:
        loss:  scalar
        stats: dict of floats for logging
    """
    probs = net(v, mask=mask, temperature=cfg.temperature)  # [B, K]

    # ── Welfare ──────────────────────────────────────────────────────────────
    EU = _expected_utility(probs, U)       # [B, A]
    welfare = EU.sum(dim=1).mean()

    # ── NOM ──────────────────────────────────────────────────────────────────
    nom, _ = nom_loss(cfg, net, v, endow_idx, S=cfg.S, M=cfg.M)

    # ── Total loss: -welfare + AL penalty on NOM ─────────────────────────────
    loss = (
        -cfg.welfare_weight * welfare
        + cfg.lambda_nom * nom
        + (cfg.rho / 2.0) * nom * nom
    )

    stats = {
        "loss":    float(loss.detach()),
        "welfare": float(welfare.detach()),
        "nom":     float(nom.detach()),
    }
    return loss, stats
