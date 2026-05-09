"""NOM + welfare losses with domain-aware sampling.

NOM definition (Troyan-Morrill 2020):
  obvious manipulation = min(relu(BC_lie - BC_truth), relu(WC_lie - WC_truth))

Domain-aware sampling: opponent profiles and misreports are drawn from the
same domain (respecting owned/unowned rank constraints) so that NOM is
evaluated on in-domain preference profiles only.
"""
from __future__ import annotations

import torch
from .config import Config
from .domains import DomainSpec, DOMAINS
from .allocations import score_matrix, ir_pe_mask
from .data_gen import sample_domain_mr_flat
from .model import AllocationNet


def _expected_score(probs: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """ES[b, a] = sum_k probs[b,k] * S[b,a,k]."""
    return torch.einsum("bk,bak->ba", probs, S)


def nom_loss(
    cfg: Config,
    domain: DomainSpec,
    net: AllocationNet,
    marginal_rank: torch.Tensor,   # [B, A, m]
    endow_idx: torch.Tensor,       # [B]
    S_true: torch.Tensor,          # [B, A, K]
    S: int | None = None,
    M: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NOM violation loss (obvious manipulation).

    For each agent i:
      1. Fix i's true rank; sample S_nom domain-consistent opponent profiles.
      2. Compute BC_truth = max_s EU_i(truth), WC_truth = min_s EU_i(truth).
      3. Sample M_nom domain-consistent misreports for agent i.
      4. obvious_gain = min(relu(BC_lie-BC_truth), relu(WC_lie-WC_truth)).
      5. violation_i = max over M misreports of obvious_gain.

    Returns:
        nom:        scalar mean violation
        violations: [B, A]
    """
    S_nom = S or cfg.S
    M_nom = M or cfg.M
    B, A, m = marginal_rank.shape
    device   = marginal_rank.device

    # Sample S_nom domain-consistent opponent profiles (full profiles)
    mr_opp_flat = sample_domain_mr_flat(cfg, domain, endow_idx, S_nom, device)  # [B*S, A, m]
    mr_opp = mr_opp_flat.reshape(B, S_nom, A, m)

    # Sample M_nom domain-consistent misreports (per-agent, will be injected)
    mr_mis_flat = sample_domain_mr_flat(cfg, domain, endow_idx, M_nom, device)  # [B*M, A, m]
    mr_mis = mr_mis_flat.reshape(B, M_nom, A, m)

    all_violations = []

    for i in range(A):
        # Fix agent i's true rank in the opponent pool
        mr_opp_i = mr_opp.clone()
        mr_opp_i[:, :, i, :] = marginal_rank[:, i, :].unsqueeze(1).expand(B, S_nom, m)

        # ── truth: BC / WC ────────────────────────────────────────────────
        mr_flat    = mr_opp_i.reshape(B * S_nom, A, m)
        endow_rep  = endow_idx.unsqueeze(1).expand(B, S_nom).reshape(B * S_nom)

        S_flat     = score_matrix(cfg, mr_flat)
        mask_flat  = ir_pe_mask(cfg, S_flat, endow_rep)
        probs_flat = net(mr_flat, mask=mask_flat)              # [B*S, K]

        S_i_true = S_true[:, i, :]                             # [B, K]
        S_i_flat = S_i_true.unsqueeze(1).expand(B, S_nom, -1).reshape(B * S_nom, -1)

        u_truth  = (probs_flat * S_i_flat).sum(1).reshape(B, S_nom)
        BC_truth = u_truth.max(1).values   # [B]
        WC_truth = u_truth.min(1).values   # [B]

        # ── lie: BC / WC across M misreports × S opponent profiles ────────
        # Replace agent i's slot with M_nom domain-consistent misreports
        mr_mis_full = mr_opp.unsqueeze(1).expand(B, M_nom, S_nom, A, m).clone()
        mr_mis_full[:, :, :, i, :] = mr_mis[:, :, i, :].unsqueeze(2).expand(B, M_nom, S_nom, m)

        BMS       = B * M_nom * S_nom
        mr_mis_f  = mr_mis_full.reshape(BMS, A, m)
        endow_r2  = endow_idx.view(B, 1, 1).expand(B, M_nom, S_nom).reshape(BMS)

        S_mis_f   = score_matrix(cfg, mr_mis_f)
        mask_mis  = ir_pe_mask(cfg, S_mis_f, endow_r2)
        probs_mis = net(mr_mis_f, mask=mask_mis)               # [BMS, K]

        S_i_bms = (S_i_true.view(B, 1, 1, -1)
                            .expand(B, M_nom, S_nom, -1)
                            .reshape(BMS, -1))
        u_lie = (probs_mis * S_i_bms).sum(1).reshape(B, M_nom, S_nom)

        BC_lie = u_lie.max(2).values   # [B, M]
        WC_lie = u_lie.min(2).values

        bc_gain  = torch.relu(BC_lie - BC_truth.unsqueeze(1))
        wc_gain  = torch.relu(WC_lie - WC_truth.unsqueeze(1))
        obvious  = torch.min(bc_gain, wc_gain)
        all_violations.append(obvious.max(1).values)           # [B]

    violations = torch.stack(all_violations, dim=1)   # [B, A]
    return violations.mean(), violations


def augmented_objective(
    cfg: Config,
    domain: DomainSpec,
    net: AllocationNet,
    marginal_rank: torch.Tensor,
    endow_idx: torch.Tensor,
    S: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Augmented Lagrangian: -welfare_weight*welfare + lambda*NOM + rho/2*NOM^2."""
    probs   = net(marginal_rank, mask=mask, temperature=cfg.temperature)
    ES      = _expected_score(probs, S)
    welfare = ES.sum(dim=1).mean()

    nom, _ = nom_loss(cfg, domain, net, marginal_rank, endow_idx, S,
                      S=cfg.S, M=cfg.M)

    loss = (
        -cfg.welfare_weight * welfare
        + cfg.lambda_nom * nom
        + (cfg.rho / 2.0) * nom * nom
    )
    return loss, {
        "loss":    float(loss.detach()),
        "welfare": float(welfare.detach()),
        "nom":     float(nom.detach()),
    }
