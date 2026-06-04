"""NOM + IR + PE losses for ordinal preferences.

Structure mirrors nom_ir_pe_3x4/losses.py.  Key substitutions:
  v  [B,A,m] float  ->  marginal_rank [B,A,m] int
  U  [B,A,K] float  ->  S [B,A,K] float  (score = -rank_sum, higher = better)
  all_utilities()   ->  score_matrix()

For the NOM lie-evaluation, agent i's score under the TRUE preferences is
S_true[:, i, :] regardless of what was reported (no v_eval reconstruction
needed -- ordinal simplification).

NOM definition follows Troyan-Morrill (2020):
  obvious manipulation = min(relu(BC_lie - BC_truth), relu(WC_lie - WC_truth))
"""
from __future__ import annotations

import torch
from .config import Config
from .allocations import score_matrix, ir_pe_mask, build_all_allocs
from .model import AllocationNet


def _sample_trichotomous_mr(cfg: Config, endow_idx: torch.Tensor,
                             B: int, N: int, device: torch.device) -> torch.Tensor:
    """Sample [B*N, A, m] domain-consistent marginal ranks for trichotomous.

    Owned items (those in endow_idx allocation) → rank in {0,...,R-2} (ε(R)=0).
    Unowned items → rank in {0,...,R-1}.
    """
    A, m, R = cfg.num_agents, cfg.num_items, cfg.num_ranks
    allocs      = build_all_allocs(cfg)
    endow_alloc = allocs[endow_idx.cpu()]            # [B, m]
    endow_exp   = endow_alloc.unsqueeze(1).expand(B, N, m).reshape(B * N, m).to(device)

    BN = B * N
    mr = torch.zeros(BN, A, m, dtype=torch.long, device=device)
    for a in range(A):
        for j in range(m):
            owned      = (endow_exp[:, j] == a)
            r_owned    = torch.randint(0, R - 1, (BN,), device=device)
            r_unowned  = torch.randint(0, R,     (BN,), device=device)
            mr[:, a, j] = torch.where(owned, r_owned, r_unowned)
    return mr


def _expected_score(probs: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """ES[b, a] = sum_k probs[b,k] * S[b,a,k]."""
    return torch.einsum("bk,bak->ba", probs, S)


def nom_loss(
    cfg: Config,
    net: AllocationNet,
    marginal_rank: torch.Tensor,   # [B, A, m]
    endow_idx: torch.Tensor,       # [B]
    S_true: torch.Tensor,          # [B, A, K]  precomputed score matrix
    S: int | None = None,
    M: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NOM violation loss (obvious manipulation, Troyan-Morrill 2020).

    For each agent i:
      1. Fix i's true marginal rank; sample S opponent profiles for others.
      2. Compute BC_truth = max_s EU_i(truth) and WC_truth = min_s EU_i(truth).
      3. For M misreports: compute BC_lie, WC_lie under TRUE i-preferences.
      4. obvious_gain = min(relu(BC_lie-BC_truth), relu(WC_lie-WC_truth)).
      5. violation_i = max over M misreports of obvious_gain.

    Returns:
        nom:        scalar mean violation
        violations: [B, A]
    """
    S_nom = S or cfg.S
    M_nom = M or cfg.M
    B, A, m = marginal_rank.shape
    R = cfg.num_ranks
    device = marginal_rank.device

    # Domain-consistent opponent and misreport pools
    mr_opp_flat = _sample_trichotomous_mr(cfg, endow_idx, B, S_nom, device)
    mr_opp      = mr_opp_flat.reshape(B, S_nom, A, m)                # [B,S,A,m]
    # Per-agent misreports: agent i's owned items constrained to {0,...,R-2}
    mr_mis_flat = _sample_trichotomous_mr(cfg, endow_idx, B, M_nom, device)
    mr_mis      = mr_mis_flat.reshape(B, M_nom, A, m)                # [B,M,A,m]

    all_violations = []

    for i in range(A):
        # Fix agent i's true marginal rank in the opponent pool
        mr_opp_i = mr_opp.clone()
        mr_opp_i[:, :, i, :] = marginal_rank[:, i, :].unsqueeze(1).expand(B, S_nom, m)

        # ── truth: BC / WC ──────────────────────────────────────────────
        mr_flat   = mr_opp_i.reshape(B * S_nom, A, m)
        endow_rep = endow_idx.unsqueeze(1).expand(B, S_nom).reshape(B * S_nom)

        S_flat     = score_matrix(cfg, mr_flat)           # [B*S, A, K]
        mask_flat  = ir_pe_mask(cfg, S_flat, endow_rep)
        probs_flat = net(mr_flat, mask=mask_flat)         # [B*S, K]

        # Agent i's score under TRUE preferences (same as S_true[:, i, :])
        S_i_true = S_true[:, i, :]                                      # [B, K]
        S_i_rep  = S_i_true.unsqueeze(1).expand(B, S_nom, -1)          # [B, S, K]
        S_i_flat = S_i_rep.reshape(B * S_nom, -1)                       # [B*S, K]

        u_truth = (probs_flat * S_i_flat).sum(1).reshape(B, S_nom)     # [B, S]
        BC_truth = u_truth.max(1).values   # [B]
        WC_truth = u_truth.min(1).values   # [B]

        # ── lie: BC / WC across M misreports × S opponent profiles ──────
        # mr_opp (not mr_opp_i): opponents are random; agent i's slot
        # will be overwritten with the misreport
        mr_mis_full = mr_opp.unsqueeze(1).expand(B, M_nom, S_nom, A, m).clone()
        mr_mis_full[:, :, :, i, :] = mr_mis[:, :, i, :].unsqueeze(2).expand(B, M_nom, S_nom, m)

        BMS         = B * M_nom * S_nom
        mr_mis_f    = mr_mis_full.reshape(BMS, A, m)
        endow_rep2  = endow_idx.view(B, 1, 1).expand(B, M_nom, S_nom).reshape(BMS)

        S_mis_f    = score_matrix(cfg, mr_mis_f)
        mask_mis   = ir_pe_mask(cfg, S_mis_f, endow_rep2)
        probs_mis  = net(mr_mis_f, mask=mask_mis)                      # [BMS, K]

        # Evaluate under agent i's TRUE score (no reconstruction needed)
        S_i_bms = (S_i_true.view(B, 1, 1, -1)
                            .expand(B, M_nom, S_nom, -1)
                            .reshape(BMS, -1))                          # [BMS, K]
        u_lie = (probs_mis * S_i_bms).sum(1).reshape(B, M_nom, S_nom)  # [B, M, S]

        BC_lie = u_lie.max(2).values   # [B, M]
        WC_lie = u_lie.min(2).values   # [B, M]

        bc_gain  = torch.relu(BC_lie - BC_truth.unsqueeze(1))   # [B, M]
        wc_gain  = torch.relu(WC_lie - WC_truth.unsqueeze(1))
        obvious  = torch.max(bc_gain, wc_gain)                  # [B, M]
        all_violations.append(obvious.max(1).values)            # [B]

    violations = torch.stack(all_violations, dim=1)   # [B, A]
    return violations.mean(), violations


def augmented_objective(
    cfg: Config,
    net: AllocationNet,
    marginal_rank: torch.Tensor,   # [B, A, m]
    endow_idx: torch.Tensor,       # [B]
    S: torch.Tensor,               # [B, A, K]
    mask: torch.Tensor,            # [B, K]
) -> tuple[torch.Tensor, dict]:
    """Augmented Lagrangian: -welfare_weight*welfare + lambda*NOM + rho/2*NOM^2."""
    probs   = net(marginal_rank, mask=mask, temperature=cfg.temperature)
    ES      = _expected_score(probs, S)     # [B, A]
    welfare = ES.sum(dim=1).mean()

    nom, _ = nom_loss(cfg, net, marginal_rank, endow_idx, S,
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
