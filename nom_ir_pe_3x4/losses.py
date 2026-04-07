"""NOM + IR + PE losses for A-agent M-item economy.

NOM loss は 2x3 と同じサンプリングベース。
A>2 への一般化: エージェント i 以外の全エージェントをまとめてサンプリング。
"""
from __future__ import annotations
import torch
from .config import Config
from .allocations import all_utilities, ir_pe_mask
from .model import AllocationNet


def _expected_utility(probs: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """EU[b, a] = sum_k probs[b,k] * U[b,a,k]."""
    return torch.einsum("bk,bak->ba", probs, U)


def nom_loss(
    cfg: Config,
    net: AllocationNet,
    v: torch.Tensor,          # [B, A, m]
    endow_idx: torch.Tensor,  # [B]
    S: int | None = None,
    M: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NOM loss: obvious manipulation gain averaged over agents.

    Returns:
        nom:        scalar
        violations: [B, A]
    """
    S = S or cfg.S
    M = M or cfg.M
    B, A, m = v.shape
    device = v.device

    # S joint opponent profiles (全エージェントをランダムサンプル → agent iの分だけ真値で上書き)
    v_opp = torch.empty(B, S, A, m, device=device).uniform_(cfg.v_min, cfg.v_max)
    # M misreport candidates
    v_mis = torch.empty(B, M, m, device=device).uniform_(cfg.v_min, cfg.v_max)

    all_violations = []

    for i in range(A):
        # agent i の真値を固定
        v_opp_i = v_opp.clone()
        v_opp_i[:, :, i, :] = v[:, i, :].unsqueeze(1).expand(B, S, m)

        # ── 真の報告での BC/WC ────────────────────────────────
        v_flat = v_opp_i.reshape(B * S, A, m)
        endow_rep = endow_idx.unsqueeze(1).expand(B, S).reshape(B * S)
        U_flat = all_utilities(cfg, v_flat)
        mask_flat = ir_pe_mask(cfg, U_flat, endow_rep)
        probs_flat = net(v_flat, mask=mask_flat)              # [B*S, K]

        U_i_flat = U_flat[:, i, :]                           # [B*S, K]
        u_truth = (probs_flat * U_i_flat).sum(1).reshape(B, S)  # [B, S]
        BC_truth = u_truth.max(1).values                     # [B]
        WC_truth = u_truth.min(1).values                     # [B]

        # ── misreport v_i' での BC/WC ─────────────────────────
        # [B, M, S, A, m]: agent i をmisreport、他はv_opp
        v_mis_full = v_opp.unsqueeze(1).expand(B, M, S, A, m).clone()
        v_mis_full[:, :, :, i, :] = v_mis.unsqueeze(2).expand(B, M, S, m)

        BMS = B * M * S
        v_mis_f = v_mis_full.reshape(BMS, A, m)
        endow_rep2 = endow_idx.view(B, 1, 1).expand(B, M, S).reshape(BMS)
        U_mis_f = all_utilities(cfg, v_mis_f)
        mask_mis = ir_pe_mask(cfg, U_mis_f, endow_rep2)
        probs_mis = net(v_mis_f, mask=mask_mis)               # [BMS, K]

        # 真の v_i で評価
        v_eval_f = v_mis_f.clone()
        v_eval_f[:, i, :] = (v[:, i, :]
                             .view(B, 1, 1, m)
                             .expand(B, M, S, m)
                             .reshape(BMS, m))
        U_eval_f = all_utilities(cfg, v_eval_f)
        u_lie = (probs_mis * U_eval_f[:, i, :]).sum(1).reshape(B, M, S)  # [B, M, S]

        BC_lie = u_lie.max(2).values    # [B, M]
        WC_lie = u_lie.min(2).values    # [B, M]

        bc_gain = torch.relu(BC_lie - BC_truth.unsqueeze(1))
        wc_gain = torch.relu(WC_lie - WC_truth.unsqueeze(1))
        obvious  = torch.min(bc_gain, wc_gain)
        all_violations.append(obvious.max(1).values)          # [B]

    violations = torch.stack(all_violations, dim=1)           # [B, A]
    return violations.mean(), violations


def augmented_objective(
    cfg: Config,
    net: AllocationNet,
    v: torch.Tensor,
    endow_idx: torch.Tensor,
    U: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    probs = net(v, mask=mask, temperature=cfg.temperature)
    EU = _expected_utility(probs, U)
    welfare = EU.sum(dim=1).mean()

    nom, _ = nom_loss(cfg, net, v, endow_idx, S=cfg.S, M=cfg.M)

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
