"""Evaluation script: compare learned mechanism vs benchmarks.

Usage:
    python -m nom_ir_pe_2x3.evaluate                              # random net
    python -m nom_ir_pe_2x3.evaluate --checkpoint allocation_net.pt

Metrics (estimated on N_eval=4000 profiles):
  - NOM violation rate:   fraction of profiles where any agent has an obvious manipulation
  - NOM mean regret:      mean obvious-manipulation gain across agents & profiles
  - IR violation rate:    fraction of profiles where any agent's EU < endowment utility
  - PE rate:              fraction of profiles where the chosen allocation is PE
  - IR∩PE rate:           fraction satisfying both simultaneously
  - Mean welfare:         mean social welfare (sum of agent utilities)
  - Max-welfare ratio:    mean welfare / mean WMAX welfare (efficiency relative to first-best IR-free)
"""
from __future__ import annotations

import argparse
from typing import Callable

import torch

from .config import Config
from .allocations import (
    K,
    all_utilities,
    ir_pe_mask,
    ir_mask,
    pareto_mask,
    endowment_utilities,
)
from .data_gen import sample_batch
from .model import AllocationNet
from .losses import nom_loss
from .benchmarks import BENCHMARKS


N_EVAL   = 1_000
S_EVAL   = 32     # opponent samples for NOM evaluation
M_EVAL   = 32     # misreport samples for NOM evaluation


@torch.no_grad()
def evaluate_mechanism(
    name: str,
    mech: Callable,
    cfg: Config,
    v: torch.Tensor,
    endow_idx: torch.Tensor,
    U: torch.Tensor,
    wmax_welfare: torch.Tensor,
    is_nn: bool = False,
) -> dict:
    """Evaluate a single mechanism.

    Args:
        mech:          callable(v, endow_idx, U) → probs [B, K]
        wmax_welfare:  [B] unconstrained welfare-maximizing welfare (for ratio)

    Returns:
        dict of scalar metrics
    """
    B = v.shape[0]

    probs = mech(v, endow_idx, U)               # [B, K]

    # ── Expected utilities ────────────────────────────────────────────────────
    EU = torch.einsum("bk,bak->ba", probs, U)   # [B, A]
    welfare = EU.sum(dim=1)                     # [B]

    # ── IR ───────────────────────────────────────────────────────────────────
    u0 = endowment_utilities(U, endow_idx)      # [B, A]
    ir_viol = (EU < u0 - 1e-6).any(dim=1)       # [B]
    ir_viol_rate = float(ir_viol.float().mean())

    # ── PE of chosen allocation ───────────────────────────────────────────────
    # Use argmax allocation for deterministic mechanisms, or expected over probs
    chosen_idx = probs.argmax(dim=1)            # [B]  (deterministic if one-hot)
    pe_mask = pareto_mask(U)                    # [B, K]
    chosen_is_pe = pe_mask.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)  # [B]
    pe_rate = float(chosen_is_pe.mean())

    # ── IR ∩ PE rate ──────────────────────────────────────────────────────────
    ir_pe = ir_pe_mask(U, endow_idx)            # [B, K]
    chosen_is_irpe = ir_pe.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)
    irpe_rate = float(chosen_is_irpe.mean())

    # ── NOM: sampling-based ──────────────────────────────────────────────────
    if is_nn:
        # NOM requires gradient tracking for the net, but we call without grad
        # Re-enable grad temporarily just to build the computation
        with torch.enable_grad():
            nom_val, violations = nom_loss(cfg, mech, v, endow_idx, S=S_EVAL, M=M_EVAL)
        nom_mean   = float(nom_val.detach())
        nom_viol_rate = float((violations.detach().max(dim=1).values > 1e-5).float().mean())
    else:
        # For benchmark mechanisms: wrap as a pseudo-net-callable
        nom_mean, nom_viol_rate = _nom_for_benchmark(cfg, mech, v, endow_idx, U)

    welfare_ratio = float(welfare.mean() / wmax_welfare.mean().clamp(min=1e-9))

    return {
        "name":           name,
        "welfare":        float(welfare.mean()),
        "welfare_ratio":  welfare_ratio,
        "ir_viol_rate":   ir_viol_rate,
        "pe_rate":        pe_rate,
        "irpe_rate":      irpe_rate,
        "nom_mean":       nom_mean,
        "nom_viol_rate":  nom_viol_rate,
    }


def _nom_for_benchmark(
    cfg: Config,
    bench_fn: Callable,
    v: torch.Tensor,
    endow_idx: torch.Tensor,
    U: torch.Tensor,
) -> tuple[float, float]:
    """Compute NOM violation for a deterministic benchmark via vectorized sampling."""
    B, A, m = v.shape
    device = v.device
    S = S_EVAL
    M_mis = M_EVAL

    all_violations = []
    for i in range(A):
        # ── Sample S opponent profiles (vectorized) ──────────────────────────
        v_opp = torch.empty(B, S, A, m, device=device).uniform_(cfg.v_min, cfg.v_max)
        v_opp[:, :, i, :] = v[:, i, :].unsqueeze(1).expand(B, S, m)

        # Flatten to (B*S, A, m) and run mechanism
        v_opp_f = v_opp.reshape(B * S, A, m)
        endow_rep = endow_idx.unsqueeze(1).expand(B, S).reshape(B * S)
        U_opp_f = all_utilities(v_opp_f)
        p_opp_f = bench_fn(v_opp_f, endow_rep, U_opp_f)   # [B*S, K]

        # Utility for agent i at true v_i (already embedded in v_opp)
        U_i_f = U_opp_f[:, i, :]                           # [B*S, K]
        u_truth_f = (p_opp_f * U_i_f).sum(dim=1)           # [B*S]
        u_truth = u_truth_f.reshape(B, S)                  # [B, S]
        BC_truth = u_truth.max(dim=1).values               # [B]
        WC_truth = u_truth.min(dim=1).values               # [B]

        # ── Sample M misreports (vectorized) ────────────────────────────────
        v_mis = torch.empty(B, M_mis, m, device=device).uniform_(cfg.v_min, cfg.v_max)

        # Build (B, M, S, A, m) mis-profiles
        # agent i = misreport, agent j = opponent sample
        v_mis_full = v_opp.unsqueeze(1).expand(B, M_mis, S, A, m).clone()
        v_mis_full[:, :, :, i, :] = v_mis.unsqueeze(2).expand(B, M_mis, S, m)

        # Flatten to (B*M*S, A, m)
        BMS = B * M_mis * S
        v_mis_f = v_mis_full.reshape(BMS, A, m)
        endow_rep2 = endow_idx.view(B, 1, 1).expand(B, M_mis, S).reshape(BMS)
        U_mis_f = all_utilities(v_mis_f)
        p_mis_f = bench_fn(v_mis_f, endow_rep2, U_mis_f)   # [BMS, K]

        # Evaluate utility at TRUE v_i
        v_eval_f = v_mis_f.clone()
        v_eval_f[:, i, :] = v[:, i, :].view(B, 1, 1, m).expand(B, M_mis, S, m).reshape(BMS, m)
        U_eval_f = all_utilities(v_eval_f)
        U_eval_i_f = U_eval_f[:, i, :]                     # [BMS, K]
        u_lie_f = (p_mis_f * U_eval_i_f).sum(dim=1)        # [BMS]
        u_lie = u_lie_f.reshape(B, M_mis, S)               # [B, M, S]

        BC_lie = u_lie.max(dim=2).values                   # [B, M]
        WC_lie = u_lie.min(dim=2).values                   # [B, M]

        bc_gain = torch.relu(BC_lie - BC_truth.unsqueeze(1))
        wc_gain = torch.relu(WC_lie - WC_truth.unsqueeze(1))
        obvious = torch.min(bc_gain, wc_gain)
        max_obvious = obvious.max(dim=1).values            # [B]

        all_violations.append(max_obvious)

    viol = torch.stack(all_violations, dim=1)   # [B, A]
    nom_mean = float(viol.mean())
    nom_viol_rate = float((viol.max(dim=1).values > 1e-5).float().mean())
    return nom_mean, nom_viol_rate


def print_table(results: list[dict]) -> None:
    header = (
        f"{'Mechanism':<18} {'Welfare':>8} {'W/WMAX':>7} "
        f"{'IR-viol%':>9} {'PE%':>7} {'IR∩PE%':>8} "
        f"{'NOM-mean':>9} {'NOM-viol%':>10}"
    )
    sep = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['name']:<18} {r['welfare']:>8.4f} {r['welfare_ratio']:>7.3f} "
            f"{r['ir_viol_rate']*100:>9.1f} {r['pe_rate']*100:>7.1f} {r['irpe_rate']*100:>8.1f} "
            f"{r['nom_mean']:>9.5f} {r['nom_viol_rate']*100:>10.1f}"
        )
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_eval",     type=int, default=N_EVAL)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    cfg.device     = args.device
    cfg.batch_size = args.n_eval
    torch.manual_seed(args.seed)

    # Sample evaluation data
    device = torch.device(cfg.device)
    batch = sample_batch(cfg)
    v         = batch["v"]
    endow_idx = batch["endow_idx"]
    U         = batch["U"]

    # Unconstrained WMAX welfare for ratio baseline
    wmax_welfare = U.sum(dim=1).max(dim=1).values   # [B]

    results = []

    # ── Benchmark mechanisms ──────────────────────────────────────────────────
    for bname, bfn in BENCHMARKS.items():
        print(f"Evaluating {bname}...")
        r = evaluate_mechanism(bname, bfn, cfg, v, endow_idx, U, wmax_welfare, is_nn=False)
        results.append(r)

    # ── Learned mechanism ─────────────────────────────────────────────────────
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        net = AllocationNet(cfg).to(device)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()

        print("Evaluating learned mechanism...")
        # Wrap as mech(v, endow_idx, U)
        from .allocations import ir_pe_mask

        def net_mech(v_, endow_idx_, U_):
            mask = ir_pe_mask(U_, endow_idx_)
            return net(v_, mask=mask, temperature=1e-3)   # near-deterministic

        r = evaluate_mechanism("LearnedNet", net_mech, cfg, v, endow_idx, U, wmax_welfare, is_nn=False)
        results.append(r)

    print_table(results)


if __name__ == "__main__":
    main()
