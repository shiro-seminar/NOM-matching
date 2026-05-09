"""Evaluation for ordinal NOM mechanism.

Metrics (all deterministic argmax):
  welfare:   mean score of chosen bundle over agents (higher = better)
  ir_rate:   fraction of profiles where ALL agents are IR
  pe_rate:   fraction of profiles where chosen allocation is PE
  irpe_rate: fraction of profiles where chosen alloc is IR + PE + Balanced
  nom_mean:  mean NOM violation (obvious manipulation gain)
  nom_viol:  fraction of profiles with any obvious manipulation > 1e-5
"""
from __future__ import annotations
import argparse
import torch
from .config import Config
from .allocations import (
    score_matrix, ir_pe_mask, ir_mask, pareto_mask,
    endowment_scores, num_allocations,
)
from .data_gen import sample_batch
from .model import AllocationNet
from .losses import nom_loss
from .benchmarks import BENCHMARKS

S_EVAL = 16
M_EVAL = 16


@torch.no_grad()
def evaluate_mechanism(name, mech_fn, cfg, marginal_rank, endow_idx, S, wmax_s, is_nn=False):
    B = marginal_rank.shape[0]
    K = num_allocations(cfg)

    probs = mech_fn(cfg, marginal_rank, endow_idx, S)   # [B, K]
    ES    = torch.einsum("bk,bak->ba", probs, S)         # [B, A]
    welfare = ES.sum(1).mean()

    s0      = endowment_scores(S, endow_idx)             # [B, A]
    ir_viol = (ES < s0 - 1e-6).any(1).float().mean()

    chosen    = probs.argmax(1)
    pe_m      = pareto_mask(S)
    irpe_m    = ir_pe_mask(cfg, S, endow_idx)
    pe_rate   = pe_m.gather(1, chosen.unsqueeze(1)).squeeze(1).mean()
    irpe_rate = irpe_m.gather(1, chosen.unsqueeze(1)).squeeze(1).mean()

    nom_mean, nom_viol_rate = _nom_benchmark(cfg, mech_fn, marginal_rank, endow_idx, S,
                                             S_EVAL, M_EVAL)

    wmax_m = float(wmax_s.mean())
    welfare_ratio = float(welfare) / wmax_m if abs(wmax_m) > 1e-9 else 1.0

    return {
        "name":          name,
        "welfare":       float(welfare),
        "welfare_ratio": welfare_ratio,
        "ir_viol":       float(ir_viol),
        "pe_rate":       float(pe_rate),
        "irpe_rate":     float(irpe_rate),
        "nom_mean":      nom_mean,
        "nom_viol":      nom_viol_rate,
    }


def _nom_benchmark(cfg, mech_fn, marginal_rank, endow_idx, S_true, S_nom, M_nom):
    """NOM computation for arbitrary mechanisms."""
    B, A, m = marginal_rank.shape
    R = cfg.num_ranks
    device = marginal_rank.device
    all_viol = []

    for i in range(A):
        mr_opp = torch.randint(0, R, (B, S_nom, A, m), device=device)
        mr_opp[:, :, i, :] = marginal_rank[:, i, :].unsqueeze(1).expand(B, S_nom, m)

        mr_flat   = mr_opp.reshape(B * S_nom, A, m)
        endow_rep = endow_idx.unsqueeze(1).expand(B, S_nom).reshape(B * S_nom)
        S_flat    = score_matrix(cfg, mr_flat)
        p_flat    = mech_fn(cfg, mr_flat, endow_rep, S_flat)         # [B*S, K]

        S_i_true = S_true[:, i, :]                                    # [B, K]
        S_i_flat = S_i_true.unsqueeze(1).expand(B, S_nom, -1).reshape(B * S_nom, -1)
        u_truth  = (p_flat * S_i_flat).sum(1).reshape(B, S_nom)
        BC_t = u_truth.max(1).values
        WC_t = u_truth.min(1).values

        mr_mis = torch.randint(0, R, (B, M_nom, m), device=device)
        mr_mis_full = mr_opp.unsqueeze(1).expand(B, M_nom, S_nom, A, m).clone()
        mr_mis_full[:, :, :, i, :] = mr_mis.unsqueeze(2).expand(B, M_nom, S_nom, m)

        BMS       = B * M_nom * S_nom
        mr_mis_f  = mr_mis_full.reshape(BMS, A, m)
        endow_r2  = endow_idx.view(B, 1, 1).expand(B, M_nom, S_nom).reshape(BMS)
        S_mis_f   = score_matrix(cfg, mr_mis_f)
        p_mis_f   = mech_fn(cfg, mr_mis_f, endow_r2, S_mis_f)        # [BMS, K]

        S_i_bms = S_i_true.view(B, 1, 1, -1).expand(B, M_nom, S_nom, -1).reshape(BMS, -1)
        u_lie   = (p_mis_f * S_i_bms).sum(1).reshape(B, M_nom, S_nom)

        BC_l = u_lie.max(2).values   # [B, M]
        WC_l = u_lie.min(2).values

        bc_gain  = torch.relu(BC_l - BC_t.unsqueeze(1))
        wc_gain  = torch.relu(WC_l - WC_t.unsqueeze(1))
        max_obv  = torch.min(bc_gain, wc_gain).max(1).values   # [B]
        all_viol.append(max_obv)

    viol = torch.stack(all_viol, 1)   # [B, A]
    return float(viol.mean()), float((viol.max(1).values > 1e-5).float().mean())


def print_table(results):
    hdr = (f"{'Mechanism':<16} {'Welfare':>8} {'W/WMAX':>7} "
           f"{'IR-viol%':>9} {'PE%':>7} {'IR+PE%':>8} {'NOM-mean':>9} {'NOM-viol%':>10}")
    sep = "-" * len(hdr)
    print("\n" + sep); print(hdr); print(sep)
    for r in results:
        print(f"{r['name']:<16} {r['welfare']:>8.4f} {r['welfare_ratio']:>7.3f} "
              f"{r['ir_viol']*100:>9.1f} {r['pe_rate']*100:>7.1f} {r['irpe_rate']*100:>8.1f} "
              f"{r['nom_mean']:>9.5f} {r['nom_viol']*100:>10.1f}")
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_eval",     type=int, default=200)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    cfg.device     = args.device
    cfg.batch_size = args.n_eval
    torch.manual_seed(args.seed)

    batch         = sample_batch(cfg)
    marginal_rank = batch["marginal_rank"]
    endow_idx     = batch["endow_idx"]
    S             = batch["S"]

    # WMAX score: oracle upper bound on welfare
    irpe_m  = ir_pe_mask(cfg, S, endow_idx)
    wmax_s  = (S.sum(1) + (1 - irpe_m) * (-1e9)).max(1).values   # [B]

    results = []
    for bname, bfn in BENCHMARKS.items():
        print(f"Evaluating {bname}...")
        results.append(evaluate_mechanism(bname, bfn, cfg, marginal_rank, endow_idx, S, wmax_s))

    if args.checkpoint:
        print(f"Loading {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        net  = AllocationNet(cfg)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()

        def net_mech(cfg_, mr_, ei_, S_):
            mask = ir_pe_mask(cfg, S_, ei_)
            return net(mr_, mask=mask, temperature=1e-3)

        results.append(evaluate_mechanism("LearnedNet", net_mech, cfg,
                                          marginal_rank, endow_idx, S, wmax_s, is_nn=True))

    print_table(results)


if __name__ == "__main__":
    main()
