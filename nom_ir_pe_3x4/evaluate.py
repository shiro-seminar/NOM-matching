from __future__ import annotations
import argparse
import torch
from .config import Config
from .allocations import all_utilities, ir_pe_mask, ir_mask, pareto_mask, endowment_utilities, num_allocations
from .data_gen import sample_batch
from .model import AllocationNet
from .losses import nom_loss
from .benchmarks import BENCHMARKS

S_EVAL = 16
M_EVAL = 16


@torch.no_grad()
def evaluate_mechanism(name, mech_fn, cfg, v, endow_idx, U, wmax_w, is_nn=False):
    B = v.shape[0]
    K = num_allocations(cfg)
    probs = mech_fn(cfg, v, endow_idx, U)                    # [B, K]
    EU = torch.einsum("bk,bak->ba", probs, U)                # [B, A]
    welfare = EU.sum(1).mean()

    u0 = endowment_utilities(U, endow_idx)
    ir_viol = (EU < u0 - 1e-6).any(1).float().mean()

    chosen = probs.argmax(1)
    pe_m = pareto_mask(U)
    irpe_m = ir_pe_mask(cfg, U, endow_idx)
    pe_rate   = pe_m.gather(1, chosen.unsqueeze(1)).squeeze(1).mean()
    irpe_rate = irpe_m.gather(1, chosen.unsqueeze(1)).squeeze(1).mean()

    # NOM（ベンチマーク用: small S/M）
    nom_mean, nom_viol = _nom_benchmark(cfg, mech_fn, v, endow_idx, U, S_EVAL, M_EVAL)

    return {
        "name":          name,
        "welfare":       float(welfare),
        "welfare_ratio": float(welfare / wmax_w.mean().clamp(min=1e-9)),
        "ir_viol":       float(ir_viol),
        "pe_rate":       float(pe_rate),
        "irpe_rate":     float(irpe_rate),
        "nom_mean":      nom_mean,
        "nom_viol":      nom_viol,
    }


def _nom_benchmark(cfg, mech_fn, v, endow_idx, U, S, M):
    """Vectorized NOM computation for benchmark mechanisms."""
    B, A, m = v.shape
    device = v.device
    all_viol = []

    for i in range(A):
        # ── truth BC/WC (vectorized over S) ──────────────────────────────
        v_opp = torch.empty(B, S, A, m, device=device).uniform_(cfg.v_min, cfg.v_max)
        v_opp[:, :, i, :] = v[:, i, :].unsqueeze(1).expand(B, S, m)

        v_flat = v_opp.reshape(B * S, A, m)
        endow_rep = endow_idx.unsqueeze(1).expand(B, S).reshape(B * S)
        U_flat = all_utilities(cfg, v_flat)
        p_flat = mech_fn(cfg, v_flat, endow_rep, U_flat)          # [B*S, K]
        u_truth = (p_flat * U_flat[:, i, :]).sum(1).reshape(B, S) # [B, S]
        BC_t = u_truth.max(1).values
        WC_t = u_truth.min(1).values

        # ── misreport BC/WC (vectorized over M*S) ────────────────────────
        v_mis = torch.empty(B, M, m, device=device).uniform_(cfg.v_min, cfg.v_max)

        v_mis_full = v_opp.unsqueeze(1).expand(B, M, S, A, m).clone()
        v_mis_full[:, :, :, i, :] = v_mis.unsqueeze(2).expand(B, M, S, m)

        BMS = B * M * S
        v_mis_f = v_mis_full.reshape(BMS, A, m)
        endow_rep2 = endow_idx.view(B, 1, 1).expand(B, M, S).reshape(BMS)
        U_mis_f = all_utilities(cfg, v_mis_f)
        p_mis_f = mech_fn(cfg, v_mis_f, endow_rep2, U_mis_f)      # [BMS, K]

        v_eval_f = v_mis_f.clone()
        v_eval_f[:, i, :] = v[:, i, :].view(B,1,1,m).expand(B,M,S,m).reshape(BMS, m)
        U_eval_f = all_utilities(cfg, v_eval_f)
        u_lie = (p_mis_f * U_eval_f[:, i, :]).sum(1).reshape(B, M, S)

        BC_l = u_lie.max(2).values    # [B, M]
        WC_l = u_lie.min(2).values

        bc_gain = torch.relu(BC_l - BC_t.unsqueeze(1))
        wc_gain = torch.relu(WC_l - WC_t.unsqueeze(1))
        max_obv = torch.min(bc_gain, wc_gain).max(1).values       # [B]
        all_viol.append(max_obv)

    viol = torch.stack(all_viol, 1)
    return float(viol.mean()), float((viol.max(1).values > 1e-5).float().mean())


def print_table(results):
    hdr = (f"{'Mechanism':<16} {'Welfare':>8} {'W/WMAX':>7} "
           f"{'IR-viol%':>9} {'PE%':>7} {'IR∩PE%':>8} {'NOM-mean':>9} {'NOM-viol%':>10}")
    sep = "─" * len(hdr)
    print("\n" + sep); print(hdr); print(sep)
    for r in results:
        print(f"{r['name']:<16} {r['welfare']:>8.4f} {r['welfare_ratio']:>7.3f} "
              f"{r['ir_viol']*100:>9.1f} {r['pe_rate']*100:>7.1f} {r['irpe_rate']*100:>8.1f} "
              f"{r['nom_mean']:>9.5f} {r['nom_viol']*100:>10.1f}")
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_eval",     type=int, default=500)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    cfg = Config(); cfg.device = args.device; cfg.batch_size = args.n_eval
    torch.manual_seed(args.seed)

    batch = sample_batch(cfg)
    v, endow_idx, U = batch["v"], batch["endow_idx"], batch["U"]
    wmax_w = U.sum(1).max(1).values

    results = []
    for bname, bfn in BENCHMARKS.items():
        print(f"Evaluating {bname}...")
        results.append(evaluate_mechanism(bname, bfn, cfg, v, endow_idx, U, wmax_w))

    if args.checkpoint:
        print(f"Loading {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        net = AllocationNet(cfg); net.load_state_dict(ckpt["state_dict"]); net.eval()
        def net_mech(cfg_, v_, ei_, U_):
            mask = ir_pe_mask(cfg, U_, ei_)
            return net(v_, mask=mask, temperature=1e-3)
        results.append(evaluate_mechanism("LearnedNet", net_mech, cfg, v, endow_idx, U, wmax_w))

    print_table(results)

if __name__ == "__main__":
    main()
