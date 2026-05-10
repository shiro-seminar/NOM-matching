"""Run domain expansion experiments for all 4 domains.

Usage:
    python -m domain_expansion_experiments.run_experiments
    python -m domain_expansion_experiments.run_experiments --steps 1000 --n_eval 200

Output: results saved to domain_expansion_results.json
"""
from __future__ import annotations

import argparse
import json
import time
import torch

from .config import Config
from .domains import DOMAINS
from .allocations import ir_pe_mask, score_matrix
from .data_gen import sample_batch
from .model import AllocationNet
from .train import train
from .evaluate import (
    evaluate_mechanism, record_violations,
    print_table, print_violations,
)
from .benchmarks import BENCHMARKS

DOMAIN_ORDER = ["trichotomous", "trichotomous_extended_e3", "four_chotomous_e4", "strict"]


def run_one_domain(domain_name: str, args) -> dict:
    print(f"\n{'='*60}")
    print(f"  Domain: {domain_name}")
    print(f"{'='*60}")

    cfg = Config(
        domain=domain_name,
        steps=args.steps,
        S=args.S,
        M=args.M,
        batch_size=args.batch,
        device=args.device,
        seed=args.seed,
    )
    eval_S = getattr(args, "eval_S", None) or args.S * 4
    eval_M = getattr(args, "eval_M", None) or args.M * 4
    print(f"  num_ranks={cfg.num_ranks}  steps={cfg.steps}  "
          f"train S={cfg.S} M={cfg.M}  eval S={eval_S} M={eval_M}")

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\n[Training]")
    t0  = time.time()
    net = train(cfg, verbose=True)
    net.eval()
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s")

    # ── Evaluation batch ─────────────────────────────────────────────────
    torch.manual_seed(args.seed + 1)
    cfg_eval = Config(domain=domain_name, batch_size=args.n_eval,
                      S=args.S, M=args.M, device=args.device, seed=args.seed + 1)
    batch         = sample_batch(cfg_eval)
    marginal_rank = batch["marginal_rank"]
    endow_idx     = batch["endow_idx"]
    S             = batch["S"]
    domain        = DOMAINS[domain_name]

    irpe_m = ir_pe_mask(cfg_eval, S, endow_idx)
    wmax_s = (S.sum(1) + (1 - irpe_m) * (-1e9)).max(1).values

    # ── Benchmark evaluation ──────────────────────────────────────────────
    print(f"\n[Benchmarks]  (eval_S={eval_S}, eval_M={eval_M}, n={args.n_eval})")
    results = []
    for bname, bfn in BENCHMARKS.items():
        r = evaluate_mechanism(bname, bfn, cfg_eval, domain,
                               marginal_rank, endow_idx, S, wmax_s,
                               eval_S=eval_S, eval_M=eval_M)
        results.append(r)

    # ── Learned net evaluation ────────────────────────────────────────────
    def net_mech(cfg_, mr_, ei_, S_):
        mask_ = ir_pe_mask(cfg_, S_, ei_)
        return net(mr_, mask=mask_, temperature=1e-3)

    r_net = evaluate_mechanism("LearnedNet", net_mech, cfg_eval, domain,
                               marginal_rank, endow_idx, S, wmax_s,
                               eval_S=eval_S, eval_M=eval_M)
    results.append(r_net)

    print_table(results)

    # ── Violation structure ───────────────────────────────────────────────
    print(f"[Violation structure (n=min(100,{args.n_eval}))]")
    viol_batch = {k: v[:100] for k, v in batch.items()}
    viol_records = record_violations(
        net, cfg_eval, domain,
        viol_batch["marginal_rank"],
        viol_batch["endow_idx"],
        viol_batch["S"],
        max_records=10,
    )
    print_violations(viol_records, max_show=3)

    return {
        "domain":    domain_name,
        "num_ranks": cfg.num_ranks,
        "results":   results,
        "violations": {
            "ir":  len(viol_records["ir"]),
            "nom": len(viol_records["nom"]),
        },
        "violation_details": viol_records,
        "training_time_s": elapsed,
    }


def compare_with_baseline(domain_results: dict, args):
    """Compare trichotomous results with nom_ir_pe_ordinal baseline if available."""
    import os
    ckpt_path = "ordinal_net.pt"
    if not os.path.exists(ckpt_path):
        print("\n[Baseline comparison] ordinal_net.pt not found, skipping.")
        return

    print("\n[Baseline comparison: nom_ir_pe_ordinal vs domain_expansion trichotomous]")

    # Load baseline
    import sys
    sys.path.insert(0, ".")
    try:
        from nom_ir_pe_ordinal.config import Config as BaseConfig
        from nom_ir_pe_ordinal.model import AllocationNet as BaseNet
        from nom_ir_pe_ordinal.allocations import ir_pe_mask as base_irpe
        from nom_ir_pe_ordinal.data_gen import sample_batch as base_sample

        base_cfg = BaseConfig()
        base_cfg.batch_size = args.n_eval
        torch.manual_seed(args.seed + 99)
        base_batch = base_sample(base_cfg)

        ckpt     = torch.load(ckpt_path, map_location="cpu")
        base_net = BaseNet(base_cfg)
        base_net.load_state_dict(ckpt["state_dict"])
        base_net.eval()

        def base_mech(cfg_, mr_, ei_, S_):
            mask_ = base_irpe(cfg_, S_, ei_)
            return base_net(mr_, mask=mask_, temperature=1e-3)

        irpe_m  = base_irpe(base_cfg, base_batch["S"], base_batch["endow_idx"])
        wmax_s  = (base_batch["S"].sum(1) + (1 - irpe_m) * (-1e9)).max(1).values

        from nom_ir_pe_ordinal.evaluate import evaluate_mechanism as base_eval
        r_base = base_eval("BaseLine(ordinal)", base_mech, base_cfg,
                           base_batch["marginal_rank"], base_batch["endow_idx"],
                           base_batch["S"], wmax_s, is_nn=True)

        trich_net_r = next((r for r in domain_results.get("trichotomous", {}).get("results", [])
                            if r["name"] == "LearnedNet"), None)

        print(f"  {'Metric':<15} {'BaseLine':>12} {'DomainExp':>12}")
        print(f"  {'-'*39}")
        if trich_net_r:
            for key in ("welfare", "ir_viol", "pe_rate", "irpe_rate", "nom_mean", "nom_viol"):
                bv = r_base.get(key, float("nan"))
                dv = trich_net_r.get(key, float("nan"))
                print(f"  {key:<15} {bv:>12.4f} {dv:>12.4f}")
    except Exception as e:
        print(f"  Baseline comparison failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int, default=1000)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--S",      type=int, default=4,  help="training NOM opponent samples")
    parser.add_argument("--M",      type=int, default=4,  help="training NOM misreport samples")
    parser.add_argument("--eval_S", type=int, default=None, help="eval NOM opponents (default: S*4)")
    parser.add_argument("--eval_M", type=int, default=None, help="eval NOM misreports (default: M*4)")
    parser.add_argument("--batch",  type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--out",    type=str, default="domain_expansion_results.json")
    parser.add_argument("--domains", type=str, nargs="+", default=DOMAIN_ORDER)
    args = parser.parse_args()

    all_results = {}

    for dname in args.domains:
        result = run_one_domain(dname, args)
        all_results[dname] = result

    # Compare trichotomous with baseline
    compare_with_baseline(all_results, args)

    # Save results
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print(f"\n[Saved] {args.out}")

    # Final summary table
    print("\n" + "="*60)
    print("  SUMMARY ACROSS DOMAINS")
    print("="*60)
    hdr = f"{'Domain':<28} {'W/WMAX':>7} {'IR-viol%':>9} {'PE%':>7} {'NOM-viol%':>10}"
    print(hdr)
    print("-" * len(hdr))
    for dname, dr in all_results.items():
        net_r = next((r for r in dr["results"] if r["name"] == "LearnedNet"), None)
        if net_r:
            print(f"{dname:<28} {net_r['welfare_ratio']:>7.3f} "
                  f"{net_r['ir_viol']*100:>9.1f} {net_r['pe_rate']*100:>7.1f} "
                  f"{net_r['nom_viol']*100:>10.1f}")


if __name__ == "__main__":
    main()
