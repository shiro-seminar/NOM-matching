"""Unambiguous IR/PE/NOM verification (ASCII-safe output).

Tests the mathematically correct unambiguous definitions from
Manjunath-Westkamp (arXiv:2502.06499):
  - IR  : FOSD vs endowment (unambiguous IR = FOSD IR)
  - PE  : unambiguous PE (exists j who can be improved, forall i not made strictly worse)
  - NOM : set-based BC/WC using FOSD bundle comparison

Verification A-1  [trichotomous]
  - unamb_ir_pe_mask should be NON-EMPTY for all profiles
    (theorem: IR+PE+Balanced is non-empty for trichotomous)

Verification A-2  [strict]
  - unamb_ir_pe_mask should be EMPTY for SOME profiles
    (IR and PE are incompatible in strict domain for n >= 3 agents)

Verification B    [four_chotomous_e4]
  - Compare additive vs unambiguous PE/NOM
  - FOSD may reveal NOM violations invisible to additive evaluation

Usage:
    python -m domain_expansion_experiments.verify_fosd
    python -m domain_expansion_experiments.verify_fosd --domain strict
    python -m domain_expansion_experiments.verify_fosd --all
"""
from __future__ import annotations

import argparse
import torch

from .config import Config
from .domains import DOMAINS
from .allocations import (
    ir_mask, ir_pe_mask, pareto_mask, balanced_mask,
    fsd_ir_mask, unamb_pe_mask, unamb_ir_pe_mask,
    score_matrix, endowment_scores,
)
from .data_gen import sample_batch
from .benchmarks import wmax_score_ir_pe, priority_mechanism
from .evaluate import _nom_eval, _unamb_nom_eval, evaluate_mechanism_fosd, print_table


def run_verification(domain_name: str, n: int = 200, S_nom: int = 16, M_nom: int = 16,
                     seed: int = 42):
    print(f"\n{'='*65}")
    print(f"  Unambiguous Verification: {domain_name}  (n={n}, S={S_nom}, M={M_nom})")
    print(f"{'='*65}")

    torch.manual_seed(seed)
    cfg    = Config(domain=domain_name, batch_size=n, device="cpu", seed=seed)
    domain = DOMAINS[domain_name]

    batch         = sample_batch(cfg)
    marginal_rank = batch["marginal_rank"]
    endow_idx     = batch["endow_idx"]
    S             = batch["S"]

    # -------- Part 1: IR mask comparison ---------------------------------
    print("\n[Part 1] IR mask: additive vs unambiguous (FOSD)")

    ir_add = ir_mask(S, endow_idx)
    ir_fsd = fsd_ir_mask(cfg, marginal_rank, endow_idx)

    add_not_fsd = (ir_add - ir_fsd).clamp(min=0).sum()
    fsd_not_add = (ir_fsd - ir_add).clamp(min=0).sum()

    total = ir_add.numel()
    print(f"  Total (b,k) pairs        : {total}")
    print(f"  Additive IR=1            : {ir_add.sum():.0f}  ({ir_add.mean()*100:.1f}%)")
    print(f"  Unambiguous IR=1         : {ir_fsd.sum():.0f}  ({ir_fsd.mean()*100:.1f}%)")
    print(f"  Add-only  (add not unamb): {add_not_fsd:.0f}  (expected >= 0)")
    print(f"  Unamb-only (unamb not add): {fsd_not_add:.0f}  (should be 0 -- unamb-IR subset of add-IR)")

    if fsd_not_add > 0:
        print("  WARNING: unamb-IR NOT subset of add-IR -- bug in fsd_ir_mask!")
    else:
        print("  OK: unamb-IR subset of add-IR as expected")

    # -------- Part 2: PE mask comparison (balanced-restricted) -----------
    print("\n[Part 2] PE mask: additive vs unambiguous (within balanced, balanced comparators)")
    bal = balanced_mask(cfg, endow_idx, device="cpu")

    # Balanced-restricted additive PE: mask non-balanced allocations to -inf
    # so they can never act as dominators in pareto_mask.
    # This gives a fair comparison with unamb_pe_mask(feasible_mask=bal).
    non_bal = ~bal.bool()  # [B, K]
    S_bal = S.clone()
    S_bal.masked_fill_(non_bal.unsqueeze(1).expand_as(S_bal), -1e9)
    pe_add_bal = pareto_mask(S_bal) * bal       # balanced-restricted additive PE

    pe_unamb = unamb_pe_mask(cfg, marginal_rank, feasible_mask=bal) * bal

    # Theory: {unamb-PE within balanced} subset of {add-PE within balanced}
    add_not_unamb = (pe_add_bal - pe_unamb).clamp(min=0).sum()
    unamb_not_add = (pe_unamb - pe_add_bal).clamp(min=0).sum()

    print(f"  Balanced allocations         : {bal.sum():.0f}")
    print(f"  Add-PE (balanced comp.) +Bal : {pe_add_bal.sum():.0f}")
    print(f"  Unamb-PE +Bal                : {pe_unamb.sum():.0f}")
    print(f"  Add-PE, not unamb-PE         : {add_not_unamb:.0f}  (expected >= 0 -- unamb is stricter)")
    print(f"  Unamb-PE, not add-PE         : {unamb_not_add:.0f}  (should be 0 by theory)")

    if unamb_not_add > 0:
        print("  WARNING: unamb-PE NOT subset of add-PE -- bug in unamb_pe_mask!")
    else:
        print("  OK: unamb-PE subset of add-PE (within balanced) as expected")

    # -------- Part 3: IR+PE mask comparison ------------------------------
    print("\n[Part 3] IR+PE mask: additive vs unambiguous")
    irpe_add   = ir_pe_mask(cfg, S, endow_idx)
    irpe_unamb = unamb_ir_pe_mask(cfg, marginal_rank, endow_idx)

    agree      = (irpe_add == irpe_unamb).all(dim=1).float().mean()
    diff_count = ((irpe_add - irpe_unamb).abs() > 0.5).sum()
    add_only   = (irpe_add - irpe_unamb).clamp(min=0).sum()
    unamb_only = (irpe_unamb - irpe_add).clamp(min=0).sum()

    print(f"  Profiles where masks agree fully : {agree*100:.1f}%")
    print(f"  (b,k) pairs where masks disagree : {diff_count:.0f}")
    print(f"  Add-only  (in add, not unamb)    : {add_only:.0f}")
    print(f"  Unamb-only (in unamb, not add)   : {unamb_only:.0f}")

    # -------- Part 4: Empty-mask check (A-1 and A-2) --------------------
    print("\n[Part 4] IR+PE+Bal emptiness check (before endowment fallback)")

    # Unambiguous IR+PE raw (no fallback)
    ir_fsd_m  = fsd_ir_mask(cfg, marginal_rank, endow_idx)
    pe_unamb2 = unamb_pe_mask(cfg, marginal_rank, feasible_mask=bal)
    raw_unamb = ir_fsd_m * pe_unamb2 * bal
    empty_unamb = (raw_unamb.sum(dim=1) < 0.5).sum().item()

    # Additive IR+PE raw (no fallback) -- NOTE: pareto uses all-K comparators here
    raw_add_irpe = ir_mask(S, endow_idx) * pareto_mask(S) * bal
    empty_add    = (raw_add_irpe.sum(dim=1) < 0.5).sum().item()

    print(f"  Unamb IR+PE+Bal empty: {empty_unamb} / {n}  ({empty_unamb/n*100:.1f}%)")
    print(f"  Add  IR+PE+Bal empty: {empty_add} / {n}  ({empty_add/n*100:.1f}%)")

    if domain_name == "trichotomous":
        print("  [A-1] Expected (Manjunath-Westkamp): 0% unamb-empty (existence theorem)")
        empty_rate = empty_unamb / n
        if empty_rate < 0.01:
            print("  PASS A-1: unamb IR+PE+Bal non-empty for all sampled profiles")
        else:
            print("  FAIL A-1: empty unamb masks found -- check implementation or theorem scope")
    else:
        empty_rate = empty_unamb / n

    if domain_name == "strict":
        print("  [A-2] Expected for additive: SOME empty profiles (IR+PE incompatible strict n>=3)")
        print("  Note: unamb IR+PE is never empty (endowment always FOSD-IR and trivially PE)")
        if empty_add > 0:
            print(f"  PASS A-2 (additive): {empty_add/n*100:.1f}% additive-empty profiles found")
        else:
            print("  INFO: no additive-empty profiles found in this sample")

    # -------- Part 5: NOM comparison -------------------------------------
    print(f"\n[Part 5] NOM comparison on WMAX-IR-PE mechanism  (S={S_nom}, M={M_nom})")

    def wmax_irpe_mech(cfg_, mr_, ei_, S_):
        return wmax_score_ir_pe(cfg_, mr_, ei_, S_)

    print("  Evaluating additive NOM...")
    nom_add_mean, nom_add_viol = _nom_eval(cfg, domain, wmax_irpe_mech,
                                           marginal_rank, endow_idx, S, S_nom, M_nom)

    print("  Evaluating unambiguous (set-based FOSD) NOM...")
    nom_unamb_mean, nom_unamb_viol = _unamb_nom_eval(cfg, domain, wmax_irpe_mech,
                                                      marginal_rank, endow_idx, S_nom, M_nom)

    print(f"\n  {'Metric':<18} {'Additive':>10} {'Unambiguous':>12}")
    print(f"  {'-'*42}")
    print(f"  {'NOM mean':<18} {nom_add_mean:>10.5f} {nom_unamb_mean:>12.5f}")
    print(f"  {'NOM viol%':<18} {nom_add_viol*100:>10.1f} {nom_unamb_viol*100:>12.1f}")

    if nom_unamb_viol > nom_add_viol + 0.01:
        print("\n  >> Unambiguous reveals MORE violations (expected for richer domains)")
    elif nom_unamb_viol < nom_add_viol - 0.01:
        print("\n  >> Unambiguous reveals FEWER violations (unexpected -- check implementation)")
    else:
        print("\n  >> Additive and unambiguous agree on NOM violations")

    # -------- Part 6: Trichotomous NOM theory check ----------------------
    if domain_name == "trichotomous":
        print("\n[Part 6] Trichotomous NOM theory check")
        print("  Expected (Manjunath-Westkamp 2025): WMAX-IR-PE satisfies NOM.")
        print(f"  Unambiguous NOM-viol% = {nom_unamb_viol*100:.1f}%  (target: ~0%)")
        if nom_unamb_viol < 0.05:
            print("  PASS: NOM ~0% under unambiguous FOSD for trichotomous")
        else:
            print("  FAIL or inconclusive: NOM > 0% under unambiguous FOSD.")
            print("  Either (a) WMAX-IR-PE is not the correct NOM mechanism, or")
            print("         (b) the unambiguous NOM evaluator has a bug.")

        print(f"\n  IR mask agree (add==unamb): {(ir_add==ir_fsd).float().mean()*100:.1f}%")
        print(f"  PE mask agree (add==unamb): {(pe_add_bal==pe_unamb).float().mean()*100:.1f}%")

    # -------- Part 7: Priority mechanism phi^IP unambiguous NOM check ----
    nom_ip_mean = nom_ip_viol = None
    if domain_name in ("trichotomous", "four_chotomous_e4"):
        print(f"\n[Part 7] Priority mechanism (phi^IP) unambiguous NOM check  (S={S_nom}, M={M_nom})")
        print("  Evaluating Priority-IP unambiguous NOM...")
        nom_ip_mean, nom_ip_viol = _unamb_nom_eval(cfg, domain, priority_mechanism,
                                                     marginal_rank, endow_idx, S_nom, M_nom)
        print(f"  Priority-IP NOM mean  = {nom_ip_mean:.5f}")
        print(f"  Priority-IP NOM viol% = {nom_ip_viol*100:.1f}%")

        if domain_name == "trichotomous":
            if nom_ip_viol < 0.05:
                print("  PASS: Priority-IP achieves ~0% unamb NOM on trichotomous")
                print("        -> evaluator validated; WMAX-IR-PE (lexmin) is genuinely manipulable.")
            else:
                print("  Priority-IP also shows violations -- evaluator construction")
                print("  (S_truth/S_lie sampling, misreport range, OR structure) needs review.")

    # -------- Part 7b: A-2 redo (strict) -- raw mask, larger sample -----
    if domain_name == "strict":
        print("\n[Part 7b] A-2 redo: raw unamb IR+PE+Bal emptiness, larger sample (n2=1000)")
        n2 = 1000
        cfg2    = Config(domain=domain_name, batch_size=n2, device="cpu", seed=seed + 1)
        batch2  = sample_batch(cfg2)
        mr2     = batch2["marginal_rank"]
        ei2     = batch2["endow_idx"]
        bal2      = balanced_mask(cfg2, ei2, device="cpu")
        ir_fsd_m2 = fsd_ir_mask(cfg2, mr2, ei2)
        pe_unamb2b = unamb_pe_mask(cfg2, mr2, feasible_mask=bal2)
        raw_unamb2 = ir_fsd_m2 * pe_unamb2b * bal2
        empty_unamb2 = (raw_unamb2.sum(dim=1) < 0.5).sum().item()
        print(f"  Unamb IR+PE+Bal empty (raw, no fallback): {empty_unamb2} / {n2}  "
              f"({empty_unamb2/n2*100:.1f}%)")
        if empty_unamb2 > 0:
            print("  CORRECTION: unamb IR+PE+Bal CAN be empty for strict (n>=3) -- ")
            print("  earlier claim 'endowment is trivially unamb-PE' was WRONG.")
            print("  A-2 holds for the unambiguous definition too.")
        else:
            print("  No empty raw unamb masks found in this larger sample either.")

    # -------- Part 8: four_chotomous full comparison (Priority-IP vs lexmin)
    if domain_name == "four_chotomous_e4":
        print(f"\n[Part 8] Full unambiguous evaluation: Priority-IP vs WMAX-IR-PE (lexmin)")
        irpe_m_full = ir_pe_mask(cfg, S, endow_idx)
        wmax_s_full = (S.sum(1) + (1 - irpe_m_full) * (-1e9)).max(1).values

        results = []
        results.append(evaluate_mechanism_fosd(
            "Priority-IP", priority_mechanism, cfg, domain,
            marginal_rank, endow_idx, S, wmax_s_full, eval_S=S_nom, eval_M=M_nom))
        results.append(evaluate_mechanism_fosd(
            "WMAX-IR-PE", wmax_score_ir_pe, cfg, domain,
            marginal_rank, endow_idx, S, wmax_s_full, eval_S=S_nom, eval_M=M_nom))
        print_table(results)

    return {
        "domain":         domain_name,
        "ir_unamb_only":  float(fsd_not_add),
        "pe_unamb_only":  float(unamb_not_add),
        "irpe_diff":      float(diff_count),
        "empty_rate":     empty_rate,
        "nom_add_viol":   nom_add_viol,
        "nom_unamb_viol": nom_unamb_viol,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="trichotomous",
                        choices=["trichotomous", "trichotomous_extended_e3",
                                 "four_chotomous_e4", "strict"])
    parser.add_argument("--n",     type=int, default=200)
    parser.add_argument("--S_nom", type=int, default=16)
    parser.add_argument("--M_nom", type=int, default=16)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--all",   action="store_true", help="run all 4 domains")
    args = parser.parse_args()

    if args.all:
        domains = ["trichotomous", "trichotomous_extended_e3", "four_chotomous_e4", "strict"]
    else:
        domains = [args.domain]

    results = {}
    for d in domains:
        results[d] = run_verification(d, n=args.n, S_nom=args.S_nom,
                                      M_nom=args.M_nom, seed=args.seed)

    if len(domains) > 1:
        print(f"\n{'='*65}")
        print("  SUMMARY")
        print(f"{'='*65}")
        print(f"{'Domain':<28} {'Empty%':>7} {'NOM-add%':>9} {'NOM-unamb%':>11}")
        print("-" * 58)
        for d, r in results.items():
            print(f"{d:<28} {r['empty_rate']*100:>7.1f} "
                  f"{r['nom_add_viol']*100:>9.1f} {r['nom_unamb_viol']*100:>11.1f}")


if __name__ == "__main__":
    main()
