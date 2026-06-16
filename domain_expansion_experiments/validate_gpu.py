"""GPU equivalence validation before the four_chotomous_e4 full run.

Confirms that the vectorised full_enum_v2 evaluator gives identical results to
the verified CPU reference on trichotomous, then runs the negative control.

Usage on Colab:
    !python -m domain_expansion_experiments.validate_gpu

Expected output (must match before proceeding to run_fourd_gpu.py):
  [CHECK 1] Priority-IP  : 0/108 viol, bc=0, wc=0    PASS
  [CHECK 2] WMAX-IR-PE   : 36/108 viol, bc=0, wc=36  PASS
  [CHECK 3] WMAX pattern : each endowment 1 agent, all smallest-bundle agent  PASS
  [CHECK 4] Negative ctrl: unamb IR+PE mask empty (strict, Example 1)  PASS
  All checks PASS -- GPU evaluator verified, safe to run run_fourd_gpu.py
"""
from __future__ import annotations

import sys
import torch

from .config import Config
from .domains import DOMAINS
from .allocations import (
    build_all_allocs, balanced_mask, fsd_ir_mask, unamb_pe_mask,
)
from .benchmarks import BENCHMARKS
from .full_enum_v2 import build_mask_table, eval_all_true_prefs


def valid_endowments(cfg: Config) -> list[int]:
    allocs = build_all_allocs(cfg)
    counts = torch.stack([(allocs == i).sum(dim=1) for i in range(cfg.num_agents)], dim=1)
    return (counts.min(dim=1).values >= 1).nonzero(as_tuple=True)[0].tolist()


def run_trichotomous_checks(device: str) -> bool:
    cfg    = Config(domain="trichotomous")
    allocs = build_all_allocs(cfg)
    domain = DOMAINS["trichotomous"]
    endows = valid_endowments(cfg)
    A      = cfg.num_agents
    total  = len(endows) * A
    ok     = True

    for mech_name, expected_viol, expect_bc_zero in [
        ("Priority-IP",  0,  True),
        ("WMAX-IR-PE",  36,  True),
    ]:
        mech_fn = BENCHMARKS[mech_name]
        viol_cells = 0
        bc_cells   = 0
        wc_cells   = 0
        viol_agents_per_endow: dict[int, list[int]] = {}

        for k_e in endows:
            for agent_i in range(A):
                reports_i, mask_codes, Ni, P = build_mask_table(
                    cfg, domain, mech_fn, k_e, agent_i, device=device)
                stats = eval_all_true_prefs(cfg, reports_i, mask_codes)
                if stats["viol_rate"] > 0:
                    viol_cells += 1
                    if stats["bc_fire_rate"] > 0:
                        bc_cells += 1
                    if stats["wc_fire_rate"] > 0:
                        wc_cells += 1
                    viol_agents_per_endow.setdefault(k_e, []).append(agent_i)

        pass1 = (viol_cells == expected_viol)
        pass2 = (bc_cells == 0) if expect_bc_zero else True
        tag_bc = f"bc={bc_cells}"

        if mech_name == "Priority-IP":
            status = "PASS" if pass1 and pass2 else "FAIL"
            print(f"  [CHECK 1] Priority-IP  : {viol_cells}/{total} viol, "
                  f"{tag_bc}, wc={wc_cells}    {status}", flush=True)
            ok &= (status == "PASS")

        else:  # WMAX-IR-PE
            status12 = "PASS" if pass1 and pass2 else "FAIL"
            print(f"  [CHECK 2] WMAX-IR-PE   : {viol_cells}/{total} viol, "
                  f"{tag_bc}, wc={wc_cells}    {status12}", flush=True)
            ok &= (status12 == "PASS")

            # Check 3: each violating endowment has exactly 1 agent, and it is
            # the agent with the LARGEST bundle (most owned items).
            # (In trichotomous (2,1,1) distribution the 2-item agent violates.)
            pat_ok = True
            for k_e, agents in viol_agents_per_endow.items():
                if len(agents) != 1:
                    pat_ok = False
                    break
                agent_counts = [(allocs[k_e] == i).sum().item() for i in range(A)]
                max_count = max(agent_counts)
                max_agents = [i for i, c in enumerate(agent_counts) if c == max_count]
                if agents[0] not in max_agents:
                    pat_ok = False
                    break
            endows_with_viol = len(viol_agents_per_endow)
            status3 = "PASS" if pat_ok and endows_with_viol == len(endows) else "FAIL"
            print(f"  [CHECK 3] WMAX pattern : {endows_with_viol}/{len(endows)} endows "
                  f"1-agent-each largest-bundle    {status3}", flush=True)
            ok &= (status3 == "PASS")

    return ok


def run_negative_control() -> bool:
    """Example 1 from Manjunath-Westkamp (2025): strict domain, A=2, m=4.
    marginal_rank = [[0,3,1,2],[0,3,1,2]], endow = items [0,0,1,1] -> alloc index 12.
    unamb IR ∩ PE ∩ balanced mask must be empty (falls back to endowment-only).
    """
    cfg = Config(domain="strict", num_agents=2, num_items=4)
    # allocs[12] for A=2,m=4: k=12 -> binary: item0=0,item1=0,item2=1,item3=1
    allocs = build_all_allocs(cfg)
    endow_idx = torch.tensor([12])
    mr = torch.tensor([[[0, 3, 1, 2], [0, 3, 1, 2]]])   # [1, 2, 4]

    bal      = balanced_mask(cfg, endow_idx, "cpu")        # [1, K]
    ir_mask  = fsd_ir_mask(cfg, mr, endow_idx)             # [1, K]
    pe_mask  = unamb_pe_mask(cfg, mr, feasible_mask=bal)   # [1, K]
    raw      = (bal * ir_mask * pe_mask)
    n_feasible = (raw > 0.5).sum().item()

    status = "PASS" if n_feasible == 0 else f"FAIL (n={n_feasible})"
    print(f"  [CHECK 4] Negative ctrl: IR+PE+balanced mask size = {n_feasible}    {status}",
          flush=True)
    return n_feasible == 0


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("\n--- trichotomous equivalence checks ---", flush=True)
    ok1 = run_trichotomous_checks(device)
    print("\n--- negative control (strict, Example 1) ---", flush=True)
    ok2 = run_negative_control()

    print()
    if ok1 and ok2:
        print("All checks PASS -- evaluator verified, safe to run run_fourd_gpu.py",
              flush=True)
    else:
        print("FAILED -- do NOT run run_fourd_gpu.py until issues are resolved",
              flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
