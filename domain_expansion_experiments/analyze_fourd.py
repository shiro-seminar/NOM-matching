"""Analyze fourd_nom_results.json produced by run_fourd_gpu.py.

Usage (in repo root):
    python -m domain_expansion_experiments.analyze_fourd [path/to/fourd_nom_results.json]

Prints:
  - Per-mechanism: total violation cells, bc/wc breakdown
  - For Priority-IP: which endowments/agents violate (if any)
  - For WMAX-IR-PE: pattern (which agent per endowment violates)
  - Conclusion: (a) NOM∩IR∩PE  (b) IR+PE but NOM violated  (c) IR+PE fails
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from .allocations import build_all_allocs
from .config import Config


def analyze(path: str = "fourd_nom_results.json") -> None:
    with open(path) as f:
        data = json.load(f)

    cfg = Config(domain="four_chotomous_e4")
    allocs = build_all_allocs(cfg)
    A = cfg.num_agents

    for mech_name, records in data.items():
        total = len(records)
        viol = [r for r in records if r["viol_rate"] > 0]
        bc   = [r for r in records if r["bc_fire_rate"] > 0]
        wc   = [r for r in records if r["wc_fire_rate"] > 0]

        print(f"\n=== {mech_name} ===")
        print(f"  cells: {total}  viol: {len(viol)}/{total}  "
              f"bc: {len(bc)}  wc: {len(wc)}")

        if not viol:
            print("  No violations. NOM satisfied for all (endowment, agent) cells.")
            continue

        # Group by endowment
        by_endow: dict[int, list[dict]] = {}
        for r in viol:
            by_endow.setdefault(r["k_e"], []).append(r)

        endows_one_agent   = sum(1 for v in by_endow.values() if len(v) == 1)
        endows_multi_agent = sum(1 for v in by_endow.values() if len(v) > 1)
        print(f"  endowments with ≥1 viol: {len(by_endow)}  "
              f"(1-agent: {endows_one_agent}  multi-agent: {endows_multi_agent})")

        # Which agent violates per endowment (bundle size analysis)
        for k_e, cells in sorted(by_endow.items()):
            agent_counts = [(allocs[k_e] == i).sum().item() for i in range(A)]
            max_count = max(agent_counts)
            viol_agents = [c["agent_i"] for c in cells]
            largest = [i for i, cnt in enumerate(agent_counts) if cnt == max_count]
            is_largest = all(a in largest for a in viol_agents)
            tag = "largest" if is_largest else "NOT-largest"
            print(f"  k_e={k_e:3d}  allocs={cells[0]['allocs']}  "
                  f"viol_agents={viol_agents}  bundle_counts={agent_counts}  ({tag})")

    # ---------- conclusion ----------
    print("\n=== CONCLUSION ===")
    ip_records  = data.get("Priority-IP", [])
    wmax_records = data.get("WMAX-IR-PE", [])

    ip_viol   = sum(1 for r in ip_records  if r["viol_rate"] > 0)
    wmax_viol = sum(1 for r in wmax_records if r["viol_rate"] > 0)
    ip_total   = len(ip_records)
    wmax_total = len(wmax_records)

    # (c) IR+PE fails: check wmax viol==0 (if wmax has no viol, IR+PE themselves fail)
    if wmax_total > 0 and wmax_viol == 0:
        print("  (c) WMAX-IR-PE has 0 violations → IR+PE criteria may be too strict / always empty.")
    elif ip_viol == 0:
        print(f"  (a) Priority-IP: NOM satisfied for all {ip_total} cells.  "
              f"WMAX-IR-PE (control): {wmax_viol}/{wmax_total} violations as expected.")
        print("      => four_chotomous_e4: phi^IP satisfies NOM ∩ unambiguous-IR ∩ unambiguous-PE")
    else:
        ip_bc = sum(1 for r in ip_records if r["bc_fire_rate"] > 0)
        ip_wc = sum(1 for r in ip_records if r["wc_fire_rate"] > 0)
        print(f"  (b) Priority-IP: NOM VIOLATED in {ip_viol}/{ip_total} cells "
              f"(bc={ip_bc}, wc={ip_wc}).")
        print("      => phi^IP does NOT satisfy NOM for four_chotomous_e4.")
        print("         Consider phi^IP M^t recursion correction (paper §5).")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "fourd_nom_results.json"
    analyze(path)
