"""Quick validation + timing of the (endowment, agent) table-based NOM evaluator."""
from __future__ import annotations

import time
import torch

from .config import Config
from .domains import DOMAINS
from .allocations import build_all_allocs, random_endowment
from .benchmarks import BENCHMARKS
from .full_enum_v2 import build_mask_table, eval_all_true_prefs


def run_one(cfg: Config, domain_name: str, mech_name: str, k_e: int, agent_i: int,
            chunk: int = 16384):
    domain = DOMAINS[domain_name]
    mech_fn = BENCHMARKS[mech_name]
    t0 = time.time()
    reports_i, mask_codes, Ni, P = build_mask_table(cfg, domain, mech_fn, k_e, agent_i, chunk)
    t1 = time.time()
    stats = eval_all_true_prefs(cfg, reports_i, mask_codes)
    t2 = time.time()
    return {
        "Ni": Ni, "P": P, "NixP": Ni * P,
        "table_time": t1 - t0, "eval_time": t2 - t1,
        **stats,
    }


if __name__ == "__main__":
    # --- Part A: trichotomous, one (endowment, agent) cell, both mechanisms ---
    cfg_tri = Config(domain="trichotomous")
    allocs = build_all_allocs(cfg_tri)
    endow_idx = random_endowment(cfg_tri, 1)
    k_e = int(endow_idx[0].item())
    print(f"[trichotomous] endowment k_e={k_e}, allocs={allocs[k_e].tolist()}")

    for mech_name in ["Priority-IP", "WMAX-IR-PE"]:
        for agent_i in range(cfg_tri.num_agents):
            r = run_one(cfg_tri, "trichotomous", mech_name, k_e, agent_i)
            print(f"  {mech_name} agent={agent_i}: Ni={r['Ni']} P={r['P']} NixP={r['NixP']} "
                  f"table_time={r['table_time']:.3f}s eval_time={r['eval_time']:.3f}s "
                  f"viol={r['viol_rate']*100:.1f}% bc={r['bc_fire_rate']*100:.1f}% "
                  f"wc={r['wc_fire_rate']*100:.1f}%")

    # --- Part B: four_chotomous_e4, ONE (endowment, agent) cell, both mechanisms ---
    cfg4 = Config(domain="four_chotomous_e4")
    allocs4 = build_all_allocs(cfg4)
    endow_idx4 = random_endowment(cfg4, 1)
    k_e4 = int(endow_idx4[0].item())
    print(f"[four_chotomous_e4] endowment k_e={k_e4}, allocs={allocs4[k_e4].tolist()}")

    for mech_name in ["Priority-IP", "WMAX-IR-PE"]:
        agent_i = 0
        r = run_one(cfg4, "four_chotomous_e4", mech_name, k_e4, agent_i, chunk=16384)
        print(f"  {mech_name} agent={agent_i}: Ni={r['Ni']} P={r['P']} NixP={r['NixP']} "
              f"table_time={r['table_time']:.3f}s eval_time={r['eval_time']:.3f}s "
              f"viol={r['viol_rate']*100:.1f}% bc={r['bc_fire_rate']*100:.1f}% "
              f"wc={r['wc_fire_rate']*100:.1f}%")
