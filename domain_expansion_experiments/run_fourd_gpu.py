"""Full-enumeration unambiguous NOM evaluation for four_chotomous_e4.

Run this on Colab GPU:
  !git clone https://github.com/shiro-seminar/NOM-matching
  %cd NOM-matching
  !python -m domain_expansion_experiments.run_fourd_gpu

Detects CUDA automatically; falls back to CPU.
Saves results to fourd_nom_results.json after completion.
"""
from __future__ import annotations

import json
import time
import torch

from .config import Config
from .domains import DOMAINS
from .allocations import build_all_allocs
from .benchmarks import BENCHMARKS
from .full_enum_v2 import build_mask_table, eval_all_true_prefs


def valid_endowments(cfg: Config) -> list[int]:
    allocs = build_all_allocs(cfg)
    counts = torch.stack([(allocs == i).sum(dim=1) for i in range(cfg.num_agents)], dim=1)
    valid = (counts.min(dim=1).values >= 1).nonzero(as_tuple=True)[0]
    return valid.tolist()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        # Larger chunks exploit GPU parallelism better.
        # WMAX-IR-PE needs smaller chunks (pareto_mask is O(B*K^2)).
        chunk_ip   = 131072
        chunk_wmax = 32768
    else:
        chunk_ip   = 16384
        chunk_wmax = 16384

    cfg = Config(domain="four_chotomous_e4")
    allocs = build_all_allocs(cfg)
    domain = DOMAINS["four_chotomous_e4"]
    endow_list = valid_endowments(cfg)
    print(f"domain: {cfg.domain}  endowments: {len(endow_list)}  agents: {cfg.num_agents}", flush=True)

    mechanisms = [
        ("Priority-IP",  BENCHMARKS["Priority-IP"],  chunk_ip),
        ("WMAX-IR-PE",   BENCHMARKS["WMAX-IR-PE"],   chunk_wmax),
    ]

    all_results = {}
    for mech_name, mech_fn, chunk in mechanisms:
        print(f"\n=== {mech_name} ===", flush=True)
        total = len(endow_list) * cfg.num_agents
        done  = 0
        t_start = time.time()
        mech_results = {}

        for k_e in endow_list:
            for agent_i in range(cfg.num_agents):
                t0 = time.time()
                reports_i, mask_codes, Ni, P = build_mask_table(
                    cfg, domain, mech_fn, k_e, agent_i,
                    chunk=chunk, device=device)
                t1 = time.time()
                stats = eval_all_true_prefs(cfg, reports_i, mask_codes)
                t2 = time.time()

                done += 1
                elapsed = time.time() - t_start
                eta = elapsed / done * (total - done)
                v = stats["viol_rate"]
                print(f"[{done}/{total}] k_e={k_e} agent={agent_i}  allocs={allocs[k_e].tolist()}  "
                      f"viol={v*100:.2f}%  bc={stats['bc_fire_rate']*100:.2f}%  "
                      f"wc={stats['wc_fire_rate']*100:.2f}%  "
                      f"table={t1-t0:.1f}s  eval={t2-t1:.3f}s  ETA={eta/60:.1f}min",
                      flush=True)
                mech_results[(k_e, agent_i)] = {
                    "viol_rate":    stats["viol_rate"],
                    "bc_fire_rate": stats["bc_fire_rate"],
                    "wc_fire_rate": stats["wc_fire_rate"],
                    "k_e": k_e,
                    "agent_i": agent_i,
                    "allocs": allocs[k_e].tolist(),
                }

        nom_cells = sum(1 for v in mech_results.values() if v["viol_rate"] > 0)
        print(f"{mech_name}: NOM-violation cells = {nom_cells}/{total}", flush=True)
        all_results[mech_name] = [
            {**v, "k_e": k, "agent_i": a}
            for (k, a), v in mech_results.items()
        ]

    out_path = "fourd_nom_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
