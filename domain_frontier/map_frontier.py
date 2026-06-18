"""Driver: map D_SP subset D_NOM subset D_IRPE over a domain lattice.

For each candidate (eps, nu) domain at the repo's config (n=3, m=4) it reports:

  IRPE   : is the domain unambiguously IR+PE feasible? (mechanism-free, exact/sampled)
  NOM    : does a trained, FOSD-verified witness mechanism exist? (optional, --nom)
  SP-viol: unambiguous-SP violation rate of a reference mechanism (priority_mechanism)

D_IRPE is mechanism-free and decisive. NOM uses a search witness (sound for
membership; absence is not proof). SP-viol>0 on a domain that is NOM-feasible
demonstrates D_SP subsetneq D_NOM.

Usage:
    python -m domain_frontier.map_frontier                 # feasibility + SP
    python -m domain_frontier.map_frontier --nom --steps 2000 --device cuda
"""
from __future__ import annotations

import argparse
import json
import time

from domain_expansion_experiments.config import Config
from domain_expansion_experiments.domains import DOMAINS, domain_lattice
from domain_expansion_experiments.benchmarks import priority_mechanism

from domain_frontier.feasibility import domain_feasible
from domain_frontier.sp_test import unamb_sp_violation_rate


def run(args) -> dict:
    rows = []
    for dom in domain_lattice():
        cfg = Config(domain=dom.name, steps=args.steps, S=args.S, M=args.M,
                     batch_size=args.batch, device=args.device)
        print(f"\n=== {dom.name} ===", flush=True)

        t = time.time()
        feas = domain_feasible(cfg, dom, verbose=False, device=args.device)
        print(f"  IRPE feasible : {feas['feasible']}  (empties={feas['n_empty']}, "
              f"mode={feas['mode']}, {time.time()-t:.1f}s)", flush=True)

        t = time.time()
        sp = unamb_sp_violation_rate(cfg, dom, priority_mechanism,
                                     n_profiles=args.sp_n, seed=0)
        print(f"  SP-viol (ref) : {sp['sp_viol']}/{sp['n']}  ({time.time()-t:.1f}s)", flush=True)

        nom = None
        if args.nom and feas["feasible"]:
            from domain_frontier.search_nom import train_witness, verify_nom_fullenum
            t = time.time()
            print(f"  [NOM witness] training {cfg.steps} steps ...", flush=True)
            net = train_witness(cfg, verbose=False)
            net.eval()
            nom = verify_nom_fullenum(cfg, dom, net, chunk=args.verify_chunk,
                                      device=args.device, verbose=False)
            print(f"  NOM witness   : {'FOUND' if nom['witness'] else 'not found'}  "
                  f"(viol cells {nom['nom_viol_cells']}/{nom['total_cells']}, "
                  f"{time.time()-t:.0f}s)", flush=True)

        rows.append({
            "domain": dom.name,
            "irpe_feasible": feas["feasible"],
            "irpe_empties": feas["n_empty"],
            "sp_viol": sp["sp_viol"], "sp_n": sp["n"],
            "nom_witness": (nom["witness"] if nom else None),
            "nom_viol_cells": (nom["nom_viol_cells"] if nom else None),
        })

    print("\n" + "=" * 72)
    print(f"{'domain':26s} {'IRPE':>6} {'NOM-witness':>12} {'SP-viol':>10}")
    print("-" * 72)
    for r in rows:
        nomw = "-" if r["nom_witness"] is None else ("yes" if r["nom_witness"] else "NO")
        print(f"{r['domain']:26s} {str(r['irpe_feasible']):>6} {nomw:>12} "
              f"{r['sp_viol']:>4}/{r['sp_n']:<5}")
    print("=" * 72)
    print("Reading: D_IRPE = {domains with IRPE=True}; D_NOM = {NOM-witness=yes};")
    print("D_SP separation shown where a NOM-feasible domain has SP-viol > 0.")
    return {"config": {"n": 3, "m": 4}, "rows": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nom", action="store_true", help="also search+verify a NOM witness (slow)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--sp_n", type=int, default=400)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--verify_chunk", type=int, default=65536)
    p.add_argument("--out", type=str, default="frontier_map.json")
    args = p.parse_args()

    result = run(args)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
