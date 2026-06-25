"""Map the MAXIMUM unambiguous IR+PE domain as objects-per-agent grows.

At m=4 (the repo config) four_chotomous is already the ordinal top (4 objects
=> at most 4 indifference classes => all weak orders), and D_IRPE reaches it.
To probe "larger than four_chotomous" we must add objects: with k objects per
agent there can be up to 3k objects and genuinely richer R-chotomous domains.

For each objects-per-agent k we fix the balanced (k,k,k) endowment and walk the
richness lattice (strongly_tri -> trichotomous -> eps(3) -> four_chotomous ->
5-chotomous -> 6-chotomous), reporting unambiguous IR+PE feasibility. The
largest feasible domain is D_IRPE(k). This is mechanism-free and decisive
(one empty profile proves infeasibility).

Theory anchor (Manjunath-Westkamp Thm 1): for k>=4, D_IRPE collapses to
trichotomous. This sweep maps the small-k boundary the theorem leaves open.

NOTE on cost: the unambiguous-PE check is O(nbal^2) where nbal = (3k)!/(k!^3)
(90 at k=2, 1680 at k=3, 34650 at k=4). k<=3 is tractable here; k=4 is covered
by Thm 1 + the 12-object reproduction and is left to an exact cycle-based check.
"""
from __future__ import annotations

import torch

from domain_expansion_experiments.config import Config
from domain_expansion_experiments.domains import DOMAINS, richness_lattice
from domain_expansion_experiments.allocations import build_all_allocs
from domain_frontier.feasibility import domain_feasible, feasibility_by_shape, endowment_shapes


def contiguous_endow_idx(num_agents: int, k: int) -> tuple[int, int]:
    """Index of the endowment where agent a owns objects [a*k, (a+1)*k).
    Returns (endow_idx, num_items)."""
    m = num_agents * k
    allocs = build_all_allocs(Config(num_agents=num_agents, num_items=m))
    target = torch.tensor([j // k for j in range(m)])           # [m]
    idx = (allocs == target.unsqueeze(0)).all(1).nonzero(as_tuple=True)[0]
    return int(idx[0]), m


def _chunk_for(nbal: int) -> int:
    """Pick a batch size so the O(B*nbal^2*A) PE tensor stays ~<5e8 elements."""
    return max(1, int(5e8 / (nbal * nbal * 3)))


def _nbal(num_agents: int, k: int) -> int:
    import math
    return math.factorial(num_agents * k) // (math.factorial(k) ** num_agents)


def sweep(ks=(2, 3), max_R: int = 6, n_samples: int = 30000,
          device: str = "cpu", verbose: bool = True) -> dict:
    A = 3
    out = {}
    for k in ks:
        endow, m = contiguous_endow_idx(A, k)
        nbal = _nbal(A, k)
        chunk = _chunk_for(nbal)
        print(f"\n=== objects/agent k={k}  (n={A}, m={m}, balanced nbal={nbal}, "
              f"chunk={chunk}) ===", flush=True)
        max_feasible = None
        rows = []
        for dom in richness_lattice(max_R):
            cfg = Config(domain=dom.name, num_agents=A, num_items=m)
            res = domain_feasible(cfg, dom, n_samples=n_samples, chunk=chunk,
                                  device=device, verbose=False, endow_list=[endow])
            rows.append((dom.name, res["feasible"], res["n_empty"]))
            if res["feasible"]:
                max_feasible = dom.name
            if verbose:
                tag = "feasible" if res["feasible"] else f"INFEASIBLE (empties={res['n_empty']})"
                print(f"  {dom.name:14s}: {tag}", flush=True)
        print(f"  -> MAX D_IRPE(k={k}) = {max_feasible}", flush=True)
        out[k] = {"endow": endow, "m": m, "max_irpe": max_feasible, "rows": rows}
    return out


def sweep_by_m(ms=(4, 5, 6, 7), max_R: int = 6, n_samples: int = 20000,
               device: str = "cpu", verbose: bool = True) -> dict:
    """For each TOTAL object count m, map IR+PE feasibility per endowment SHAPE
    (all distributions, not just balanced), across the richness lattice.

    Answers "does the maximal IR+PE domain depend on the endowment SHAPE, not just
    the total m?" By symmetry one representative per shape is exact. GPU-friendly:
    pass device='cuda'. Heavy m (>=7) and rich domains rely on sampling -> a
    'feasible' verdict is "no empty found", an 'INFEASIBLE' verdict is a proof.
    """
    A = 3
    out = {}
    for m in ms:
        print(f"\n############ total objects m={m} (n={A}) ############", flush=True)
        m_rows = {}
        for dom in richness_lattice(max_R):
            cfg = Config(domain=dom.name, num_agents=A, num_items=m)
            if verbose:
                print(f"  {dom.name}:", flush=True)
            res = feasibility_by_shape(cfg, dom, n_samples=n_samples,
                                       device=device, verbose=verbose)
            m_rows[dom.name] = res
        # per-shape max feasible domain
        all_shapes = sorted({sh for r in m_rows.values() for sh in r["by_shape"]}, reverse=True)
        print(f"  --- MAX D_IRPE(m={m}) per shape ---", flush=True)
        for sh in all_shapes:
            feas_doms = [d for d in richness_lattice(max_R)
                         if m_rows[d.name]["by_shape"].get(sh, {}).get("feasible")]
            maxd = feas_doms[-1].name if feas_doms else "(none)"
            print(f"    shape {str(sh):14s}: MAX = {maxd}", flush=True)
        out[m] = m_rows
    return out


def sp_map_by_shape(ms=(4, 6, 8), max_R: int = 6, n_profiles: int = 300,
                    device: str = "cpu", verbose: bool = True) -> dict:
    """SP shape-map: for each total m and endowment SHAPE, find the richest domain
    on which the REFERENCE mechanism (priority_mechanism) is unambiguously SP
    (sampled). SP-viol==0 -> that domain is a sound LOWER BOUND member of D_SP at
    the shape (the mechanism is an explicit SP witness). Combined with the IR+PE
    shape-map this gives D_SP(shape) <= D_NOM(shape) <= D_IRPE(shape).

    Cheap (sampling, no full-enum) -> GPU-friendly. Question: does the SP-achievable
    boundary, like IR+PE, jump up for singleton-heavy shapes (k,1,1)?
    """
    from domain_frontier.sp_test import unamb_sp_violation_rate
    from domain_expansion_experiments.benchmarks import priority_mechanism
    A = 3
    out = {}
    for m in ms:
        print(f"\n############ SP shape-map  m={m} (n={A}) ############", flush=True)
        shapes = endowment_shapes(Config(num_agents=A, num_items=m), device)
        m_out = {}
        for shape in sorted(shapes, reverse=True):
            rep = shapes[shape][0]
            sp_ok_max = None
            row = []
            for dom in richness_lattice(max_R):
                cfg = Config(domain=dom.name, num_agents=A, num_items=m)
                r = unamb_sp_violation_rate(cfg, dom, priority_mechanism,
                                            n_profiles=n_profiles, seed=0,
                                            endow_idx=rep)
                row.append((dom.name, r["sp_viol"]))
                if r["sp_viol"] == 0:
                    sp_ok_max = dom.name
            m_out[shape] = {"sp_witness_max": sp_ok_max, "row": row}
            if verbose:
                detail = "  ".join(f"{d.split('_')[0]}:{v}" for d, v in row)
                print(f"    shape {str(shape):14s}: SP-witness MAX = {sp_ok_max}"
                      f"   [{detail}]", flush=True)
        out[m] = m_out
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["balanced", "by_m", "sp_by_shape"], default="balanced",
                   help="'balanced': fixed (k,k,k); 'by_m': IR+PE per shape; "
                        "'sp_by_shape': SP-witness per shape")
    p.add_argument("--ks", type=int, nargs="+", default=[2, 3])
    p.add_argument("--ms", type=int, nargs="+", default=[4, 5, 6, 7])
    p.add_argument("--max_R", type=int, default=6)
    p.add_argument("--n_samples", type=int, default=20000)
    p.add_argument("--n_profiles", type=int, default=300)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    if args.mode == "by_m":
        sweep_by_m(ms=tuple(args.ms), max_R=args.max_R,
                   n_samples=args.n_samples, device=args.device)
    elif args.mode == "sp_by_shape":
        sp_map_by_shape(ms=tuple(args.ms), max_R=args.max_R,
                        n_profiles=args.n_profiles, device=args.device)
    else:
        sweep(ks=tuple(args.ks), max_R=args.max_R,
              n_samples=args.n_samples, device=args.device)
