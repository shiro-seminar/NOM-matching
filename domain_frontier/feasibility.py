"""Mechanism-free unambiguous IR+PE existence check (the D_IRPE ceiling).

A marginal mechanism that is unambiguously IR and unambiguously PE on a domain D
exists IFF every profile in D admits at least one matching that is simultaneously
balanced, component-wise IR (= unambiguous IR, Prop. 1) and unambiguously PE.
So feasibility reduces to a per-profile non-emptiness test of

    fsd_ir_mask * unamb_pe_mask(feasible=balanced) * balanced_mask

(the RAW masks, before the endowment-fallback in unamb_ir_pe_mask). Finding ONE
profile with an empty set PROVES the domain is infeasible (not in D_IRPE).

Reuses the verified masks in domain_expansion_experiments/allocations.py. Profiles
are enumerated exactly for small domains and sampled for large ones.

Verified numbers (3 agents, 4 items): trichotomous / four_chotomous_e4 -> 0 empty
(D_IRPE strictly exceeds trichotomous at this small config). The o/p/p'/q
counterexample only fits once each agent holds >= 4 objects.
"""
from __future__ import annotations

import itertools
import torch

from domain_expansion_experiments.config import Config
from domain_expansion_experiments.domains import DomainSpec
from domain_expansion_experiments.allocations import build_all_allocs
from domain_expansion_experiments.full_enum import enumerate_reports


def valid_endowments(cfg: Config) -> list[int]:
    allocs = build_all_allocs(cfg)
    counts = torch.stack([(allocs == i).sum(dim=1) for i in range(cfg.num_agents)], dim=1)
    return (counts.min(dim=1).values >= 1).nonzero(as_tuple=True)[0].tolist()


def _agent_reports(cfg: Config, domain: DomainSpec, k_e: int, device: str):
    """Per-agent domain-valid reports for endowment k_e. Returns list of [Na, m]."""
    A, m = cfg.num_agents, cfg.num_items
    alloc_e = build_all_allocs(cfg).to(device)[k_e]
    return [enumerate_reports(domain, [(alloc_e[j].item() == a) for j in range(m)]).to(device)
            for a in range(A)]


def _balanced_data(cfg: Config, k_e: int, device: str):
    """Precompute the BALANCED allocations for endowment k_e (the only relevant
    matchings). Returns (owned [nbal, A, m] float, be int, nbal int).

    Restricting to balanced allocations (a small set, e.g. 12 for a (2,1,1)
    endowment) instead of all K = A^m makes the unambiguous-PE check ~1000x
    cheaper than the dense allocations.unamb_pe_mask path.
    """
    A, m = cfg.num_agents, cfg.num_items
    allocs = build_all_allocs(cfg).to(device)               # [K, m]
    counts = torch.stack([(allocs == i).sum(1) for i in range(A)], 1)   # [K, A]
    ecount = counts[k_e]
    bal_idx = (counts == ecount).all(1).nonzero(as_tuple=True)[0]       # [nbal]
    bal_allocs = allocs[bal_idx]                            # [nbal, m]
    owned = torch.stack([(bal_allocs == a).float() for a in range(A)], 1)  # [nbal, A, m]
    be = int((bal_idx == k_e).nonzero(as_tuple=True)[0].item())          # endow position
    return owned, be, bal_idx.shape[0]


def _empty_in_chunk(cfg: Config, mr_chunk: torch.Tensor, bdata) -> torch.Tensor:
    """Boolean [B]: True where NO balanced matching is simultaneously
    component-wise IR (= unambiguous IR) and unambiguously PE.

    Unambiguous PE (matching unamb_pe_mask semantics): nu improves mu iff some
    agent could be strictly better under nu AND no agent is *strictly* FOSD-worse
    (incomparable bundles allowed, since responsive extensions are per-agent).
    """
    owned, be, nbal = bdata
    B, A, m = mr_chunk.shape
    R = float(cfg.num_ranks)

    r = mr_chunk.float().unsqueeze(1)                       # [B, 1, A, m]
    own = owned.unsqueeze(0)                                # [1, nbal, A, m]
    bundle = r * own + R * (1.0 - own)                      # [B, nbal, A, m]
    sb, _ = torch.sort(bundle, dim=-1)                     # [B, nbal, A, m]

    esb = sb[:, be:be + 1, :, :]                           # [B, 1, A, m] endowment bundle
    ir = ((sb <= esb + 1e-8).all(-1)).all(-1)              # [B, nbal] component-wise IR

    x = sb.unsqueeze(2)                                     # [B, mu, 1, A, m]
    y = sb.unsqueeze(1)                                     # [B, 1, nu, A, m]
    mu_weak = (x <= y + 1e-8).all(-1)                      # [B, mu, nu, A]  mu >= nu
    nu_weak = (y <= x + 1e-8).all(-1)                      # [B, mu, nu, A]
    mu_strict = mu_weak & ~nu_weak                         # agent strictly worse under nu
    cond1 = (~mu_weak).any(-1)                             # some agent could be better
    improves = cond1 & ~mu_strict.any(-1)                 # [B, mu, nu]: nu improves mu
    eye = torch.eye(nbal, dtype=torch.bool, device=mr_chunk.device).unsqueeze(0)
    improves = improves & ~eye
    pe = ~improves.any(-1)                                 # [B, mu] unambiguously PE

    feasible = (ir & pe).any(-1)                           # [B]
    return ~feasible


@torch.no_grad()
def feasibility_enumerate(cfg: Config, domain: DomainSpec, k_e: int,
                          chunk: int = 128, device: str = "cpu") -> dict:
    """Exact: enumerate ALL domain profiles for endowment k_e, count empties."""
    reps = _agent_reports(cfg, domain, k_e, device)
    sizes = [r.shape[0] for r in reps]
    total = 1
    for s in sizes:
        total *= s
    A, m = cfg.num_agents, cfg.num_items
    # mixed-radix decode of the flat profile index into per-agent report indices
    radix = []
    acc = 1
    for s in reversed(sizes):
        radix.append(acc)
        acc *= s
    radix = list(reversed(radix))   # radix[a] = stride of agent a

    bdata = _balanced_data(cfg, k_e, device)
    n_empty = 0
    example = None
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        flat = torch.arange(start, end, device=device)
        mr = torch.zeros(end - start, A, m, dtype=torch.long, device=device)
        for a in range(A):
            idx_a = (flat // radix[a]) % sizes[a]
            mr[:, a, :] = reps[a][idx_a]
        empt = _empty_in_chunk(cfg, mr, bdata)
        if empt.any():
            n_empty += int(empt.sum())
            if example is None:
                example = mr[empt.nonzero(as_tuple=True)[0][0]].tolist()
    return {"k_e": k_e, "total": total, "n_empty": n_empty, "example": example}


@torch.no_grad()
def feasibility_sample(cfg: Config, domain: DomainSpec, k_e: int,
                       n_samples: int = 20000, chunk: int = 128,
                       seed: int = 0, device: str = "cpu") -> dict:
    """Sampling: draw n_samples random domain profiles for endowment k_e."""
    from domain_expansion_experiments.data_gen import sample_domain_marginal_rank
    torch.manual_seed(seed)
    bdata = _balanced_data(cfg, k_e, device)
    n_empty = 0
    example = None
    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        ei = torch.full((b,), k_e, dtype=torch.long, device=device)
        mr = sample_domain_marginal_rank(cfg, domain, ei, device)
        empt = _empty_in_chunk(cfg, mr, bdata)
        if empt.any():
            n_empty += int(empt.sum())
            if example is None:
                example = mr[empt.nonzero(as_tuple=True)[0][0]].tolist()
        done += b
    return {"k_e": k_e, "total": n_samples, "n_empty": n_empty, "example": example}


def endowment_shapes(cfg: Config, device: str = "cpu") -> dict:
    """Group valid endowments by sorted bundle-size shape.

    Because (eps, nu) domains and unambiguous IR+PE are symmetric in objects and
    agents, feasibility depends ONLY on the sorted size tuple (e.g. (2,2,2),
    (3,2,1), (4,1,1) for m=6). So one representative per shape is exact -- we need
    not loop over all endowments. Returns {shape_tuple: [endow_idx, ...]}.
    """
    A = cfg.num_agents
    allocs = build_all_allocs(cfg).to(device)
    counts = torch.stack([(allocs == i).sum(1) for i in range(A)], 1)   # [K, A]
    valid = (counts.min(1).values >= 1).nonzero(as_tuple=True)[0]
    shapes: dict = {}
    for k in valid.tolist():
        shape = tuple(sorted((int(c) for c in counts[k].tolist()), reverse=True))
        shapes.setdefault(shape, []).append(k)
    return shapes


@torch.no_grad()
def feasibility_by_shape(cfg: Config, domain: DomainSpec,
                         n_samples: int = 20000, chunk: int = 128,
                         device: str = "cpu", verbose: bool = True) -> dict:
    """IR+PE feasibility per endowment SHAPE (one representative each, exact by
    symmetry). Far cheaper than looping all endowments. 'feasible' means no empty
    found (sampled when full enumeration is too large -> not a proof); 'INFEASIBLE'
    means an empty profile was found (a proof)."""
    import math
    shapes = endowment_shapes(cfg, device)
    out = {}
    any_infeasible = False
    for shape in sorted(shapes, reverse=True):
        rep = shapes[shape][0]
        # nbal = multinomial(m; sizes); pick chunk so B*nbal^2*A stays ~<3e8
        nbal = math.factorial(cfg.num_items)
        for s in shape:
            nbal //= math.factorial(s)
        sh_chunk = max(1, min(chunk, int(3e8 / (max(nbal, 1) ** 2 * cfg.num_agents))))
        res = domain_feasible(cfg, domain, n_samples=n_samples, chunk=sh_chunk,
                              device=device, verbose=False, endow_list=[rep])
        out[shape] = res
        if not res["feasible"]:
            any_infeasible = True
        if verbose:
            tag = "feasible" if res["feasible"] else f"INFEASIBLE (empties={res['n_empty']})"
            print(f"    shape {str(shape):14s}: {tag}", flush=True)
    return {"domain": domain.name, "by_shape": out, "feasible_all_shapes": not any_infeasible}


@torch.no_grad()
def domain_feasible(cfg: Config, domain: DomainSpec,
                    max_enum_per_endow: int = 200_000,
                    n_samples: int = 20000, chunk: int = 128,
                    device: str = "cpu", verbose: bool = True,
                    endow_list: list[int] | None = None) -> dict:
    """Check IR+PE feasibility of `domain` over endowments.

    Uses exact enumeration when an endowment's profile count <= max_enum_per_endow,
    else sampling. Returns the first empty profile found (a witness of
    INfeasibility) if any.

    endow_list: restrict to these endowments (e.g. the single balanced (k,k,k)
    endowment for an objects-per-agent study). Defaults to all valid endowments.
    """
    endows = endow_list if endow_list is not None else valid_endowments(cfg)
    total_empty = 0
    example = None
    mode_used = "enumerate"
    for k_e in endows:
        reps = _agent_reports(cfg, domain, k_e, device)
        cnt = 1
        for r in reps:
            cnt *= r.shape[0]
        if cnt <= max_enum_per_endow:
            res = feasibility_enumerate(cfg, domain, k_e, chunk, device)
        else:
            mode_used = "sample"
            res = feasibility_sample(cfg, domain, k_e, n_samples, chunk, device=device)
        total_empty += res["n_empty"]
        if example is None and res["example"] is not None:
            example = {"k_e": k_e, "profile": res["example"]}
        if verbose:
            print(f"  endow {k_e:3d}: total={res['total']:>9d}  empty={res['n_empty']}",
                  flush=True)
    feasible = (total_empty == 0)
    return {"domain": domain.name, "feasible": feasible,
            "n_empty": total_empty, "mode": mode_used, "example_empty": example}


if __name__ == "__main__":
    from domain_expansion_experiments.domains import domain_lattice
    for dom in domain_lattice():
        cfg = Config(domain=dom.name)
        print(f"\n=== {dom.name} (n={cfg.num_agents}, m={cfg.num_items}) ===", flush=True)
        res = domain_feasible(cfg, dom, verbose=True)
        tag = "FEASIBLE (in D_IRPE)" if res["feasible"] else "INFEASIBLE"
        print(f"  -> {tag}  (empties={res['n_empty']}, mode={res['mode']})", flush=True)
