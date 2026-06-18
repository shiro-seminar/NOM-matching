"""Unambiguous strategy-proofness tester for marginal mechanisms.

A marginal mechanism phi is *unambiguously strategy-proof* on domain D if, for
every profile, every agent i, and every misreport r'_i (opponents truthful),
agent i's truthful bundle weakly first-order-stochastically dominates the bundle
i would get by misreporting -- i.e. truth is weakly preferred under EVERY
responsive extension of i's marginal. Equivalently:

    sorted(rank of truthful bundle)  <=  sorted(rank of misreport bundle)   (elementwise)

If this fails for some (profile, i, r'_i), there is a responsive extension under
which the misreport is strictly better -> not unambiguously SP (manipulable).

This is *cheaper* than NOM (opponents are fixed at their truthful report; no
best-/worst-case set over opponent profiles). Used to map D_SP and to exhibit
D_SP subsetneq D_NOM (a domain where phi is NOM but not SP).

Verified controls (3 agents, 4 items, repo priority_mechanism):
    strongly_tri      : 0   / 400   (Thm 3: phi^IP is SP here)
    trichotomous      : 11  / 400   (Thm 4: not SP, but NOM)
    four_chotomous_e4 : 88  / 400
"""
from __future__ import annotations

import torch

from domain_expansion_experiments.config import Config
from domain_expansion_experiments.domains import DomainSpec
from domain_expansion_experiments.allocations import build_all_allocs, score_matrix, random_endowment
from domain_expansion_experiments.data_gen import sample_domain_marginal_rank
from domain_expansion_experiments.full_enum import enumerate_reports


@torch.no_grad()
def _sorted_received_bundle(cfg: Config, allocs: torch.Tensor, chosen_k: int,
                            agent_i: int, true_rank_i: torch.Tensor) -> torch.Tensor:
    """Sorted rank vector of the bundle agent i receives in allocation chosen_k,
    scored under i's TRUE marginal ranks. Unowned slots get sentinel R (sort last)."""
    R = float(cfg.num_ranks)
    owned = (allocs[chosen_k] == agent_i)                       # [m] bool
    bundle = torch.where(owned, true_rank_i.float(),
                         torch.full_like(true_rank_i.float(), R))
    return torch.sort(bundle).values                            # [m]


@torch.no_grad()
def unamb_sp_violation_rate(cfg: Config, domain: DomainSpec, mech_fn,
                            n_profiles: int = 400, seed: int = 0,
                            return_example: bool = False) -> dict:
    """Estimate the fraction of profiles with an unambiguous-SP violation.

    Samples n_profiles domain-consistent profiles. For each, scans every agent and
    every domain-valid misreport (opponents truthful) for an FOSD non-domination.

    Returns dict: {sp_viol, n, (example)}.
    """
    torch.manual_seed(seed)
    A, m = cfg.num_agents, cfg.num_items
    allocs = build_all_allocs(cfg)
    sp_viol = 0
    example = None

    for _ in range(n_profiles):
        ei = random_endowment(cfg, 1)
        k_e = int(ei[0])
        endow = allocs[k_e]
        mr = sample_domain_marginal_rank(cfg, domain, ei)[0]    # [A, m] truthful
        viol = False

        for i in range(A):
            owned = [(endow[j].item() == i) for j in range(m)]
            reps = enumerate_reports(domain, owned)              # [Ni, m] incl. truth
            true_i = mr[i]

            prof = mr.clone().unsqueeze(0)
            S = score_matrix(cfg, prof)
            ti = int(mech_fn(cfg, prof, ei, S).argmax(1)[0])
            t_sorted = _sorted_received_bundle(cfg, allocs, ti, i, true_i)

            for r in reps:
                if torch.equal(r, true_i):
                    continue
                prof2 = mr.clone()
                prof2[i] = r
                prof2 = prof2.unsqueeze(0)
                S2 = score_matrix(cfg, prof2)
                li = int(mech_fn(cfg, prof2, ei, S2).argmax(1)[0])
                l_sorted = _sorted_received_bundle(cfg, allocs, li, i, true_i)
                # unambiguous-SP violation: truth does NOT FOSD-weakly dominate lie
                if not bool((t_sorted <= l_sorted + 1e-9).all()):
                    viol = True
                    if example is None:
                        example = {"endow_idx": k_e, "agent": i,
                                   "marginal_rank": mr.tolist(),
                                   "misreport": r.tolist()}
                    break
            if viol:
                break
        if viol:
            sp_viol += 1

    out = {"sp_viol": sp_viol, "n": n_profiles,
           "sp_viol_rate": sp_viol / n_profiles}
    if return_example:
        out["example"] = example
    return out


if __name__ == "__main__":
    from domain_expansion_experiments.domains import DOMAINS
    from domain_expansion_experiments.benchmarks import priority_mechanism

    for dname in ["strongly_tri", "trichotomous", "four_chotomous_e4"]:
        cfg = Config(domain=dname)
        res = unamb_sp_violation_rate(cfg, DOMAINS[dname], priority_mechanism,
                                      n_profiles=400, seed=0)
        print(f"{dname:18s}: unambiguous-SP violating profiles = "
              f"{res['sp_viol']}/{res['n']}", flush=True)
