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
                            return_example: bool = False,
                            endow_idx: int | None = None) -> dict:
    """Estimate the fraction of profiles with an unambiguous-SP violation.

    Samples n_profiles domain-consistent profiles. For each, scans every agent and
    every domain-valid misreport (opponents truthful) for an FOSD non-domination.

    endow_idx: if given, fix the endowment (e.g. a shape representative) instead of
    sampling a random one -- used by the SP shape-map. A 0 violation rate means the
    reference mechanism IS unambiguously SP there -> that domain is a sound LOWER
    BOUND member of D_SP at this shape.

    Returns dict: {sp_viol, n, (example)}.
    """
    torch.manual_seed(seed)
    A, m = cfg.num_agents, cfg.num_items
    allocs = build_all_allocs(cfg)
    sp_viol = 0
    example = None

    for _ in range(n_profiles):
        if endow_idx is None:
            ei = random_endowment(cfg, 1)
        else:
            ei = torch.tensor([endow_idx], dtype=torch.long)
        k_e = int(ei[0])
        endow = allocs[k_e]
        mr = sample_domain_marginal_rank(cfg, domain, ei)[0]    # [A, m] truthful
        viol = False

        dev = mr.device
        R = float(cfg.num_ranks)
        for i in range(A):
            owned = [(endow[j].item() == i) for j in range(m)]
            reps = enumerate_reports(domain, owned).to(dev)      # [Ni, m] incl. truth
            Ni = reps.shape[0]
            true_i = mr[i]

            # All Ni profiles at once: opponents truthful (=mr), agent i -> each report.
            batch = mr.unsqueeze(0).repeat(Ni, 1, 1)             # [Ni, A, m]
            batch[:, i, :] = reps
            ei_b = torch.full((Ni,), k_e, dtype=torch.long, device=dev)
            chosen = mech_fn(cfg, batch, ei_b, score_matrix(cfg, batch)).argmax(1)  # [Ni]

            got = (allocs[chosen] == i)                          # [Ni, m] items i receives
            bundle = torch.where(got, true_i.float().unsqueeze(0),
                                 torch.full((Ni, m), R, device=dev))
            l_sorted, _ = torch.sort(bundle, dim=1)              # [Ni, m] under TRUE ranks
            truth_mask = (reps == true_i.unsqueeze(0)).all(1)    # [Ni]
            t_sorted = l_sorted[truth_mask][0]                   # [m] truthful bundle
            # SP violation: a misreport whose bundle is NOT FOSD-weakly dominated by truth
            dominated = (t_sorted.unsqueeze(0) <= l_sorted + 1e-9).all(1)   # [Ni]
            viol_mask = (~dominated) & (~truth_mask)
            if bool(viol_mask.any()):
                viol = True
                if example is None:
                    j = int(viol_mask.nonzero(as_tuple=True)[0][0])
                    example = {"endow_idx": k_e, "agent": i,
                               "marginal_rank": mr.tolist(),
                               "misreport": reps[j].tolist()}
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
