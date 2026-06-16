"""Full-enumeration unambiguous NOM evaluation (no sampling).

For each agent i and profile b, enumerate ALL domain-valid opponent report
profiles and ALL domain-valid misreports of agent i, compute the resulting
S_truth / S_lie sets of agent i's bundles (sorted under i's TRUE ranks,
regardless of which report the mechanism was queried with), and apply the
set-based FOSD BC/WC criterion:

  BC violation: exists a lie bundle not FOSD-dominated by ANY truth bundle
  WC violation: exists a truth bundle that does not FOSD-dominate ANY lie bundle
  NOM violation (for misreport r') = BC viol OR WC viol
  Profile violation = exists r' with NOM violation (for some agent i)
"""
from __future__ import annotations

import itertools
import torch

from .config import Config
from .domains import DomainSpec
from .allocations import build_all_allocs, score_matrix


def enumerate_reports(domain: DomainSpec, owned_mask: list[bool]) -> torch.Tensor:
    """All domain-valid marginal-rank reports for an agent.

    owned_mask[j] = True iff the agent holds item j in their endowment.
    Returns [N, m] long tensor.
    """
    m = len(owned_mask)
    if domain.strict:
        perms = list(itertools.permutations(range(m)))
        return torch.tensor(perms, dtype=torch.long)
    choices = [domain.owned_ranks if owned_mask[j] else domain.unowned_ranks
               for j in range(m)]
    combos = list(itertools.product(*choices))
    return torch.tensor(combos, dtype=torch.long)


@torch.no_grad()
def _chosen_alloc(cfg: Config, mech_fn, mr: torch.Tensor, endow_rep: torch.Tensor,
                  chunk: int = 4096) -> torch.Tensor:
    N = mr.shape[0]
    parts = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        S_ch = score_matrix(cfg, mr[s:e])
        p_ch = mech_fn(cfg, mr[s:e], endow_rep[s:e], S_ch)
        parts.append(p_ch.argmax(dim=1))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def full_enum_nom_eval(cfg: Config, domain: DomainSpec, mech_fn,
                       marginal_rank: torch.Tensor, endow_idx: torch.Tensor,
                       chunk: int = 4096) -> tuple[float, float, dict]:
    """Full-enumeration unambiguous NOM evaluation.

    Returns (mean_viol, viol_rate, stats) where stats contains diagnostic
    counters (e.g. how often WC fired, for the blocking-profile sanity check).
    """
    B, A, m = marginal_rank.shape
    device = marginal_rank.device
    allocs_all = build_all_allocs(cfg).to(device)
    R = cfg.num_ranks

    viol    = torch.zeros(B, A)
    bc_fire = torch.zeros(B, A)
    wc_fire = torch.zeros(B, A)

    for b in range(B):
        alloc_e = allocs_all[endow_idx[b]]   # [m]
        for i in range(A):
            owned_mask_i = [(alloc_e[j].item() == i) for j in range(m)]
            true_rank_i  = marginal_rank[b, i, :]                 # [m]
            truth_reports_i = enumerate_reports(domain, owned_mask_i).to(device)  # [Ni, m]
            Ni = truth_reports_i.shape[0]

            opponents = [a for a in range(A) if a != i]
            opp_reports = []
            for a in opponents:
                owned_mask_a = [(alloc_e[j].item() == a) for j in range(m)]
                opp_reports.append(enumerate_reports(domain, owned_mask_a).to(device))

            grids = torch.meshgrid(*[torch.arange(r.shape[0]) for r in opp_reports],
                                    indexing="ij")
            idx_combo = torch.stack([g.reshape(-1) for g in grids], dim=1)  # [P, n_opp]
            P = idx_combo.shape[0]
            opp_joint = torch.stack(
                [opp_reports[oi][idx_combo[:, oi]] for oi in range(len(opponents))], dim=1
            )  # [P, n_opp, m]

            # ---- S_truth: agent i reports truthfully ----
            mr_full_truth = torch.zeros(P, A, m, dtype=torch.long, device=device)
            mr_full_truth[:, i, :] = true_rank_i.view(1, m).expand(P, m)
            for oi, a in enumerate(opponents):
                mr_full_truth[:, a, :] = opp_joint[:, oi, :]

            endow_rep = endow_idx[b].view(1).expand(P)
            chosen_t  = _chosen_alloc(cfg, mech_fn, mr_full_truth, endow_rep, chunk)
            alloc_chosen_t = allocs_all[chosen_t]                          # [P, m]
            agent_mask_t   = (alloc_chosen_t == i).float()
            bundle_t = true_rank_i.view(1, m).float() * agent_mask_t + R * (1 - agent_mask_t)
            S_truth, _ = torch.sort(bundle_t, dim=-1)                       # [P, m]

            # ---- S_lie(r') for every domain-valid misreport r' ----
            mr_full_lie = mr_full_truth.unsqueeze(0).expand(Ni, P, A, m).clone()
            mr_full_lie[:, :, i, :] = truth_reports_i.view(Ni, 1, m).expand(Ni, P, m)
            mr_full_lie_flat = mr_full_lie.reshape(Ni * P, A, m)

            endow_rep2 = endow_idx[b].view(1).expand(Ni * P)
            chosen_l = _chosen_alloc(cfg, mech_fn, mr_full_lie_flat, endow_rep2, chunk)
            alloc_chosen_l = allocs_all[chosen_l]
            agent_mask_l = (alloc_chosen_l == i).float()
            bundle_l = (true_rank_i.view(1, m).float().expand(Ni * P, m) * agent_mask_l
                        + R * (1 - agent_mask_l))
            sorted_l, _ = torch.sort(bundle_l, dim=-1)
            S_lie = sorted_l.reshape(Ni, P, m)                               # [Ni, P, m]

            # fosd_T_over_L[s_t, r', s_l] = truth(s_t) FOSD-weakly-dom lie(r', s_l)
            diff = S_truth.view(P, 1, 1, m) - S_lie.view(1, Ni, P, m)
            fosd = (diff <= 1e-8).all(dim=-1)            # [P, Ni, P]  (s_t, r', s_l)

            any_truth_dom = fosd.any(dim=0)               # [Ni, P]  over s_t
            bc_viol = (~any_truth_dom).any(dim=1)         # [Ni]   exists s_l undominated

            any_lie_dom = fosd.any(dim=2)                 # [P, Ni]  over s_l
            wc_viol = (~any_lie_dom).any(dim=0)           # [Ni]   exists s_t dominating none

            obvious = bc_viol | wc_viol
            viol[b, i]    = obvious.any().float()
            bc_fire[b, i] = bc_viol.any().float()
            wc_fire[b, i] = wc_viol.any().float()

    stats = {
        "bc_fire_rate": float(bc_fire.mean()),
        "wc_fire_rate": float(wc_fire.mean()),
    }
    return float(viol.mean()), float((viol.max(1).values > 0.5).float().mean()), stats
