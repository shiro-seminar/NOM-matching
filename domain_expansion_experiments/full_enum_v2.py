"""Full-enumeration unambiguous NOM evaluation, table-based (endowment, agent).

Key fact: which item-subset agent i receives from the mechanism depends only
on (endowment, agent i's report, opponents' reports) -- NOT on agent i's true
preferences. So for a fixed (endowment k_e, agent i):

  1. Enumerate all reports r_i for agent i (Ni) and all opponent report
     combinations (P). Run the mechanism once for each of the Ni*P profiles
     and record which item-subset (mask) agent i receives. This is the only
     expensive step (Ni*P mechanism evaluations).
  2. For each report r_i, the SET of distinct masks agent i can receive
     across all P opponent combos is small (<= C(m, |Omega_i|)).
  3. For ANY true-preference vector of agent i (there are exactly Ni of
     them, since true preferences are domain-valid reports too), the
     FOSD-sorted bundle for each mask is a cheap lookup. S_truth / S_lie
     and the BC/WC criterion are then evaluated for all Ni true-preference
     vectors essentially for free.
"""
from __future__ import annotations

import time
import torch

from .config import Config
from .domains import DomainSpec
from .allocations import build_all_allocs, score_matrix
from .full_enum import enumerate_reports


@torch.no_grad()
def build_mask_table(cfg: Config, domain: DomainSpec, mech_fn,
                     k_e: int, agent_i: int,
                     chunk: int = 65536,
                     device: str = "cpu",
                     ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build the (report_i, opponent-combo) -> received-item-mask table.

    Returns:
      reports_i:  [Ni, m] long  -- agent i's domain-valid reports (on device)
      mask_codes: [Ni, P] long  -- bitmask (0..2^m-1) of items agent i receives
      Ni, P
    """
    A, m = cfg.num_agents, cfg.num_items
    allocs_all = build_all_allocs(cfg).to(device)
    alloc_e = allocs_all[k_e]

    owned_mask_i = [(alloc_e[j].item() == agent_i) for j in range(m)]
    reports_i = enumerate_reports(domain, owned_mask_i).to(device)   # [Ni, m]
    Ni = reports_i.shape[0]

    opponents = [a for a in range(A) if a != agent_i]
    opp_reports = [
        enumerate_reports(domain, [(alloc_e[j].item() == a) for j in range(m)]).to(device)
        for a in opponents
    ]

    grids = torch.meshgrid(*[torch.arange(r.shape[0], device=device) for r in opp_reports],
                            indexing="ij")
    idx_combo = torch.stack([g.reshape(-1) for g in grids], dim=1)   # [P, n_opp]
    P = idx_combo.shape[0]
    opp_joint = torch.stack(
        [opp_reports[oi][idx_combo[:, oi]] for oi in range(len(opponents))], dim=1
    )   # [P, n_opp, m]

    # Build the (report_i, opp_combo) table chunk-by-chunk over the flattened
    # Ni*P space WITHOUT materialising the full [Ni*P, A, m] tensor in memory.
    N = Ni * P
    powers = (2 ** torch.arange(m, dtype=torch.long, device=device))
    mask_codes_flat = torch.empty(N, dtype=torch.long, device=device)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        flat  = torch.arange(s, e, device=device)
        r_idx = flat // P
        p_idx = flat % P

        mr_chunk = torch.zeros(e - s, A, m, dtype=torch.long, device=device)
        mr_chunk[:, agent_i, :] = reports_i[r_idx]
        for oi, a in enumerate(opponents):
            mr_chunk[:, a, :] = opp_joint[p_idx, oi, :]

        endow_rep = torch.full((e - s,), k_e, dtype=torch.long, device=device)
        S_chunk   = score_matrix(cfg, mr_chunk)
        p_chunk   = mech_fn(cfg, mr_chunk, endow_rep, S_chunk)
        chosen    = p_chunk.argmax(dim=1)

        alloc_chosen = allocs_all[chosen]                   # [chunk, m]
        agent_mask   = (alloc_chosen == agent_i).long()     # [chunk, m]
        mask_codes_flat[s:e] = (agent_mask * powers).sum(-1)

    mask_codes = mask_codes_flat.reshape(Ni, P)
    return reports_i, mask_codes, Ni, P


def _decode_mask(code: int, m: int, device: str) -> torch.Tensor:
    return torch.tensor([(code >> j) & 1 for j in range(m)],
                        dtype=torch.float32, device=device)


@torch.no_grad()
def eval_all_true_prefs(cfg: Config,
                        reports_i: torch.Tensor,
                        mask_codes: torch.Tensor,
                        ) -> dict:
    """Evaluate unambiguous NOM for ALL Ni possible true-preference vectors.

    Each true-pref is itself one of agent i's domain-valid reports.
    Reuses the precomputed mask_codes table -- zero additional mechanism evals.
    """
    device = mask_codes.device
    reports_i = reports_i.to(device)
    Ni, P = mask_codes.shape
    m = cfg.num_items
    R = cfg.num_ranks

    # For each report index r, collect the small set of distinct item-masks
    # (at most C(m, |Omega_i|) distinct values -- typically 4-6 for m=4).
    unique_per_r = [mask_codes[r].unique() for r in range(Ni)]
    MAXU = max(u.numel() for u in unique_per_r)
    mask_set = torch.zeros(Ni, MAXU, m, device=device)
    valid    = torch.zeros(Ni, MAXU, dtype=torch.bool, device=device)
    for r in range(Ni):
        u = unique_per_r[r]
        for j, code in enumerate(u.tolist()):
            mask_set[r, j] = _decode_mask(int(code), m, device)
            valid[r, j]    = True

    viol    = torch.zeros(Ni, dtype=torch.bool, device=device)
    bc_fire = torch.zeros(Ni, dtype=torch.bool, device=device)
    wc_fire = torch.zeros(Ni, dtype=torch.bool, device=device)

    # Pre-sort bundles for ALL true-rank / report-mask combinations at once.
    # sorted_bundle[t, r, j, :] = sorted ranks agent i gets under mask_set[r,j]
    # when true rank is reports_i[t].  Shape: [Ni, Ni, MAXU, m].
    # Doing this in one shot avoids the inner python loop entirely.
    true_ranks = reports_i.float()                              # [Ni, m]
    # bundle[t, r, j, :] = true_ranks[t] * mask_set[r,j] + R*(1-mask_set[r,j])
    bundle = (true_ranks.view(Ni, 1, 1, m) * mask_set.view(1, Ni, MAXU, m)
              + R * (1.0 - mask_set.view(1, Ni, MAXU, m)))     # [Ni, Ni, MAXU, m]
    # Mask out padding slots with sentinel R so they sort last.
    bundle = torch.where(valid.view(1, Ni, MAXU, 1), bundle,
                         torch.full_like(bundle, float(R)))
    sorted_bundle, _ = torch.sort(bundle, dim=-1)              # [Ni, Ni, MAXU, m]

    for t in range(Ni):
        # S_truth: all bundles reachable by reporting truthfully (report_idx=t),
        # across all opponent combos -> union over j of mask_set[t,j] (small set).
        Ut = valid[t].sum().item()
        S_truth = sorted_bundle[t, t, :Ut, :]                 # [Ut, m]  (valid entries first)

        # BC/WC over all misreports r != t (include r==t too; S_lie[t]=S_truth -> never fires)
        # Vectorise over r and j simultaneously.
        # sorted_bundle[t, r, :, :] for r in 0..Ni-1: bundles under each misreport
        # Shape [Ni, MAXU, m]; valid mask [Ni, MAXU].
        S_lie_all  = sorted_bundle[t]       # [Ni, MAXU, m]  (lie bundles for each misreport r)
        valid_lie  = valid                  # [Ni, MAXU]

        # diff[st, r, j] = S_truth[st] - S_lie_all[r,j]  (truthful - lie)
        # [Ut, 1, 1, m] - [1, Ni, MAXU, m] -> [Ut, Ni, MAXU, m]
        diff = (S_truth.view(Ut, 1, 1, m) - S_lie_all.view(1, Ni, MAXU, m))
        fosd = (diff <= 1e-8).all(dim=-1)                      # [Ut, Ni, MAXU]

        # Ignore invalid padding slots in S_lie.
        fosd = fosd & valid_lie.view(1, Ni, MAXU)              # [Ut, Ni, MAXU]

        # BC: exists a lie bundle (r,j) not dominated by ANY truth bundle st
        any_truth_dom = fosd.any(dim=0)                        # [Ni, MAXU]
        bc = ((~any_truth_dom) & valid_lie).any()

        # WC: exists a truth bundle st that dominates NO lie bundle across ALL r
        any_lie_dom = fosd.any(dim=2)                          # [Ut, Ni]
        any_lie_dom_per_st = any_lie_dom.all(dim=1)            # [Ut] false -> wc
        wc = (~any_lie_dom_per_st).any()

        viol[t]    = bc | wc
        bc_fire[t] = bc
        wc_fire[t] = wc

    return {
        "viol_rate":    float(viol.float().mean()),
        "bc_fire_rate": float(bc_fire.float().mean()),
        "wc_fire_rate": float(wc_fire.float().mean()),
        "Ni": Ni,
    }


def run_full_enum(cfg: Config, domain: DomainSpec, mech_fn,
                  valid_endow_idx: list[int],
                  chunk: int = 65536,
                  device: str = "cpu",
                  verbose: bool = True,
                  ) -> dict:
    """Run full-enumeration NOM eval over all (endowment, agent) cells.

    Returns per-cell results dict keyed by (k_e, agent_i).
    """
    results = {}
    total = len(valid_endow_idx) * cfg.num_agents
    done  = 0
    t_start = time.time()

    for k_e in valid_endow_idx:
        for agent_i in range(cfg.num_agents):
            t0 = time.time()
            reports_i, mask_codes, Ni, P = build_mask_table(
                cfg, domain, mech_fn, k_e, agent_i, chunk=chunk, device=device)
            t1 = time.time()
            stats = eval_all_true_prefs(cfg, reports_i, mask_codes)
            t2 = time.time()

            results[(k_e, agent_i)] = {
                **stats, "table_t": t1 - t0, "eval_t": t2 - t1
            }
            done += 1
            if verbose:
                elapsed = time.time() - t_start
                eta = elapsed / done * (total - done)
                v = stats["viol_rate"]
                print(f"[{done}/{total}] k_e={k_e} agent={agent_i}  "
                      f"viol={v*100:.2f}%  "
                      f"table={t1-t0:.1f}s  eval={t2-t1:.2f}s  "
                      f"ETA={eta/60:.1f}min", flush=True)

    return results
