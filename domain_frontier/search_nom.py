"""NOM witness search: train a mechanism, then VERIFY it with the FOSD oracle.

We do not implement phi^IP. Instead we search the space of marginal mechanisms
directly: an AllocationNet, hard-masked to the *unambiguous* IR+PE feasible set
(`unamb_ir_pe_mask`), trained to minimise a NOM surrogate. The trained net's
argmax is a deterministic marginal mechanism that is IR+PE by construction
(the mask), so only NOM needs checking.

Soundness: a VERIFIED NOM=0 (via the full-enum FOSD oracle in full_enum_v2)
PROVES the domain admits an IR+PE+NOM mechanism (witness found) -> D in D_NOM.
Training NOM->0 alone is not proof; verification is mandatory. This closes the
open experiment E in PROJECT_OVERVIEW.md (learned model never FOSD-NOM-checked).
"""
from __future__ import annotations

import time
import torch
import torch.nn as nn

from domain_expansion_experiments.config import Config
from domain_expansion_experiments.domains import DomainSpec, DOMAINS
from domain_expansion_experiments.allocations import (
    score_matrix, unamb_ir_pe_mask, build_all_allocs,
)
from domain_expansion_experiments.data_gen import sample_batch, sample_domain_mr_flat
from domain_expansion_experiments.model import AllocationNet
from domain_expansion_experiments.full_enum_v2 import build_mask_table, eval_all_true_prefs


# ---------------------------------------------------------------------------
# Differentiable NOM surrogate, FOSD-masked (mirrors losses.nom_loss but the net
# is always restricted to the unambiguous IR+PE set, not the score-based one).
# ---------------------------------------------------------------------------
def _nom_surrogate(cfg: Config, domain: DomainSpec, net: AllocationNet,
                   marginal_rank: torch.Tensor, endow_idx: torch.Tensor,
                   S_true: torch.Tensor, S: int, M: int) -> torch.Tensor:
    B, A, m = marginal_rank.shape
    device = marginal_rank.device

    mr_opp = sample_domain_mr_flat(cfg, domain, endow_idx, S, device).reshape(B, S, A, m)
    mr_mis = sample_domain_mr_flat(cfg, domain, endow_idx, M, device).reshape(B, M, A, m)

    violations = []
    for i in range(A):
        mr_opp_i = mr_opp.clone()
        mr_opp_i[:, :, i, :] = marginal_rank[:, i, :].unsqueeze(1).expand(B, S, m)

        mr_flat = mr_opp_i.reshape(B * S, A, m)
        endow_rep = endow_idx.unsqueeze(1).expand(B, S).reshape(B * S)
        mask = fast_irpe_mask_batched(cfg, mr_flat, endow_rep, device)
        probs = net(mr_flat, mask=mask)
        S_i = S_true[:, i, :].unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)
        u_truth = (probs * S_i).sum(1).reshape(B, S)
        BC_t, WC_t = u_truth.max(1).values, u_truth.min(1).values

        mr_mis_full = mr_opp.unsqueeze(1).expand(B, M, S, A, m).clone()
        mr_mis_full[:, :, :, i, :] = mr_mis[:, :, i, :].unsqueeze(2).expand(B, M, S, m)
        BMS = B * M * S
        mr_mis_f = mr_mis_full.reshape(BMS, A, m)
        endow_r2 = endow_idx.view(B, 1, 1).expand(B, M, S).reshape(BMS)
        mask2 = fast_irpe_mask_batched(cfg, mr_mis_f, endow_r2, device)
        probs2 = net(mr_mis_f, mask=mask2)
        S_i2 = S_true[:, i, :].view(B, 1, 1, -1).expand(B, M, S, -1).reshape(BMS, -1)
        u_lie = (probs2 * S_i2).sum(1).reshape(B, M, S)
        BC_l, WC_l = u_lie.max(2).values, u_lie.min(2).values

        bc = torch.relu(BC_l - BC_t.unsqueeze(1))
        wc = torch.relu(WC_l - WC_t.unsqueeze(1))
        violations.append(torch.max(bc, wc).max(1).values)
    return torch.stack(violations, 1).mean()


def train_witness(cfg: Config, steps: int | None = None, verbose: bool = True) -> AllocationNet:
    """Train an IR+PE-masked AllocationNet to minimise the NOM surrogate."""
    steps = steps or cfg.steps
    torch.manual_seed(cfg.seed)
    domain = DOMAINS[cfg.domain]
    device = torch.device(cfg.device)
    net = AllocationNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    lam, rho = cfg.lambda_nom, cfg.rho
    t0 = time.time()

    for step in range(1, steps + 1):
        batch = sample_batch(cfg)
        mr, ei, S = batch["marginal_rank"], batch["endow_idx"], batch["S"]
        mask = fast_irpe_mask_batched(cfg, mr, ei, device)
        probs = net(mr, mask=mask, temperature=cfg.temperature)
        welfare = torch.einsum("bk,bak->ba", probs, S).sum(1).mean()
        nom = _nom_surrogate(cfg, domain, net, mr, ei, S, cfg.S, cfg.M)
        loss = -cfg.welfare_weight * welfare + lam * nom + (rho / 2.0) * nom * nom

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()

        nom_v = float(nom.detach())
        if step % cfg.dual_update_every == 0:
            lam = max(0.0, lam + rho * nom_v)
            if nom_v > cfg.nom_target:
                rho = min(rho * cfg.rho_mult, cfg.rho_max)
        if verbose and (step % 200 == 0 or step == 1):
            print(f"  step={step:5d}  welfare={float(welfare.detach()):.4f}  "
                  f"nom_surrogate={nom_v:.5f}  lambda={lam:.3f}  rho={rho:.2f}  "
                  f"t={time.time()-t0:.0f}s", flush=True)
    return net


_BAL_CACHE: dict = {}


def _balanced_idx(cfg: Config, k_e: int, device):
    """Cached: balanced allocations + endowment position for endowment k_e.
    Recomputed every mech_fn call otherwise -- dominated full-enum verify time."""
    key = (cfg.num_agents, cfg.num_items, k_e, str(device))
    hit = _BAL_CACHE.get(key)
    if hit is not None:
        return hit
    A = cfg.num_agents
    allocs = build_all_allocs(cfg).to(device)
    counts = torch.stack([(allocs == i).sum(1) for i in range(A)], 1)
    bal_idx = (counts == counts[k_e]).all(1).nonzero(as_tuple=True)[0]
    owned = torch.stack([(allocs[bal_idx] == a).float() for a in range(A)], 1)
    be = int((bal_idx == k_e).nonzero(as_tuple=True)[0].item())
    _BAL_CACHE[key] = (bal_idx, owned, be)
    return bal_idx, owned, be


@torch.no_grad()
def fast_irpe_mask(cfg: Config, mr: torch.Tensor, k_e: int, device) -> torch.Tensor:
    """[B, K] unambiguous IR+PE+balanced mask for a FIXED endowment k_e.

    Equivalent to allocations.unamb_ir_pe_mask but restricted to balanced
    allocations (small set), so it is O(B*nbal^2) instead of O(B*K^2) -- safe for
    the large chunks build_mask_table feeds during verification.
    """
    A, m = cfg.num_agents, cfg.num_items
    K = A ** m
    R = float(cfg.num_ranks)
    bal_idx, owned, be = _balanced_idx(cfg, k_e, device)
    nbal, B = bal_idx.shape[0], mr.shape[0]

    r = mr.float().unsqueeze(1)
    own = owned.unsqueeze(0)
    sb, _ = torch.sort(r * own + R * (1.0 - own), dim=-1)        # [B, nbal, A, m]
    esb = sb[:, be:be + 1]
    ir = ((sb <= esb + 1e-8).all(-1)).all(-1)                    # [B, nbal]
    x, y = sb.unsqueeze(2), sb.unsqueeze(1)
    mu_weak = (x <= y + 1e-8).all(-1)
    nu_weak = (y <= x + 1e-8).all(-1)
    cond1 = (~mu_weak).any(-1)
    improves = cond1 & ~(mu_weak & ~nu_weak).any(-1)
    eye = torch.eye(nbal, dtype=torch.bool, device=device).unsqueeze(0)
    pe = ~(improves & ~eye).any(-1)
    feas = (ir & pe).float()                                     # [B, nbal]

    empty = feas.sum(1) < 0.5
    if empty.any():
        feas = feas.clone(); feas[empty] = 0.0; feas[empty, be] = 1.0
    mask = torch.zeros(B, K, device=device)
    mask[:, bal_idx] = feas
    return mask


@torch.no_grad()
def fast_irpe_mask_batched(cfg: Config, mr: torch.Tensor, ei: torch.Tensor,
                          device) -> torch.Tensor:
    """[B, K] unambiguous IR+PE mask for a batch with MIXED endowments.

    Groups rows by endowment (few distinct values) and applies fast_irpe_mask per
    group. Equivalent to allocations.unamb_ir_pe_mask but far cheaper. The mask is
    a constant w.r.t. net parameters, so no_grad is fine inside training.
    """
    B = mr.shape[0]
    K = cfg.num_agents ** cfg.num_items
    mask = torch.zeros(B, K, device=device)
    for k_e in torch.unique(ei).tolist():
        rows = (ei == k_e).nonzero(as_tuple=True)[0]
        mask[rows] = fast_irpe_mask(cfg, mr[rows], int(k_e), device)
    return mask


def make_mech_fn(net: AllocationNet):
    """Wrap a trained net as a deterministic marginal mechanism for full_enum_v2.

    Uses the fast (balanced-restricted) IR+PE mask since full_enum_v2 calls this
    with a single fixed endowment per (endowment, agent) cell.
    """
    @torch.no_grad()
    def mech_fn(cfg, mr, ei, S):
        mask = fast_irpe_mask(cfg, mr, int(ei[0].item()), mr.device)
        return net(mr, mask=mask, temperature=1e-3)
    return mech_fn


@torch.no_grad()
def verify_nom_fullenum(cfg: Config, domain: DomainSpec, net: AllocationNet,
                        endow_list: list[int] | None = None,
                        chunk: int = 65536, device: str = "cpu",
                        verbose: bool = True) -> dict:
    """Run the full-enum FOSD-NOM oracle over (endowment, agent) cells.

    Returns the number of NOM-violating cells. 0 -> verified witness (D in D_NOM).
    IR+PE holds by construction of the mask, so only NOM is checked here.
    """
    allocs = build_all_allocs(cfg)
    counts = torch.stack([(allocs == i).sum(1) for i in range(cfg.num_agents)], 1)
    if endow_list is None:
        endow_list = (counts.min(1).values >= 1).nonzero(as_tuple=True)[0].tolist()
    mech_fn = make_mech_fn(net)

    viol_cells = 0
    total = len(endow_list) * cfg.num_agents
    done = 0
    for k_e in endow_list:
        for agent_i in range(cfg.num_agents):
            reports_i, mask_codes, Ni, P = build_mask_table(
                cfg, domain, mech_fn, k_e, agent_i, chunk=chunk, device=device)
            stats = eval_all_true_prefs(cfg, reports_i, mask_codes)
            done += 1
            if stats["viol_rate"] > 0:
                viol_cells += 1
            if verbose:
                print(f"  [{done}/{total}] k_e={k_e} agent={agent_i}  "
                      f"viol={stats['viol_rate']*100:.2f}%", flush=True)
    return {"domain": domain.name, "nom_viol_cells": viol_cells, "total_cells": total,
            "witness": viol_cells == 0}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--domain", type=str, default="trichotomous")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--verify_chunk", type=int, default=65536)
    args = p.parse_args()

    cfg = Config(domain=args.domain, steps=args.steps, S=args.S, M=args.M,
                 batch_size=args.batch, device=args.device)
    print(f"[train witness] domain={cfg.domain} steps={cfg.steps} device={cfg.device}", flush=True)
    net = train_witness(cfg)
    net.eval()
    print(f"[verify NOM via full-enum FOSD oracle] domain={cfg.domain}", flush=True)
    res = verify_nom_fullenum(cfg, DOMAINS[cfg.domain], net,
                              chunk=args.verify_chunk, device=args.device)
    tag = "WITNESS FOUND (domain in D_NOM)" if res["witness"] else \
          f"NOM violated in {res['nom_viol_cells']}/{res['total_cells']} cells"
    print(f"  -> {tag}", flush=True)
