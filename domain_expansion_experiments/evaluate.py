"""Evaluation with violation structure recording.

Metrics (argmax / deterministic):
  welfare:   mean score of chosen bundle (higher = better)
  ir_viol:   fraction of profiles where some agent is IR-violated
  pe_rate:   fraction where chosen allocation is PE
  irpe_rate: fraction where chosen allocation is IR+PE+Balanced
  nom_mean:  mean NOM violation
  nom_viol:  fraction of profiles with obvious manipulation > 1e-5

Violation records capture the profile details when IR or NOM is violated.
"""
from __future__ import annotations

import torch
from .config import Config
from .domains import DomainSpec, DOMAINS
from .allocations import (
    score_matrix, build_all_allocs, endowment_scores,
    ir_pe_mask, ir_mask, pareto_mask, num_allocations,
    fsd_ir_mask, unamb_pe_mask, unamb_ir_pe_mask,
)
from .data_gen import sample_batch, sample_domain_mr_flat
from .model import AllocationNet
from .benchmarks import BENCHMARKS

S_EVAL = 16
M_EVAL = 16
CHUNK  = 2048   # max batch size for chunked NOM forward passes


def _mech_fn_chunked(mech_fn, cfg, mr_flat, endow_rep, chunk=CHUNK):
    """Call mech_fn in chunks to avoid OOM with large S/M."""
    N = mr_flat.shape[0]
    if N <= chunk:
        S_flat = score_matrix(cfg, mr_flat)
        return mech_fn(cfg, mr_flat, endow_rep, S_flat)
    parts = []
    for start in range(0, N, chunk):
        end  = min(start + chunk, N)
        S_ch = score_matrix(cfg, mr_flat[start:end])
        p_ch = mech_fn(cfg, mr_flat[start:end], endow_rep[start:end], S_ch)
        parts.append(p_ch.detach())
    return torch.cat(parts, dim=0)


@torch.no_grad()
def evaluate_mechanism(name, mech_fn, cfg: Config, domain: DomainSpec,
                       marginal_rank, endow_idx, S, wmax_s,
                       eval_S: int | None = None,
                       eval_M: int | None = None) -> dict:
    """Evaluate a mechanism.

    eval_S / eval_M: opponent/misreport samples for NOM evaluation.
    Larger values give more reliable violation detection (default: S_EVAL/M_EVAL).
    Uses chunked forward passes so large eval_S × eval_M values are feasible.
    """
    s_nom = eval_S or S_EVAL
    m_nom = eval_M or M_EVAL

    B = marginal_rank.shape[0]

    probs   = _mech_fn_chunked(mech_fn, cfg, marginal_rank, endow_idx)
    ES      = torch.einsum("bk,bak->ba", probs, S)
    welfare = ES.sum(1).mean()

    s0      = endowment_scores(S, endow_idx)
    ir_viol = (ES < s0 - 1e-6).any(1).float().mean()

    chosen    = probs.argmax(1)
    pe_m      = pareto_mask(S)
    irpe_m    = ir_pe_mask(cfg, S, endow_idx)
    pe_rate   = pe_m.gather(1, chosen.unsqueeze(1)).squeeze(1).mean()
    irpe_rate = irpe_m.gather(1, chosen.unsqueeze(1)).squeeze(1).mean()

    nom_mean, nom_viol_rate = _nom_eval(cfg, domain, mech_fn,
                                        marginal_rank, endow_idx, S,
                                        s_nom, m_nom)
    wmax_m = float(wmax_s.mean())
    welfare_ratio = float(welfare) / wmax_m if abs(wmax_m) > 1e-9 else 1.0

    return {
        "name":          name,
        "welfare":       float(welfare),
        "welfare_ratio": welfare_ratio,
        "ir_viol":       float(ir_viol),
        "pe_rate":       float(pe_rate),
        "irpe_rate":     float(irpe_rate),
        "nom_mean":      nom_mean,
        "nom_viol":      nom_viol_rate,
    }


def _nom_eval(cfg, domain, mech_fn, marginal_rank, endow_idx, S_true, S_nom, M_nom):
    """NOM evaluation with chunked forward passes (handles large S_nom × M_nom)."""
    B, A, m = marginal_rank.shape
    device   = marginal_rank.device
    all_viol = []

    for i in range(A):
        mr_opp_flat = sample_domain_mr_flat(cfg, domain, endow_idx, S_nom, device)
        mr_opp = mr_opp_flat.reshape(B, S_nom, A, m)
        mr_opp[:, :, i, :] = marginal_rank[:, i, :].unsqueeze(1).expand(B, S_nom, m)

        mr_flat   = mr_opp.reshape(B * S_nom, A, m)
        endow_rep = endow_idx.unsqueeze(1).expand(B, S_nom).reshape(B * S_nom)
        p_flat    = _mech_fn_chunked(mech_fn, cfg, mr_flat, endow_rep)

        S_i_true = S_true[:, i, :]
        S_i_flat = S_i_true.unsqueeze(1).expand(B, S_nom, -1).reshape(B * S_nom, -1)
        u_truth  = (p_flat * S_i_flat).sum(1).reshape(B, S_nom)
        BC_t = u_truth.max(1).values
        WC_t = u_truth.min(1).values

        mr_mis_flat = sample_domain_mr_flat(cfg, domain, endow_idx, M_nom, device)
        mr_mis      = mr_mis_flat.reshape(B, M_nom, A, m)

        mr_mis_full = mr_opp.unsqueeze(1).expand(B, M_nom, S_nom, A, m).clone()
        mr_mis_full[:, :, :, i, :] = mr_mis[:, :, i, :].unsqueeze(2).expand(B, M_nom, S_nom, m)

        BMS      = B * M_nom * S_nom
        mr_mis_f = mr_mis_full.reshape(BMS, A, m)
        endow_r2 = endow_idx.view(B, 1, 1).expand(B, M_nom, S_nom).reshape(BMS)
        p_mis_f  = _mech_fn_chunked(mech_fn, cfg, mr_mis_f, endow_r2)

        S_i_bms = S_i_true.view(B, 1, 1, -1).expand(B, M_nom, S_nom, -1).reshape(BMS, -1)
        u_lie   = (p_mis_f * S_i_bms).sum(1).reshape(B, M_nom, S_nom)

        BC_l = u_lie.max(2).values
        WC_l = u_lie.min(2).values

        bc_gain = torch.relu(BC_l - BC_t.unsqueeze(1))
        wc_gain = torch.relu(WC_l - WC_t.unsqueeze(1))
        max_obv = torch.max(bc_gain, wc_gain).max(1).values
        all_viol.append(max_obv)

    viol = torch.stack(all_viol, 1)
    return float(viol.mean()), float((viol.max(1).values > 1e-5).float().mean())


# ---------------------------------------------------------------------------
# Violation structure recording
# ---------------------------------------------------------------------------

@torch.no_grad()
def record_violations(
    net: AllocationNet,
    cfg: Config,
    domain: DomainSpec,
    marginal_rank: torch.Tensor,
    endow_idx: torch.Tensor,
    S: torch.Tensor,
    max_records: int = 10,
) -> dict[str, list]:
    """Record profiles where IR or NOM violations occur.

    Returns a dict with keys 'ir' and 'nom', each a list of record dicts.
    Each record contains:
      - marginal_rank: [A, m] the profile
      - endow_idx:     int
      - chosen_alloc:  int
      - agent:         int (which agent violated)
      - violation:     float (magnitude)
      - details:       str (human-readable description)
    """
    allocs_all = build_all_allocs(cfg)    # [K, m]
    B, A, m    = marginal_rank.shape
    device     = marginal_rank.device

    mask    = ir_pe_mask(cfg, S, endow_idx)
    chosen  = net.argmax_alloc(marginal_rank, mask=mask)    # [B]
    ES      = torch.einsum("bk,bak->ba",
                  net(marginal_rank, mask=mask, temperature=1e-3), S)
    s0      = endowment_scores(S, endow_idx)

    ir_records  = []
    nom_records = []

    for b in range(B):
        k   = endow_idx[b].item()
        mr  = marginal_rank[b]           # [A, m]
        ch  = chosen[b].item()
        alloc = allocs_all[ch]           # [m]

        # ── IR violations ──────────────────────────────────────────────
        if len(ir_records) < max_records:
            for a in range(A):
                if ES[b, a] < s0[b, a] - 1e-6:
                    details = _ir_details(a, mr, allocs_all, k, ch, domain)
                    ir_records.append({
                        "marginal_rank": mr.cpu().tolist(),
                        "endow_idx":     k,
                        "chosen_alloc":  ch,
                        "agent":         a,
                        "violation":     float(s0[b, a] - ES[b, a]),
                        "details":       details,
                    })
                    break   # one record per profile

    # ── NOM violations ─────────────────────────────────────────────────
    def net_mech(cfg_, mr_, ei_, S_):
        mask_ = ir_pe_mask(cfg_, S_, ei_)
        return net(mr_, mask=mask_, temperature=1e-3)

    S_nom, M_nom = 4, 4
    for i in range(A):
        mr_opp_flat = sample_domain_mr_flat(cfg, domain, endow_idx, S_nom, device)
        mr_opp = mr_opp_flat.reshape(B, S_nom, A, m)
        mr_opp[:, :, i, :] = marginal_rank[:, i, :].unsqueeze(1).expand(B, S_nom, m)

        mr_flat   = mr_opp.reshape(B * S_nom, A, m)
        endow_rep = endow_idx.unsqueeze(1).expand(B, S_nom).reshape(B * S_nom)
        S_flat    = score_matrix(cfg, mr_flat)
        p_flat    = net_mech(cfg, mr_flat, endow_rep, S_flat)

        S_i_true = S[:, i, :]
        S_i_flat = S_i_true.unsqueeze(1).expand(B, S_nom, -1).reshape(B * S_nom, -1)
        u_truth  = (p_flat * S_i_flat).sum(1).reshape(B, S_nom)
        BC_t = u_truth.max(1).values
        WC_t = u_truth.min(1).values

        mr_mis_flat = sample_domain_mr_flat(cfg, domain, endow_idx, M_nom, device)
        mr_mis      = mr_mis_flat.reshape(B, M_nom, A, m)

        mr_mis_full = mr_opp.unsqueeze(1).expand(B, M_nom, S_nom, A, m).clone()
        mr_mis_full[:, :, :, i, :] = mr_mis[:, :, i, :].unsqueeze(2).expand(B, M_nom, S_nom, m)

        BMS      = B * M_nom * S_nom
        mr_mis_f = mr_mis_full.reshape(BMS, A, m)
        endow_r2 = endow_idx.view(B, 1, 1).expand(B, M_nom, S_nom).reshape(BMS)
        S_mis_f  = score_matrix(cfg, mr_mis_f)
        p_mis_f  = net_mech(cfg, mr_mis_f, endow_r2, S_mis_f)

        S_i_bms = S_i_true.view(B, 1, 1, -1).expand(B, M_nom, S_nom, -1).reshape(BMS, -1)
        u_lie   = (p_mis_f * S_i_bms).sum(1).reshape(B, M_nom, S_nom)

        BC_l = u_lie.max(2).values
        WC_l = u_lie.min(2).values

        bc_gain = torch.relu(BC_l - BC_t.unsqueeze(1))
        wc_gain = torch.relu(WC_l - WC_t.unsqueeze(1))
        obvious = torch.max(bc_gain, wc_gain)               # [B, M]
        best_m  = obvious.argmax(1)                          # [B]
        max_obv = obvious.max(1).values                      # [B]

        for b in range(B):
            if len(nom_records) >= max_records:
                break
            gain = max_obv[b].item()
            if gain > 1e-5:
                best_mis = mr_mis[b, best_m[b].item(), i, :].cpu().tolist()
                true_r   = marginal_rank[b, i, :].cpu().tolist()
                k        = endow_idx[b].item()
                details  = _nom_details(i, true_r, best_mis, gain, domain, allocs_all, k, cfg)
                nom_records.append({
                    "marginal_rank": marginal_rank[b].cpu().tolist(),
                    "endow_idx":     k,
                    "chosen_alloc":  chosen[b].item(),
                    "agent":         i,
                    "violation":     gain,
                    "misreport":     best_mis,
                    "details":       details,
                })

    return {"ir": ir_records, "nom": nom_records}


def _ir_details(a, mr, allocs_all, endow_k, chosen_k, domain):
    endow_alloc  = allocs_all[endow_k].tolist()
    chosen_alloc = allocs_all[chosen_k].tolist()
    m = len(endow_alloc)
    owned_end = [j for j in range(m) if endow_alloc[j] == a]
    owned_ch  = [j for j in range(m) if chosen_alloc[j] == a]
    lines = [
        f"agent {a}: endowment items {owned_end} -> chosen items {owned_ch}",
        f"  ranks: {mr[a].tolist()}",
    ]
    for j in owned_end:
        r = mr[a, j].item()
        lines.append(f"  item {j}: rank {r} (class {r+1})"
                     + (" [owned, left endowment]" if j not in owned_ch else ""))
    return " | ".join(lines)


def _nom_details(i, true_r, mis_r, gain, domain, allocs_all, endow_k, cfg):
    m = cfg.num_items
    alloc = allocs_all[endow_k].tolist()
    diffs = []
    for j in range(m):
        if true_r[j] != mis_r[j]:
            own = "owned" if alloc[j] == i else "unowned"
            diffs.append(f"item{j}:{true_r[j]}->class{mis_r[j]+1}({own})")
    return f"agent {i} gain={gain:.5f} | misreport changes: {', '.join(diffs) or 'none'}"


def print_table(results):
    hdr = (f"{'Mechanism':<20} {'Welfare':>8} {'W/WMAX':>7} "
           f"{'IR-viol%':>9} {'PE%':>7} {'IR+PE%':>8} {'NOM-mean':>9} {'NOM-viol%':>10}")
    sep = "-" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for r in results:
        print(f"{r['name']:<20} {r['welfare']:>8.4f} {r['welfare_ratio']:>7.3f} "
              f"{r['ir_viol']*100:>9.1f} {r['pe_rate']*100:>7.1f} {r['irpe_rate']*100:>8.1f} "
              f"{r['nom_mean']:>9.5f} {r['nom_viol']*100:>10.1f}")
    print(sep + "\n")


def print_violations(viol_records: dict, max_show: int = 3):
    for vtype, records in viol_records.items():
        if not records:
            print(f"  [{vtype.upper()}] no violations recorded")
            continue
        print(f"  [{vtype.upper()}] {len(records)} violation(s) recorded (showing up to {max_show}):")
        for rec in records[:max_show]:
            mr = rec["marginal_rank"]
            for a, row in enumerate(mr):
                label = "(*)" if a == rec["agent"] else "   "
                print(f"    {label} agent {a} ranks: {row}")
            print(f"       endow={rec['endow_idx']}  chosen={rec['chosen_alloc']}  "
                  f"violation={rec['violation']:.5f}")
            print(f"       {rec['details']}")


# ---------------------------------------------------------------------------
# Unambiguous (FOSD set-based) NOM evaluation
# ---------------------------------------------------------------------------

def _get_chosen_bundle_sorted(cfg: Config, mech_fn, mr_flat: torch.Tensor,
                               endow_rep: torch.Tensor, agent_i: int,
                               chunk: int = CHUNK) -> torch.Tensor:
    """Sorted rank vector of agent i's bundle under deterministic mechanism output.

    Returns [N, m] sorted ranks. Lower rank = better.
    Unowned item slots padded with cfg.num_ranks (sentinel, sorts to end).
    """
    N = mr_flat.shape[0]
    R = cfg.num_ranks
    device = mr_flat.device
    allocs_all = build_all_allocs(cfg).to(device)   # [K, m]

    chosen_parts = []
    for start in range(0, N, chunk):
        end  = min(start + chunk, N)
        S_ch = score_matrix(cfg, mr_flat[start:end])
        p_ch = mech_fn(cfg, mr_flat[start:end], endow_rep[start:end], S_ch)
        chosen_parts.append(p_ch.argmax(dim=1))
    chosen = torch.cat(chosen_parts, dim=0)    # [N]

    alloc_chosen = allocs_all[chosen]                          # [N, m]
    agent_mask   = (alloc_chosen == agent_i).float()           # [N, m]
    ranks_i      = mr_flat[:, agent_i, :].float()              # [N, m]
    bundle       = ranks_i * agent_mask + float(R) * (1.0 - agent_mask)
    sorted_b, _  = torch.sort(bundle, dim=-1)                  # [N, m]
    return sorted_b


@torch.no_grad()
def _unamb_nom_eval(cfg: Config, domain: DomainSpec, mech_fn,
                    marginal_rank: torch.Tensor, endow_idx: torch.Tensor,
                    S_nom: int, M_nom: int) -> tuple[float, float]:
    """Unambiguous NOM evaluation using set-based FOSD comparison.

    For each agent i and misreport r':
      S_truth   = {bundle i gets under truth  × S_nom opponent profiles}
      S_lie(r') = {bundle i gets under r'     × S_nom opponent profiles}

      fosd_T_over_L[s_t, s_l] = 1 iff truth(s_t) ⪰_FOSD lie(r', s_l)

      BC violation: ∃s_l: ∀s_t, NOT fosd_T_over_L[s_t, s_l]
        (some lie scenario whose bundle no truth bundle can dominate)
      WC violation: ∃s_t: ∀s_l, NOT fosd_T_over_L[s_t, s_l]
        (some truth scenario whose bundle can't dominate any lie bundle)
      NOM violation = BC OR WC
    """
    B, A, m_ = marginal_rank.shape
    device   = marginal_rank.device
    all_viol = []

    for i in range(A):
        # --- Truth: S_nom opponent profiles with agent i fixed to truth ---
        mr_opp_flat = sample_domain_mr_flat(cfg, domain, endow_idx, S_nom, device)
        mr_opp = mr_opp_flat.reshape(B, S_nom, A, m_)
        mr_opp[:, :, i, :] = marginal_rank[:, i, :].unsqueeze(1).expand(B, S_nom, m_)

        mr_flat_t = mr_opp.reshape(B * S_nom, A, m_)
        endow_rep = endow_idx.unsqueeze(1).expand(B, S_nom).reshape(B * S_nom)

        sorted_truth_flat = _get_chosen_bundle_sorted(cfg, mech_fn, mr_flat_t, endow_rep, i)
        sorted_truth = sorted_truth_flat.reshape(B, S_nom, m_)   # [B, S_t, m]

        # --- Lie: M_nom misreports × S_nom opponent profiles ---
        mr_mis_flat = sample_domain_mr_flat(cfg, domain, endow_idx, M_nom, device)
        mr_mis = mr_mis_flat.reshape(B, M_nom, A, m_)

        mr_mis_full = mr_opp.unsqueeze(1).expand(B, M_nom, S_nom, A, m_).clone()
        mr_mis_full[:, :, :, i, :] = mr_mis[:, :, i, :].unsqueeze(2).expand(B, M_nom, S_nom, m_)

        BMS      = B * M_nom * S_nom
        mr_mis_f = mr_mis_full.reshape(BMS, A, m_)
        endow_r2 = endow_idx.view(B, 1, 1).expand(B, M_nom, S_nom).reshape(BMS)

        sorted_lie_flat = _get_chosen_bundle_sorted(cfg, mech_fn, mr_mis_f, endow_r2, i)
        sorted_lie = sorted_lie_flat.reshape(B, M_nom, S_nom, m_)   # [B, M, S_l, m]

        # fosd_T_over_L[b, m, s_t, s_l]: truth(s_t) ⪰_FOSD lie(m, s_l)
        #   = sorted_truth[s_t] ≤ sorted_lie[m,s_l] elementwise (lower rank = better)
        truth_exp = sorted_truth.unsqueeze(1).unsqueeze(3)  # [B, 1, S_t, 1, m]
        lie_exp   = sorted_lie.unsqueeze(2)                  # [B, M, 1,   S_l, m]
        diff      = truth_exp - lie_exp                      # [B, M, S_t, S_l, m]
        fosd_T_over_L = (diff <= 1e-8).all(dim=-1)          # [B, M, S_t, S_l]

        # BC: ∃s_l: NO truth bundle dominates it → lie bundle undominated by all truth
        any_truth_dom = fosd_T_over_L.any(dim=2)             # [B, M, S_l]
        bc_viol       = (~any_truth_dom).any(dim=2)           # [B, M]

        # WC: ∃s_t: truth bundle can't dominate ANY lie bundle
        any_lie_dom   = fosd_T_over_L.any(dim=3)             # [B, M, S_t]
        wc_viol       = (~any_lie_dom).any(dim=2)             # [B, M]

        obvious  = (bc_viol | wc_viol).float()               # [B, M]
        max_obv  = obvious.max(dim=1).values                  # [B]
        all_viol.append(max_obv)

    viol = torch.stack(all_viol, dim=1)   # [B, A]
    return float(viol.mean()), float((viol.max(1).values > 0.5).float().mean())


@torch.no_grad()
def evaluate_mechanism_fosd(name: str, mech_fn, cfg: Config, domain: DomainSpec,
                             marginal_rank: torch.Tensor, endow_idx: torch.Tensor,
                             S: torch.Tensor, wmax_s: torch.Tensor,
                             eval_S: int | None = None,
                             eval_M: int | None = None) -> dict:
    """Evaluate a mechanism using unambiguous (FOSD) IR / PE / NOM.

    IR and PE use fsd_ir_mask / unamb_ir_pe_mask (all responsive extensions).
    NOM uses _unamb_nom_eval (set-based FOSD comparison, deterministic argmax).
    Welfare is additive rank-sum (for comparability with existing tables).
    """
    s_nom = eval_S or S_EVAL
    m_nom = eval_M or M_EVAL

    probs   = _mech_fn_chunked(mech_fn, cfg, marginal_rank, endow_idx)
    ES      = torch.einsum("bk,bak->ba", probs, S)
    welfare = ES.sum(1).mean()

    unamb_ir   = fsd_ir_mask(cfg, marginal_rank, endow_idx)       # [B, K]
    unamb_irpe = unamb_ir_pe_mask(cfg, marginal_rank, endow_idx)  # [B, K]
    chosen     = probs.argmax(1)                                   # [B]

    ir_viol   = 1.0 - unamb_ir.gather(1, chosen.unsqueeze(1)).squeeze(1).mean()
    irpe_rate = unamb_irpe.gather(1, chosen.unsqueeze(1)).squeeze(1).mean()

    pe_m    = pareto_mask(S)
    pe_rate = pe_m.gather(1, chosen.unsqueeze(1)).squeeze(1).mean()

    nom_mean, nom_viol = _unamb_nom_eval(cfg, domain, mech_fn,
                                         marginal_rank, endow_idx, s_nom, m_nom)

    wmax_m = float(wmax_s.mean())
    welfare_ratio = float(welfare) / wmax_m if abs(wmax_m) > 1e-9 else 1.0

    return {
        "name":          name,
        "welfare":       float(welfare),
        "welfare_ratio": welfare_ratio,
        "ir_viol":       float(ir_viol),
        "pe_rate":       float(pe_rate),
        "irpe_rate":     float(irpe_rate),
        "nom_mean":      nom_mean,
        "nom_viol":      nom_viol,
    }
