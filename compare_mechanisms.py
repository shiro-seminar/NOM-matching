"""Compare phi_ML vs Endowment / TTC-general / WMAX-IR-PE / WMAX-PE
on 1000 random valuation profiles.

Metrics per baseline:
  - Exact match rate       : alloc(net) == alloc(baseline)
  - Item match distribution: # items assigned to same agent (0-4)
  - Welfare difference     : W(net) - W(baseline)  (signed)
"""
import sys; sys.path.insert(0, '.')
import torch
from nom_ir_pe_3x4.config import Config
from nom_ir_pe_3x4.allocations import (
    all_utilities, ir_pe_mask, build_all_allocs, random_endowment,
)
from nom_ir_pe_3x4.model import AllocationNet
from nom_ir_pe_3x4.benchmarks import (
    endowment_mechanism, ttc_generalized, wmax_ir_pe, wmax_pe,
)

# ── setup ──────────────────────────────────────────────────────────────
cfg   = Config()
torch.manual_seed(0)
B     = 1000
m     = cfg.num_items   # 4

v         = torch.rand(B, cfg.num_agents, m)
endow_idx = random_endowment(cfg, B)
U         = all_utilities(cfg, v)
mask      = ir_pe_mask(cfg, U, endow_idx)
allocs    = build_all_allocs(cfg)          # [K, m]

# ── LearnedNet argmax ───────────────────────────────────────────────────
ckpt = torch.load('alloc_net_3x4.pt', map_location='cpu')
net  = AllocationNet(cfg)
net.load_state_dict(ckpt['state_dict'])
net.eval()
with torch.no_grad():
    net_idx = net.argmax_alloc(v, mask=mask)   # [B]

net_alloc = allocs[net_idx]   # [B, m]  各財を受け取るエージェント番号
net_W     = U[torch.arange(B), :, net_idx].sum(1)   # [B] welfare


# ── baseline helper ────────────────────────────────────────────────────
def get_argmax_idx(probs: torch.Tensor) -> torch.Tensor:
    """Probabilistic output → argmax index [B]."""
    return probs.argmax(1)


BASELINES = {
    "Endowment":  endowment_mechanism,
    "TTC-general": ttc_generalized,
    "WMAX-IR-PE": wmax_ir_pe,
    "WMAX-PE":    wmax_pe,
}


# ── comparison ─────────────────────────────────────────────────────────
def item_overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Number of items with same agent assignment. a,b: [B, m] -> [B]."""
    return (a == b).sum(1)


results = {}
for name, fn in BASELINES.items():
    with torch.no_grad():
        probs     = fn(cfg, v, endow_idx, U)   # [B, K]
    base_idx  = get_argmax_idx(probs)           # [B]
    base_alloc = allocs[base_idx]               # [B, m]
    base_W     = U[torch.arange(B), :, base_idx].sum(1)   # [B]

    exact   = (net_idx == base_idx).float()                # [B]
    overlap = item_overlap(net_alloc, base_alloc).float()  # [B]  0-4
    w_diff  = net_W - base_W                               # [B]

    results[name] = dict(
        exact   = exact,
        overlap = overlap,
        w_diff  = w_diff,
        base_W  = base_W,
    )


# ── print ───────────────────────────────────────────────────────────────
sep = "=" * 62
print(sep)
print(f"  phi_ML vs baselines  (B={B}, 3 agents, 4 items)")
print(sep)

net_W_mean = net_W.mean().item()
print(f"\nphi_ML welfare:  {net_W_mean:.4f}\n")

for name, r in results.items():
    exact_rate = r['exact'].mean().item()
    ov         = r['overlap']
    w_diff     = r['w_diff']
    base_W_mean = r['base_W'].mean().item()

    print(f"── vs {name} {'─'*(40-len(name))}")
    print(f"  Exact match rate    : {exact_rate*100:6.1f}%")
    print(f"  Item match dist (# items same agent):")
    for k in range(m + 1):
        cnt  = (ov == k).sum().item()
        pct  = cnt / B * 100
        bar  = '#' * int(pct / 2)
        print(f"    {k} items: {cnt:4d} ({pct:5.1f}%)  {bar}")
    print(f"  Mean item overlap   : {ov.mean().item():.3f} / {m}")
    print(f"  Welfare (baseline)  : {base_W_mean:.4f}")
    print(f"  Welfare diff (net-base):")
    print(f"    mean  = {w_diff.mean().item():+.4f}")
    print(f"    std   = {w_diff.std().item():.4f}")
    print(f"    min   = {w_diff.min().item():+.4f}")
    print(f"    max   = {w_diff.max().item():+.4f}")
    net_better  = (w_diff > 1e-5).float().mean().item()
    base_better = (w_diff < -1e-5).float().mean().item()
    tie         = 1.0 - net_better - base_better
    print(f"    net>base: {net_better*100:.1f}%  tie: {tie*100:.1f}%  base>net: {base_better*100:.1f}%")
    print()
