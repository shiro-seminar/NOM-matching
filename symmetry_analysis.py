"""Symmetry (permutation equivariance) analysis of phi_ML.

A symmetric mechanism satisfies:
  phi(pi(v), pi(endow))[j] = pi( phi(v, endow)[j] )  for all j, all permutations pi

Tests all 6 permutations of {a0, a1, a2}.
For each permutation, reports:
  - Exact equivariance rate  (all 4 items match expected permuted output)
  - Item-level equivariance rate
  - Welfare gap: E[u_i under original] vs E[u_i under permuted input]
"""
import sys; sys.path.insert(0, '.')
import torch
import numpy as np
from itertools import permutations

from nom_ir_pe_3x4.config import Config
from nom_ir_pe_3x4.allocations import (
    all_utilities, ir_pe_mask, build_all_allocs, random_endowment,
)
from nom_ir_pe_3x4.model import AllocationNet

# ── setup ──────────────────────────────────────────────────────────────
cfg = Config()
torch.manual_seed(0)
B = 2000
A, m = cfg.num_agents, cfg.num_items

v         = torch.rand(B, A, m)
endow_idx = random_endowment(cfg, B)
U         = all_utilities(cfg, v)
mask      = ir_pe_mask(cfg, U, endow_idx)
allocs    = build_all_allocs(cfg)   # [K, m]

# encoding: k = sum_j alloc[j] * A^j
powers = torch.tensor([A ** j for j in range(m)], dtype=torch.long)

ckpt = torch.load('alloc_net_3x4.pt', map_location='cpu')
net  = AllocationNet(cfg)
net.load_state_dict(ckpt['state_dict'])
net.eval()

with torch.no_grad():
    net_idx   = net.argmax_alloc(v, mask=mask)   # [B]
net_alloc = allocs[net_idx]                       # [B, m]
U_orig    = U[torch.arange(B), :, net_idx]        # [B, A]  true utilities


# ── helper: apply permutation to a [B, m] allocation ──────────────────
def permute_alloc(alloc: torch.Tensor, perm: list) -> torch.Tensor:
    """alloc [B, m] with values in {0,..,A-1} -> relabelled [B, m]."""
    out = torch.zeros_like(alloc)
    for a in range(A):
        out[alloc == a] = perm[a]
    return out


# ── main loop over all 6 permutations ─────────────────────────────────
all_perms = list(permutations(range(A)))

print("=" * 68)
print("  Permutation equivariance test  (B=2000)")
print("=" * 68)

welfare_by_perm = {}   # perm -> [B, A] utility of each agent under permuted run

for perm in all_perms:
    perm = list(perm)
    perm_inv = [perm.index(i) for i in range(A)]

    # --- permute input ---
    # v_perm[b, perm[i], j] = v[b, i, j]  =>  v_perm[b, i, j] = v[b, perm_inv[i], j]
    v_perm = v[:, perm_inv, :]                           # [B, A, m]

    # permute endowment allocation labels
    endow_alloc_orig = allocs[endow_idx]                 # [B, m]
    endow_alloc_perm = permute_alloc(endow_alloc_orig, perm)   # [B, m]
    endow_idx_perm   = (endow_alloc_perm * powers).sum(1)      # [B]

    # --- run net on permuted input ---
    U_perm   = all_utilities(cfg, v_perm)
    mask_perm = ir_pe_mask(cfg, U_perm, endow_idx_perm)
    with torch.no_grad():
        net_idx_perm = net.argmax_alloc(v_perm, mask=mask_perm)
    alloc_perm_actual   = allocs[net_idx_perm]           # [B, m]

    # --- expected output if equivariant ---
    alloc_perm_expected = permute_alloc(net_alloc, perm) # [B, m]

    # --- metrics ---
    item_match  = (alloc_perm_actual == alloc_perm_expected).float()  # [B, m]
    exact_match = item_match.all(dim=1).float()                       # [B]

    exact_rate = exact_match.mean().item()
    item_rate  = item_match.mean().item()

    # welfare of each agent under PERMUTED run, evaluated at TRUE (permuted) preferences
    # U_perm[b, i, k] = utility of perm-agent i under allocation k given v_perm
    U_perm_chosen = U_perm[torch.arange(B), :, net_idx_perm]   # [B, A]
    # Undo permutation to get "original agent index" welfare
    # Agent perm[i] in permuted world = agent i in original world
    # So utility of original agent i = U_perm_chosen[:, perm[i]]
    U_orig_agent_in_perm = torch.stack(
        [U_perm_chosen[:, perm[i]] for i in range(A)], dim=1
    )  # [B, A]  -- utility of original agent i when perm is applied

    welfare_by_perm[tuple(perm)] = U_orig_agent_in_perm

    label = "".join(str(p) for p in perm)
    identity = (perm == list(range(A)))
    tag = " [identity]" if identity else ""
    print(f"\n  pi=({label}){tag}")
    print(f"    Exact equivariance : {exact_rate*100:6.2f}%")
    print(f"    Item-level         : {item_rate*100:6.2f}%")
    print(f"    Welfare per agent (original agent index):")
    for i in range(A):
        orig_u = U_orig[:, i].mean().item()
        perm_u = U_orig_agent_in_perm[:, i].mean().item()
        diff   = perm_u - orig_u
        print(f"      a{i}: orig={orig_u:.4f}  perm={perm_u:.4f}  diff={diff:+.4f}")


# ── symmetry summary across all non-identity permutations ─────────────
print("\n" + "=" * 68)
print("  SYMMETRY SUMMARY")
print("=" * 68)

non_id_exact = []
non_id_item  = []
for perm in all_perms:
    if perm == tuple(range(A)):
        continue
    perm = list(perm)
    perm_inv = [perm.index(i) for i in range(A)]
    v_perm = v[:, perm_inv, :]
    endow_alloc_perm = permute_alloc(allocs[endow_idx], perm)
    endow_idx_perm   = (endow_alloc_perm * powers).sum(1)
    U_perm = all_utilities(cfg, v_perm)
    mask_perm = ir_pe_mask(cfg, U_perm, endow_idx_perm)
    with torch.no_grad():
        net_idx_perm = net.argmax_alloc(v_perm, mask=mask_perm)
    actual   = allocs[net_idx_perm]
    expected = permute_alloc(net_alloc, perm)
    item_m  = (actual == expected).float().mean().item()
    exact_m = (actual == expected).all(1).float().mean().item()
    non_id_exact.append(exact_m)
    non_id_item.append(item_m)

print(f"\n  Avg exact equivariance (non-identity perms): {np.mean(non_id_exact)*100:.2f}%")
print(f"  Avg item-level equivariance                : {np.mean(non_id_item)*100:.2f}%")


# ── agent-level welfare bias ───────────────────────────────────────────
print("\n" + "=" * 68)
print("  AGENT WELFARE BIAS")
print("=" * 68)
print("""
If phi_ML is symmetric, then for any permutation pi:
  E[u_i | original] == E[u_{pi(i)} | permuted by pi]
Bias = mean over all permutations of |E[u_i|orig] - E[u_{pi(i)}|perm]|
""")

# Mean utility per agent under original
u_orig_mean = U_orig.mean(0)   # [A]
print(f"  Mean utility (original labelling):")
for i in range(A):
    print(f"    a{i}: {u_orig_mean[i].item():.4f}")

# Collect mean utility for each original agent across all permutations
print(f"\n  Mean utility of original agent i, averaged over all 6 permutations:")
for i in range(A):
    vals = [welfare_by_perm[perm][:, i].mean().item() for perm in all_perms]
    grand_mean = np.mean(vals)
    grand_std  = np.std(vals)
    print(f"    a{i}: grand_mean={grand_mean:.4f}  std_across_perms={grand_std:.4f}")

print(f"\n  Welfare gap between most-favored and least-favored agent:")
u_means = [U_orig[:, i].mean().item() for i in range(A)]
print(f"    Max: a{np.argmax(u_means)} = {max(u_means):.4f}")
print(f"    Min: a{np.argmin(u_means)} = {min(u_means):.4f}")
print(f"    Gap: {max(u_means) - min(u_means):.4f}")


# ── pairwise swap: focus on a1 <-> a2 ─────────────────────────────────
print("\n" + "=" * 68)
print("  PAIRWISE SWAP DETAIL: a1 <-> a2")
print("=" * 68)

perm12 = [0, 2, 1]
v_perm = v[:, [0, 2, 1], :]
endow_alloc_perm = permute_alloc(allocs[endow_idx], perm12)
endow_idx_perm   = (endow_alloc_perm * powers).sum(1)
U_perm = all_utilities(cfg, v_perm)
mask_perm = ir_pe_mask(cfg, U_perm, endow_idx_perm)
with torch.no_grad():
    net_idx_swap = net.argmax_alloc(v_perm, mask=mask_perm)
alloc_swap    = allocs[net_idx_swap]
alloc_expected = permute_alloc(net_alloc, perm12)

# per-profile: does the swap commute?
equivariant = (alloc_swap == alloc_expected).all(1)   # [B]
not_equiv   = ~equivariant

print(f"\n  Profiles where swap commutes    : {equivariant.sum().item()} / {B} ({equivariant.float().mean()*100:.1f}%)")
print(f"  Profiles where swap breaks equiv: {not_equiv.sum().item()} / {B} ({not_equiv.float().mean()*100:.1f}%)")

if not_equiv.any():
    U_swap_chosen = U_perm[torch.arange(B), :, net_idx_swap]   # [B, A] in permuted labelling
    # original agent welfare under swap
    u_a1_orig  = U_orig[:, 1]       # a1's welfare under original input
    u_a2_orig  = U_orig[:, 2]       # a2's welfare under original input
    u_a1_swap  = U_swap_chosen[:, 2]  # a1 is now slot 2 in permuted world
    u_a2_swap  = U_swap_chosen[:, 1]  # a2 is now slot 1 in permuted world

    # Among non-equivariant profiles: who benefits from the swap?
    ne_idx = not_equiv.nonzero(as_tuple=True)[0]
    a1_gains = (u_a1_swap[ne_idx] - u_a1_orig[ne_idx]).mean().item()
    a2_gains = (u_a2_swap[ne_idx] - u_a2_orig[ne_idx]).mean().item()
    print(f"\n  Among non-equivariant profiles (a1<->a2 swap):")
    print(f"    a1 welfare change: {a1_gains:+.4f}")
    print(f"    a2 welfare change: {a2_gains:+.4f}")
    favored = "a1" if a1_gains > a2_gains else "a2"
    print(f"    => '{favored}' is favored in original labelling")
