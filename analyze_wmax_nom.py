"""WMAX-IR-PE の NOM 2% 違反を詳細解析（メモリ節約版）"""
import sys; sys.path.insert(0, '.')
import torch
from nom_ir_pe_3x4.config import Config
from nom_ir_pe_3x4.allocations import (
    all_utilities, ir_pe_mask, ir_mask, pareto_mask,
    endowment_utilities, build_all_allocs, random_endowment
)
from nom_ir_pe_3x4.model import AllocationNet

cfg = Config()
A, m = cfg.num_agents, cfg.num_items
K = cfg.num_agents ** cfg.num_items

# ── WMAX-IR-PE をシンプルに実装（メモリ節約）────────────────────────────
def wmax_ir_pe_fn(v_, ei_, U_):
    msk = ir_pe_mask(cfg, U_, ei_)
    scores = U_.sum(1) + (1.0 - msk) * (-1e9)
    return scores.argmax(1)  # [B] index

# ── データ ────────────────────────────────────────────────────────────
torch.manual_seed(42)
B = 200
v = torch.rand(B, A, m)
endow_idx = random_endowment(cfg, B)
U = all_utilities(cfg, v)
wmax_ch = wmax_ir_pe_fn(v, endow_idx, U)

# ── LearnedNet ────────────────────────────────────────────────────────
ckpt = torch.load('alloc_net_3x4.pt', map_location='cpu')
net = AllocationNet(cfg); net.load_state_dict(ckpt['state_dict']); net.eval()
mask = ir_pe_mask(cfg, U, endow_idx)
with torch.no_grad():
    net_ch = net.argmax_alloc(v, mask=mask)

pe_m = pareto_mask(U)
ir_m = ir_mask(U, endow_idx)
print('=== LearnedNet ===')
print(f'IR  : {ir_m[torch.arange(B), net_ch].mean()*100:.2f}%')
print(f'PE  : {pe_m[torch.arange(B), net_ch].mean()*100:.2f}%')
print(f'IR∩PE: {mask[torch.arange(B), net_ch].mean()*100:.2f}%')
print()

# ── WMAX-IR-PE NOM 違反解析（1サンプルずつ、小さいS/M）─────────────────
S, M = 128, 128
violations = torch.zeros(B, A)
bc_truth_all = torch.zeros(B, A)
wc_truth_all = torch.zeros(B, A)
bc_lie_all   = torch.zeros(B, A)
wc_lie_all   = torch.zeros(B, A)
best_mis_all = torch.zeros(B, A, m)

for i in range(A):
    print(f'computing agent {i}...')
    for b in range(B):
        v_b = v[b:b+1]       # [1, A, m]
        ei_b = endow_idx[b:b+1]

        # S opponent samples
        v_opp = torch.rand(S, A, m)
        v_opp[:, i, :] = v_b[0, i, :].unsqueeze(0).expand(S, m)
        U_opp = all_utilities(cfg, v_opp)
        ei_rep = ei_b.expand(S)
        wmax_opp = wmax_ir_pe_fn(v_opp, ei_rep, U_opp)
        u_truth = U_opp[torch.arange(S), i, wmax_opp]  # [S]
        BC_t = u_truth.max().item()
        WC_t = u_truth.min().item()

        # M misreports
        v_mis = torch.rand(M, m)
        best_gain = -1.0
        best_bc_l = BC_t
        best_wc_l = WC_t
        best_mis_v = v_b[0, i, :]

        for ms in range(M):
            # For each misreport, use same S opponents
            v_full = v_opp.clone()      # [S, A, m]
            v_full[:, i, :] = v_mis[ms].unsqueeze(0).expand(S, m)
            U_full = all_utilities(cfg, v_full)
            wmax_ms = wmax_ir_pe_fn(v_full, ei_rep, U_full)

            # evaluate at TRUE v_i
            v_eval = v_full.clone()
            v_eval[:, i, :] = v_b[0, i, :].unsqueeze(0).expand(S, m)
            U_eval = all_utilities(cfg, v_eval)
            u_lie = U_eval[torch.arange(S), i, wmax_ms]  # [S]
            BC_l = u_lie.max().item()
            WC_l = u_lie.min().item()

            bc_g = max(0., BC_l - BC_t)
            wc_g = max(0., WC_l - WC_t)
            gain = min(bc_g, wc_g)
            if gain > best_gain:
                best_gain = gain
                best_bc_l = BC_l
                best_wc_l = WC_l
                best_mis_v = v_mis[ms]

        violations[b, i] = best_gain
        bc_truth_all[b, i] = BC_t
        wc_truth_all[b, i] = WC_t
        bc_lie_all[b, i]   = best_bc_l
        wc_lie_all[b, i]   = best_wc_l
        best_mis_all[b, i] = best_mis_v

viol_flag = violations.max(1).values > 1e-5
print(f'\nNOM違反: {viol_flag.sum().item()} / {B} ({viol_flag.float().mean()*100:.1f}%)')

# 違反例の詳細
viol_idx = viol_flag.nonzero(as_tuple=True)[0]
allocs = build_all_allocs(cfg)
print(f'\n=== 違反サンプル詳細 ===')
print('配分表: k -> [item0のエージェント, item1のエージェント, item2のエージェント, item3のエージェント]')
print()

for n, b in enumerate(viol_idx[:8]):
    b = b.item()
    ag = violations[b].argmax().item()
    e = endow_idx[b].item()
    alloc_e = allocs[e].tolist()
    alloc_w = allocs[wmax_ch[b]].tolist()

    print(f'[{n+1}] 違反エージェント=a{ag}')
    print(f'     endow  k={e:2d} {alloc_e}')
    print(f'     wmax出力 k={wmax_ch[b].item():2d} {alloc_w}')
    for aa in range(A):
        marker = ' ← 違反者' if aa == ag else ''
        print(f'     v[a{aa}]={[f"{x:.2f}" for x in v[b,aa].tolist()]}{marker}')
    print(f'     真の報告  : {[f"{x:.2f}" for x in v[b,ag].tolist()]}')
    print(f'     最適misrep: {[f"{x:.2f}" for x in best_mis_all[b,ag].tolist()]}')
    print(f'     BC: {bc_truth_all[b,ag]:.4f} → {bc_lie_all[b,ag]:.4f}  gain={bc_lie_all[b,ag]-bc_truth_all[b,ag]:.4f}')
    print(f'     WC: {wc_truth_all[b,ag]:.4f} → {wc_lie_all[b,ag]:.4f}  gain={wc_lie_all[b,ag]-wc_truth_all[b,ag]:.4f}')
    print(f'     obvious gain = {violations[b,ag].item():.5f}')
    print()

# 違反エージェントの分布
print('=== 違反エージェント分布 ===')
for i in range(A):
    n = (violations[:, i].max(0).values > 1e-5 if B==1 else
         (violations[:, i] > 1e-5)).sum().item()
    print(f'  agent {i}: {n}件')

# ── LearnedNet argmax NOM 違反解析 ──────────────────────────────────────
def net_argmax_fn(v_, ei_, U_):
    msk = ir_pe_mask(cfg, U_, ei_)
    return net.argmax_alloc(v_, mask=msk)

print()
print('=== LearnedNet argmax NOM 違反解析 ===')
net_violations    = torch.zeros(B, A)
net_bc_truth_all  = torch.zeros(B, A)
net_wc_truth_all  = torch.zeros(B, A)
net_bc_lie_all    = torch.zeros(B, A)
net_wc_lie_all    = torch.zeros(B, A)
net_best_mis_all  = torch.zeros(B, A, m)

for i in range(A):
    print(f'computing agent {i}...')
    for b in range(B):
        v_b = v[b:b+1]
        ei_b = endow_idx[b:b+1]

        v_opp = torch.rand(S, A, m)
        v_opp[:, i, :] = v_b[0, i, :].unsqueeze(0).expand(S, m)
        U_opp = all_utilities(cfg, v_opp)
        ei_rep = ei_b.expand(S)
        net_opp = net_argmax_fn(v_opp, ei_rep, U_opp)
        u_truth = U_opp[torch.arange(S), i, net_opp]
        BC_t = u_truth.max().item()
        WC_t = u_truth.min().item()

        v_mis = torch.rand(M, m)
        best_gain  = -1.0
        best_bc_l  = BC_t
        best_wc_l  = WC_t
        best_mis_v = v_b[0, i, :]

        for ms in range(M):
            v_full = v_opp.clone()
            v_full[:, i, :] = v_mis[ms].unsqueeze(0).expand(S, m)
            U_full = all_utilities(cfg, v_full)
            net_ms = net_argmax_fn(v_full, ei_rep, U_full)

            v_eval = v_full.clone()
            v_eval[:, i, :] = v_b[0, i, :].unsqueeze(0).expand(S, m)
            U_eval = all_utilities(cfg, v_eval)
            u_lie = U_eval[torch.arange(S), i, net_ms]
            BC_l = u_lie.max().item()
            WC_l = u_lie.min().item()

            bc_g = max(0., BC_l - BC_t)
            wc_g = max(0., WC_l - WC_t)
            gain = min(bc_g, wc_g)
            if gain > best_gain:
                best_gain  = gain
                best_bc_l  = BC_l
                best_wc_l  = WC_l
                best_mis_v = v_mis[ms]

        net_violations[b, i]    = best_gain
        net_bc_truth_all[b, i]  = BC_t
        net_wc_truth_all[b, i]  = WC_t
        net_bc_lie_all[b, i]    = best_bc_l
        net_wc_lie_all[b, i]    = best_wc_l
        net_best_mis_all[b, i]  = best_mis_v

net_viol_flag = net_violations.max(1).values > 1e-5
print(f'\nLearnedNet argmax NOM違反: {net_viol_flag.sum().item()} / {B} ({net_viol_flag.float().mean()*100:.1f}%)')

# 違反例の詳細
net_viol_idx = net_viol_flag.nonzero(as_tuple=True)[0]
print(f'\n=== LearnedNet argmax 違反サンプル詳細 ===')
print('配分表: k -> [item0のエージェント, item1のエージェント, item2のエージェント, item3のエージェント]')
print()

for n, b in enumerate(net_viol_idx[:8]):
    b = b.item()
    ag = net_violations[b].argmax().item()
    e = endow_idx[b].item()
    alloc_e = allocs[e].tolist()
    alloc_n = allocs[net_ch[b]].tolist()

    print(f'[{n+1}] 違反エージェント=a{ag}')
    print(f'     endow    k={e:2d} {alloc_e}')
    print(f'     net出力  k={net_ch[b].item():2d} {alloc_n}')
    for aa in range(A):
        marker = ' ← 違反者' if aa == ag else ''
        print(f'     v[a{aa}]={[f"{x:.2f}" for x in v[b,aa].tolist()]}{marker}')
    print(f'     真の報告  : {[f"{x:.2f}" for x in v[b,ag].tolist()]}')
    print(f'     最適misrep: {[f"{x:.2f}" for x in net_best_mis_all[b,ag].tolist()]}')
    print(f'     BC: {net_bc_truth_all[b,ag]:.4f} → {net_bc_lie_all[b,ag]:.4f}  gain={net_bc_lie_all[b,ag]-net_bc_truth_all[b,ag]:.4f}')
    print(f'     WC: {net_wc_truth_all[b,ag]:.4f} → {net_wc_lie_all[b,ag]:.4f}  gain={net_wc_lie_all[b,ag]-net_wc_truth_all[b,ag]:.4f}')
    print(f'     obvious gain = {net_violations[b,ag].item():.5f}')
    print()

print('=== LearnedNet argmax 違反エージェント分布 ===')
for i in range(A):
    n = (net_violations[:, i] > 1e-5).sum().item()
    print(f'  agent {i}: {n}件')
