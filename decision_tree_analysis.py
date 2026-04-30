"""Decision tree analysis of phi_ML allocation decisions.

Features extracted from valuation profiles:
  - Raw valuations v[agent][item]
  - Endowment structure (who owns each item)
  - Per-item: argmax agent, top-2 gap
  - Per-agent: max/2nd-max/mean valuation, endowment utility

Targets:
  - For each item j: which agent receives it (3-class, trained separately)
  - Full allocation index k (81-class, for reference)
"""
import sys; sys.path.insert(0, '.')
import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report

from nom_ir_pe_3x4.config import Config
from nom_ir_pe_3x4.allocations import (
    all_utilities, ir_pe_mask, build_all_allocs, random_endowment,
)
from nom_ir_pe_3x4.model import AllocationNet

# ── data ──────────────────────────────────────────────────────────────
cfg = Config()
torch.manual_seed(0)
B = 1000
A, m = cfg.num_agents, cfg.num_items

v         = torch.rand(B, A, m)
endow_idx = random_endowment(cfg, B)
U         = all_utilities(cfg, v)
mask      = ir_pe_mask(cfg, U, endow_idx)
allocs    = build_all_allocs(cfg)   # [K, m]

ckpt = torch.load('alloc_net_3x4.pt', map_location='cpu')
net  = AllocationNet(cfg)
net.load_state_dict(ckpt['state_dict'])
net.eval()
with torch.no_grad():
    net_idx = net.argmax_alloc(v, mask=mask)   # [B]

net_alloc   = allocs[net_idx].numpy()          # [B, m]  int
endow_alloc = allocs[endow_idx].numpy()        # [B, m]  int
v_np        = v.numpy()                        # [B, A, m]

# ── feature engineering ────────────────────────────────────────────────
feat_list  = []
feat_names = []

# 1. raw valuations v[i][j]  (12 features)
for i in range(A):
    for j in range(m):
        feat_list.append(v_np[:, i, j])
        feat_names.append(f"v[a{i}][o{j}]")

# 2. endowment: agent owning each item  (4 features, categorical 0/1/2)
for j in range(m):
    feat_list.append(endow_alloc[:, j].astype(float))
    feat_names.append(f"endow_owner[o{j}]")

# 3. per-item: highest-valuing agent  (4 features)
for j in range(m):
    feat_list.append(v_np[:, :, j].argmax(axis=1).astype(float))
    feat_names.append(f"argmax_agent[o{j}]")

# 4. per-item: valuation of best agent  (4 features)
for j in range(m):
    feat_list.append(v_np[:, :, j].max(axis=1))
    feat_names.append(f"max_val[o{j}]")

# 5. per-item: gap between top-2 agents  (4 features)
for j in range(m):
    sorted_v = np.sort(v_np[:, :, j], axis=1)[:, ::-1]
    feat_list.append(sorted_v[:, 0] - sorted_v[:, 1])
    feat_names.append(f"gap_top2[o{j}]")

# 6. per-agent: max / 2nd-max valuation  (6 features)
for i in range(A):
    sorted_v = np.sort(v_np[:, i, :], axis=1)[:, ::-1]
    feat_list.append(sorted_v[:, 0])
    feat_names.append(f"max_v[a{i}]")
    feat_list.append(sorted_v[:, 1])
    feat_names.append(f"2nd_max_v[a{i}]")

# 7. per-agent: endowment utility = sum of v[i][j] for owned items  (3 features)
U_np = U.numpy()   # [B, A, K]
for i in range(A):
    u_end = U_np[np.arange(B), i, endow_idx.numpy()]
    feat_list.append(u_end)
    feat_names.append(f"endow_util[a{i}]")

# 8. per-agent: best item NOT in endowment  (3 features)
for i in range(A):
    best_non_endow = []
    for b in range(B):
        non_endow_items = [j for j in range(m) if endow_alloc[b, j] != i]
        best = max((v_np[b, i, j] for j in non_endow_items), default=0.0)
        best_non_endow.append(best)
    feat_list.append(np.array(best_non_endow))
    feat_names.append(f"best_non_endow_v[a{i}]")

X = np.column_stack(feat_list)   # [B, n_features]
print(f"Feature matrix: {X.shape}  ({len(feat_names)} features)")
print(f"Features: {feat_names}\n")


# ── Decision tree: per-item allocation  ───────────────────────────────
print("=" * 65)
print("  DECISION TREES: who receives each item?")
print("=" * 65)

per_item_acc = []
for j in range(m):
    y = net_alloc[:, j]   # [B]  which agent gets item j
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=0)
    clf.fit(X, y)
    acc = clf.score(X, y)
    per_item_acc.append(acc)

    print(f"\n{'─'*65}")
    print(f"  Item o{j}  |  accuracy={acc*100:.1f}%  |  "
          f"class dist: " +
          "  ".join(f"a{k}:{(y==k).mean()*100:.0f}%" for k in range(A)))
    print(f"{'─'*65}")
    print(export_text(clf, feature_names=feat_names, max_depth=4,
                      spacing=3, decimals=3, show_weights=True))

    # feature importances (top 5)
    imp = clf.feature_importances_
    top5 = np.argsort(imp)[::-1][:5]
    print(f"  Top-5 features:")
    for rank, fi in enumerate(top5, 1):
        print(f"    {rank}. {feat_names[fi]:<28} importance={imp[fi]:.4f}")


# ── Decision tree: full allocation index k  ───────────────────────────
print("\n" + "=" * 65)
print("  DECISION TREE: full allocation index k (81 classes)")
print("=" * 65)

y_k = net_idx.numpy()
clf_k = DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=0)
clf_k.fit(X, y_k)
acc_k = clf_k.score(X, y_k)
n_leaves = clf_k.get_n_leaves()
print(f"\n  Accuracy: {acc_k*100:.1f}%  |  Leaves: {n_leaves}  |  "
      f"Unique k in data: {len(np.unique(y_k))}")
print(f"\n  Top-10 feature importances:")
imp_k = clf_k.feature_importances_
top10 = np.argsort(imp_k)[::-1][:10]
for rank, fi in enumerate(top10, 1):
    print(f"    {rank:2d}. {feat_names[fi]:<28} importance={imp_k[fi]:.4f}")

print(f"\n  Tree (depth<=5):")
print(export_text(clf_k, feature_names=feat_names, max_depth=5,
                  spacing=3, decimals=3, show_weights=True))


# ── Global feature importance summary  ────────────────────────────────
print("=" * 65)
print("  GLOBAL FEATURE IMPORTANCE (avg across 4 per-item trees)")
print("=" * 65)
all_imp = np.zeros(len(feat_names))
for j in range(m):
    y = net_alloc[:, j]
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=0)
    clf.fit(X, y)
    all_imp += clf.feature_importances_
all_imp /= m
order = np.argsort(all_imp)[::-1]
print()
for rank, fi in enumerate(order[:15], 1):
    bar = '#' * int(all_imp[fi] * 200)
    print(f"  {rank:2d}. {feat_names[fi]:<28}  {all_imp[fi]:.4f}  {bar}")
