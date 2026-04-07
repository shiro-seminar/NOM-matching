"""Training loop with Augmented Lagrangian for NOM or SP constraint.

Usage:
  python -m nom_ir_pe.train --loss_type nom --steps 50000
  python -m nom_ir_pe.train --loss_type sp  --steps 50000
  python -m nom_ir_pe.train --num_agents 2 --num_items 3 --steps 200   # smoke test
"""
from __future__ import annotations

import argparse
import torch
import torch.nn as nn

from .config import Config
from .allocations import AllocationIndex
from .data_gen import sample_types, compute_all_utilities
from .model import AllocationNet
from .losses import augmented_loss, compute_irpe_mask


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=None)
    parser.add_argument("--num_items", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--loss_type", type=str, default=None, choices=["nom", "sp"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)

    aidx = AllocationIndex(num_agents=cfg.num_agents, num_items=cfg.num_items)
    K = aidx.num_allocations
    device = torch.device(cfg.device)

    print(f"=== NOM+IR+PE Training ===")
    print(f"  agents={cfg.num_agents}, items={cfg.num_items}, K={K}")
    print(f"  loss_type={cfg.loss_type}, steps={cfg.steps}")
    print(f"  batch={cfg.batch_size}, lr={cfg.lr}, hidden={cfg.hidden}, depth={cfg.depth}")
    print()

    net = AllocationNet(cfg, aidx).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    milestones = list(cfg.lr_milestones)
    scheduler = None
    if milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=milestones, gamma=cfg.lr_gamma,
        )

    for step in range(1, cfg.steps + 1):
        # Dynamic misreport samples (scale up over training)
        if cfg.loss_type == "sp":
            if step < cfg.steps * 0.4:
                cfg.misreport_samples = 16
            elif step < cfg.steps * 0.8:
                cfg.misreport_samples = 32
            else:
                cfg.misreport_samples = 64
        else:
            # NOM: scale opponent samples
            if step < cfg.steps * 0.4:
                cfg.nom_opponent_samples = 32
                cfg.nom_misreport_samples = 16
            elif step < cfg.steps * 0.8:
                cfg.nom_opponent_samples = 64
                cfg.nom_misreport_samples = 32
            else:
                cfg.nom_opponent_samples = 96
                cfg.nom_misreport_samples = 48

        # Temperature annealing in last 10% of training
        anneal_start = int(cfg.steps * 0.9)
        if step >= anneal_start:
            progress = (step - anneal_start) / (cfg.steps - anneal_start)
            cfg.temperature = 1.0 - progress * 0.95  # 1.0 → 0.05
        else:
            cfg.temperature = 1.0

        # Sample batch
        batch = sample_types(cfg, aidx, cfg.batch_size)
        v_true = batch["v_true"]
        endow_idx = batch["endow_idx"]
        U_true = compute_all_utilities(v_true, aidx)

        # Compute loss
        loss, stats = augmented_loss(cfg, aidx, net, v_true, U_true, endow_idx)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()

        if scheduler is not None:
            scheduler.step()

        # Dual variable update
        if step % cfg.dual_update_every == 0:
            c_val = stats["constraint"]
            cfg.lambda_constraint = max(
                0.0, cfg.lambda_constraint + cfg.rho * c_val,
            )
            if c_val > cfg.constraint_target:
                cfg.rho = min(cfg.rho * cfg.rho_mult, cfg.rho_max)

        # Logging
        if step % 200 == 0 or step == 1:
            # Oracle welfare (unconstrained)
            with torch.no_grad():
                oracle = U_true.sum(dim=1).max(dim=1).values.mean().item()
                gap = oracle - stats["welfare"]
            print(
                f"step={step:5d}  tau={cfg.temperature:.3f}  "
                f"loss={stats['loss']:.4f}  welfare={stats['welfare']:.4f}  "
                f"oracle={oracle:.4f}  gap={gap:.4f}  "
                f"{cfg.loss_type.upper()}={stats['constraint']:.4f}  "
                f"λ={cfg.lambda_constraint:.2f}  ρ={cfg.rho:.2f}"
            )

        # Diagnostic sample
        if step % 1000 == 0 or step == 1:
            _print_diagnostic(cfg, aidx, net, v_true, U_true, endow_idx, step)

        # Checkpoint
        if step % 5000 == 0:
            ckpt = f"nom_ir_pe_{cfg.loss_type}_step{step}.pt"
            torch.save({
                "state_dict": net.state_dict(),
                "cfg": cfg.__dict__,
                "step": step,
                "optimizer": opt.state_dict(),
            }, ckpt)
            print(f"[Checkpoint] {ckpt}")

    # Final save
    final_path = f"nom_ir_pe_{cfg.loss_type}_final.pt"
    torch.save({"state_dict": net.state_dict(), "cfg": cfg.__dict__}, final_path)
    print(f"Saved final model to {final_path}")


def _print_diagnostic(cfg, aidx, net, v_true, U_true, endow_idx, step):
    """Print detailed diagnostics for sample 0."""
    with torch.no_grad():
        A, m = cfg.num_agents, cfg.num_items
        mask = compute_irpe_mask(cfg, aidx, U_true, endow_idx)

        endow_masks = aidx.allocation_to_agent_masks(endow_idx)
        alloc_idx = net.predict_argmax(v_true, endow_idx, mask=mask)
        alloc_masks = aidx.allocation_to_agent_masks(alloc_idx)

        endow_u = U_true[0, :, endow_idx[0].long()]
        alloc_u = U_true[0, :, alloc_idx[0].long()]

        oracle_idx = U_true[0].sum(dim=0).argmax()
        oracle_u = U_true[0, :, oracle_idx.long()]

        print("=" * 60)
        print(f"  [Diag] step={step}  (sample 0)")
        for i in range(A):
            v_str = ", ".join(f"{v_true[0,i,j]:.3f}" for j in range(m))
            print(f"    Agent {i}: v=[{v_str}]")

        print(f"\n  Endowment (idx={endow_idx[0].item()}):")
        for i in range(A):
            items = [j for j in range(m) if endow_masks[0, i, j] > 0.5]
            print(f"    Agent {i}: items={items}  u={endow_u[i]:.4f}")

        print(f"\n  Allocation (idx={alloc_idx[0].item()}):")
        for i in range(A):
            items = [j for j in range(m) if alloc_masks[0, i, j] > 0.5]
            print(f"    Agent {i}: items={items}  u={alloc_u[i]:.4f}")

        ir_ok = ["OK" if alloc_u[i] >= endow_u[i] - 1e-5 else "VIOL" for i in range(A)]
        print(f"\n  IR: {', '.join(f'Agent {i}={s}' for i,s in enumerate(ir_ok))}")
        print(f"  Welfare: endow={endow_u.sum():.4f}  alloc={alloc_u.sum():.4f}  oracle={oracle_u.sum():.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
