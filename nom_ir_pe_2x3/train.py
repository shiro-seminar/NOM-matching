"""Training script: NOM + IR + PE mechanism learning.

Usage:
    python -m nom_ir_pe_2x3.train                    # full run
    python -m nom_ir_pe_2x3.train --steps 200        # quick smoke test

The mechanism:
  - IR and PE are hard constraints enforced via logit masking.
  - NOM is minimized via Augmented Lagrangian.
  - Welfare is maximized as a soft regularizer (weight=welfare_weight).

Checkpoints are saved every 10,000 steps as allocation_net_step{N}.pt.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from .config import Config
from .allocations import all_utilities, ir_pe_mask
from .data_gen import sample_batch
from .model import AllocationNet
from .losses import augmented_objective


def train(cfg: Config | None = None) -> AllocationNet:
    if cfg is None:
        cfg = Config()

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    net = AllocationNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        batch = sample_batch(cfg)
        v         = batch["v"]           # [B, 2, 3]
        endow_idx = batch["endow_idx"]   # [B]
        U         = batch["U"]           # [B, 2, 8]

        # Hard IR ∩ PE mask
        mask = ir_pe_mask(U, endow_idx)  # [B, 8]

        # Augmented Lagrangian objective
        loss, stats = augmented_objective(cfg, net, v, endow_idx, U, mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()

        # ── Dual update (Augmented Lagrangian) ────────────────────────────────
        if step % cfg.dual_update_every == 0:
            nom_val = stats["nom"]
            cfg.lambda_nom = max(0.0, cfg.lambda_nom + cfg.rho * nom_val)
            if nom_val > cfg.nom_target:
                cfg.rho = min(cfg.rho * cfg.rho_mult, cfg.rho_max)

        # ── Logging ───────────────────────────────────────────────────────────
        if step % 200 == 0 or step == 1:
            elapsed = time.time() - t0
            print(
                f"step={step:6d}  loss={stats['loss']:+.4f}  "
                f"welfare={stats['welfare']:.4f}  "
                f"nom={stats['nom']:.5f}  "
                f"λ_nom={cfg.lambda_nom:.3f}  ρ={cfg.rho:.2f}  "
                f"elapsed={elapsed:.0f}s"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % 10_000 == 0:
            path = f"allocation_net_step{step}.pt"
            torch.save({"state_dict": net.state_dict(), "cfg": cfg.__dict__, "step": step}, path)
            print(f"[ckpt] saved {path}")

    # Final save
    path = "allocation_net.pt"
    torch.save({"state_dict": net.state_dict(), "cfg": cfg.__dict__, "step": cfg.steps}, path)
    print(f"[done] saved {path}")
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",    type=int,   default=None)
    parser.add_argument("--lr",       type=float, default=None)
    parser.add_argument("--hidden",   type=int,   default=None)
    parser.add_argument("--depth",    type=int,   default=None)
    parser.add_argument("--S",        type=int,   default=None, help="opponent samples for NOM")
    parser.add_argument("--M",        type=int,   default=None, help="misreport samples for NOM")
    parser.add_argument("--batch",    type=int,   default=None)
    parser.add_argument("--device",   type=str,   default=None)
    parser.add_argument("--seed",     type=int,   default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.steps  is not None: cfg.steps      = args.steps
    if args.lr     is not None: cfg.lr         = args.lr
    if args.hidden is not None: cfg.hidden     = args.hidden
    if args.depth  is not None: cfg.depth      = args.depth
    if args.S      is not None: cfg.S          = args.S
    if args.M      is not None: cfg.M          = args.M
    if args.batch  is not None: cfg.batch_size = args.batch
    if args.device is not None: cfg.device     = args.device
    if args.seed   is not None: cfg.seed       = args.seed

    train(cfg)


if __name__ == "__main__":
    main()
