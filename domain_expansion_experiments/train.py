from __future__ import annotations
import time
import torch
import torch.nn as nn
from .config import Config
from .domains import DOMAINS
from .allocations import ir_pe_mask
from .data_gen import sample_batch
from .model import AllocationNet
from .losses import augmented_objective


def train(cfg: Config | None = None, verbose: bool = True) -> AllocationNet:
    if cfg is None:
        cfg = Config()
    torch.manual_seed(cfg.seed)
    domain = DOMAINS[cfg.domain]
    device = torch.device(cfg.device)

    net = AllocationNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    t0  = time.time()

    for step in range(1, cfg.steps + 1):
        batch         = sample_batch(cfg)
        marginal_rank = batch["marginal_rank"]
        endow_idx     = batch["endow_idx"]
        S             = batch["S"]
        mask          = ir_pe_mask(cfg, S, endow_idx)

        loss, stats = augmented_objective(cfg, domain, net,
                                          marginal_rank, endow_idx, S, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.dual_update_every == 0:
            nom_val = stats["nom"]
            cfg.lambda_nom = max(0.0, cfg.lambda_nom + cfg.rho * nom_val)
            if nom_val > cfg.nom_target:
                cfg.rho = min(cfg.rho * cfg.rho_mult, cfg.rho_max)

        if verbose and (step % 200 == 0 or step == 1):
            print(
                f"  step={step:6d}  welfare={stats['welfare']:.4f}  "
                f"nom={stats['nom']:.5f}  lambda={cfg.lambda_nom:.3f}  "
                f"rho={cfg.rho:.2f}  elapsed={time.time()-t0:.0f}s"
            )

    return net


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",  type=str, default="trichotomous")
    parser.add_argument("--steps",   type=int, default=None)
    parser.add_argument("--S",       type=int, default=None)
    parser.add_argument("--M",       type=int, default=None)
    parser.add_argument("--batch",   type=int, default=None)
    parser.add_argument("--device",  type=str, default=None)
    parser.add_argument("--out",     type=str, default=None)
    args = parser.parse_args()

    cfg = Config(domain=args.domain)
    if args.steps:  cfg.steps      = args.steps
    if args.S:      cfg.S          = args.S
    if args.M:      cfg.M          = args.M
    if args.batch:  cfg.batch_size = args.batch
    if args.device: cfg.device     = args.device

    print(f"Training domain={cfg.domain}  num_ranks={cfg.num_ranks}  steps={cfg.steps}")
    net = train(cfg)

    out = args.out or f"ordinal_net_{cfg.domain}.pt"
    torch.save({"state_dict": net.state_dict(), "cfg": cfg.__dict__}, out)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
