from __future__ import annotations
import torch
from .config import Config
from .allocations import all_utilities, random_endowment


def sample_batch(cfg: Config) -> dict[str, torch.Tensor]:
    device = torch.device(cfg.device)
    B = cfg.batch_size
    v = torch.empty(B, cfg.num_agents, cfg.num_items, device=device).uniform_(cfg.v_min, cfg.v_max)
    endow_idx = random_endowment(cfg, B, device)
    U = all_utilities(cfg, v)
    return {"v": v, "endow_idx": endow_idx, "U": U}
