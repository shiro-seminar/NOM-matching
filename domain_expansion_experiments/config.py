from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Problem ───────────────────────────────────────────────
    num_agents: int = 3
    num_items:  int = 4      # K = 3^4 = 81 allocations

    # ── Ordinal preference domain ─────────────────────────────
    domain: str = "trichotomous"
    # num_ranks is set automatically from domain in __post_init__
    num_ranks: int = 3

    # ── Model ─────────────────────────────────────────────────
    hidden: int = 256
    depth:  int = 4

    # ── Training ──────────────────────────────────────────────
    batch_size: int = 64
    steps:      int = 1_000
    lr:         float = 3e-4
    grad_clip:  float = 1.0
    seed:       int = 42

    # ── NOM sampling ─────────────────────────────────────────
    S: int = 4    # opponent profiles for BC/WC estimation
    M: int = 4    # misreport samples

    # ── Temperature ──────────────────────────────────────────
    temperature: float = 1.0

    # ── Augmented Lagrangian ──────────────────────────────────
    lambda_nom:        float = 0.0
    rho:               float = 1.0
    rho_mult:          float = 1.005
    rho_max:           float = 200.0
    dual_update_every: int   = 100
    nom_target:        float = 5e-3

    welfare_weight: float = 0.02

    # ── Device ────────────────────────────────────────────────
    device: str = "cpu"

    def __post_init__(self) -> None:
        from .domains import DOMAINS
        if self.domain not in DOMAINS:
            raise ValueError(f"Unknown domain '{self.domain}'. "
                             f"Available: {list(DOMAINS.keys())}")
        self.num_ranks = DOMAINS[self.domain].num_ranks
