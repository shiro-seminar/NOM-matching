from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ---- Problem size (parametric) ----
    num_agents: int = 3
    num_items: int = 4

    # ---- Utility model (pure additive) ----
    v_min: float = 0.0
    v_max: float = 1.0

    # ---- Endowment ----
    random_endowment: bool = True

    # ---- Model ----
    hidden: int = 128
    depth: int = 3
    dropout: float = 0.0

    # ---- Training ----
    batch_size: int = 512
    steps: int = 50000
    lr: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 0

    # ---- Training mode ----
    soft_training_only: bool = True
    temperature: float = 1.0

    # ---- LR scheduler ----
    lr_milestones: Tuple[int, int] = (35000, 45000)
    lr_gamma: float = 0.5

    # ---- Loss type: "nom" or "sp" ----
    loss_type: str = "nom"

    # ---- Loss weights (Augmented Lagrangian) ----
    lambda_constraint: float = 0.0   # dual variable for NOM or SP
    rho: float = 5.0

    # ---- AL update knobs ----
    dual_update_every: int = 200
    rho_mult: float = 1.01
    rho_max: float = 100.0
    constraint_target: float = 0.10

    # ---- SP approximation ----
    misreport_samples: int = 64
    misreport_noise_v: float = 0.35

    # ---- NOM approximation ----
    nom_opponent_samples: int = 64   # S: number of opponent-type samples
    nom_misreport_samples: int = 32  # M: misreport samples per agent

    # ---- Device ----
    device: str = "cpu"
