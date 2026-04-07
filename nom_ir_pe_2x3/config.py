from dataclasses import dataclass


@dataclass
class Config:
    # ── Problem ──────────────────────────────────────────────
    num_agents: int = 2
    num_items: int = 3      # K = 2^3 = 8 allocations

    # ── Types ────────────────────────────────────────────────
    v_min: float = 0.0
    v_max: float = 1.0

    # ── Model ────────────────────────────────────────────────
    hidden: int = 128
    depth: int = 4          # MLP layers

    # ── Training ─────────────────────────────────────────────
    batch_size: int = 128
    steps: int = 30_000
    lr: float = 3e-4
    grad_clip: float = 1.0
    seed: int = 42

    # ── NOM sampling (training) ──────────────────────────────
    # S: opponent samples for BC/WC estimation
    # M: misreport samples per training step
    S: int = 4
    M: int = 4

    # ── NOM sampling (evaluation) ────────────────────────────
    S_eval: int = 128
    M_eval: int = 128

    # ── Temperature (softmax) ────────────────────────────────
    temperature: float = 1.0

    # ── Augmented Lagrangian for NOM ─────────────────────────
    lambda_nom: float = 0.0
    rho: float = 1.0
    rho_mult: float = 1.005
    rho_max: float = 200.0
    dual_update_every: int = 100
    nom_target: float = 5e-3   # target NOM violation threshold

    # ── Secondary welfare weight ─────────────────────────────
    # Tiebreak toward higher welfare when NOM ≈ 0
    welfare_weight: float = 0.02

    # ── Device ───────────────────────────────────────────────
    device: str = "cpu"
