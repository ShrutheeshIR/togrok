from dataclasses import dataclass
from typing import Literal

import torch

LossType = Literal["cross_entropy", "mse", "mse_cross_entropy"]


@dataclass
class TrainerConfig:
    MOD: int = 113
    op: str = "/"
    train_fraction: float = 0.5
    batch_size: int = 1024
    seed: int = 0
    num_workers: int = 0

    num_layers: int = 1
    embed_dim: int = 512
    num_heads: int = 8
    context_size: int = 4

    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    epochs: int = 20

    loss_type: LossType = "cross_entropy"
    ce_weight: float = 1.0
    mse_weight: float = 1.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def vocab_size(self) -> int:
        # Values are in [0, p-1] plus operator and '=' tokens.
        return self.MOD + 2
