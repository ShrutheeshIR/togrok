from dataclasses import dataclass
from typing import Literal

import torch
import json

import argparse

LossType = Literal["cross_entropy", "mse", "mse_cross_entropy"]


def build_argparse_for_config():
    parser = argparse.ArgumentParser(description="Train a Grokker model with specified configuration.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["transformer", "mlp"],
        default="mlp",
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16384,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for the model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=2e-2,
        help="Weight decay (L2 regularization) coefficient.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam"],
        default="sgd",
        help="Optimizer to use for training.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for the optimizer.",
    )

    return parser

@dataclass
class TrainerConfig:
    model: Literal["transformer", "mlp"] = "transformer"
    p: int = 97
    op: str = "/"
    train_fraction: float = 0.5
    batch_size: int = 16384
    seed: int = 42
    num_workers: int = 4
    dropout: float = 0.1

    num_layers: int = 2
    embed_dim: int = 256
    num_heads: int = 2
    context_size: int = 3

    lr: float = 1e-2
    weight_decay: float = 2e-2
    momentum: float = 0.9
    epochs: int = 100000
    beta1: float = 0.9
    beta2: float = 0.98
    log_dir: str = "experiments_scripting_adam_transformer/logs_v2"

    optimizer: Literal["sgd", "adam"] = "adam"

    loss_type: LossType = "cross_entropy"
    ce_weight: float = 1.0
    mse_weight: float = 1.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def vocab_size(self) -> int:
        # Values are in [0, p-1] plus operator and '=' tokens.
        return self.p + 2


    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=4)

    def to_json_file(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)


# last expt - 20260416-164901 - 1e-2 + 0 weight decay
# 2nd last expt - 20260416-164736 - 1e-3 + 0 weight decay
# 3rd last expt - 20260416-164507 - 1e-3 + 2e-3 weight decay
# 4th last expt - 20260416-162728 - 1e-2 + 2e-3 weight decay
# 20260416-180521 - 1e-2 + 0.5 weight decay
# 20260416-180614 - 1e-2 + 2e-2 weight decay
# 20260416-184351 - 1e-2 + 2e-1 weight decay
# 20260416-184604 - 1e-3 + 2e-2 weight decay
# Go through this - https://www.lesswrong.com/posts/Fjoy5SxgBmxfy7FNB/gradient-surfing-the-hidden-role-of-regularization

# simulated annealing
# increase batch size as reg
# try to fit the whole thing into mem
# weight norm to be constant
# try dropout