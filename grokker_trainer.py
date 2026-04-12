from typing import Dict, List, Literal, Tuple
import os
from datetime import datetime

from trainer_config import TrainerConfig
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader import build_grokking_dataloaders
from grokker_og import TransformerTorch
from transformer_model import GrokModularModel
from mlp_model import GrokMLP
from custom_optimizer import AdamCustom

torch.set_default_dtype(torch.float64)
class GrokkerTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device(config.device)
        log_dir = os.path.join(config.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.train_writer = SummaryWriter(log_dir=os.path.join(log_dir,  "train"))
        self.val_writer = SummaryWriter(log_dir=os.path.join(log_dir, "val"))

        self.train_loader, self.test_loader = build_grokking_dataloaders(
            p=config.p,
            op=config.op,
            train_fraction=config.train_fraction,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        self.model = None

        if config.model == "transformer":
            self.model = GrokModularModel(
                vocab_size=config.vocab_size,
                num_layers=config.num_layers,
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                context_size=config.context_size,
            ).to(self.device)

        # update later if we want variable layers
        elif config.model == "mlp":
            self.model = GrokMLP(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
            ).to(self.device)


        # self.model = TransformerTorch(
        #     depth=config.num_layers,
        #     dim=config.embed_dim,
        #     heads=config.num_heads,
        #     n_tokens=config.vocab_size,
        #     seq_len=config.context_size,
        #     dropout=config.dropout,
        # ).to(self.device)



        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=config.lr,
        #     weight_decay=config.weight_decay,
        #     momentum=config.momentum,
        # )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        self.loss_fn = F.cross_entropy


    def train_epoch(self) -> Dict[str, float]:
        self.model.train()

        running_total = 0.0
        running_ce = 0.0
        running_mse = 0.0
        total_correct = 0
        total_seen = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

          # Convert input to float for MLP
            # x = x.float()
            logits_last = self.model(x)
            # print(logits_last.shape)
            # logits_last = logits[:, -1, :]
            total_loss = self.loss_fn(logits_last, y)

            preds = torch.argmax(logits_last, dim=-1)
            correct = int((preds == y).sum().item())
            total = int(y.numel())

            metrics = {
                "total_loss": float(total_loss.detach().item()),
                "ce_loss": float(total_loss.detach().item()),
            }
            total_loss.backward()
            self.optimizer.step()

            running_total += metrics["total_loss"]
            running_ce += metrics["ce_loss"]
            total_correct += correct
            total_seen += total

        n_batches = len(self.train_loader)
        return {
            "loss": running_total / n_batches,
            "ce_loss": running_ce / n_batches,
            "accuracy": total_correct / total_seen,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()

        running_total = 0.0
        running_ce = 0.0
        running_mse = 0.0
        total_correct = 0
        total_seen = 0

        for batch in self.test_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            # x = x.float()
            logits_last = self.model(x)
            total_loss = self.loss_fn(logits_last, y)

            preds = torch.argmax(logits_last, dim=-1)
            correct = int((preds == y).sum().item())
            total = int(y.numel())

            metrics = {
                "total_loss": float(total_loss.detach().item()),
                "ce_loss": float(total_loss.detach().item()),
            }
            running_total += metrics["total_loss"]
            running_ce += metrics["ce_loss"]
            total_correct += correct
            total_seen += total

        n_batches = len(self.test_loader)
        return {
            "loss": running_total / n_batches,
            "ce_loss": running_ce / n_batches,
            "accuracy": total_correct / total_seen,
        }

    def fit(self) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        try:
            for epoch in range(1, self.config.epochs + 1):
                train_metrics = self.train_epoch()
                val_metrics = self.evaluate()

                row = {
                    "epoch": float(epoch),
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "train_ce": train_metrics["ce_loss"],
                    "val_ce": val_metrics["ce_loss"],
                }
                history.append(row)

                self.train_writer.add_scalar("Loss", row["train_loss"], epoch)
                self.train_writer.add_scalar("Accuracy", row["train_acc"], epoch)
                self.train_writer.add_scalar("CrossEntropy", row["train_ce"], epoch)

                self.val_writer.add_scalar("Loss", row["val_loss"], epoch)
                self.val_writer.add_scalar("Accuracy", row["val_acc"], epoch)
                self.val_writer.add_scalar("CrossEntropy", row["val_ce"], epoch)

                print(
                    f"Epoch {epoch:03d}/{self.config.epochs} | "
                    f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} | "
                    f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f}"
                )
        finally:
            self.train_writer.flush()
            self.val_writer.flush()
            self.train_writer.close()
            self.val_writer.close()

        return history


def train_grokker(config: TrainerConfig | None = None):
    cfg = config or TrainerConfig()
    trainer = GrokkerTrainer(cfg)
    history = trainer.fit()
    return trainer.model, history


if __name__ == "__main__":
    default_config = TrainerConfig()
    train_grokker(default_config)
