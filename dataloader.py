import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from trainer_config import TrainerConfig


class GrokkingModularDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same length")
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def ground_truth_data_generator(p: int, op: str, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
    operations = {
        "*": lambda a, b: (a * b) % p,
        "/": lambda a, b: (a * pow(int(b), p - 2, p)) % p,
        "+": lambda a, b: (a + b) % p,
        "-": lambda a, b: (a - b) % p,
    }

    if op not in operations:
        raise ValueError("Unsupported operation, choose from ['*', '/', '+', '-']")

    x_pairs = np.array(
        [(a, b) for a in range(p) for b in range(1 if op == "/" else 0, p)]
    )
    targets = np.array([operations[op](a, b) for a, b in x_pairs])

    embed = {"*": p, "/": p, "+": p, "-": p, "=": p + 1}
    # inputs = np.array([[a, embed[op], b, embed["="]] for (a, b) in x_pairs])
    inputs = np.array([[a, b, embed["="]] for (a, b) in x_pairs])

    return inputs, targets

def ground_truth_data_generator_torch(
    p: int,
    op: str,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device)

    # Define operations
    def mul(a, b): return (a * b) % p
    def add(a, b): return (a + b) % p
    def sub(a, b): return (a - b) % p
    def div(a, b): return (a * pow(b, p - 2, p)) % p  # b inverse via Fermat

    operations = {
        "*": mul,
        "/": div,
        "+": add,
        "-": sub,
    }

    if op not in operations:
        raise ValueError("Unsupported operation, choose from ['*', '/', '+', '-']")

    # Create all (a, b) pairs
    a = torch.arange(p, device=device)
    b_start = 1 if op == "/" else 0
    b = torch.arange(b_start, p, device=device)

    # Cartesian product
    aa = a.repeat_interleave(len(b))
    bb = b.repeat(len(a))

    x_pairs = torch.stack([aa, bb], dim=1)  # shape: (N, 2)
    print("x_pairs shape:", x_pairs.shape, "device:", x_pairs.device)

    # Compute targets
    if op == "/":
        # Need modular inverse in tensor form
        inv_b = torch.tensor(
            [pow(int(x), p - 2, p) for x in bb.tolist()],
            device=device
        )
        targets = (aa * inv_b) % p
    else:
        targets = operations[op](aa, bb)

    # Embedding tokens
    embed = {"*": p, "/": p, "+": p, "-": p, "=": p + 1}

    # Build inputs
    eq_token = torch.full((x_pairs.shape[0], 1), embed["="], device=device)
    inputs = torch.cat([x_pairs, eq_token], dim=1)

    return inputs, targets


def build_grokking_dataloaders(
    p: int = TrainerConfig.p,
    op: str = TrainerConfig.op,
    train_fraction: float = TrainerConfig.train_fraction,
    batch_size: int = TrainerConfig.batch_size,
    seed: int = TrainerConfig.seed,
    num_workers: int = TrainerConfig.num_workers,
):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    inputs, targets = ground_truth_data_generator(p, op)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(inputs))
    n_train = int(train_fraction * len(inputs))

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_dataset = GrokkingModularDataset(inputs[train_idx], targets[train_idx])
    test_dataset = GrokkingModularDataset(inputs[test_idx], targets[test_idx])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def grokking_data_torch(
    p: int = TrainerConfig.p,
    op: str = TrainerConfig.op,
    train_fraction: float = TrainerConfig.train_fraction,
    batch_size: int = TrainerConfig.batch_size,
    seed: int = TrainerConfig.seed,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    inputs, targets = ground_truth_data_generator(p, op)

    n_train = int(train_fraction * len(inputs))
    indices = np.random.permutation(len(inputs))
    x_train, t_train = inputs[indices[:n_train]], targets[indices[:n_train]]
    x_test, t_test = inputs[indices[n_train:]], targets[indices[n_train:]]

    x_train_torch = torch.tensor(x_train, dtype=torch.long, device=device)
    t_train_torch = torch.tensor(t_train, dtype=torch.long, device=device)
    x_test_torch = torch.tensor(x_test, dtype=torch.long, device=device)
    t_test_torch = torch.tensor(t_test, dtype=torch.long, device=device)

    return x_train_torch, t_train_torch, x_test_torch, t_test_torch


if __name__ == "__main__":
    train_loader, test_loader = build_grokking_dataloaders(113, op="/", batch_size=8)
    x_batch, t_batch = next(iter(train_loader))
    print("Train batches:", len(train_loader), "Test batches:", len(test_loader))
    print("Sample batch shapes:", x_batch.shape, t_batch.shape)
