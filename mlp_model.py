import torch
import torch.nn as nn
import torch.nn.functional as F

class GrokMLP(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed_a = nn.Embedding(vocab_size, vocab_size)
        self.embed_b = nn.Embedding(vocab_size, vocab_size)
        self.layers = nn.Sequential(
            nn.Linear(vocab_size, vocab_size),
            nn.ReLU(),
            nn.Linear(vocab_size, vocab_size),
            nn.ReLU()
        )
        self.fc_out = nn.Linear(vocab_size, vocab_size)

    def forward(self, x):
        a = self.embed_a(x[:, 0])
        b = self.embed_b(x[:, 1])
        x = a + b
        # print(x.shape)
        x = self.layers(x)
        x = self.fc_out(x)
        return x