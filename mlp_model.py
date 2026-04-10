import torch
import torch.nn as nn
import torch.nn.functional as F

class GrokMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()
        # self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.Sequential(
            nn.Linear(3, 8 * embed_dim),
            nn.SELU(),
            nn.Linear(8 * embed_dim, 4 * embed_dim),
            nn.SELU(),
            nn.Linear(4 * embed_dim, 2 * embed_dim),
            nn.SELU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.SELU(),
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # print(x.shape)
        x = self.layers(x)
        x = self.fc_out(x)
        return x