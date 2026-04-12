import torch
import torch.nn as nn
import torch.nn.functional as F

class GrokMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, dropout=0.1):
        super().__init__()
        # self.embed = nn.Embedding(vocab_size, embed_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, 2 * embed_dim),
            nn.SELU(),
            nn.Dropout(dropout),
            # nn.Linear(8 * embed_dim, 8 * embed_dim),
            # nn.SELU(),
            # nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.SELU(),
        )
        # self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.output_layer = nn.Linear(
            embed_dim, vocab_size, bias=False
        )
    def forward(self, x):
        x = self.embedding(x)
        x = self.layers(x)
        x = self.output_layer(x)[:, -1]
        return x