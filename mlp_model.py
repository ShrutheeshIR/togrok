import torch
import torch.nn as nn
import torch.nn.functional as F



class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expand_size: int,
        act: nn.Module = nn.GELU,
        drop: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        # project input to expanded dimension
        self.fc1 = nn.Linear(hidden_size, expand_size, bias=bias)
        # activation function to introduce non-linearity
        self.act = act()
        # project back to the input dimension
        self.fc2 = nn.Linear(expand_size, hidden_size, bias=bias)
        # optional dropout layer to prevent overfitting
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)  # apply first linear layer
        x = self.act(x)  # apply activation function
        x = self.fc2(x)  # apply second linear layer
        x = self.drop(x)  # optionally apply dropout layer
        return x

class MLPGrokBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        expand_size: int = 2048,
        act: nn.Module = nn.GELU,
        ffn_drop: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = FeedForward(
            hidden_size=embed_dim,
            expand_size=embed_dim,
            act=act,
            drop=ffn_drop,
            bias=False,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn2 = FeedForward(
            hidden_size=embed_dim,
            expand_size=expand_size,
            act=act,
            drop=ffn_drop,
            bias=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.ffn1(self.norm1(x))
        # normalize input then add residual to feedforward output
        return x + self.ffn2(self.norm2(x))


class MLPGrokModel(nn.Module):
    def __init__(
        self, vocab_size: int, num_layers: int = 2, **kwargs
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, kwargs.get("embed_dim", 256))
        # self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [MLPGrokBlock(**kwargs) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(kwargs.get("embed_dim", 256))
        self.output_layer = nn.Linear(
            kwargs.get("embed_dim", 256), vocab_size, bias=False
        )

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.layer_norm(x)
        return self.output_layer(x)[:, -1]


class GrokMLP(nn.Module):
    def __init__(self, vocab_size, num_layers: int = 2, embed_dim: int = 256):
        super().__init__()
        self.embed_a = nn.Embedding(vocab_size, embed_dim)
        self.embed_b = nn.Embedding(vocab_size, embed_dim)


        # a single layer is just a linear transformation followed by a non-linearity, which is equivalent to a feedforward network with no hidden layer
        self.layer_block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias = True),
            nn.GELU(),
        )

        self.layers = nn.Sequential(*[self.layer_block for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        a = self.embed_a(x[:, 0])
        b = self.embed_b(x[:, 1])
        x = a + b
        x = self.layers(x)
        return x