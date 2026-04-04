import torch
import torch.nn as nn
import torch.nn.functional as F


EMBED_DIM = 512
NUM_HEADS = 8
EXPAND_SIZE = 2048
ATTENTION_DROPOUT = 0.2
OUTPUT_DROPOUT = 0.2
FEEDFORWARD_DROPOUT = 0.2
SEQ_LEN = 4  # For modular arithmetic we just have a op b =
NUM_LAYERS = 1


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        attn_drop=ATTENTION_DROPOUT,
        out_drop=OUTPUT_DROPOUT,
    ):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim**-0.5

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), "Embedding size needs to be divisible by heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x, mask=None):
        B, seq_length, embed_dim = x.shape

        assert (
            embed_dim == self.embed_dim
        ), "Input embedding dimension must match model embedding dimension"

        # Split the embedding into self.num_heads different pieces
        q = (
            self.query(x)
            .view(B, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(x)
            .view(B, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(x)
            .view(B, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # pass q and k through activation

        # Scaled dot product attention
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        if mask is not None:
            scores = scores + mask  # broadcast
        attn = torch.softmax(scores, dim=-1)

        # optinally add dropout to attention weights here
        out = self.attn_drop(attn)

        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)

        # (b, heads, n, dim_head) -> (b, n, heads*dim_head)
        out = out.transpose(1, 2).reshape(B, seq_length, embed_dim)
        out = self.fc_out(out)

        out = self.out_drop(out)
        return out


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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        context_size: int = SEQ_LEN,
        expand_size: int = EXPAND_SIZE,
        attention: nn.Module = SelfAttention,
        act: nn.Module = nn.GELU,
        attn_drop: float = ATTENTION_DROPOUT,
        out_drop: float = OUTPUT_DROPOUT,
        ffn_drop: float = FEEDFORWARD_DROPOUT,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            out_drop=out_drop,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(
            hidden_size=embed_dim,
            expand_size=expand_size,
            act=act,
            drop=ffn_drop,
            bias=True,
        )

        self.register_buffer("causal_mask", self._causal_mask(context_size))

    def _causal_mask(self, n):
        # shape: (1, 1, n, n) for broadcasting in multi-head attention
        # Required for torch.export compatibility with shape inference
        mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # (n, n) -> (1, 1, n, n)

    def forward(self, x: torch.Tensor):
        # normalize input then add residual to attention output
        x = x + self.attn(self.norm1(x), mask=self.causal_mask)
        # normalize input then add residual to feedforward output
        return x + self.ffn(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int = NUM_LAYERS, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, kwargs.get("embed_dim", EMBED_DIM))
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(**kwargs) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(kwargs.get("embed_dim", EMBED_DIM))

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        for layer in self.transformer_blocks:
            x = layer(x)
        x = self.layer_norm(x)
        return x


class GrokModularModel(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int = NUM_LAYERS, **kwargs):
        super().__init__()
        self.transformer = Transformer(vocab_size, num_layers, **kwargs)
        self.output_layer = nn.Linear(
            kwargs.get("embed_dim", EMBED_DIM), vocab_size, bias=False
        )

    def forward(self, x: torch.Tensor):
        x = self.transformer(x)
        return self.output_layer(x)[:, -1]
