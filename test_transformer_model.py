import torch
from transformer_model import GrokModularModel


def test_transformer_model():
    kwargs = {
        "num_layers": 1,
        "embed_dim": 512,
        "num_heads": 8,
        "vocab_size": 115,
        "context_size": 4,  # typically X shape is (N, 4) for [a, op, b, '=']
    }

    model = GrokModularModel(**kwargs)
    B = 8
    x = torch.randint(0, kwargs["vocab_size"], (B, kwargs["context_size"]))
    output = model(x)
    assert output.shape == (
        B,
        kwargs["context_size"],
        kwargs["vocab_size"],
    ), f"Expected output shape {(B, kwargs['context_size'], kwargs['vocab_size'])}, got {output.shape}"


if __name__ == "__main__":
    test_transformer_model()
    print("All tests passed!")
