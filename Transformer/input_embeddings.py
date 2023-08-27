import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    Generate input Embeddings with a specified vocabulary size & Embedding dimension
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        # i.e. Dot Product Scaling
        return self.embedding(x) * math.sqrt(self.d_model)
