import torch
import torch.nn as nn
from encoder_block import EncoderBlock
from encoder_utils import batch_size, seq_len, pos_idx


class Encoder(nn.Module):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim, src_vocab_size,
                 max_seq_len, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(src_vocab_size, self.d_model)
        self.pos_embed = nn.Embedding(max_seq_len, self.d_model)

        # The original Attention Is All You Need paper applied dropout to the
        # input before feeding it to the first encoder block.
        self.dropout = nn.Dropout(dropout_rate)

        # Create encoder blocks.
        self.blocks = nn.ModuleList([
            EncoderBlock(self.d_model, num_heads, hidden_dim, dropout_rate)
            for _ in range(num_blocks)
        ])

    def forward(self, inputs, training, mask):
        token_embeds = self.token_embed(inputs)

        # Generate position indices for a batch of input sequences.
        num_pos = inputs.size(0) * self.max_seq_len
        torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        pos_embeds = self.pos_embed(pos_idx)

        x = self.dropout(token_embeds + pos_embeds)

        # Run input through successive encoder blocks.
        weights = []
        for block in self.blocks:
            x, weight = block(x, mask)
            weights.append(weight)

        return x, weights
