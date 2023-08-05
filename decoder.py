import torch.nn as nn
from decoder_block import DecoderBlock
import torch
from encoder_utils import batch_size


class Decoder(nn.Module):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim, target_vocab_size,
                 max_seq_len, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(target_vocab_size, self.d_model)
        self.pos_embed = nn.Embedding(max_seq_len, self.d_model)

        self.dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList(
            [DecoderBlock(self.d_model, num_heads, hidden_dim, dropout_rate) for _ in range(num_blocks)])

    def forward(self, encoder_output, target, training, decoder_mask, memory_mask):
        token_embeds = self.token_embed(target)

        # Generate position indices.
        num_pos = target.shape[0] * self.max_seq_len
        pos_idx = torch.arange(self.max_seq_len).unsqueeze(0).repeat(batch_size, 1)
        pos_idx = torch.reshape(pos_idx, target.shape)

        pos_embeds = self.pos_embed(pos_idx)

        x = self.dropout(token_embeds + pos_embeds, training=training)

        for block in self.blocks:
            x, weights = block(encoder_output, x, training, decoder_mask, memory_mask)

        return x, weights
