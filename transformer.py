from encoder import Encoder
from decoder import Decoder
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, num_blocks, d_model, num_heads, hidden_dim, source_vocab_size,
                 target_vocab_size, max_input_len, max_target_len, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_blocks, d_model, num_heads, hidden_dim, source_vocab_size,
                               max_input_len, dropout_rate)

        self.decoder = Decoder(num_blocks, d_model, num_heads, hidden_dim, target_vocab_size,
                               max_target_len, dropout_rate)

        # The final dense layer to generate logits from the decoder output.
        self.output_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, input_seqs, target_input_seqs, training, encoder_mask,
                decoder_mask, memory_mask):
        encoder_output, encoder_attn_weights = self.encoder(input_seqs,
                                                            training, encoder_mask)

        decoder_output, decoder_attn_weights = self.decoder(encoder_output,
                                                            target_input_seqs, training,
                                                            decoder_mask, memory_mask)

        return self.output_layer(decoder_output), encoder_attn_weights, decoder_attn_weights
