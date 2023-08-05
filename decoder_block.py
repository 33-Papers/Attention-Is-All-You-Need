import torch.nn as nn
from multi_head_self_attention import MultiHeadSelfAttention
from feed_forward import feed_forward_network


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout_rate=0.1):
        super(DecoderBlock, self).__init__()

        self.mhsa1 = MultiHeadSelfAttention(d_model, num_heads)
        self.mhsa2 = MultiHeadSelfAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, hidden_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    # Note the decoder block takes two masks. One for the first MHSA, another
    # for the second MHSA.
    def forward(self, encoder_output, target, training, decoder_mask, memory_mask):
        mhsa_output1, attn_weights = self.mhsa1(target, target, target, decoder_mask)
        mhsa_output1 = self.dropout1(mhsa_output1, training=training)
        mhsa_output1 = self.layernorm1(mhsa_output1 + target)

        mhsa_output2, attn_weights = self.mhsa2(mhsa_output1, encoder_output,
                                                encoder_output,
                                                memory_mask)
        mhsa_output2 = self.dropout2(mhsa_output2, training=training)
        mhsa_output2 = self.layernorm2(mhsa_output2 + mhsa_output1)

        ffn_output = self.ffn(mhsa_output2)
        ffn_output = self.dropout3(ffn_output, training=training)
        output = self.layernorm3(ffn_output + mhsa_output2)

        return output, attn_weights
