import torch.nn as nn
from multi_head_self_attention import MultiHeadSelfAttention
from feed_forward import feed_forward_network


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.mhsa = MultiHeadSelfAttention(d_model, num_heads)  # You'll need to define MultiHeadSelfAttention
        self.ffn = feed_forward_network(d_model, hidden_dim)  # You'll need to define FeedForwardNetwork

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        mhsa_output, attn_weights = self.mhsa(x, x, x, mask)
        mhsa_output = self.dropout1(mhsa_output)
        mhsa_output = self.layernorm1(x + mhsa_output)

        ffn_output = self.ffn(mhsa_output)
        ffn_output = self.dropout2(ffn_output)
        output = self.layernorm2(mhsa_output + ffn_output)

        return output, attn_weights
