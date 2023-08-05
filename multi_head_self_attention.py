import torch.nn as nn
from scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_head = self.d_model // self.num_heads

        self.wq = nn.Linear(self.d_model, self.d_model)
        self.wk = nn.Linear(self.d_model, self.d_model)
        self.wv = nn.Linear(self.d_model, self.d_model)

        # Linear layer to generate the final output.
        self.dense = nn.Linear(self.d_model, self.d_model)

    def split_heads(self, x):
        batch_size = x.size(0)

        split_inputs = x.view(batch_size, -1, self.num_heads, self.d_head)
        return split_inputs.permute(0, 2, 1, 3)

    def merge_heads(self, x):
        batch_size = x.size(0)

        merged_inputs = x.permute(0, 2, 1, 3)
        return merged_inputs.reshape(batch_size, -1, self.d_model)

    def forward(self, q, k, v, mask):
        qs = self.wq(q)
        ks = self.wk(k)
        vs = self.wv(v)

        qs = self.split_heads(qs)
        ks = self.split_heads(ks)
        vs = self.split_heads(vs)

        output, attn_weights = scaled_dot_product_attention(qs, ks, vs, mask)
        output = self.merge_heads(output)

        return self.dense(output), attn_weights
