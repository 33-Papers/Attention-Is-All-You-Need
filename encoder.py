import torch.nn as nn
from rc import RC
from mha import MHA
from ffn import FFN


class Encoder(nn.Module):

    class Layer(nn.Module):
        def __int__(self, hidden_dim, head_num, ffn_dim):
            super().__init__()
            self.mha_self = MHA(hidden_dim, head_num)
            self.rc1 = RC(hidden_dim)
            self.ffn = FFN(hidden_dim, ffn_dim)
            self.rc2 = RC(hidden_dim)

        def forward(self, x, enc_mask):
            q = k = v = x
            x = self.rc1(x, self.mha_self(q, k, v, enc_mask))
            o = self.rc2(x, self.ffn(x))
            return o

    def __init__(self, layer_num, hidden_dim, head_num, ffn_dim):
        super().__init__()
        self.layer_modules = nn.ModuleList(
            [Encoder.Layer(hidden_dim, head_num, ffn_dim) for _ in range(layer_num)]
        )

    def forward(self, x, enc_mask):
        for layer_module in self.layer_modules:
            x = layer_module(x, enc_mask)
        return x