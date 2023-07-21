import torch
import torch.nn as nn
import math


class EMB(nn.Module):
    def __int__(self, num_embeddings, embedding_dim, max_len=60, cross_idx=True, dropout_rate=0.1):

        super().__int__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.cross_idx = cross_idx
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lut = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
        pe_slots = self.get_pe_slots(seq_len=self.max_len, feat_dim=embedding_dim, cross_idx=self.cross_idx)
        self.register_buffer("pe_slots", pe_slots)

    # staticmethod to generate positional encodings
    # We will later add this with the input embeddings
    @staticmethod
    def get_pe_slots(seq_len, feat_dim, cross_idx):
        assert feat_dim % 2 == 0, "feat_dim must be even number but it is %s" % feat_dim
        pe_slots = torch.zeros(seq_len, feat_dim)
        pos = torch.arange(0, seq_len).unsqueeze(1)
        even_dim = torch.arange(0, feat_dim, 2)
        power = -even_dim * math.log(10000) / feat_dim
        if cross_idx:
            pe_slots[:, 0::2] = torch.sin(torch.mul(pos, torch.exp(power)))
            pe_slots[:, 1::2] = torch.cos(torch.mul(pos, torch.exp(power)))
        else:
            pe_slots[:, 0:feat_dim / 2] = torch.sin(pos * torch.exp(power))
            pe_slots[:, feat_dim / 2:] = torch.cos(pos * torch.exp(power))
        return pe_slots

    def value_encoding(self, x, use_dim_sqrt_weight=True):
        if not use_dim_sqrt_weight:
            return self.lut(x)
        return self.lut(x) * math.sqrt(self.embedding_dim)

    def positional_encoding(self, x):
        return x + self.pe_slots.unsqueeze(0)[:, :x.size(1), :].requires_grad(False)

    def forward(self, x):
        x = self.value_encoding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        return x
